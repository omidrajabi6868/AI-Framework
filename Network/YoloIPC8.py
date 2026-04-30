import torch
import torchvision
import torch.nn.functional as F
import numpy as np

from typing import Optional
from torch import nn
from .utils import make_anchors,decode_boxes, dist2bbox, box2dist, bbox_ciou
from .heads import build_task_head
from .SAHI import SAHIWrapper, SAHIConfig

# YOLOIPC8 scaling coefficients
YOLOV8_SCALES = {
    "n": dict(depth=0.33, width=0.25),
    "s": dict(depth=0.33, width=0.50),
    "m": dict(depth=0.67, width=0.75),
    "l": dict(depth=1.00, width=1.00),
    "x": dict(depth=1.33, width=1.25),
}

def make_divisible(x, divisor=8):
    return int((x + divisor / 2) // divisor * divisor)

def yolo_ipc8_postprocess(preds, strides=(8, 16, 32),
                        conf_thres=0.25, iou_thres=0.7, max_det=300):
    if isinstance(preds, dict):
        preds = preds.get("det")
    if preds is None:
        raise ValueError("Model output does not contain detection predictions under key 'det'.")

    device = preds[0].device
    bs = preds[0].shape[0]
    reg_max = 16
    nc = preds[0].shape[1] - 4 * reg_max

    outputs = []
    anchor_points, stride_tensor = make_anchors(preds, strides, device)

    for b in range(bs):
        pred_dist, pred_cls = [], []

        for p in preds:
            p = p[b]
            pred_dist.append(
                p[:4 * reg_max].permute(1, 2, 0).reshape(-1, 4 * reg_max)
            )
            pred_cls.append(
                p[4 * reg_max:].permute(1, 2, 0).reshape(-1, nc)
            )

        pred_dist = torch.cat(pred_dist)
        pred_cls = torch.cat(pred_cls).sigmoid()

        assert pred_dist.shape[0] == anchor_points.shape[0]

        boxes = decode_boxes(pred_dist, anchor_points, stride_tensor, reg_max)

        scores, labels = pred_cls.max(dim=1)
        mask = scores > conf_thres

        boxes = boxes[mask].float()
        scores = scores[mask]
        labels = labels[mask]

        if boxes.numel() == 0:
            outputs.append(torch.zeros((0, 6), device=device))
            continue

        keep = torchvision.ops.batched_nms(
            boxes, scores, labels, iou_thres
        )[:max_det]

        outputs.append(torch.cat([
            boxes[keep],
            scores[keep, None],
            labels[keep, None].float()
        ], dim=1))

    return outputs

# ---------------------------
# Yolo Predictor
# ---------------------------
class IPC8Predictor:
    def __init__(
        self,
        model,
        strides=(8, 16, 32),
        max_input_size=1024,
        class_names=None,
        conf_thres=0.25,
        iou_thres=0.7,
        device="cpu",
        mask_threshold: float = 0.5,
        enable_masks: bool = True,
        sahi_config: Optional[SAHIConfig] = None,
    ):
        self.model = model.to(device).eval()
        self.strides = strides
        self.class_names = class_names
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.max_input_size = max_input_size
        self.mask_threshold = float(mask_threshold)
        self.enable_masks = bool(enable_masks)
        self.sahi = None

        if sahi_config is not None:
            self.sahi = self._build_sahi_wrapper(sahi_config)

    def _to_nchw_float(self, image_input):
        if isinstance(image_input, np.ndarray):
            # Ensure positive, contiguous strides for torch.from_numpy.
            t = torch.from_numpy(np.ascontiguousarray(image_input))
        elif torch.is_tensor(image_input):
            t = image_input
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")

        if t.ndim == 4:
            if t.shape[0] != 1:
                raise ValueError("IPC8Predictor only supports a single image at a time.")
            image = t
        elif t.ndim == 3:
            if t.shape[0] in (1, 3):
                image = t.unsqueeze(0)
            else:
                image = t.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError("Expected image input with 3 or 4 dimensions.")

        image = image.to(self.device).float()
        if image.max() > 1.0:
            image = image / 255.0
        return image

    def _preprocess_for_model(self, image_input):
        """Preprocess a single image for the model.

        Returns:
            image_tensor: Tensor ready for model input.
            meta: dict with details needed for postprocessing.
        """
        image = self._to_nchw_float(image_input)
        image, (orig_h, orig_w), resize_scale = self._resize_to_max_side(
            image, max_side=self.max_input_size
        )
        image, _, (pad_left, pad_top) = self._pad_to_stride(image)

        meta = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "resize_scale": resize_scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
        }
        return image, meta

    def _resize_to_max_side(self, image: torch.Tensor, max_side: int = 10):
        _, _, orig_h, orig_w = image.shape
        largest_side = max(orig_h, orig_w)
        if largest_side <= 0:
            raise ValueError("Invalid image size for resizing.")
        scale = float(max_side) / float(largest_side)
        new_h = max(1, int(round(orig_h * scale)))
        new_w = max(1, int(round(orig_w * scale)))
        if new_h != orig_h or new_w != orig_w:
            image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return image, (orig_h, orig_w), scale

    def _pad_to_stride(self, image: torch.Tensor):
        _, _, h, w = image.shape
        stride = max(int(s) for s in self.strides)
        pad_h = (stride - (h % stride)) % stride
        pad_w = (stride - (w % stride)) % stride

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        if pad_h == 0 and pad_w == 0:
            return image, (h, w), (left, top)

        image = F.pad(image, (left, right, top, bottom), value=114.0 / 255.0)
        return image, (h, w), (left, top)

    def _postprocess_model_output(self, preds, meta):
        """Postprocess raw model outputs into standard prediction dict."""
        orig_h = meta["orig_h"]
        orig_w = meta["orig_w"]
        resize_scale = meta["resize_scale"]
        pad_left = meta["pad_left"]
        pad_top = meta["pad_top"]

        # --- YOLOIPC8 postprocess ---
        dets = yolo_ipc8_postprocess(
            preds,
            strides=self.strides,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            max_det=300,
        )

        det = dets[0]

        boxes = det[:, :4].clone()
        scores = det[:, 4]
        classes = det[:, 5].long()

        if boxes.numel() > 0:
            boxes[:, [0, 2]] -= pad_left
            boxes[:, [1, 3]] -= pad_top
            boxes /= resize_scale
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, max(0, orig_w - 1))
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, max(0, orig_h - 1))

        out = {
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "box_format": "xyxy",
            "normalized": False,
            "image_size": (orig_h, orig_w),
        }
        if self.enable_masks and isinstance(preds, dict) and ("mask_coeffs" in preds) and ("proto" in preds):
            masks = self._build_instance_masks(
                boxes=boxes,
                mask_coeffs=preds.get("mask_coeffs"),
                proto=preds.get("proto"),
                orig_h=orig_h,
                orig_w=orig_w,
                device=boxes.device,
            )
            out["masks"] = masks
            out["mask_threshold"] = self.mask_threshold

        return out

    def _build_sahi_wrapper(self, sahi_config: SAHIConfig) -> SAHIWrapper:
        """Build a SAHI wrapper around this predictor's model.

        The wrapper executes the underlying model on tiles and merges the
        per-tile detections into a single set of detections.
        """
        def preprocess_fn(image_slice):
            return self._preprocess_for_model(image_slice)

        def postprocess_fn(raw_output, meta):
            return self._postprocess_model_output(raw_output, meta)

        return SAHIWrapper(
            model_fn=self.model,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn,
            slice_size=sahi_config.slice_size,
            overlap=sahi_config.overlap,
            merge_mode=sahi_config.merge_mode,
            nms_threshold=sahi_config.nms_threshold,
            class_agnostic_nms=sahi_config.class_agnostic_nms,
            device=self.device,
            min_box_area=getattr(sahi_config, "min_box_area", 9.0),
            min_aspect_ratio=getattr(sahi_config, "min_aspect_ratio", 0.02),
        )

    @torch.no_grad()
    def __call__(self, image_input):
        if self.sahi is not None:
            return self.sahi(image_input)

        image, meta = self._preprocess_for_model(image_input)
        preds = self.model(image)
        return self._postprocess_model_output(preds, meta)

    def _build_instance_masks(
        self,
        boxes: torch.Tensor,
        mask_coeffs,
        proto,
        orig_h: int,
        orig_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.zeros((0, int(orig_h), int(orig_w)), device=device, dtype=torch.float32)
        if mask_coeffs is None or proto is None:
            return torch.zeros((0, int(orig_h), int(orig_w)), device=device, dtype=torch.float32)
        if not isinstance(mask_coeffs, (list, tuple)) or len(mask_coeffs) == 0:
            return torch.zeros((0, int(orig_h), int(orig_w)), device=device, dtype=torch.float32)

        coeff = mask_coeffs[0]
        if not torch.is_tensor(coeff):
            return torch.zeros((0, int(orig_h), int(orig_w)), device=device, dtype=torch.float32)
        if coeff.ndim != 4:
            return torch.zeros((0, int(orig_h), int(orig_w)), device=device, dtype=torch.float32)

        proto_t = proto[0] if torch.is_tensor(proto) and proto.ndim == 4 else proto
        if not torch.is_tensor(proto_t) or proto_t.ndim != 3:
            return torch.zeros((0, int(orig_h), int(orig_w)), device=device, dtype=torch.float32)

        bsz, nm, hf, wf = coeff.shape
        if bsz == 0:
            return torch.zeros((0, int(orig_h), int(orig_w)), device=device, dtype=torch.float32)

        k = min(int(nm), int(proto_t.shape[0]))
        if k <= 0:
            return torch.zeros((0, int(orig_h), int(orig_w)), device=device, dtype=torch.float32)

        coeff_b = coeff[0, :k].to(device)
        proto_b = proto_t[:k].to(device)

        centers_x = ((boxes[:, 0] + boxes[:, 2]) * 0.5 / float(max(orig_w, 1)) * float(wf))
        centers_y = ((boxes[:, 1] + boxes[:, 3]) * 0.5 / float(max(orig_h, 1)) * float(hf))
        centers_x = centers_x.long().clamp(0, wf - 1)
        centers_y = centers_y.long().clamp(0, hf - 1)
        idx = centers_y * wf + centers_x  # [N]

        coeff_flat = coeff_b.reshape(k, -1).transpose(0, 1)  # [Hf*Wf, k]
        sel_coeff = coeff_flat[idx]  # [N, k]
        mask_logits = torch.einsum("nk,khw->nhw", sel_coeff, proto_b)  # [N, Hp, Wp]
        masks = torch.sigmoid(mask_logits)
        masks = F.interpolate(
            masks.unsqueeze(1),
            size=(int(orig_h), int(orig_w)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return masks

# ---------------------------
# Basic Convolution Layer
# ---------------------------
class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p or k//2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ---------------------------
# Bottleneck
# ---------------------------
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

# ---------------------------
# C2f Block
# ---------------------------
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


# ---------------------------
# SPPF Block
# ---------------------------
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c_*4, c2, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))

# ---------------------------
# Backbone
# ---------------------------
class Backbone(nn.Module):
    def __init__(self, model_size="l"):
        super().__init__()
        s = YOLOV8_SCALES[model_size]
        d, w = s["depth"], s["width"]

        def c(ch): return make_divisible(ch * w)
        def n(rep): return max(1, int(round(rep * d)))

        # Stem
        self.stem = Conv(3, c(64), 3, 2)

        self.stage1 = nn.Sequential(
            Conv(c(64), c(128), 3, 2),
            C2f(c(128), c(128), n(3), shortcut=True)
        )

        # Stage2 (P3)
        self.stage2 = nn.Sequential(
            Conv(c(128), c(256), 3, 2),
            C2f(c(256), c(256), n(6), shortcut=True)
        )

        # Stage3 (P4)
        self.stage3 = nn.Sequential(
            Conv(c(256), c(512), 3, 2),
            C2f(c(512), c(512), n(6), shortcut=True)
        )

         # Stage4 
        self.stage4 = nn.Sequential(
            Conv(c(512), c(1024), 3, 2),
            C2f(c(1024), c(1024), n(3), shortcut=True)
        )

        # (P5)
        self.sppf = SPPF(c(1024), c(1024))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.sppf(self.stage4(p4))
        return p3, p4, p5

# ---------------------------
# Neck
# ---------------------------
class Neck(nn.Module):
    def __init__(self, model_size="l"):
        super().__init__()
        s = YOLOV8_SCALES[model_size]
        d, w = s["depth"], s["width"]
        def n(rep): return max(1, int(round(rep * d)))
        c = lambda x: make_divisible(x * w)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-down
        self.c2f_p4 = C2f(c(1024)+c(512), c(512), n(3))
        self.c2f_p3 = C2f(c(512)+c(256), c(256), n(3))

        # Bottom-up
        self.down_p3 = Conv(c(256), c(256), 3, 2)
        self.c2f_n4 = C2f(c(256)+c(512), c(512), n(3))

        self.down_p4 = Conv(c(512), c(512), 3, 2)
        self.c2f_n5 = C2f(c(512)+c(1024), c(1024), n(3))

    def forward(self, feats):
        p3, p4, p5 = feats

        # Top-down pathway
        p5_up = self.upsample(p5)
        p4_td = self.c2f_p4(torch.cat([p5_up, p4], dim=1))

        p4_up = self.upsample(p4_td)
        p3_td = self.c2f_p3(torch.cat([p4_up, p3], dim=1))

        # Bottom-up pathway
        p3_down = self.down_p3(p3_td)                      # 256 → 256, 48x48
        p4_out = self.c2f_n4(torch.cat([p3_down, p4_td], dim=1)) # 256+512=768 → 512

        p4_down = self.down_p4(p4_out)                     # 512 → 512, 24x24
        p5_out = self.c2f_n5(torch.cat([p4_down, p5], dim=1))     # 512+1024=1536 →1024

        return p3_td, p4_out, p5_out

# ---------------------------
# YOLO Model
# ---------------------------
class YOLOIPC8(nn.Module):

    def __init__(
        self,
        model_size="l",
        num_classes=80,
        reg_max=16,
        task: str = "detection",
        num_masks: int = 32,
        num_protos: int = 32,
    ):
        super().__init__()
        self.task = str(task).lower()
        self.backbone = Backbone(model_size)
        self.neck = Neck(model_size)

        w = YOLOV8_SCALES[model_size]["width"]
        ch = [
            make_divisible(256*w),
            make_divisible(512*w),
            make_divisible(1024*w),
        ]
        self.task_head = build_task_head(
            task=self.task,
            ch=ch,
            nc=num_classes,
            reg_max=reg_max,
            decoupled=False,
            num_masks=num_masks,
            num_protos=num_protos,
        )

    def forward(self, x):
        feats = self.neck(self.backbone(x))
        return self.task_head(feats)

    @property
    def detect(self):
        return self.task_head

# -----------------------------
# DFL Loss
# -----------------------------
class DFLLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred, target):
        """
        pred:   [N, 4, reg_max] logits
        target: [N, 4] float distances (feature-map units)
        """
        target = target.clamp(0, self.reg_max - 1e-4)

        left = target.floor().long()
        right = (left + 1).clamp(max=self.reg_max - 1)

        wl = right.float() - target
        wr = target - left.float()

        pred = pred.view(-1, self.reg_max)
        left = left.view(-1)
        right = right.view(-1)

        loss = (
            F.cross_entropy(pred, left, reduction="none") * wl.view(-1) +
            F.cross_entropy(pred, right, reduction="none") * wr.view(-1)
        )

        return loss.mean()

# -----------------------------
# YOLOIPC8 Task Aligned Assigner
# -----------------------------
class TaskAlignedAssigner:
    def __init__(self, topk=10, alpha=1.0, beta=6.0):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def assign(self, pred_boxes, pred_scores, gt_boxes, gt_labels, anchor_points, stride_tensor):
        device = pred_boxes.device
        N = pred_boxes.shape[0]
        M = gt_boxes.shape[0]

        if stride_tensor.ndim == 1:
            stride_tensor = stride_tensor.unsqueeze(1)

        anchor_points_img = anchor_points * stride_tensor

        if M == 0:
            return (
                torch.zeros(N, dtype=torch.bool, device=device),
                torch.zeros(N, dtype=torch.long, device=device),
                torch.zeros(N, 4, device=device),
                torch.zeros(N, device=device),
            )

        in_gt = (
            (anchor_points_img[:, None, 0] >= gt_boxes[None, :, 0]) &
            (anchor_points_img[:, None, 0] <= gt_boxes[None, :, 2]) &
            (anchor_points_img[:, None, 1] >= gt_boxes[None, :, 1]) &
            (anchor_points_img[:, None, 1] <= gt_boxes[None, :, 3])
        ) 

        ious = bbox_ciou(
            pred_boxes[:, None, :],
            gt_boxes[None, :, :]
        ) 
        assert gt_labels.max() < pred_scores.shape[1]
        cls_scores = pred_scores[:, gt_labels.long()]  # [N, M]

        ious = ious.clamp(min=0)
        alignment = (cls_scores ** self.alpha) * (ious ** self.beta)
        alignment = alignment * in_gt

        eps = 1e-9
        alignment = alignment / (alignment.max(dim=0, keepdim=True).values + eps)
        alignment = alignment * (ious > 0.1)

        fg_mask = torch.zeros(N, dtype=torch.bool, device=device)
        assigned_gt = torch.full((N,), -1, device=device)
        assigned_score = torch.zeros(N, device=device)

        topk = min(self.topk, N)

        for gt_idx in range(M):
            scores = alignment[:, gt_idx]
            topk_scores, idx = scores.topk(topk)

            valid = topk_scores > 0
            idx = idx[valid]

            better = scores[idx] > assigned_score[idx]
            idx = idx[better]

            fg_mask[idx] = True
            assigned_gt[idx] = gt_idx
            assigned_score[idx] = scores[idx]

        # ---------------------------
        # Safe label & box gathering
        # ---------------------------
        labels = torch.zeros(N, dtype=torch.long, device=device)
        boxes = torch.zeros(N, 4, device=device)

        pos = assigned_gt >= 0
        labels[pos] = gt_labels[assigned_gt[pos]]
        boxes[pos] = gt_boxes[assigned_gt[pos]]

        return fg_mask, labels, boxes, assigned_score

# -----------------------------
# YOLOIPC8 Loss (DFL + BCE + CIoU)
# -----------------------------
class YOLOIPC8Loss(nn.Module):
    def __init__(
        self,
        nc,
        reg_max=16,
        class_balance=True,
        class_balance_power=1.0,
        class_balance_min=0.25,
        class_balance_max=4.0,
        class_balance_eps=1e-6,
    ):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max

        self.box_weight = 7.5
        self.cls_weight = 0.5
        self.dfl_weight = 1.5

        self.assigner = TaskAlignedAssigner()
        self.dfl = DFLLoss(reg_max)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.class_balance = bool(class_balance)
        self.class_balance_power = float(class_balance_power)
        self.class_balance_min = float(class_balance_min)
        self.class_balance_max = float(class_balance_max)
        self.class_balance_eps = float(class_balance_eps)

        self.register_buffer("proj", torch.arange(reg_max).float())
        self.strides = [8, 16, 32]

    def _build_batch_class_weights(self, labels_per_image, device):
        weights = torch.ones(self.nc, device=device, dtype=torch.float32)
        if (not self.class_balance) or self.nc <= 1:
            return weights

        all_labels = [lab for lab in labels_per_image if lab.numel() > 0]
        if not all_labels:
            return weights

        labels = torch.cat(all_labels, dim=0)
        counts = torch.bincount(labels, minlength=self.nc).to(torch.float32)
        present = counts > 0
        if not present.any():
            return weights

        mean_count = counts[present].mean()
        raw = (mean_count / (counts[present] + self.class_balance_eps)).pow(self.class_balance_power)
        raw = raw.clamp(min=self.class_balance_min, max=self.class_balance_max)
        weights[present] = raw
        return weights

    def forward(self, preds, targets, return_items=False):
        if isinstance(preds, dict):
            preds = preds.get("det")
        if preds is None:
            raise ValueError("YOLOIPC8Loss expected detection predictions (list) or dict with 'det'.")

        device = preds[0].device
        total_box = torch.zeros(1, device=device)
        total_cls = torch.zeros(1, device=device)
        total_dfl = torch.zeros(1, device=device)
        batch_size = max(len(targets), 1)

        anchor_points, stride_tensor = make_anchors(preds, self.strides, device)
        cleaned_targets = []
        labels_per_image = []
        for target in targets:
            gt_boxes = target["boxes"].to(device).to(torch.float32)
            gt_labels = target["labels"].to(device).to(torch.long)

            valid_label = (gt_labels >= 0) & (gt_labels < self.nc)
            valid_box = (
                torch.isfinite(gt_boxes).all(dim=1)
                & (gt_boxes[:, 2] > gt_boxes[:, 0])
                & (gt_boxes[:, 3] > gt_boxes[:, 1])
            )
            valid = valid_label & valid_box
            gt_boxes = gt_boxes[valid]
            gt_labels = gt_labels[valid]
            cleaned_targets.append((gt_boxes, gt_labels))
            labels_per_image.append(gt_labels)

        class_weights = self._build_batch_class_weights(labels_per_image, device=device)

        for b, (gt_boxes, gt_labels) in enumerate(cleaned_targets):

            pred_dist, pred_cls = [], []

            for p in preds:
                p = p[b]
                pred_dist.append(
                    p[:4*self.reg_max].view(4, self.reg_max, -1).permute(2,0,1)
                )
                pred_cls.append(
                    p[4*self.reg_max:].permute(1,2,0).reshape(-1, self.nc)
                )

            pred_dist = torch.cat(pred_dist)
            pred_cls = torch.cat(pred_cls)

            pred_boxes = decode_boxes(pred_dist, anchor_points, stride_tensor, self.reg_max)

            fg_mask, labels, boxes, scores = self.assigner.assign(
                pred_boxes,
                pred_cls.sigmoid(),
                gt_boxes,
                gt_labels,
                anchor_points,
                stride_tensor
                )

            if fg_mask.sum() == 0:
                total_box += pred_boxes.sum() * 0.0
                total_cls += pred_cls.sum() * 0.0
                total_dfl += pred_dist.sum() * 0.0
                continue

            total_box += (1 - bbox_ciou(
                pred_boxes[fg_mask],
                boxes[fg_mask]
            )).mean()

            dist_targets = box2dist(
                anchor_points[fg_mask],
                boxes[fg_mask],
                stride_tensor[fg_mask],
                self.reg_max
            )

            total_dfl += self.dfl(pred_dist[fg_mask], dist_targets)

            cls_target = torch.zeros_like(pred_cls)
            cls_target[fg_mask, labels[fg_mask]] = scores[fg_mask].to(cls_target.dtype)

            cls_loss = self.bce(pred_cls, cls_target)
            positive_weight = class_weights.view(1, -1).to(cls_loss.dtype)
            cls_scale = torch.ones_like(cls_loss)
            cls_scale = cls_scale + (positive_weight - 1.0) * (cls_target > 0).to(cls_loss.dtype)
            cls_loss = cls_loss * cls_scale

            weighted_cls_target = cls_target * positive_weight
            target_scores_sum = weighted_cls_target.sum().clamp(min=1.0)
            cls_loss = cls_loss.sum() / target_scores_sum
            total_cls += cls_loss

        total_box = total_box / batch_size
        total_cls = total_cls / batch_size
        total_dfl = total_dfl / batch_size

        weighted_box = self.box_weight * total_box
        weighted_cls = self.cls_weight * total_cls
        weighted_dfl = self.dfl_weight * total_dfl
        total_loss = (
            weighted_box +
            weighted_cls +
            weighted_dfl
        ).sum()

        if return_items:
            items = {
                "box": float(weighted_box.detach().item()),
                "cls": float(weighted_cls.detach().item()),
                "dfl": float(weighted_dfl.detach().item()),
            }
            return total_loss, items

        return total_loss


class YOLOIPC8InstanceSegLoss(YOLOIPC8Loss):
    """Detection loss + lightweight instance mask loss."""

    def __init__(
        self,
        nc,
        reg_max=16,
        class_balance=True,
        class_balance_power=1.0,
        class_balance_min=0.25,
        class_balance_max=4.0,
        class_balance_eps=1e-6,
        seg_weight=1.0,
        mask_bce_weight=1.0,
        mask_dice_weight=1.0,
    ):
        super().__init__(
            nc=nc,
            reg_max=reg_max,
            class_balance=class_balance,
            class_balance_power=class_balance_power,
            class_balance_min=class_balance_min,
            class_balance_max=class_balance_max,
            class_balance_eps=class_balance_eps,
        )
        self.seg_weight = float(seg_weight)
        self.mask_bce_weight = float(mask_bce_weight)
        self.mask_dice_weight = float(mask_dice_weight)

    def _instance_mask_loss(self, preds, targets):
        default_device = targets[0]["boxes"].device if len(targets) > 0 else torch.device("cpu")
        if not isinstance(preds, dict):
            # Fallback to zero seg loss if model is detection-only.
            return torch.zeros(1, device=default_device)

        mask_coeffs = preds.get("mask_coeffs")
        proto = preds.get("proto")
        det = preds.get("det")
        if mask_coeffs is None or proto is None or det is None:
            if isinstance(det, list) and len(det) > 0 and torch.is_tensor(det[0]):
                device = det[0].device
            elif torch.is_tensor(proto):
                device = proto.device
            else:
                device = default_device
            return torch.zeros(1, device=device)
        if len(mask_coeffs) == 0:
            return torch.zeros(1, device=proto.device)

        coeff = mask_coeffs[0]  # [B, Nm, Hf, Wf]
        bsz, nm, hf, wf = coeff.shape
        hp, wp = proto.shape[-2], proto.shape[-1]
        k = min(int(nm), int(proto.shape[1]))

        total = torch.zeros(1, device=proto.device)
        n_inst = 0
        eps = 1.0

        # Infer image size from per-image detection targets (absolute xyxy).
        for b in range(min(bsz, len(targets))):
            tgt = targets[b]
            boxes = tgt.get("boxes")
            masks = tgt.get("masks")
            if boxes is None or masks is None or boxes.numel() == 0 or masks.numel() == 0:
                continue

            boxes = boxes.to(proto.device, dtype=torch.float32)
            masks = masks.to(proto.device, dtype=torch.float32)
            if masks.ndim != 3:
                continue

            n = min(int(boxes.shape[0]), int(masks.shape[0]))
            if n <= 0:
                continue
            boxes = boxes[:n]
            masks = masks[:n]

            img_h = float(max(int(masks.shape[-2]), 1))
            img_w = float(max(int(masks.shape[-1]), 1))

            gt_masks = F.interpolate(
                masks.unsqueeze(1),
                size=(hp, wp),
                mode="nearest",
            ).squeeze(1)

            centers_x = ((boxes[:, 0] + boxes[:, 2]) * 0.5 / img_w * wf).long().clamp(0, wf - 1)
            centers_y = ((boxes[:, 1] + boxes[:, 3]) * 0.5 / img_h * hf).long().clamp(0, hf - 1)
            idx = centers_y * wf + centers_x  # [N]

            coeff_flat = coeff[b, :k].reshape(k, -1).transpose(0, 1)  # [Hf*Wf, k]
            sel_coeff = coeff_flat[idx]  # [N, k]
            pred_logits = torch.einsum("nk,khw->nhw", sel_coeff, proto[b, :k])  # [N, Hp, Wp]

            bce_per = F.binary_cross_entropy_with_logits(
                pred_logits,
                gt_masks,
                reduction="none",
            ).mean(dim=(1, 2))
            pred_prob = pred_logits.sigmoid()
            inter = (pred_prob * gt_masks).sum(dim=(1, 2))
            union = pred_prob.sum(dim=(1, 2)) + gt_masks.sum(dim=(1, 2))
            dice_per = 1.0 - ((2.0 * inter + eps) / (union + eps))

            inst_loss = self.mask_bce_weight * bce_per + self.mask_dice_weight * dice_per
            total = total + inst_loss.sum()
            n_inst += int(n)

        if n_inst <= 0:
            return total * 0.0
        return total / float(n_inst)

    def forward(self, preds, targets, return_items=False):
        det_preds = preds.get("det") if isinstance(preds, dict) else preds
        det_loss, det_items = super().forward(det_preds, targets, return_items=True)
        seg_loss = self._instance_mask_loss(preds, targets)
        total = det_loss + self.seg_weight * seg_loss

        if return_items:
            items = dict(det_items)
            items["seg"] = float((self.seg_weight * seg_loss).detach().item())
            return total, items
        return total


        det = preds.get("det")
        mask_coeffs = preds.get("mask_coeffs")
        proto = preds.get("proto")
        if det is None or mask_coeffs is None or proto is None:
            raise ValueError("Segmentation ONNX wrapper expected det, mask_coeffs, and proto.")

        # Reuse detection decoding from base wrapper.
        anchor_points, stride_tensor = make_anchors(det, self.strides, x.device)

        pred_dist = []
        pred_cls = []
        for p in det:
            b = p.shape[0]
            pred_dist.append(
                p[:, : 4 * self.reg_max].permute(0, 2, 3, 1).reshape(b, -1, 4 * self.reg_max)
            )
            pred_cls.append(
                p[:, 4 * self.reg_max :].permute(0, 2, 3, 1).reshape(b, -1, self.num_classes)
            )

        pred_dist = torch.cat(pred_dist, dim=1)
        pred_cls = torch.cat(pred_cls, dim=1).sigmoid()

        b, n, _ = pred_dist.shape
        pred_dist = pred_dist.view(b, n, 4, self.reg_max).softmax(dim=-1)
        proj = torch.arange(self.reg_max, device=pred_dist.device, dtype=pred_dist.dtype).view(1, 1, 1, -1)
        dist = (pred_dist * proj).sum(dim=-1)

        anchors = anchor_points.view(1, n, 2)
        stride = stride_tensor.view(1, n, 1)

        x1y1 = anchors - dist[:, :, :2]
        x2y2 = anchors + dist[:, :, 2:]
        boxes = torch.cat([x1y1, x2y2], dim=-1) * stride

        boxes[:, :, [0, 2]] = (boxes[:, :, [0, 2]] - float(self.pad_left)) / float(self.resize_scale)
        boxes[:, :, [1, 3]] = (boxes[:, :, [1, 3]] - float(self.pad_top)) / float(self.resize_scale)

        zero = torch.tensor(0.0, device=boxes.device, dtype=boxes.dtype)
        max_w = torch.maximum(orig_w - 1.0, zero)
        max_h = torch.maximum(orig_h - 1.0, zero)
        boxes_x = boxes[:, :, [0, 2]]
        boxes_y = boxes[:, :, [1, 3]]
        boxes_x = torch.minimum(torch.maximum(boxes_x, zero), max_w)
        boxes_y = torch.minimum(torch.maximum(boxes_y, zero), max_h)
        boxes = torch.cat([boxes_x[:, :, 0:1], boxes_y[:, :, 0:1], boxes_x[:, :, 1:2], boxes_y[:, :, 1:2]], dim=-1)

        if self.normalize_boxes:
            x1 = boxes[:, :, 0:1] / orig_w
            y1 = boxes[:, :, 1:2] / orig_h
            x2 = boxes[:, :, 2:3] / orig_w
            y2 = boxes[:, :, 3:4] / orig_h
            boxes = torch.cat([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)

        det_out = torch.cat([boxes, pred_cls], dim=-1)
        # Return det + per-level mask coeffs + proto
        return (det_out, mask_coeffs[0], mask_coeffs[1], mask_coeffs[2], proto)
