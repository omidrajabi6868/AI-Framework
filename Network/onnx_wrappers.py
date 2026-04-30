from typing import Tuple

import torch
import torch.nn as nn
from pathlib import Path
import json
import torch.nn.functional as F

from .utils import make_anchors


SEG_OUTPUT_SUFFIXES = ["mask_coeffs_p3", "mask_coeffs_p4", "mask_coeffs_p5", "proto"]


def build_output_names(model_output_name: str, export_segmentation: bool) -> list:
    if export_segmentation:
        return [model_output_name] + SEG_OUTPUT_SUFFIXES
    return [model_output_name]


def _normalize_input_if_needed(x: torch.Tensor, normalize_input: bool) -> torch.Tensor:
    if not normalize_input:
        return x
    max_val = x.max()
    scale = torch.where(
        max_val > 1.0,
        torch.tensor(255.0, device=x.device, dtype=x.dtype),
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
    )
    return x / scale


def _clamp_and_normalize_boxes(
    boxes: torch.Tensor,
    orig_h: torch.Tensor,
    orig_w: torch.Tensor,
    normalize_boxes: bool,
) -> torch.Tensor:
    zero = torch.tensor(0.0, device=boxes.device, dtype=boxes.dtype)
    max_w = torch.maximum(orig_w - 1.0, zero)
    max_h = torch.maximum(orig_h - 1.0, zero)
    boxes_x = boxes[:, :, [0, 2]]
    boxes_y = boxes[:, :, [1, 3]]
    boxes_x = torch.minimum(torch.maximum(boxes_x, zero), max_w)
    boxes_y = torch.minimum(torch.maximum(boxes_y, zero), max_h)
    boxes = torch.cat([boxes_x[:, :, 0:1], boxes_y[:, :, 0:1], boxes_x[:, :, 1:2], boxes_y[:, :, 1:2]], dim=-1)

    if normalize_boxes:
        x1 = boxes[:, :, 0:1] / orig_w
        y1 = boxes[:, :, 1:2] / orig_h
        x2 = boxes[:, :, 2:3] / orig_w
        y2 = boxes[:, :, 3:4] / orig_h
        boxes = torch.cat([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)
    return boxes


class BaseONNXWrapper(nn.Module):
    def __init__(self, model, normalize_input: bool = True):
        super().__init__()
        self.model = model.eval()
        self.normalize_input = bool(normalize_input)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return _normalize_input_if_needed(x, self.normalize_input)


class IPC8ONNXExportWrapper(BaseONNXWrapper):
    def __init__(
        self,
        model,
        num_classes,
        reg_max,
        strides,
        normalize_boxes=True,
        normalize_input=True,
        max_input_side=1024,
        pad_to_stride=True,
        input_size=(640, 640),
        resize_to_max_side: bool = True,
    ):
        super().__init__(model=model, normalize_input=normalize_input)
        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self.strides = tuple(int(s) for s in strides)
        self.normalize_boxes = bool(normalize_boxes)
        self.max_input_side = int(max_input_side)
        self.pad_to_stride = bool(pad_to_stride)
        self.resize_to_max_side = bool(resize_to_max_side)
        self.input_h = int(input_size[0])
        self.input_w = int(input_size[1])
        self._build_static_preprocess_params()

    def _build_static_preprocess_params(self):
        if not self.resize_to_max_side:
            self.resize_scale = 1.0
            self.resized_h = int(self.input_h)
            self.resized_w = int(self.input_w)
        else:
            largest_side = max(self.input_h, self.input_w)
            if largest_side <= 0:
                raise ValueError("Invalid input size for preprocessing.")
            self.resize_scale = float(self.max_input_side) / float(largest_side)
            self.resized_h = max(1, int(round(self.input_h * self.resize_scale)))
            self.resized_w = max(1, int(round(self.input_w * self.resize_scale)))

        self.pad_left = 0
        self.pad_top = 0
        self.pad_right = 0
        self.pad_bottom = 0
        if self.pad_to_stride:
            stride = max(self.strides)
            pad_h = (stride - (self.resized_h % stride)) % stride
            pad_w = (stride - (self.resized_w % stride)) % stride
            self.pad_top = pad_h // 2
            self.pad_bottom = pad_h - self.pad_top
            self.pad_left = pad_w // 2
            self.pad_right = pad_w - self.pad_left

    def _preprocess_input(self, x):
        x = self._normalize_input(x)
        if self.resize_to_max_side and self.resize_scale != 1.0:
            x = F.interpolate(x, size=(self.resized_h, self.resized_w), mode="bilinear", align_corners=False)
        if self.pad_left or self.pad_right or self.pad_top or self.pad_bottom:
            x = F.pad(x, (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom), value=114.0 / 255.0)
        return x

    def forward(self, x):
        # x: [B,3,H,W]
        orig_shape = torch._shape_as_tensor(x)
        orig_h = orig_shape[2].to(dtype=x.dtype, device=x.device)
        orig_w = orig_shape[3].to(dtype=x.dtype, device=x.device)

        x = self._preprocess_input(x)
        preds = self.model(x)
        if isinstance(preds, dict):
            preds = preds.get("det")
        if preds is None:
            raise ValueError("ONNX export wrapper expected detection predictions or dict with 'det'.")
        anchor_points, stride_tensor = make_anchors(preds, self.strides, x.device)

        pred_dist = []
        pred_cls = []
        for p in preds:
            b = p.shape[0]
            pred_dist.append(
                p[:, : 4 * self.reg_max].permute(0, 2, 3, 1).reshape(b, -1, 4 * self.reg_max)
            )
            pred_cls.append(
                p[:, 4 * self.reg_max :].permute(0, 2, 3, 1).reshape(b, -1, self.num_classes)
            )

        pred_dist = torch.cat(pred_dist, dim=1)  # [B, N, 4*reg_max]
        pred_cls = torch.cat(pred_cls, dim=1).sigmoid()  # [B, N, nc]

        b, n, _ = pred_dist.shape
        pred_dist = pred_dist.view(b, n, 4, self.reg_max).softmax(dim=-1)
        proj = torch.arange(self.reg_max, device=pred_dist.device, dtype=pred_dist.dtype).view(1, 1, 1, -1)
        dist = (pred_dist * proj).sum(dim=-1)  # [B, N, 4]

        anchors = anchor_points.view(1, n, 2)
        stride = stride_tensor.view(1, n, 1)

        x1y1 = anchors - dist[:, :, :2]
        x2y2 = anchors + dist[:, :, 2:]
        boxes = torch.cat([x1y1, x2y2], dim=-1) * stride  # xyxy image coords

        boxes[:, :, [0, 2]] = (boxes[:, :, [0, 2]] - float(self.pad_left)) / float(self.resize_scale)
        boxes[:, :, [1, 3]] = (boxes[:, :, [1, 3]] - float(self.pad_top)) / float(self.resize_scale)

        boxes = _clamp_and_normalize_boxes(boxes, orig_h, orig_w, self.normalize_boxes)

        return torch.cat([boxes, pred_cls], dim=-1)


class IPC8SegONNXExportWrapper(IPC8ONNXExportWrapper):
    def forward(self, x):
        orig_shape = torch._shape_as_tensor(x)
        orig_h = orig_shape[2].to(dtype=x.dtype, device=x.device)
        orig_w = orig_shape[3].to(dtype=x.dtype, device=x.device)

        x = self._preprocess_input(x)
        preds = self.model(x)
        if not isinstance(preds, dict):
            raise ValueError("Segmentation ONNX wrapper expected dict outputs.")

        det = preds.get("det")
        mask_coeffs = preds.get("mask_coeffs")
        proto = preds.get("proto")
        if det is None or mask_coeffs is None or proto is None:
            raise ValueError("Segmentation ONNX wrapper expected det, mask_coeffs, and proto.")

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
        boxes = _clamp_and_normalize_boxes(boxes, orig_h, orig_w, self.normalize_boxes)

        det_out = torch.cat([boxes, pred_cls], dim=-1)
        return (det_out, mask_coeffs[0], mask_coeffs[1], mask_coeffs[2], proto)


class IPC4ONNXExportWrapper(BaseONNXWrapper):
    def __init__(
        self,
        model,
        normalize_boxes: bool = True,
        normalize_input: bool = True,
    ):
        super().__init__(model=model, normalize_input=normalize_input)
        self.normalize_boxes = bool(normalize_boxes)
        self.num_classes = int(model.num_classes)
        self.net_w = int(getattr(model, "net_w", 608))
        self.net_h = int(getattr(model, "net_h", 608))

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalize_input(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = torch._shape_as_tensor(x)
        orig_h = orig_shape[2].to(dtype=x.dtype, device=x.device)
        orig_w = orig_shape[3].to(dtype=x.dtype, device=x.device)

        x = self._preprocess_input(x)
        outputs = self.model(x)
        if isinstance(outputs, dict):
            outputs = outputs.get("det")
        if outputs is None:
            raise ValueError("IPC4 ONNX export wrapper expected detection predictions.")

        boxes_all = []
        scores_all = []
        for out, yolo in zip(outputs, self.model.yolo_layers):
            b, c, h, w = out.shape
            na = len(yolo.mask)
            out = out.view(b, na, self.num_classes + 5, h, w).permute(0, 1, 3, 4, 2).contiguous()

            xy = torch.sigmoid(out[..., 0:2])
            wh = out[..., 2:4]
            obj = torch.sigmoid(out[..., 4:5])
            cls = torch.sigmoid(out[..., 5:])

            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=out.device),
                torch.arange(w, device=out.device),
                indexing="ij",
            )
            grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, h, w, 2).float()

            anchors = torch.tensor([yolo.anchors[i] for i in yolo.mask], device=out.device).view(1, na, 1, 1, 2)
            scale = torch.tensor(
                [orig_w / float(self.net_w), orig_h / float(self.net_h)],
                device=out.device,
            ).view(1, 1, 1, 1, 2)
            anchors = anchors * scale
            stride = torch.tensor([orig_w / w, orig_h / h], device=out.device).view(1, 1, 1, 1, 2)

            box_xy = (xy + grid) * stride
            box_wh = torch.exp(wh) * anchors
            x1y1 = box_xy - box_wh / 2
            x2y2 = box_xy + box_wh / 2
            boxes = torch.cat([x1y1, x2y2], dim=-1).view(b, -1, 4)

            scores = (obj * cls).view(b, -1, self.num_classes)
            boxes_all.append(boxes)
            scores_all.append(scores)

        boxes = torch.cat(boxes_all, dim=1)
        scores = torch.cat(scores_all, dim=1)

        boxes = _clamp_and_normalize_boxes(boxes, orig_h, orig_w, self.normalize_boxes)

        return torch.cat([boxes, scores], dim=-1)


def export_onnx_model(
    *,
    export_model: nn.Module,
    onnx_path: str,
    input_size: Tuple[int, int],
    opset_version: int,
    dynamic_axes: bool,
    output_names: list,
    model_output_name: str,
    embed_metadata_in_onnx: bool,
    write_metadata_json: bool,
    metadata_json_path: str,
    gst_tensor_id: str,
) -> str:
    h, w = int(input_size[0]), int(input_size[1])
    dummy = torch.zeros(1, 3, h, w, dtype=torch.float32)

    onnx_file = Path(onnx_path)
    if onnx_file.suffix.lower() != ".onnx":
        if onnx_file.exists() and onnx_file.is_dir():
            onnx_file = onnx_file / "model.onnx"
        else:
            onnx_file = onnx_file.with_suffix(".onnx")
    onnx_file.parent.mkdir(parents=True, exist_ok=True)

    dyn = None
    if dynamic_axes:
        dyn = {
            "images": {0: "batch", 2: "height", 3: "width"},
            model_output_name: {0: "batch", 1: "num_predictions"},
        }

    torch.onnx.export(
        export_model,
        dummy,
        str(onnx_file),
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dyn,
        dynamo=False,
    )

    if embed_metadata_in_onnx:
        try:
            import onnx

            model = onnx.load(str(onnx_file))
            filtered = [p for p in model.metadata_props if p.key != "detections"]
            model.ClearField("metadata_props")
            model.metadata_props.extend(filtered)
            model.metadata_props.append(onnx.StringStringEntryProto(key="detections", value=gst_tensor_id))
            onnx.save(model, str(onnx_file))
        except Exception as exc:
            print(f"[WARN] Failed to embed ONNX metadata: {exc}")

    if write_metadata_json:
        metadata_file = Path(metadata_json_path) if metadata_json_path else (onnx_file.parent / "out.json")
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.write_text(json.dumps({"detections": gst_tensor_id}))

    return str(onnx_file)
