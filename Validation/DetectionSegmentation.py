from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import colorsys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torchvision.ops import box_iou
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .Validation import ValidationBase


class ValDetectionSegmentation(ValidationBase):
    def __init__(
        self,
        predictor,
        dataset,
        conf_tresh: float = 0.5,
        iou_tresh: float = 0.5,
        task: str = "detection",
        split: str = "val",
        device: Optional[str] = None,
        gt_box_format: str = "auto",
        gt_normalized: Optional[bool] = None,
        pred_box_format: str = "auto",
        pred_normalized: Optional[bool] = None,
        ap_iou_thresholds: Optional[List[float]] = None,
    ):
        self.predictor = predictor
        self.dataset = dataset
        self.task = str(task).lower()
        self.split = split
        self.iou_tresh = float(iou_tresh)
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.gt_box_format = gt_box_format.lower()
        self.gt_normalized = gt_normalized
        self.pred_box_format = pred_box_format.lower()
        self.pred_normalized = pred_normalized
        self.ap_iou_thresholds = self._normalize_ap_thresholds(ap_iou_thresholds)

        if hasattr(self.predictor, "conf_thres"):
            self.predictor.conf_thres = conf_tresh
        if hasattr(self.predictor, "iou_thres"):
            self.predictor.iou_thres = iou_tresh

        self._run_task = self._resolve_task_runner()
        super().__init__(predictor=self.predictor, dataset=self.dataset, device=self.device)

    @torch.no_grad()
    def start(self):
        return self._run_task()

    def _resolve_task_runner(self):
        task_aliases = {
            "object_detection": self._run_object_detection,
            "detection": self._run_object_detection,
            "segmentation": self._run_instance_segmentation,
            "instance_segmentation": self._run_instance_segmentation,
        }
        if self.task not in task_aliases:
            raise NotImplementedError(f"Unsupported task: {self.task}")
        return task_aliases[self.task]

    def _run_object_detection(self):
        val_loader = self.dataset.load_images_and_ground_truth(split=self.split, shuffle=False)
        class_names = getattr(self.dataset, "target_classes", None)
        num_classes = getattr(self.dataset, "class_num", 0) or (len(class_names) if class_names else 0)
        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_classes)]

        stats = {
            "tp": torch.zeros(num_classes, dtype=torch.float64),
            "fp": torch.zeros(num_classes, dtype=torch.float64),
            "fn": torch.zeros(num_classes, dtype=torch.float64),
            "iou_sum": torch.zeros(num_classes, dtype=torch.float64),
            "pred_count": torch.zeros(num_classes, dtype=torch.float64),
            "gt_count": torch.zeros(num_classes, dtype=torch.float64),
            "ap_scores": [[] for _ in range(num_classes)],
            "ap_tp_flags": self._init_ap_flags(num_classes, len(self.ap_iou_thresholds)),
            "ap_iou_thresholds": self.ap_iou_thresholds,
        }

        for images, targets, _ in val_loader:
            images = images.to(self.device, non_blocking=True)
            outputs = self.predict_batch(images)

            image_hw = (int(images.shape[2]), int(images.shape[3]))
            for pred, gt in zip(outputs, targets):
                self.tp_fp_fn_iou_detection(pred, gt, stats, image_hw=image_hw)

        results = self.report(stats, class_names)
        return results

    def _run_instance_segmentation(self):
        val_loader = self.dataset.load_images_and_ground_truth(split=self.split, shuffle=False)
        class_names = getattr(self.dataset, "target_classes", None)
        num_classes = getattr(self.dataset, "class_num", 0) or (len(class_names) if class_names else 0)
        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_classes)]

        stats = {
            "tp": torch.zeros(num_classes, dtype=torch.float64),
            "fp": torch.zeros(num_classes, dtype=torch.float64),
            "fn": torch.zeros(num_classes, dtype=torch.float64),
            "iou_sum": torch.zeros(num_classes, dtype=torch.float64),
            "pred_count": torch.zeros(num_classes, dtype=torch.float64),
            "gt_count": torch.zeros(num_classes, dtype=torch.float64),
            "ap_scores": [[] for _ in range(num_classes)],
            "ap_tp_flags": self._init_ap_flags(num_classes, len(self.ap_iou_thresholds)),
            "ap_iou_thresholds": self.ap_iou_thresholds,
        }

        for images, targets, _ in val_loader:
            images = images.to(self.device, non_blocking=True)
            outputs = self.predict_batch(images)

            image_hw = (int(images.shape[2]), int(images.shape[3]))
            for pred, gt in zip(outputs, targets):
                self.tp_fp_fn_iou_segmentation(pred, gt, stats, image_hw=image_hw)

        return self.report(stats, class_names)

    def tp_fp_fn_iou_detection(self, pred, gt, stats, image_hw: Tuple[int, int]):
        pred_boxes, pred_labels, pred_scores = self._normalize_prediction(pred, fallback_hw=image_hw)
        gt_boxes, gt_labels = self._normalize_ground_truth(gt, fallback_hw=image_hw)

        if gt_labels.numel() > 0:
            binc = torch.bincount(gt_labels, minlength=stats["gt_count"].numel()).to(stats["gt_count"].dtype)
            stats["gt_count"] += binc

        if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
            return stats
        if pred_boxes.numel() == 0:
            binc = torch.bincount(gt_labels, minlength=stats["fn"].numel()).to(stats["fn"].dtype)
            stats["fn"] += binc
            return stats
        if gt_boxes.numel() == 0:
            binc = torch.bincount(pred_labels, minlength=stats["fp"].numel()).to(stats["fp"].dtype)
            stats["fp"] += binc
            binc_pred = torch.bincount(pred_labels, minlength=stats["pred_count"].numel()).to(stats["pred_count"].dtype)
            stats["pred_count"] += binc_pred
            for c, s in zip(pred_labels.tolist(), pred_scores.tolist()):
                stats["ap_scores"][c].append(float(s))
                for t in range(len(stats["ap_iou_thresholds"])):
                    stats["ap_tp_flags"][c][t].append(0.0)
            return stats

        num_classes = stats["tp"].numel()
        for c in range(num_classes):
            pred_mask = pred_labels == c
            gt_mask = gt_labels == c

            p_boxes = pred_boxes[pred_mask]
            p_scores = pred_scores[pred_mask]
            g_boxes = gt_boxes[gt_mask]

            if p_boxes.numel() == 0 and g_boxes.numel() == 0:
                continue
            if p_boxes.numel() == 0:
                stats["fn"][c] += float(g_boxes.shape[0])
                continue
            if g_boxes.numel() == 0:
                stats["fp"][c] += float(p_boxes.shape[0])
                stats["pred_count"][c] += float(p_boxes.shape[0])
                for s in p_scores.tolist():
                    stats["ap_scores"][c].append(float(s))
                    for t in range(len(stats["ap_iou_thresholds"])):
                        stats["ap_tp_flags"][c][t].append(0.0)
                continue

            ious = box_iou(p_boxes, g_boxes)
            tp_mask, matched_iou, matched_pred_idx = self._greedy_match(ious, self.iou_tresh)

            tp = int(tp_mask.sum().item())
            fp = int(p_boxes.shape[0] - tp)
            fn = int(g_boxes.shape[0] - tp)

            stats["tp"][c] += tp
            stats["fp"][c] += fp
            stats["fn"][c] += fn
            stats["pred_count"][c] += float(p_boxes.shape[0])
            if tp > 0:
                per_pred_iou = torch.zeros((p_boxes.shape[0],), dtype=torch.float32)
                per_pred_iou[matched_pred_idx] = matched_iou
                stats["iou_sum"][c] += float(per_pred_iou.sum().item())

            order = torch.argsort(p_scores, descending=True)
            sc_sorted = p_scores[order]
            stats["ap_scores"][c].extend(sc_sorted.tolist())
            for t, thr in enumerate(stats["ap_iou_thresholds"]):
                tp_mask_thr, _, _ = self._greedy_match(ious, thr)
                tp_sorted = tp_mask_thr[order].to(torch.float32)
                stats["ap_tp_flags"][c][t].extend(tp_sorted.tolist())

        return stats

    def tp_fp_fn_iou_segmentation(self, pred, gt, stats, image_hw: Tuple[int, int]):
        pred_masks, pred_labels, pred_scores = self._normalize_prediction_masks(pred, fallback_hw=image_hw)
        gt_masks, gt_labels = self._normalize_ground_truth_masks(gt, fallback_hw=image_hw)

        if gt_labels.numel() > 0:
            binc = torch.bincount(gt_labels, minlength=stats["gt_count"].numel()).to(stats["gt_count"].dtype)
            stats["gt_count"] += binc

        if pred_masks.numel() == 0 and gt_masks.numel() == 0:
            return stats
        if pred_masks.numel() == 0:
            binc = torch.bincount(gt_labels, minlength=stats["fn"].numel()).to(stats["fn"].dtype)
            stats["fn"] += binc
            return stats
        if gt_masks.numel() == 0:
            binc = torch.bincount(pred_labels, minlength=stats["fp"].numel()).to(stats["fp"].dtype)
            stats["fp"] += binc
            binc_pred = torch.bincount(pred_labels, minlength=stats["pred_count"].numel()).to(stats["pred_count"].dtype)
            stats["pred_count"] += binc_pred
            for c, s in zip(pred_labels.tolist(), pred_scores.tolist()):
                stats["ap_scores"][c].append(float(s))
                for t in range(len(stats["ap_iou_thresholds"])):
                    stats["ap_tp_flags"][c][t].append(0.0)
            return stats

        num_classes = stats["tp"].numel()
        for c in range(num_classes):
            pred_mask = pred_labels == c
            gt_mask = gt_labels == c

            p_masks = pred_masks[pred_mask]
            p_scores = pred_scores[pred_mask]
            g_masks = gt_masks[gt_mask]

            if p_masks.numel() == 0 and g_masks.numel() == 0:
                continue
            if p_masks.numel() == 0:
                stats["fn"][c] += float(g_masks.shape[0])
                continue
            if g_masks.numel() == 0:
                stats["fp"][c] += float(p_masks.shape[0])
                stats["pred_count"][c] += float(p_masks.shape[0])
                for s in p_scores.tolist():
                    stats["ap_scores"][c].append(float(s))
                    for t in range(len(stats["ap_iou_thresholds"])):
                        stats["ap_tp_flags"][c][t].append(0.0)
                continue

            ious = self._mask_iou_matrix(p_masks, g_masks)
            tp_mask, matched_iou, matched_pred_idx = self._greedy_match(ious, self.iou_tresh)

            tp = int(tp_mask.sum().item())
            fp = int(p_masks.shape[0] - tp)
            fn = int(g_masks.shape[0] - tp)

            stats["tp"][c] += tp
            stats["fp"][c] += fp
            stats["fn"][c] += fn
            stats["pred_count"][c] += float(p_masks.shape[0])
            if tp > 0:
                per_pred_iou = torch.zeros((p_masks.shape[0],), dtype=torch.float32)
                per_pred_iou[matched_pred_idx] = matched_iou
                stats["iou_sum"][c] += float(per_pred_iou.sum().item())

            order = torch.argsort(p_scores, descending=True)
            sc_sorted = p_scores[order]
            stats["ap_scores"][c].extend(sc_sorted.tolist())
            for t, thr in enumerate(stats["ap_iou_thresholds"]):
                tp_mask_thr, _, _ = self._greedy_match(ious, thr)
                tp_sorted = tp_mask_thr[order].to(torch.float32)
                stats["ap_tp_flags"][c][t].extend(tp_sorted.tolist())

        return stats

    def average_precision(self, scores: List[float], tp_flags: List[float], n_gt: int) -> float:
        if n_gt == 0:
            return 0.0
        if len(scores) == 0:
            return 0.0

        scores_t = torch.tensor(scores, dtype=torch.float32)
        tp_t = torch.tensor(tp_flags, dtype=torch.float32)
        order = torch.argsort(scores_t, descending=True)
        tp_sorted = tp_t[order]
        fp_sorted = 1.0 - tp_sorted

        tp_cum = torch.cumsum(tp_sorted, dim=0)
        fp_cum = torch.cumsum(fp_sorted, dim=0)

        recall = tp_cum / max(float(n_gt), 1.0)
        precision = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-12)

        mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
        mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
        mpre = torch.flip(torch.cummax(torch.flip(mpre, dims=[0]), dim=0).values, dims=[0])
        idx = torch.where(mrec[1:] != mrec[:-1])[0]
        ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item()
        return float(ap)

    def report(self, stats, class_names):
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        iou_sum = stats["iou_sum"]
        pred_count = stats["pred_count"]
        gt_count = stats["gt_count"]

        eps = 1e-12
        precision = tp / torch.clamp(tp + fp, min=eps)
        recall = tp / torch.clamp(tp + fn, min=eps)
        f1 = 2.0 * precision * recall / torch.clamp(precision + recall, min=eps)
        mean_iou = iou_sum / torch.clamp(pred_count, min=1.0)

        ap_thresholds = stats.get("ap_iou_thresholds", [0.5])
        ap50_index = self._ap50_index(ap_thresholds)
        ap50 = []
        ap_all = []
        for i in range(len(class_names)):
            ap_vals = []
            for t in range(len(ap_thresholds)):
                ap_val = self.average_precision(
                    stats["ap_scores"][i],
                    stats["ap_tp_flags"][i][t],
                    int(gt_count[i].item()),
                )
                ap_vals.append(ap_val)
            ap50.append(ap_vals[ap50_index])
            ap_all.append(float(sum(ap_vals) / max(1, len(ap_vals))))
        ap50_t = torch.tensor(ap50, dtype=torch.float32)
        ap_t = torch.tensor(ap_all, dtype=torch.float32)

        print("Class                   TP      FP      FN    Prec    Recall     F1     mIoU   AP50     AP")
        for i, name in enumerate(class_names):
            print(
                f"{name[:20].ljust(20)} "
                f"{int(tp[i].item()):6d} "
                f"{int(fp[i].item()):7d} "
                f"{int(fn[i].item()):7d} "
                f"{precision[i].item():6.2f} "
                f"{recall[i].item():7.2f} "
                f"{f1[i].item():7.2f} "
                f"{mean_iou[i].item():7.2f} "
                f"{ap50_t[i].item():7.2f} "
                f"{ap_t[i].item():7.2f}"
            )

        overall_tp = tp.sum()
        overall_fp = fp.sum()
        overall_fn = fn.sum()
        overall_precision = overall_tp / torch.clamp(overall_tp + overall_fp, min=eps)
        overall_recall = overall_tp / torch.clamp(overall_tp + overall_fn, min=eps)
        overall_f1 = 2.0 * overall_precision * overall_recall / torch.clamp(overall_precision + overall_recall, min=eps)
        overall_iou = iou_sum.sum() / torch.clamp(pred_count.sum(), min=1.0)
        map50 = ap50_t.mean() if ap50_t.numel() > 0 else torch.tensor(0.0)
        map_all = ap_t.mean() if ap_t.numel() > 0 else torch.tensor(0.0)
        miou = mean_iou.mean() if mean_iou.numel() > 0 else torch.tensor(0.0)

        print("-" * 98)
        print(
            f"{'overall'.ljust(20)} "
            f"{int(overall_tp.item()):6d} "
            f"{int(overall_fp.item()):7d} "
            f"{int(overall_fn.item()):7d} "
            f"{overall_precision.item():6.3f} "
            f"{overall_recall.item():7.3f} "
            f"{overall_f1.item():7.3f} "
            f"{overall_iou.item():8.3f} "
            f"{map50.item():7.3f} "
            f"{map_all.item():7.3f}"
        )
        print(f"mIoU: {miou.item():.4f}")

        return {
            "precision": precision.cpu(),
            "recall": recall.cpu(),
            "f1": f1.cpu(),
            "iou": mean_iou.cpu(),
            "ap50": ap50_t.cpu(),
            "ap": ap_t.cpu(),
            "overall": {
                "precision": float(overall_precision.item()),
                "recall": float(overall_recall.item()),
                "f1": float(overall_f1.item()),
                "iou": float(overall_iou.item()),
                "map50": float(map50.item()),
                "map": float(map_all.item()),
                "miou": float(miou.item()),
            },
        }

    @staticmethod
    def _init_ap_flags(num_classes: int, num_thresholds: int):
        return [[[] for _ in range(num_thresholds)] for _ in range(num_classes)]

    @staticmethod
    def _normalize_ap_thresholds(thresholds: Optional[List[float]]) -> List[float]:
        if thresholds is None:
            return [round(x, 2) for x in torch.arange(0.5, 0.96, 0.05).tolist()]
        cleaned = sorted({float(t) for t in thresholds})
        if not cleaned:
            return [0.5]
        return cleaned

    @staticmethod
    def _ap50_index(thresholds: List[float]) -> int:
        if not thresholds:
            return 0
        for i, t in enumerate(thresholds):
            if abs(t - 0.5) < 1e-6:
                return i
        return 0

    def _normalize_prediction(
        self,
        pred: Dict[str, torch.Tensor],
        fallback_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes = pred.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
        labels = pred.get("classes", pred.get("labels", torch.zeros((0,), dtype=torch.long)))
        scores = pred.get("scores", torch.ones((boxes.shape[0],), dtype=torch.float32))

        boxes = boxes.detach().cpu().to(torch.float32).reshape(-1, 4)
        labels = labels.detach().cpu().to(torch.long).reshape(-1)
        scores = scores.detach().cpu().to(torch.float32).reshape(-1)

        box_format = str(pred.get("box_format", self.pred_box_format)).lower()
        normalized = pred.get("normalized", self.pred_normalized)
        image_hw = pred.get("image_size", fallback_hw)
        boxes = self._to_xyxy_abs(boxes, box_format, normalized, image_hw)

        return boxes, labels, scores

    def _normalize_ground_truth(
        self,
        gt,
        fallback_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(gt, dict):
            boxes = gt.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
            labels = gt.get("labels", torch.zeros((0,), dtype=torch.long))
            box_format = str(gt.get("box_format", self.gt_box_format)).lower()
            normalized = gt.get("normalized", self.gt_normalized)
            image_hw = gt.get("image_size", fallback_hw)
        else:
            gt_t = gt.detach().cpu().to(torch.float32)
            if gt_t.numel() == 0:
                return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
            labels = gt_t[:, 0].to(torch.long)
            boxes = gt_t[:, 1:5]
            box_format = self.gt_box_format if self.gt_box_format != "auto" else "xywh"
            normalized = self.gt_normalized if self.gt_normalized is not None else True
            image_hw = fallback_hw

        boxes = boxes.detach().cpu().to(torch.float32).reshape(-1, 4)
        labels = labels.detach().cpu().to(torch.long).reshape(-1)
        boxes = self._to_xyxy_abs(boxes, box_format, normalized, image_hw)
        return boxes, labels

    def _normalize_prediction_masks(
        self,
        pred: Dict[str, torch.Tensor],
        fallback_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes, labels, scores = self._normalize_prediction(pred, fallback_hw=fallback_hw)
        masks = pred.get("masks", pred.get("mask"))
        thr = float(pred.get("mask_threshold", 0.5))
        masks_t = self._normalize_masks_tensor(masks, fallback_hw, threshold=thr)

        n = min(int(boxes.shape[0]), int(labels.shape[0]), int(scores.shape[0]), int(masks_t.shape[0]))
        if n <= 0:
            h, w = int(fallback_hw[0]), int(fallback_hw[1])
            return (
                torch.zeros((0, h, w), dtype=torch.bool),
                torch.zeros((0,), dtype=torch.long),
                torch.zeros((0,), dtype=torch.float32),
            )
        return masks_t[:n], labels[:n], scores[:n]

    def _normalize_ground_truth_masks(
        self,
        gt,
        fallback_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(gt, dict):
            labels = gt.get("labels", torch.zeros((0,), dtype=torch.long))
            masks = gt.get("masks", gt.get("mask"))
            labels = labels.detach().cpu().to(torch.long).reshape(-1)
            masks_t = self._normalize_masks_tensor(masks, fallback_hw, threshold=0.5)
            n = min(int(labels.shape[0]), int(masks_t.shape[0]))
            if n <= 0:
                h, w = int(fallback_hw[0]), int(fallback_hw[1])
                return torch.zeros((0, h, w), dtype=torch.bool), torch.zeros((0,), dtype=torch.long)
            return masks_t[:n], labels[:n]

        return torch.zeros((0, int(fallback_hw[0]), int(fallback_hw[1])), dtype=torch.bool), torch.zeros((0,), dtype=torch.long)

    @staticmethod
    def _normalize_masks_tensor(masks, fallback_hw: Tuple[int, int], threshold: float = 0.5) -> torch.Tensor:
        h, w = int(fallback_hw[0]), int(fallback_hw[1])
        if masks is None:
            return torch.zeros((0, h, w), dtype=torch.bool)

        m = masks.detach().cpu()
        if m.ndim == 2:
            m = m.unsqueeze(0)
        elif m.ndim == 4 and m.shape[1] == 1:
            m = m.squeeze(1)
        elif m.ndim != 3:
            return torch.zeros((0, h, w), dtype=torch.bool)

        m = m.to(torch.float32)
        if m.shape[-2] != h or m.shape[-1] != w:
            m = F.interpolate(
                m.unsqueeze(1),
                size=(h, w),
                mode="nearest",
            ).squeeze(1)
        return m >= float(threshold)

    @staticmethod
    def _parse_hw(image_hw) -> Tuple[float, float]:
        if isinstance(image_hw, (list, tuple)) and len(image_hw) >= 2:
            h, w = float(image_hw[0]), float(image_hw[1])
            return h, w
        s = float(image_hw)
        return s, s

    @staticmethod
    def _infer_box_format(boxes: torch.Tensor) -> str:
        if boxes.numel() == 0:
            return "xyxy"
        xyxy_valid = ((boxes[:, 2] >= boxes[:, 0]) & (boxes[:, 3] >= boxes[:, 1])).float().mean().item()
        return "xyxy" if xyxy_valid >= 0.8 else "xywh"

    @staticmethod
    def _infer_normalized(boxes: torch.Tensor) -> bool:
        if boxes.numel() == 0:
            return False
        maxv = float(boxes.max().item())
        minv = float(boxes.min().item())
        return minv >= -1e-6 and maxv <= 1.5

    def _to_xyxy_abs(
        self,
        boxes: torch.Tensor,
        box_format: str,
        normalized: Optional[bool],
        image_hw,
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.reshape(0, 4)

        h, w = self._parse_hw(image_hw)

        if box_format == "auto":
            box_format = self._infer_box_format(boxes)
        if normalized is None:
            normalized = self._infer_normalized(boxes)

        out = boxes.clone()
        if box_format == "xywh":
            x, y, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
            out = torch.stack([x - 0.5 * bw, y - 0.5 * bh, x + 0.5 * bw, y + 0.5 * bh], dim=1)
        elif box_format != "xyxy":
            raise ValueError(f"Unsupported box_format: {box_format}. Use 'xyxy', 'xywh', or 'auto'.")

        if bool(normalized):
            out[:, [0, 2]] = out[:, [0, 2]] * w
            out[:, [1, 3]] = out[:, [1, 3]] * h

        out[:, [0, 2]] = out[:, [0, 2]].clamp(0.0, max(0.0, w - 1.0))
        out[:, [1, 3]] = out[:, [1, 3]].clamp(0.0, max(0.0, h - 1.0))
        return out

    @staticmethod
    def _greedy_match(iou_matrix: torch.Tensor, iou_thr: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_pred = iou_matrix.shape[0]
        if n_pred == 0:
            return (
                torch.zeros((0,), dtype=torch.bool),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
            )

        pairs = torch.nonzero(iou_matrix >= iou_thr, as_tuple=False)
        if pairs.numel() == 0:
            return (
                torch.zeros((n_pred,), dtype=torch.bool),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
            )

        iou_vals = iou_matrix[pairs[:, 0], pairs[:, 1]]
        order = torch.argsort(iou_vals, descending=True)
        pairs = pairs[order]
        iou_vals = iou_vals[order]

        used_pred = torch.zeros((n_pred,), dtype=torch.bool)
        used_gt = torch.zeros((iou_matrix.shape[1],), dtype=torch.bool)
        tp_mask = torch.zeros((n_pred,), dtype=torch.bool)
        matched_ious = []
        matched_pred_idx = []

        for k in range(pairs.shape[0]):
            p = int(pairs[k, 0].item())
            g = int(pairs[k, 1].item())
            if used_pred[p] or used_gt[g]:
                continue
            used_pred[p] = True
            used_gt[g] = True
            tp_mask[p] = True
            matched_ious.append(iou_vals[k].item())
            matched_pred_idx.append(p)

        if matched_ious:
            matched_ious_t = torch.tensor(matched_ious, dtype=torch.float32)
            matched_pred_idx_t = torch.tensor(matched_pred_idx, dtype=torch.long)
        else:
            matched_ious_t = torch.zeros((0,), dtype=torch.float32)
            matched_pred_idx_t = torch.zeros((0,), dtype=torch.long)
        return tp_mask, matched_ious_t, matched_pred_idx_t

    @staticmethod
    def _mask_iou_matrix(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        if pred_masks.numel() == 0 or gt_masks.numel() == 0:
            return torch.zeros((pred_masks.shape[0], gt_masks.shape[0]), dtype=torch.float32)

        pm = pred_masks.reshape(pred_masks.shape[0], -1).to(torch.float32)
        gm = gt_masks.reshape(gt_masks.shape[0], -1).to(torch.float32)

        inter = pm @ gm.t()
        area_p = pm.sum(dim=1, keepdim=True)
        area_g = gm.sum(dim=1, keepdim=True).t()
        union = area_p + area_g - inter
        return inter / torch.clamp(union, min=1e-12)

    def test_from_dataset_split(
        self,
        split: str,
        labels_dir: Path,
        visuals_dir: Path,
        save_visuals: bool,
        save_txt: bool,
    ):
        loader = self.dataset.load_images(split=split, shuffle=False)
        saved = 0
        for images, paths in loader:
            images = images.to(self.device, non_blocking=True)
            outputs = self.predict_batch(images)
            for i, (img_t, path_str, pred) in enumerate(zip(images, paths, outputs)):
                h, w = int(img_t.shape[1]), int(img_t.shape[2])
                boxes, labels, scores = self._normalize_prediction(pred, fallback_hw=(h, w))
                stem = Path(path_str).stem

                if save_txt:
                    self._save_detection_txt(labels_dir / f"{stem}.txt", boxes, labels, scores)

                if save_visuals:
                    img_np = (
                        img_t.detach()
                        .cpu()
                        .permute(1, 2, 0)
                        .mul(255.0)
                        .clamp(0, 255)
                        .to(torch.uint8)
                        .numpy()
                    )
                    self._save_visual_pil(visuals_dir / f"{stem}.jpg", img_np, boxes, labels, scores, self.dataset.target_classes)
                saved += 1

        return {
            "mode": "dataset_split",
            "split": split,
            "saved_items": saved,
            "output_dir": str(labels_dir.parent),
        }

    def test_from_images(
        self,
        source_path: Path,
        labels_dir: Path,
        visuals_dir: Path,
        save_visuals: bool,
        save_txt: bool,
    ):
        image_paths = self._collect_image_paths(source_path)
        saved = 0
        for p in image_paths:
            with Image.open(p) as img:
                img = img.convert("RGB")
                img_t = pil_to_tensor(img).float().div(255.0).to(self.device)
                img_np = pil_to_tensor(img).permute(1, 2, 0).to(torch.uint8).numpy()

            pred = self.predict_single(img_t)
            h, w = int(img_t.shape[1]), int(img_t.shape[2])
            boxes, labels, scores = self._normalize_prediction(pred, fallback_hw=(h, w))

            masks = None
            if self.task in ("segmentation", "instance_segmentation"):
                masks, _, _ = self._normalize_prediction_masks(pred, fallback_hw=(h, w))
                n = min(int(boxes.shape[0]), int(labels.shape[0]), int(scores.shape[0]), int(masks.shape[0]))
                boxes = boxes[:n]
                labels = labels[:n]
                scores = scores[:n]
                masks = masks[:n]

            if save_txt:
                self._save_detection_txt(labels_dir / f"{p.stem}.txt", boxes, labels, scores)

            if save_visuals:
                if masks is not None:
                    self._save_visual_pil_segmentation(
                        visuals_dir / f"{p.stem}.jpg",
                        img_np,
                        boxes,
                        labels,
                        scores,
                        masks,
                        self.dataset.target_classes,
                    )
                else:
                    self._save_visual_pil(
                        visuals_dir / f"{p.stem}.jpg",
                        img_np,
                        boxes,
                        labels,
                        scores,
                        self.dataset.target_classes,
                    )
            saved += 1

        return {
            "mode": "images",
            "source": str(source_path),
            "saved_items": saved,
            "output_dir": str(labels_dir.parent),
        }

    def test_from_video(
        self,
        source_path: Path,
        labels_dir: Path,
        visuals_dir: Path,
        save_visuals: bool,
        save_txt: bool,
        fps: int = 30,
    ):
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("OpenCV is required for video test_performance.") from exc

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {source_path}")

        writer = None
        out_path = None
        if save_visuals:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer, out_path = self._create_video_writer_cv2(
                visuals_dir=visuals_dir,
                source_stem=source_path.stem,
                fps=fps,
                width=width,
                height=height,
            )

        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            pred = self.predict_from_bgr_frame(frame_bgr)
            h, w = frame_bgr.shape[:2]
            boxes, labels, scores = self._normalize_prediction(pred, fallback_hw=(h, w))

            masks = None
            if self.task in ("segmentation", "instance_segmentation"):
                masks, _, _ = self._normalize_prediction_masks(pred, fallback_hw=(h, w))
                n = min(int(boxes.shape[0]), int(labels.shape[0]), int(scores.shape[0]), int(masks.shape[0]))
                boxes = boxes[:n]
                labels = labels[:n]
                scores = scores[:n]
                masks = masks[:n]
            stem = f"{source_path.stem}_frame_{frame_idx:06d}"

            if save_txt:
                self._save_detection_txt(labels_dir / f"{stem}.txt", boxes, labels, scores)

            if save_visuals and writer is not None:
                if masks is not None:
                    vis_frame = self._draw_visual_cv2_segmentation(
                        frame_bgr, boxes, labels, scores, masks, self.dataset.target_classes
                    )
                else:
                    vis_frame = self._draw_visual_cv2(frame_bgr, boxes, labels, scores, self.dataset.target_classes)
                writer.write(vis_frame)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        return {
            "mode": "video",
            "source": str(source_path),
            "saved_items": frame_idx,
            "output_dir": str(labels_dir.parent),
            "visual_output": str(out_path) if out_path is not None else None,
        }

    @staticmethod
    def _create_video_writer_cv2(visuals_dir: Path, source_stem: str, fps: float, width: int, height: int):
        import cv2

        fps = float(fps) if fps > 0 else 30.0
        candidates = [
            # Prefer fast/commonly available MKV encoders first.
            ("MJPG", ".mkv"),
            ("XVID", ".mkv"),
            ("H264", ".mkv"),
            ("avc1", ".mkv"),
            ("mp4v", ".mkv"),
            # Lossless fallback (larger files, CPU heavier).
            ("FFV1", ".mkv"),
        ]

        for codec, suffix in candidates:
            out_path = visuals_dir / f"{source_stem}_detections{suffix}"
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*codec),
                fps,
                (int(width), int(height)),
            )
            if writer is not None and writer.isOpened():
                return writer, out_path
            if writer is not None:
                writer.release()

        raise RuntimeError(
            "Failed to create an MKV output video writer with codecs: "
            + ", ".join(c for c, _ in candidates)
        )

    @staticmethod
    def _try_convert_video_to_playable_mp4(source_path: Path) -> Optional[Path]:
        try:
            import imageio.v2 as iio
        except Exception:
            return None

        target_path = source_path.with_name(f"{source_path.stem}_playable.mp4")
        try:
            reader = iio.get_reader(str(source_path))
            fps = reader.get_meta_data().get("fps", 30)
            writer = iio.get_writer(
                str(target_path),
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
            )

            for frame in reader:
                writer.append_data(frame)

            writer.close()
            reader.close()
            return target_path
        except Exception as exc:
            print(f"[WARN] Failed to convert {source_path} to playable mp4: {exc}")
            return None

    @staticmethod
    def _save_detection_txt(path: Path, boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor):
        lines = []
        for b, c, s in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
            x1, y1, x2, y2 = b
            lines.append(f"{int(c)} {float(s):.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}")
        path.write_text("\n".join(lines))

    @staticmethod
    def _save_visual_pil(
        path: Path,
        image_rgb,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        class_names: List[str],
    ):
        img = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(img, mode="RGBA")
        font = None
        for font_name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
            try:
                font = ImageFont.truetype(font_name, 16)
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()

        for b, c, s in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
            x1, y1, x2, y2 = [int(v) for v in b]
            color_rgb = ValDetectionSegmentation._class_color_rgb(int(c))
            draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)
            cname = class_names[c] if 0 <= c < len(class_names) else str(c)
            label = f"{cname} {s:.2f}"
            if hasattr(draw, "textbbox"):
                l, t, r, btm = draw.textbbox((0, 0), label, font=font)
                text_w, text_h = (r - l), (btm - t)
            else:
                text_w, text_h = draw.textsize(label, font=font)
            tx = max(0, x1)
            ty = max(0, y1 - text_h - 8)
            draw.rectangle(
                [tx, ty, tx + text_w + 8, ty + text_h + 6],
                fill=(color_rgb[0], color_rgb[1], color_rgb[2], 190),
            )
            draw.text((tx + 4, ty + 3), label, fill=(255, 255, 255, 255), font=font)
        img.save(path)

    @staticmethod
    def _save_visual_pil_segmentation(
        path: Path,
        image_rgb,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        masks: torch.Tensor,
        class_names: List[str],
        mask_alpha: int = 90,
    ):
        base = Image.fromarray(image_rgb).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

        if masks.numel() > 0:
            masks_np = masks.detach().cpu().numpy().astype(np.uint8)
            for i in range(masks_np.shape[0]):
                c = int(labels[i].item()) if labels.numel() > i else -1
                color_rgb = ValDetectionSegmentation._class_color_rgb(c if c >= 0 else 0)
                mask_img = Image.fromarray(masks_np[i] * 255, mode="L")
                overlay.paste((color_rgb[0], color_rgb[1], color_rgb[2], int(mask_alpha)), mask=mask_img)

        img = Image.alpha_composite(base, overlay).convert("RGB")
        draw = ImageDraw.Draw(img, mode="RGBA")
        font = None
        for font_name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
            try:
                font = ImageFont.truetype(font_name, 16)
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()

        class_counts: Dict[int, int] = {}
        for b, c, s in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
            class_counts[c] = class_counts.get(c, 0) + 1
            instance_idx = class_counts[c]
            x1, y1, x2, y2 = [int(v) for v in b]
            color_rgb = ValDetectionSegmentation._class_color_rgb(int(c))
            draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)
            cname = class_names[c] if 0 <= c < len(class_names) else str(c)
            label = f"{cname} #{instance_idx} {s:.2f}"
            if hasattr(draw, "textbbox"):
                l, t, r, btm = draw.textbbox((0, 0), label, font=font)
                text_w, text_h = (r - l), (btm - t)
            else:
                text_w, text_h = draw.textsize(label, font=font)
            tx = max(0, x1)
            ty = max(0, y1 - text_h - 8)
            draw.rectangle(
                [tx, ty, tx + text_w + 8, ty + text_h + 6],
                fill=(color_rgb[0], color_rgb[1], color_rgb[2], 190),
            )
            draw.text((tx + 4, ty + 3), label, fill=(255, 255, 255, 255), font=font)

        img.save(path)

    @staticmethod
    def _draw_visual_cv2(frame_bgr, boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor, class_names: List[str]):
        import cv2

        out = frame_bgr.copy()
        h, w = out.shape[:2]
        for b, c, s in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
            x1, y1, x2, y2 = [int(v) for v in b]
            color_bgr = ValDetectionSegmentation._class_color_bgr(int(c))
            cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)
            cname = class_names[c] if 0 <= c < len(class_names) else str(c)
            label = f"{cname} {s:.2f}"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.55
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            tx = max(0, min(x1, w - text_w - 8))
            if y1 - text_h - baseline - 8 >= 0:
                ty1 = y1 - text_h - baseline - 8
                ty2 = y1
                text_org = (tx + 4, y1 - baseline - 4)
            else:
                ty1 = y1
                ty2 = min(h - 1, y1 + text_h + baseline + 8)
                text_org = (tx + 4, min(h - 1, ty2 - baseline - 4))

            cv2.rectangle(out, (tx, ty1), (tx + text_w + 8, ty2), color_bgr, thickness=-1)
            cv2.putText(out, label, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return out

    @staticmethod
    def _draw_visual_cv2_segmentation(
        frame_bgr,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        masks: torch.Tensor,
        class_names: List[str],
        alpha: float = 0.3,
    ):
        import cv2

        out = frame_bgr.copy()
        if masks.numel() > 0:
            masks_np = masks.detach().cpu().numpy().astype(bool)
            for i in range(masks_np.shape[0]):
                c = int(labels[i].item()) if labels.numel() > i else 0
                color_bgr = ValDetectionSegmentation._class_color_bgr(int(c))
                m = masks_np[i]
                if m.any():
                    out[m] = (out[m].astype(np.float32) * (1.0 - alpha) + np.array(color_bgr, dtype=np.float32) * alpha).astype(np.uint8)

        h, w = out.shape[:2]
        class_counts: Dict[int, int] = {}
        for b, c, s in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
            class_counts[c] = class_counts.get(c, 0) + 1
            instance_idx = class_counts[c]
            x1, y1, x2, y2 = [int(v) for v in b]
            color_bgr = ValDetectionSegmentation._class_color_bgr(int(c))
            cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)
            cname = class_names[c] if 0 <= c < len(class_names) else str(c)
            label = f"{cname} #{instance_idx} {s:.2f}"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.55
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            tx = max(0, min(x1, w - text_w - 8))
            if y1 - text_h - baseline - 8 >= 0:
                ty1 = y1 - text_h - baseline - 8
                ty2 = y1
                text_org = (tx + 4, y1 - baseline - 4)
            else:
                ty1 = y1
                ty2 = min(h - 1, y1 + text_h + baseline + 8)
                text_org = (tx + 4, min(h - 1, ty2 - baseline - 4))

            cv2.rectangle(out, (tx, ty1), (tx + text_w + 8, ty2), color_bgr, thickness=-1)
            cv2.putText(out, label, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return out

    @staticmethod
    def _class_color_rgb(class_id: int) -> Tuple[int, int, int]:
        # Deterministic, well-separated class colors.
        hue = (int(class_id) * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        return int(r * 255), int(g * 255), int(b * 255)

    @staticmethod
    def _class_color_bgr(class_id: int) -> Tuple[int, int, int]:
        r, g, b = ValDetectionSegmentation._class_color_rgb(class_id)
        return b, g, r
