import mediapipe as mp
import torch
import numpy as np
from typing import Optional, List
from .SAHI import SAHIWrapper, SAHIConfig


class MediaPipeDetectionPredictor:
    def __init__(
        self,
        detector,
        class_names: Optional[List[str]] = None,
        sahi_config: Optional[SAHIConfig] = None,
        device: str = "cpu",
    ):
        self.detector = detector
        self.class_names = [str(c).lower() for c in class_names] if class_names else None
        self.conf_thres = None
        self.iou_thres = None
        self.device = device
        self.sahi = None

        if sahi_config is not None:
            self.sahi = self._build_sahi_wrapper(sahi_config)

    def _preprocess_for_model(self, image_input):
        """Preprocess input for MediaPipe detection.

        Returns:
            mp_image: The MediaPipe Image object.
            meta: dict with image dimensions.
        """
        if isinstance(image_input, mp.Image):
            mp_image = image_input
            image_h, image_w = int(mp_image.height), int(mp_image.width)
        else:
            if torch.is_tensor(image_input):
                t = image_input.detach().cpu()
                if t.ndim == 3:
                    if t.shape[0] in (1, 3):
                        img = t.permute(1, 2, 0).numpy()
                    else:
                        img = t.numpy()
                elif t.ndim == 4:
                    img = t[0].permute(1, 2, 0).numpy()
                else:
                    raise ValueError("Unsupported tensor shape for MediaPipe detection.")
            else:
                img = np.asarray(image_input)
            if img.dtype != np.uint8:
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            if img.shape[-1] != 3:
                raise ValueError("Expected RGB image with 3 channels for MediaPipe detection.")
            # Assume OpenCV-style BGR input for numpy frames.
            img = img[:, :, ::-1]
            img = np.ascontiguousarray(img, dtype=np.uint8)
            image_h, image_w = int(img.shape[0]), int(img.shape[1])
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        meta = {"image_h": image_h, "image_w": image_w}
        return mp_image, meta

    def _postprocess_mediapipe(self, result, meta: dict):
        """Convert MediaPipe detection output to standardized dict."""
        image_h = meta.get("image_h")
        image_w = meta.get("image_w")

        boxes = []
        scores = []
        labels = []

        for det in result.detections:
            if not det.categories:
                continue
            cat = det.categories[0]
            score = float(cat.score)
            if self.conf_thres is not None and score < float(self.conf_thres):
                continue

            label_id = None
            if hasattr(cat, "index") and cat.index is not None:
                try:
                    label_id = int(cat.index)
                except Exception:
                    label_id = None
            if label_id is None and self.class_names is not None:
                name = str(cat.category_name).lower()
                if name in self.class_names:
                    label_id = int(self.class_names.index(name))
            if label_id is None:
                label_id = 0

            bbox = det.bounding_box
            x1 = float(bbox.origin_x)
            y1 = float(bbox.origin_y)
            x2 = x1 + float(bbox.width)
            y2 = y1 + float(bbox.height)
            x1 = max(0.0, min(x1, image_w - 1))
            y1 = max(0.0, min(y1, image_h - 1))
            x2 = max(0.0, min(x2, image_w - 1))
            y2 = max(0.0, min(y2, image_h - 1))

            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            labels.append(label_id)

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            scores_t = torch.zeros((0,), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            scores_t = torch.tensor(scores, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)

        return {
            "boxes": boxes_t,
            "scores": scores_t,
            "classes": labels_t,
            "box_format": "xyxy",
            "normalized": False,
            "image_size": (image_h, image_w),
        }

    def _build_sahi_wrapper(self, sahi_config: SAHIConfig) -> SAHIWrapper:
        def preprocess_fn(image_slice):
            return self._preprocess_for_model(image_slice)

        def postprocess_fn(raw_output, meta):
            return self._postprocess_mediapipe(raw_output, meta)

        return SAHIWrapper(
            model_fn=self.detector.detect,
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

    def __call__(self, image_input):
        if self.sahi is not None:
            return self.sahi(image_input)

        mp_image, meta = self._preprocess_for_model(image_input)
        result = self.detector.detect(mp_image)
        return self._postprocess_mediapipe(result, meta)
