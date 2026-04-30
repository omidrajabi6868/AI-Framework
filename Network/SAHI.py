import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Any, Callable, Optional
from torchvision.ops import nms as torch_nms


class SAHIWrapper:
    """SAHI wrapper for model outputs."""

    def __init__(
        self,
        model_fn: Callable,
        preprocess_fn: Callable,
        postprocess_fn: Callable,
        slice_size: int = 640,
        overlap: float = 0.2,
        merge_mode: str = 'nms',
        nms_threshold: float = 0.5,
        class_agnostic_nms: bool = False,
        device: Optional[str] = None,
        min_box_area: float = 9.0,
        min_aspect_ratio: float = 0.02,
    ):
        self.model_fn = model_fn
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.slice_size = int(slice_size)
        self.overlap = float(overlap)
        self.merge_mode = str(merge_mode).lower()
        self.nms_threshold = float(nms_threshold)
        self.class_agnostic_nms = bool(class_agnostic_nms)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = str(device)

        # (No debug flags in production wrapper)
        # Filtering thresholds for removing spurious boxes after merging
        self.min_box_area = float(min_box_area)
        self.min_aspect_ratio = float(min_aspect_ratio)

        if self.overlap < 0.0 or self.overlap >= 1.0:
            raise ValueError(f"overlap must be in [0.0, 1.0), got {self.overlap}")

        if self.merge_mode not in ('nms', 'keep_all'):
            raise ValueError(f"merge_mode must be 'nms' or 'keep_all', got {self.merge_mode}")
    
    def _get_slice_coords(self, image_h: int, image_w: int) -> List[Tuple[int, int, int, int]]:
        """
        Generate slice coordinates (y1, x1, y2, x2) covering the entire image
        with specified overlap.
        """
        # Ensure we always make progress when tiling. If (1 - overlap) is very small,
        # int(...) can round down to 0, which would cause an infinite loop.
        stride = int(self.slice_size * (1.0 - self.overlap))
        if stride <= 0:
            stride = 1

        coords = []
        
        y = 0
        while y < image_h:
            x = 0
            while x < image_w:
                y1 = y
                x1 = x
                y2 = min(y + self.slice_size, image_h)
                x2 = min(x + self.slice_size, image_w)
                
                coords.append((y1, x1, y2, x2))
                
                x += stride
                if x2 == image_w:
                    break
            # CRITICAL: Increment y to process next row of slices
            y += stride
            if y2 == image_h:
                break
        return coords
    
    def _extract_slice(self, image: Union[torch.Tensor, np.ndarray],
                       y1: int, x1: int, y2: int, x2: int) -> Union[torch.Tensor, np.ndarray]:
        """Extract a slice from the image."""
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                return image[:, :, y1:y2, x1:x2]
            elif image.ndim == 3:
                return image[:, y1:y2, x1:x2]
            else:
                raise ValueError(f"Unsupported tensor shape: {image.shape}")
        elif isinstance(image, np.ndarray):
            return image[y1:y2, x1:x2]
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
    
    def _adjust_boxes_to_original(
        self,
        boxes: torch.Tensor,
        offset_y: int,
        offset_x: int,
    ) -> torch.Tensor:
        """Adjust box coordinates from slice-local to image-global coordinates."""
        if boxes.numel() == 0:
            return boxes
        
        adjusted = boxes.clone()
        adjusted[:, [0, 2]] += offset_x
        adjusted[:, [1, 3]] += offset_y
        return adjusted
    
    def _merge_predictions(self, all_predictions: List[Dict[str, Any]], full_image_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Merge predictions from multiple slices.
        
        Args:
            all_predictions: List of prediction dicts from each slice.
            
        Returns:
            Merged prediction dict.
        """
        if not all_predictions:
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                'scores': torch.zeros((0,), dtype=torch.float32, device=self.device),
                'classes': torch.zeros((0,), dtype=torch.long, device=self.device),
            }
        
        # Concatenate all predictions
        all_boxes = []
        all_scores = []
        all_classes = []
        all_masks = []
        
        for pred in all_predictions:
            boxes = pred.get('boxes')
            scores = pred.get('scores')
            classes = pred.get('classes')
            masks = pred.get('masks')
            
            if boxes is not None and boxes.numel() > 0:
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_classes.append(classes)
                if masks is not None and getattr(masks, 'numel', lambda:0)() > 0:
                    # ensure masks are same device/dtype
                    all_masks.append(masks.to(device=self.device))
        
        if not all_boxes:
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                'scores': torch.zeros((0,), dtype=torch.float32, device=self.device),
                'classes': torch.zeros((0,), dtype=torch.long, device=self.device),
            }
        
        # Use CPU for concatenation of many predictions to reduce GPU memory pressure
        concat_device = 'cpu' if (self.device == 'cuda' and len(all_predictions) > 4) else self.device
        merged_boxes = torch.cat(all_boxes, dim=0).to(concat_device)
        merged_scores = torch.cat(all_scores, dim=0).to(concat_device)
        merged_classes = torch.cat(all_classes, dim=0).to(concat_device)
        merged_masks = torch.cat(all_masks, dim=0).to(concat_device) if all_masks else None
        
        # Delete intermediate lists to free memory
        del all_boxes, all_scores, all_classes, all_masks

        # (No debug prints)

        # Basic validation and cleaning of merged boxes. Remove degenerate
        # boxes (non-positive width/height) and extremely narrow/flat boxes that
        # are likely artifacts of slicing or rounding. Keep threshold conservative
        # (>=1.0 pixel) to preserve small objects while removing invalid ones.
        if merged_boxes.numel() > 0:
            widths = merged_boxes[:, 2] - merged_boxes[:, 0]
            heights = merged_boxes[:, 3] - merged_boxes[:, 1]
            areas = widths * heights
            # Aspect ratio defined as min(width/height, height/width) to catch very thin boxes
            aspect_ratio = torch.min(widths / (heights + 1e-6), heights / (widths + 1e-6))

            # Base validity: finite and at least 1 pixel in each dimension
            valid_mask = (widths >= 1.0) & (heights >= 1.0) & torch.isfinite(widths) & torch.isfinite(heights)

            # Additional heuristics: minimum area and minimum aspect ratio
            if self.min_box_area > 0:
                valid_mask &= (areas >= float(self.min_box_area))
            if self.min_aspect_ratio > 0:
                valid_mask &= (aspect_ratio >= float(self.min_aspect_ratio))

            if valid_mask.all().item() is False:
                merged_boxes = merged_boxes[valid_mask]
                merged_scores = merged_scores[valid_mask]
                merged_classes = merged_classes[valid_mask]

        # Ensure scores are floats in reasonable range
        if merged_scores.numel() > 0:
            merged_scores = merged_scores.clamp(min=0.0, max=1.0).to(torch.float32)
        
        # Move to processing device for NMS
        merged_boxes = merged_boxes.to(self.device)
        merged_scores = merged_scores.to(self.device)
        merged_classes = merged_classes.to(self.device)
        if merged_masks is not None:
            merged_masks = merged_masks.to(self.device)
        
        if self.merge_mode == 'keep_all':
            result = {
                'boxes': merged_boxes,
                'scores': merged_scores,
                'classes': merged_classes,
            }
            if merged_masks is not None:
                result['masks'] = merged_masks
        elif self.merge_mode == 'nms':
            result = self._apply_nms(merged_boxes, merged_scores, merged_classes, masks=merged_masks)
        else:
            result = {
                'boxes': merged_boxes,
                'scores': merged_scores,
                'classes': merged_classes,
            }
            if merged_masks is not None:
                result['masks'] = merged_masks
        
        # Preserve non-spatial fields from the first prediction only.
        # Avoid copying slice-local or spatial fields (masks, proto, mask_coeffs,
        # etc.) which are not valid for the merged, full-image result.
        SPATIAL_KEYS = {'boxes', 'scores', 'classes', 'masks', 'mask_coeffs', 'proto'}
        if all_predictions and all_predictions[0]:
            for key, val in all_predictions[0].items():
                if key in SPATIAL_KEYS:
                    continue
                result[key] = val

        # Ensure merged result points to the full image size (not the per-slice image_size)
        if full_image_size is not None:
            result['image_size'] = (int(full_image_size[0]), int(full_image_size[1]))
        else:
            # If no full image size provided, keep the first prediction's image_size if present
            if 'image_size' not in result and all_predictions and isinstance(all_predictions[0], dict):
                result['image_size'] = all_predictions[0].get('image_size', None)

        # Merged predictions are in absolute pixel coordinates
        result['box_format'] = 'xyxy'
        result['normalized'] = False
        # Attach masks if we have them. If NMS was applied, _apply_nms will
        # have kept the corresponding indices; otherwise, use merged_masks directly.
        if merged_masks is not None:
            if 'boxes' in result and result['boxes'].numel() > 0:
                # need to align masks with kept boxes
                if 'masks' in result and result['masks'] is not None:
                    # already handled by _apply_nms
                    pass
                else:
                    # if no NMS was applied, or result maps directly, try to match by count
                    if merged_masks.shape[0] == merged_boxes.shape[0]:
                        result['masks'] = merged_masks
                    else:
                        # fallback: do not attach masks if counts mismatch
                        pass
        
        return result
    
    def _apply_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Apply NMS to remove duplicate detections from overlapping slices."""
        if boxes.numel() == 0:
            return {
                'boxes': boxes,
                'scores': scores,
                'classes': classes,
            }
        
        # Move to CPU for NMS to reduce GPU memory pressure
        original_device = boxes.device
        boxes_cpu = boxes.to('cpu')
        scores_cpu = scores.to('cpu')
        classes_cpu = classes.to('cpu')
        
        # Sort by score first so that higher-confidence detections are kept
        sorted_scores, sort_indices = torch.sort(scores_cpu, descending=True)
        sorted_boxes = boxes_cpu[sort_indices]
        sorted_classes = classes_cpu[sort_indices]
        
        # Compute keep indices either class-agnostic or per-class
        if self.class_agnostic_nms:
            # Global NMS across all classes
            keep = torch_nms(sorted_boxes, sorted_scores, self.nms_threshold)
        else:
            keep_list = []
            for cls_id in sorted_classes.unique():
                mask = sorted_classes == cls_id
                cls_boxes = sorted_boxes[mask]
                cls_scores = sorted_scores[mask]
                cls_indices = torch.where(mask)[0]

                if cls_boxes.numel() > 0:
                    # Apply size-aware NMS for this class
                    cls_keep = self._size_aware_nms(cls_boxes, cls_scores, self.nms_threshold)
                    keep_list.append(cls_indices[cls_keep])

            if keep_list:
                keep = torch.cat(keep_list, dim=0)
            else:
                keep = torch.tensor([], dtype=torch.long, device=sorted_boxes.device)
        
        # Map back to original indices
        final_keep = sort_indices[keep]

        # Move results back to original device
        result = {
            'boxes': boxes[final_keep].to(original_device),
            'scores': scores[final_keep].to(original_device),
            'classes': classes[final_keep].to(original_device),
        }
        if masks is not None and getattr(masks, 'numel', lambda:0)() > 0:
            try:
                result['masks'] = masks[final_keep].to(original_device)
            except Exception:
                # If masks cannot be aligned, omit them silently in production
                pass

        return result
    
    def _size_aware_nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """
        Size-aware NMS that prefers larger detections over smaller ones when IoU is high.
        This helps preserve small detections while removing face-only duplicates of larger body detections.
        """
        if boxes.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)
        
        # Calculate box areas (width * height)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        
        # Sort by score (already done by caller, but ensure consistency)
        _, order = torch.sort(scores, descending=True)
        boxes = boxes[order]
        scores = scores[order]
        areas = areas[order]
        
        keep = []
        while boxes.numel() > 0:
            # Keep the highest scoring box
            keep.append(order[0])
            
            if len(boxes) == 1:
                break
                
            # Calculate IoU with remaining boxes
            ious = self._calculate_iou(boxes[0:1], boxes[1:])
            
            # Size-aware suppression: be more lenient with small boxes
            # If the current box is much larger than overlapping boxes, suppress them more aggressively
            size_ratios = areas[0] / areas[1:]  # How much larger current box is
            
            # Adaptive threshold: lower threshold for large size differences
            adaptive_threshold = iou_threshold * torch.clamp(1.0 / (size_ratios + 0.5), 0.3, 1.0)
            
            # Suppress boxes that exceed the adaptive threshold
            suppress_mask = ious < adaptive_threshold
            
            # Keep boxes that pass the threshold
            keep_indices = torch.where(suppress_mask)[0] + 1  # +1 because we compare against boxes[1:]
            
            if keep_indices.numel() == 0:
                break
                
            # Update for next iteration
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            areas = areas[keep_indices]
            order = order[keep_indices]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    
    def _calculate_iou(self, box1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between one box and multiple boxes."""
        # box1: [1, 4], boxes2: [N, 4]
        x1 = torch.max(box1[:, 0], boxes2[:, 0])
        y1 = torch.max(box1[:, 1], boxes2[:, 1])
        x2 = torch.min(box1[:, 2], boxes2[:, 2])
        y2 = torch.min(box1[:, 3], boxes2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        union = area1 + area2 - intersection
        return intersection / torch.clamp(union, min=1e-6)
    
    def __call__(self, image_input: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """
        Run inference with SAHI.
        
        Args:
            image_input: Input image as torch.Tensor or np.ndarray.
                        Can be HxWx3 (RGB) or CxHxW or BxCxHxW.
        
        Returns:
            Prediction dict with merged results from all slices.
        """
        # Determine image dimensions
        if isinstance(image_input, torch.Tensor):
            if image_input.ndim == 4:
                _, _, h, w = image_input.shape
            elif image_input.ndim == 3:
                if image_input.shape[0] in (1, 3):
                    _, h, w = image_input.shape
                else:
                    h, w, _ = image_input.shape
            else:
                raise ValueError(f"Unsupported tensor shape: {image_input.shape}")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3:
                h, w, _ = image_input.shape
            elif image_input.ndim == 2:
                h, w = image_input.shape
            else:
                raise ValueError(f"Unsupported array shape: {image_input.shape}")
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")
        
        # If image is small enough, run inference without slicing
        if h <= self.slice_size and w <= self.slice_size:
            # Preprocess -> model -> postprocess
            pre = self.preprocess_fn(image_input)
            if isinstance(pre, tuple):
                inp, meta = pre
                raw = self.model_fn(inp)
                return self.postprocess_fn(raw, meta)
            raw = self.model_fn(pre)
            return self.postprocess_fn(raw)

        # Get slice coordinates
        slice_coords = self._get_slice_coords(h, w)
        all_predictions = []

        # Run inference on each slice
        for y1, x1, y2, x2 in slice_coords:
            slice_img = self._extract_slice(image_input, y1, x1, y2, x2)

            pre = self.preprocess_fn(slice_img)
            if isinstance(pre, tuple):
                inp, meta = pre
                raw = self.model_fn(inp)
                slice_pred = self.postprocess_fn(raw, meta)
            else:
                raw = self.model_fn(pre)
                slice_pred = self.postprocess_fn(raw)

            # Adjust box coordinates to original image space
            if slice_pred.get('boxes') is not None and slice_pred['boxes'].numel() > 0:
                slice_pred['boxes'] = self._adjust_boxes_to_original(
                    slice_pred['boxes'],
                    offset_y=y1,
                    offset_x=x1,
                )

            # If slice produced masks, paste them into a full-image canvas so
            # merged masks can be concatenated and indexed alongside boxes.
            if slice_pred.get('masks') is not None and getattr(slice_pred['masks'], 'numel', lambda:0)() > 0:
                masks = slice_pred['masks']
                # ensure tensor
                if not torch.is_tensor(masks):
                    masks = torch.tensor(masks)
                masks = masks.to(torch.float32)
                # masks shape expected [N, mh, mw]
                if masks.ndim == 2:
                    masks = masks.unsqueeze(0)
                n_masks = masks.shape[0]
                h_slice = y2 - y1
                w_slice = x2 - x1
                # resize masks to the slice size if needed
                if masks.shape[1] != h_slice or masks.shape[2] != w_slice:
                    masks = F.interpolate(masks.unsqueeze(1), size=(h_slice, w_slice), mode='bilinear', align_corners=False).squeeze(1)

                # create full-image mask canvas and place slice masks
                full_masks = torch.zeros((n_masks, h, w), dtype=masks.dtype, device=masks.device)
                full_masks[:, y1:y2, x1:x2] = masks.to(device=full_masks.device)
                slice_pred['masks'] = full_masks

            all_predictions.append(slice_pred)
            
            # Free GPU memory after processing each slice to prevent accumulation
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        # Merge all predictions (pass full image size so merged metadata is correct)
        merged = self._merge_predictions(all_predictions, full_image_size=(h, w))
        
        # Clean up
        del all_predictions
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return merged


class SAHIConfig:
    """Configuration for SAHI inference."""
    
    def __init__(
        self,
        slice_size: int = 640,
        max_input_size: Optional[int] = None,
        overlap: float = 0.2,
        merge_mode: str = 'nms',
        nms_threshold: float = 0.5,
        class_agnostic_nms: bool = False,
        min_box_area: float = 9.0,
        min_aspect_ratio: float = 0.02,
    ):
        # Accept `max_input_size` as an alias used elsewhere in the codebase.
        if max_input_size is not None:
            slice_size = max_input_size
        self.slice_size = int(slice_size)
        self.overlap = float(overlap)
        self.merge_mode = str(merge_mode).lower()
        self.nms_threshold = float(nms_threshold)
        self.class_agnostic_nms = bool(class_agnostic_nms)
        self.min_box_area = float(min_box_area)
        self.min_aspect_ratio = float(min_aspect_ratio)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'slice_size': self.slice_size,
            'overlap': self.overlap,
            'merge_mode': self.merge_mode,
            'nms_threshold': self.nms_threshold,
            'class_agnostic_nms': self.class_agnostic_nms,
            'min_box_area': self.min_box_area,
            'min_aspect_ratio': self.min_aspect_ratio,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SAHIConfig':
        """Create config from dictionary."""
        return cls(**d)
