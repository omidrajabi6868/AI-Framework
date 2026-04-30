from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Deque
from dataclasses import dataclass, field
from collections import deque

import cv2
import numpy as np
import torch

from .Validation import ValidationBase
from .ObjectTracking import IoUTracker, ByteLikeTracker, Track

@dataclass
class FallTrackState:
    track_id: int
    history: Deque[Tuple[int, np.ndarray]] = field(default_factory=lambda: deque(maxlen=30))
    fall_flags: Deque[bool] = field(default_factory=lambda: deque(maxlen=10))
    last_bbox: Optional[np.ndarray] = None
    last_seen: int = 0
    missed: int = 0
    drop_frame: Optional[int] = None
    fall: bool = False
    not_fall_count: int = 0

class ValFallDetection(ValidationBase):
    def __init__(
        self,
        predictor: Callable,
        dataset,
        person_class: str = "person",
        conf_tresh: float = 0.6,
        iou_tresh: float = 0.4,
        track_iou: float = 0.3,
        max_missed: int = 10,
        history: int = 30,
        aspect_thresh: float = 0.9,
        height_ratio: float = 0.9,
        drop_ratio: float = 0.03,
        drop_window: int = 20,
        persist_frames: int = 1,
        recover_frames: int = 1,
        tracker_type: str = "byte",
        byte_high: float = 0.6,
        byte_low: float = 0.2,
        use_hungarian: bool = True,
        use_motion: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.predictor = predictor
        self.dataset = dataset
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.person_class = str(person_class).lower()

        self.conf_tresh = float(conf_tresh)
        self.iou_tresh = float(iou_tresh)
        self.track_iou = float(track_iou)
        self.max_missed = int(max_missed)
        self.history = int(history)
        self.aspect_thresh = float(aspect_thresh)
        self.height_ratio = float(height_ratio)
        self.drop_ratio = float(drop_ratio)
        self.drop_window = int(drop_window)
        self.persist_frames = int(persist_frames)
        self.recover_frames = int(recover_frames)
        self.tracker_type = str(tracker_type).lower()
        self.byte_high = float(byte_high)
        self.byte_low = float(byte_low)
        self.use_hungarian = bool(use_hungarian)
        self.use_motion = bool(use_motion)

        if hasattr(self.predictor, "conf_thres"):
            self.predictor.conf_thres = self.conf_tresh
        if hasattr(self.predictor, "iou_thres"):
            self.predictor.iou_thres = self.iou_tresh

        if self.tracker_type in {"iou", "hungarian"}:
            self.tracker = IoUTracker(
                iou_thr=self.track_iou,
                max_missed=self.max_missed,
                use_hungarian=self.use_hungarian,
                use_motion=self.use_motion,
            )
        elif self.tracker_type in {"byte", "bytetrack"}:
            self.tracker = ByteLikeTracker(
                high_thr=self.byte_high,
                low_thr=self.byte_low,
                iou_thr=self.track_iou,
                max_missed=self.max_missed,
                use_hungarian=self.use_hungarian,
                use_motion=self.use_motion,
            )
        else:
            raise ValueError(f"Unsupported tracker_type: {self.tracker_type}")

        super().__init__(predictor=self.predictor, dataset=self.dataset, device=self.device)

    def test_from_dataset_split(self, split: str, labels_dir: Path, visuals_dir: Path, save_visuals: bool, save_txt: bool):
        raise NotImplementedError("Fall detection is a temporal task. Use test_from_video with a video source.")

    def test_from_images(self, source_path: Path, labels_dir: Path, visuals_dir: Path, save_visuals: bool, save_txt: bool):
        raise NotImplementedError("Fall detection requires a video sequence (multiple frames).")

    def test_from_video(
        self,
        source_path: Path,
        labels_dir: Path,
        visuals_dir: Path,
        save_visuals: bool,
        save_txt: bool,
        fps: Optional[float] = None,
    ):
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {source_path}")

        if fps is None or fps <= 0:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

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

        class_names = getattr(self.dataset, "target_classes", None) or []
        person_class_id = None
        for i, name in enumerate(class_names):
            if str(name).lower() == self.person_class:
                person_class_id = i
                break

        states: Dict[int, FallTrackState] = {}
        events: List[Tuple[int, int]] = []

        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            pred = self.predict_from_bgr_frame(frame_bgr)
            boxes, labels, scores = self._extract_detections(pred)
            if boxes.shape[0] > 0:
                keep = scores >= self.conf_tresh
                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]

            if person_class_id is not None and boxes.shape[0] > 0:
                keep = labels == int(person_class_id)
                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]

            tracker_out = self.tracker.update(boxes, scores)
            new_events = self.update_fall_states(
                states,
                tracker_out,
                frame_idx,
                history=self.history,
                persist_frames=self.persist_frames,
                max_missed=self.max_missed,
                aspect_thresh=self.aspect_thresh,
                height_ratio=self.height_ratio,
                drop_ratio=self.drop_ratio,
                drop_window=self.drop_window,
                recover_frames=self.recover_frames,
            )
            events.extend(new_events)

            if save_visuals and writer is not None:
                vis = self._draw_tracks(frame_bgr, states)
                writer.write(vis)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        if save_txt:
            lines = [f"{f},{tid}" for (f, tid) in events]
            (labels_dir / "falls.txt").write_text("\n".join(lines))

        return {
            "mode": "video",
            "source": str(source_path),
            "saved_items": frame_idx,
            "output_dir": str(labels_dir.parent),
            "visual_output": str(out_path) if out_path is not None else None,
            "fall_events": len(events),
        }
    
    def update_fall_states(
        self,
        states: Dict[int, FallTrackState],
        tracks: List[Track],
        frame_idx: int,
        *,
        history: int,
        persist_frames: int,
        max_missed: int,
        aspect_thresh: float,
        height_ratio: float,
        drop_ratio: float,
        drop_window: int,
        recover_frames: int,
    ) -> List[Tuple[int, int]]:
        events: List[Tuple[int, int]] = []
        live_ids = set()
        for tr in tracks:
            live_ids.add(tr.track_id)
            st = states.get(tr.track_id)
            if st is None:
                st = FallTrackState(track_id=tr.track_id)
                st.history = deque(maxlen=history)
                st.fall_flags = deque(maxlen=persist_frames)
                states[tr.track_id] = st
            event = self._update_fall_state(
                st,
                tr.bbox,
                frame_idx,
                aspect_thresh=aspect_thresh,
                height_ratio=height_ratio,
                drop_ratio=drop_ratio,
                drop_window=drop_window,
                recover_frames=recover_frames,
            )
            if event is not None:
                events.append(event)

        to_delete = []
        for tid, st in states.items():
            if tid not in live_ids:
                st.missed += 1
                if st.missed > max_missed:
                    to_delete.append(tid)
        for tid in to_delete:
            states.pop(tid, None)

        return events

    @staticmethod
    def _update_fall_state(
        st: FallTrackState,
        bbox: np.ndarray,
        frame_idx: int,
        *,
        aspect_thresh: float,
        height_ratio: float,
        drop_ratio: float,
        drop_window: int,
        recover_frames: int,
    ) -> Optional[Tuple[int, int]]:
        st.last_bbox = bbox
        st.last_seen = frame_idx
        st.missed = 0
        st.history.append((frame_idx, bbox.copy()))

        x1, y1, x2, y2 = bbox
        h = max(float(y2 - y1), 1.0)
        w = max(float(x2 - x1), 1.0)
        cy = 0.5 * (y1 + y2)

        heights = [max(float(b[3] - b[1]), 1.0) for _, b in st.history]
        median_h = float(np.median(heights)) if heights else h

        aspect = w / max(h, 1.0)
        fall_pose = (aspect >= aspect_thresh) or (h <= height_ratio * median_h)

        drop = False
        if len(st.history) >= 2:
            _, prev = st.history[-2]
            prev_cy = 0.5 * (prev[1] + prev[3])
            drop = ((cy - prev_cy) / max(median_h, 1.0)) >= drop_ratio
        if drop:
            st.drop_frame = frame_idx
        if st.drop_frame is not None and (frame_idx - st.drop_frame) > drop_window:
            st.drop_frame = None

        st.fall_flags.append(bool(fall_pose))

        fall_event = None
        if (not st.fall) and st.drop_frame is not None:
            if len(st.fall_flags) == st.fall_flags.maxlen and all(st.fall_flags):
                st.fall = True
                fall_event = (frame_idx, st.track_id)
                st.not_fall_count = 0
        else:
            if st.fall:
                if fall_pose:
                    st.not_fall_count = 0
                else:
                    st.not_fall_count += 1
                    if st.not_fall_count >= recover_frames:
                        st.fall = False
                        st.drop_frame = None
                        st.not_fall_count = 0

        return fall_event

    @staticmethod
    def _create_video_writer_cv2(visuals_dir: Path, source_stem: str, fps: float, width: int, height: int):
        fps = float(fps) if fps > 0 else 30.0
        candidates = [
            ("MJPG", ".mkv"),
            ("XVID", ".mkv"),
            ("H264", ".mkv"),
            ("avc1", ".mkv"),
            ("mp4v", ".mkv"),
            ("FFV1", ".mkv"),
        ]

        for codec, suffix in candidates:
            out_path = visuals_dir / f"{source_stem}_fall{suffix}"
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

    def _extract_detections(self, pred) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if pred is None:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        boxes = pred.get("boxes", torch.zeros((0, 4)))
        labels = pred.get("classes", pred.get("labels", torch.zeros((0,))))
        scores = pred.get("scores", torch.zeros((boxes.shape[0],)))

        boxes = boxes.detach().cpu().numpy().astype(np.float32).reshape(-1, 4)
        labels = labels.detach().cpu().numpy().astype(np.int64).reshape(-1)
        scores = scores.detach().cpu().numpy().astype(np.float32).reshape(-1)
        return boxes, labels, scores

    @staticmethod
    def _draw_tracks(frame_bgr: np.ndarray, tracks: Dict[int, FallTrackState]) -> np.ndarray:
        out = frame_bgr.copy()
        for tid, tr in tracks.items():
            if tr.last_bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in tr.last_bbox.tolist()]
            color = (0, 0, 255) if tr.fall else (0, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"id:{tid} FALL" if tr.fall else f"id:{tid}"
            cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out
