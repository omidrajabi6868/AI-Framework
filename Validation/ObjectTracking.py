from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import cv2
import numpy as np
import torch

from .Validation import ValidationBase


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    score: float
    missed: int = 0
    velocity: Optional[np.ndarray] = None


class BaseTracker:
    def update(self, boxes: np.ndarray, scores: np.ndarray) -> List[Track]:
        raise NotImplementedError


class IoUTracker(BaseTracker):
    def __init__(
        self,
        iou_thr: float = 0.3,
        max_missed: int = 10,
        use_hungarian: bool = True,
        use_motion: bool = True,
    ):
        self.iou_thr = float(iou_thr)
        self.max_missed = int(max_missed)
        self.use_hungarian = bool(use_hungarian)
        self.use_motion = bool(use_motion)
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> List[Track]:
        boxes = boxes.astype(np.float32).reshape(-1, 4)
        scores = scores.astype(np.float32).reshape(-1)

        track_ids = list(self.tracks.keys())
        matched_tracks = set()
        matched_boxes = set()

        if len(track_ids) > 0 and boxes.shape[0] > 0:
            iou_mat = np.zeros((len(track_ids), boxes.shape[0]), dtype=np.float32)
            for ti, tid in enumerate(track_ids):
                tbox = self._predict_bbox(self.tracks[tid])
                for bi in range(boxes.shape[0]):
                    iou_mat[ti, bi] = self._iou(tbox, boxes[bi])

            matches = self._hungarian_match(iou_mat, self.iou_thr) if self.use_hungarian else self._greedy_match(iou_mat, self.iou_thr)

            for ti, bi in matches:
                tid = track_ids[ti]
                matched_tracks.add(tid)
                matched_boxes.add(bi)
                tr = self.tracks[tid]
                self._update_track(tr, boxes[bi], scores[bi])

        for bi in range(boxes.shape[0]):
            if bi in matched_boxes:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(track_id=tid, bbox=boxes[bi], score=float(scores[bi]))

        to_del = []
        for tid, tr in self.tracks.items():
            if tid not in matched_tracks:
                tr.missed += 1
                if tr.missed > self.max_missed:
                    to_del.append(tid)
        for tid in to_del:
            self.tracks.pop(tid, None)

        return list(self.tracks.values())

    def _update_track(self, tr: Track, bbox: np.ndarray, score: float):
        if tr.bbox is not None:
            tr.velocity = bbox - tr.bbox
        tr.bbox = bbox
        tr.score = float(score)
        tr.missed = 0

    def _predict_bbox(self, tr: Track) -> np.ndarray:
        if not self.use_motion or tr.velocity is None:
            return tr.bbox
        return tr.bbox + tr.velocity

    @staticmethod
    def _greedy_match(iou_mat: np.ndarray, iou_thr: float) -> List[Tuple[int, int]]:
        matches: List[Tuple[int, int]] = []
        if iou_mat.size == 0:
            return matches
        used_rows = set()
        used_cols = set()
        mat = iou_mat.copy()
        while True:
            ti, bi = np.unravel_index(np.argmax(mat), mat.shape)
            if mat[ti, bi] < iou_thr:
                break
            if ti in used_rows or bi in used_cols:
                mat[ti, bi] = -1.0
                continue
            used_rows.add(ti)
            used_cols.add(bi)
            matches.append((ti, bi))
            mat[ti, bi] = -1.0
        return matches

    @staticmethod
    def _hungarian_match(iou_mat: np.ndarray, iou_thr: float) -> List[Tuple[int, int]]:
        if iou_mat.size == 0:
            return []
        n, m = iou_mat.shape
        transpose = False
        cost = 1.0 - iou_mat
        if n > m:
            cost = cost.T
            n, m = cost.shape
            transpose = True

        inf_cost = 1e6
        cost = cost.copy()
        cost[cost > (1.0 - iou_thr)] = inf_cost

        u = np.zeros(n + 1, dtype=np.float32)
        v = np.zeros(m + 1, dtype=np.float32)
        p = np.zeros(m + 1, dtype=np.int32)
        way = np.zeros(m + 1, dtype=np.int32)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = np.full(m + 1, np.inf, dtype=np.float32)
            used = np.zeros(m + 1, dtype=bool)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = np.inf
                j1 = 0
                for j in range(1, m + 1):
                    if used[j]:
                        continue
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                for j in range(m + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        assignment = -np.ones(n, dtype=np.int32)
        for j in range(1, m + 1):
            if p[j] != 0:
                assignment[p[j] - 1] = j - 1

        matches = []
        for i in range(n):
            j = assignment[i]
            if j < 0:
                continue
            if cost[i, j] >= inf_cost:
                continue
            if transpose:
                matches.append((j, i))
            else:
                matches.append((i, j))
        return matches

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return inter / union


class ByteLikeTracker(BaseTracker):
    def __init__(
        self,
        high_thr: float = 0.5,
        low_thr: float = 0.1,
        iou_thr: float = 0.3,
        max_missed: int = 10,
        use_hungarian: bool = True,
        use_motion: bool = True,
    ):
        self.high_thr = float(high_thr)
        self.low_thr = float(low_thr)
        self.iou_thr = float(iou_thr)
        self.max_missed = int(max_missed)
        self.use_hungarian = bool(use_hungarian)
        self.use_motion = bool(use_motion)
        self.tracker = IoUTracker(
            iou_thr=self.iou_thr,
            max_missed=self.max_missed,
            use_hungarian=self.use_hungarian,
            use_motion=self.use_motion,
        )

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> List[Track]:
        boxes = boxes.astype(np.float32).reshape(-1, 4)
        scores = scores.astype(np.float32).reshape(-1)

        high_mask = scores >= self.high_thr
        low_mask = (scores >= self.low_thr) & (scores < self.high_thr)

        high_boxes = boxes[high_mask]
        high_scores = scores[high_mask]
        low_boxes = boxes[low_mask]
        low_scores = scores[low_mask]

        tracks = self.tracker.update(high_boxes, high_scores)

        if low_boxes.shape[0] > 0 and len(self.tracker.tracks) > 0:
            # Try to recover unmatched tracks using low confidence detections.
            _ = self.tracker.update(low_boxes, low_scores)

        return list(self.tracker.tracks.values())


class ValObjectTracking(ValidationBase):
    def __init__(
        self,
        predictor: Callable,
        dataset,
        tracker_type: str = "byte",
        conf_tresh: float = 0.25,
        iou_tresh: float = 0.5,
        track_iou: float = 0.3,
        max_missed: int = 10,
        use_hungarian: bool = True,
        use_motion: bool = True,
        byte_high: float = 0.5,
        byte_low: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        self.predictor = predictor
        self.dataset = dataset
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        self.conf_tresh = float(conf_tresh)
        self.iou_tresh = float(iou_tresh)

        if hasattr(self.predictor, "conf_thres"):
            self.predictor.conf_thres = self.conf_tresh
        if hasattr(self.predictor, "iou_thres"):
            self.predictor.iou_thres = self.iou_tresh

        tracker_type = str(tracker_type).lower()
        if tracker_type in {"iou", "hungarian"}:
            self.tracker = IoUTracker(
                iou_thr=track_iou,
                max_missed=max_missed,
                use_hungarian=use_hungarian,
                use_motion=use_motion,
            )
        elif tracker_type in {"byte", "bytetrack"}:
            self.tracker = ByteLikeTracker(
                high_thr=byte_high,
                low_thr=byte_low,
                iou_thr=track_iou,
                max_missed=max_missed,
                use_hungarian=use_hungarian,
                use_motion=use_motion,
            )
        else:
            raise ValueError(f"Unsupported tracker_type: {tracker_type}")

        super().__init__(predictor=self.predictor, dataset=self.dataset, device=self.device)

    def test_from_dataset_split(self, split: str, labels_dir: Path, visuals_dir: Path, save_visuals: bool, save_txt: bool):
        raise NotImplementedError("Object tracking requires a video sequence.")

    def test_from_images(self, source_path: Path, labels_dir: Path, visuals_dir: Path, save_visuals: bool, save_txt: bool):
        raise NotImplementedError("Object tracking requires a video sequence.")

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

        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            pred = self.predict_from_bgr_frame(frame_bgr)
            boxes, labels, scores = self._extract_detections(pred)
            keep = scores >= self.conf_tresh
            boxes = boxes[keep]
            scores = scores[keep]

            tracks = self.tracker.update(boxes, scores)

            if save_visuals and writer is not None:
                vis = self._draw_tracks(frame_bgr, tracks)
                writer.write(vis)

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
            out_path = visuals_dir / f"{source_stem}_tracks{suffix}"
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
    def _extract_detections(pred) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    def _draw_tracks(frame_bgr: np.ndarray, tracks: List[Track]) -> np.ndarray:
        out = frame_bgr.copy()
        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.bbox.tolist()]
            color = (0, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"id:{tr.track_id}"
            cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out
