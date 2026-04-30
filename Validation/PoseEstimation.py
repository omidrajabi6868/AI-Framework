from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import colorsys
import cv2

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torchvision.ops import box_iou
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image, ImageDraw, ImageFont

from .Validation import ValidationBase

import mediapipe as mp



class ValPoseEstimation(ValidationBase):
    def __init__(
        self,
        model_name: str,
        predictor: Callable,
        dataset: Optional[TorchDataset] = None,
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        self.predictor = predictor
        self.dataset = dataset
        self.device = device if device is not None else torch.device("cpu")

    def start(self, gt, pred):
        raise NotImplementedError("Pose estimation evaluation is not implemented yet.")

    def predict_batch(self, images: torch.Tensor):
        raise NotImplementedError("Pose estimation evaluation is not implemented yet.")

    def predict_single(self, image):
        try:
            if self.model_name == "mediapipe":
                return self.predictor.detect(image)
            else:
                return self.predictor(image)
        except Exception as e:
            print(f"Error in prediction: {e}")

    def predict_from_bgr_frame(self, frame_bgr):
        raise NotImplementedError("Pose estimation evaluation is not implemented yet.")

    def test_from_dataset_split(self, split: str, labels_dir: Path, visuals_dir: Path, save_visuals: bool, save_txt: bool):
        raise NotImplementedError("Pose estimation evaluation is not implemented yet.")

    def test_from_video(self, source_path: Path, labels_dir: Path, visuals_dir: Path, save_visuals: bool, save_txt: bool):
        raise NotImplementedError("Pose estimation evaluation is not implemented yet.")

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

            if self.model_name == "mediapipe":
                img_np = np.ascontiguousarray(img_np, dtype=np.uint8)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
                pred = self.predict_single(mp_image)

            h, w = int(img_t.shape[1]), int(img_t.shape[2])

            if save_txt:
                pass
            if save_visuals:
                if self.model_name == 'mediapipe':
                    annotated_image = self._pipemedia_draw_landmarks_on_image(mp_image.numpy_view(), detection_result=pred)
                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f'{visuals_dir}/{p.stem}.jpg', annotated_image)
                        
            saved += 1

        return {
            "mode": "images",
            "source": str(source_path),
            "saved_items": saved,
            "output_dir": str(labels_dir.parent),
        }
    
    @staticmethod
    def _pipemedia_draw_landmarks_on_image(rgb_image, detection_result):
        # detection_result.pose_landmarks is a list of lists of landmarks
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)
        
        # Define connection mapping (standard Pose 33 landmarks)
        # This replaces vision.PoseLandmarksConnections.POSE_LANDMARKS
        POSE_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), (12, 24),
            (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31),
            (30, 32), (27, 31), (28, 32)
        ]

        height, width, _ = rgb_image.shape

        for pose_landmarks in pose_landmarks_list:
            # 1. Draw Connections (Lines)
            for connection in POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                # Get normalized coordinates and convert to pixel coordinates
                start_point = pose_landmarks[start_idx]
                end_point = pose_landmarks[end_idx]
                
                start_coords = (int(start_point.x * width), int(start_point.y * height))
                end_coords = (int(end_point.x * width), int(end_point.y * height))
                
                cv2.line(annotated_image, start_coords, end_coords, (0, 255, 0), 2)

            # 2. Draw Landmarks (Dots)
            for landmark in pose_landmarks:
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(annotated_image, (cx, cy), 3, (255, 255, 255), -1)
                cv2.circle(annotated_image, (cx, cy), 3, (0, 0, 255), 1)

        return annotated_image