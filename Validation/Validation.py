from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import colorsys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torchvision.ops import box_iou
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image, ImageDraw, ImageFont


class IValidation(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def test_performance(self, output_dir: str, source: Optional[str], split: str, save_visuals: bool, save_txt: bool):
        pass

    @abstractmethod
    def predict_batch(self, images: torch.Tensor):
        pass

    @abstractmethod
    def predict_single(self, image: torch.Tensor):
        pass

    @abstractmethod
    def predict_from_bgr_frame(self, frame_bgr):
        pass

    @abstractmethod
    def test_from_dataset_split(self, split: str, labels_dir: Path, visuals_dir: Path, save_visuals: bool, save_txt: bool):
        pass

    @abstractmethod
    def test_from_video(self, source_path: Path, labels_dir: Path, visuals_dir: Path, save_visuals: bool, save_txt: bool):
        pass

    @abstractmethod
    def test_from_images(self, source_path: Path, labels_dir: Path, visuals_dir: Path, save_visuals: bool, save_txt: bool):
        pass


class ValidationBase(IValidation):
    def __init__(
        self,
        predictor: Callable,
        dataset: Optional[TorchDataset] = None,
        device: Optional[torch.device] = None,
    ):
        if predictor is None:
            raise ValueError(
                "predictor is None. Call your model's build_predictor(...) and pass the returned "
                "predictor (or use model.predictor) before creating Validation."
            )
        self.predictor = predictor
        self.dataset = dataset
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")   
    
    def start(self):
        pass

    def test_performance(
        self,
        output_dir: str,
        source: Optional[str] = None,
        split: str = "test",
        save_visuals: bool = True,
        save_txt: bool = True,
    ):
        output_root = Path(output_dir)
        labels_dir = output_root / "labels"
        visuals_dir = output_root / "visuals"
        labels_dir.mkdir(parents=True, exist_ok=True)
        if save_visuals:
            visuals_dir.mkdir(parents=True, exist_ok=True)

        class_names = getattr(self.dataset, "target_classes", None)
        num_classes = getattr(self.dataset, "class_num", 0) or (len(class_names) if class_names else 0)
        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_classes)]

        if source is None:
            return self.test_from_dataset_split(
                split=split,
                labels_dir=labels_dir,
                visuals_dir=visuals_dir,
                save_visuals=save_visuals,
                save_txt=save_txt,
            )

        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source does not exist: {source}")

        if self._is_video_file(source_path):
            return self.test_from_video(
                source_path=source_path,
                labels_dir=labels_dir,
                visuals_dir=visuals_dir,
                save_visuals=save_visuals,
                save_txt=save_txt,
            )

        return self.test_from_images(
            source_path=source_path,
            labels_dir=labels_dir,
            visuals_dir=visuals_dir,
            save_visuals=save_visuals,
            save_txt=save_txt,
        )

    def predict_batch(self, images: torch.Tensor):
        try:
            out = self.predictor(images)
            if isinstance(out, list):
                return out
            if isinstance(out, dict):
                return [out]
        except Exception:
            pass

        outputs = []
        for i in range(images.shape[0]):
            outputs.append(self.predict_single(images[i]))
        return outputs

    def predict_single(self, image):
            try:
                return self.predictor(image)
            except Exception:
                image_np = (
                    image.detach()
                    .cpu()
                    .permute(1, 2, 0)
                    .mul(255.0)
                    .clamp(0, 255)
                    .to(torch.uint8)
                    .numpy()
                )
                return self.predictor(image_np)

    def predict_from_bgr_frame(self, frame_bgr):
        try:
            return self.predictor(frame_bgr)
        except Exception:
            frame_rgb = frame_bgr[:, :, ::-1].copy()
            frame_t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
            return self.predict_single(frame_t)

    def test_from_dataset_split(self, split: str, labels_dir: Path, visuals_dir: Path, save_visuals: bool, save_txt: bool):
        pass

    def test_from_images(self, source_path: Path, labels_dir: Path, visuals_dir: Path, save_visuals: bool, save_txt: bool):
        pass

    def test_from_video(self, source_path: Path, labels_dir: Path, visuals_dir: Path, save_visuals: bool, save_txt: bool):
        pass

    @staticmethod
    def _collect_image_paths(source_path: Path) -> List[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if source_path.is_file():
            if source_path.suffix.lower() not in exts:
                raise ValueError(f"Unsupported image file extension: {source_path.suffix}")
            return [source_path]
        if not source_path.is_dir():
            raise ValueError(f"Unsupported source path: {source_path}")
        return sorted([p for p in source_path.iterdir() if p.is_file() and p.suffix.lower() in exts])
    
    @staticmethod
    def _is_video_file(path: Path) -> bool:
        return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
