import mediapipe as mp
from abc import ABC, abstractmethod
import json
from pathlib import Path

import torch
from torch import nn
import numpy as np
from typing import Optional, List

from .YoloIPC8 import YOLOIPC8, IPC8Predictor
from .YoloIPC12 import YOLOv12
from .YoloIPC4 import YOLOIPC4, IPC4Predictor
from .onnx_wrappers import (
    IPC8ONNXExportWrapper,
    IPC8SegONNXExportWrapper,
    IPC4ONNXExportWrapper,
    export_onnx_model,
    build_output_names,
)
from .utils import make_anchors, load_state_dict_compat
from .CustomMediaPipe import MediaPipeDetectionPredictor
from .SAHI import SAHIConfig


class INetwork(ABC):
    @abstractmethod
    def build(self):
        pass
    @abstractmethod
    def export_onnx(self, onnx_path: str,
        checkpoint_path: str,
        input_size: tuple,
        opset_version: int,
        dynamic_axes: bool,
        normalize_boxes: bool):
        pass


class NYOLO(INetwork):
    def __init__(
        self,
        name,
        size,
        num_classes,
        reg_max=16,
        task: str = "detection",
        num_masks: int = 32,
        num_protos: int = 32,
        use_sahi: bool = False,
    ):
        self.name = name
        self.size = size
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.task = str(task).lower()
        self.num_masks = int(num_masks)
        self.num_protos = int(num_protos)
        self.strides = (8, 16, 32)
        self.use_sahi = bool(use_sahi)
        pass

    def build(self):
        if self.name in {'yolov8', 'ipc8'}:
            yolo_model = YOLOIPC8(
                self.size,
                self.num_classes,
                self.reg_max,
                task=self.task,
                num_masks=self.num_masks,
                num_protos=self.num_protos,
            )
        elif self.name == 'yolov12':
            yolo_model = YOLOv12(
                self.size,
                self.num_classes,
                self.reg_max,
                task=self.task,
                num_masks=self.num_masks,
                num_protos=self.num_protos,
            )
        elif self.name in {"ipc4", "ipc4-full", "yolov4", "yolov4-full"}:
            yolo_model = YOLOIPC4(
                self.size if self.size else "full",
                self.num_classes,
                task=self.task,
                num_masks=self.num_masks,
                num_protos=self.num_protos,
                reg_max=self.reg_max,
            )
        elif self.name in {"ipc4-tiny", "yolov4-tiny", "yolov4-tiny-custom"}:
            yolo_model = YOLOIPC4(
                "tiny",
                self.num_classes,
                task=self.task,
                num_masks=self.num_masks,
                num_protos=self.num_protos,
                reg_max=self.reg_max,
            )
        else:
            raise ValueError(f"Unsupported YOLO model name: {self.name}")
        self.model = yolo_model
        pass

    def build_predictor(self, class_names, max_input_size: int = 1024):
        if self.name in {"ipc4", "ipc4-full", "ipc4-tiny", "yolov4", "yolov4-full", "yolov4-tiny", "yolov4-tiny-custom"}:
            if self.task == "detection":
                sahi_config = SAHIConfig(class_agnostic_nms=True, slice_size=max_input_size, overlap=0.2, nms_threshold=0.4) if self.use_sahi else None
                self.predictor = IPC4Predictor(
                    self.model,
                    conf_thres=0.25,
                    iou_thres=0.5,
                    max_input_size=max_input_size,
                    sahi_config=sahi_config,
                )
            else:
                # Instance segmentation uses the shared task head outputs.
                sahi_config = SAHIConfig(class_agnostic_nms=True, slice_size=max_input_size, overlap=0.2, nms_threshold=0.4) if self.use_sahi else None
                self.predictor = IPC8Predictor(
                    self.model,
                    class_names=class_names,
                    max_input_size=max_input_size,
                    sahi_config=sahi_config,
                )
        else:
            sahi_config = SAHIConfig(class_agnostic_nms=True, slice_size=max_input_size, overlap=0.2, nms_threshold=0.4) if self.use_sahi else None
            self.predictor = IPC8Predictor(
                self.model,
                class_names=class_names,
                max_input_size=max_input_size,
                sahi_config=sahi_config
            )
        return self.predictor

    def export_onnx(
        self,
        onnx_path: str,
        checkpoint_path: str = "",
        input_size=(640, 640),
        opset_version: int = 13,
        dynamic_axes: bool = False,
        normalize_boxes: bool = True,
        model_output_name: str = "detections",
        gst_tensor_id: str = "Gst.Yolo.Output",
        write_metadata_json: bool = True,
        metadata_json_path: str = "",
        embed_metadata_in_onnx: bool = True,
        max_input_side: int = 1024,
        export_segmentation: bool = False,
    ):
        if not hasattr(self, "model"):
            raise RuntimeError("Model is not built. Call build() before export_onnx().")
        if self.task not in {"detection", "instance_segmentation"}:
            raise NotImplementedError(
                f"ONNX export supports detection and instance segmentation. Requested task: {self.task}"
            )

        if checkpoint_path:
            if checkpoint_path.lower().endswith(".weights") and hasattr(self.model, "load_darknet_weights"):
                self.model.load_darknet_weights(checkpoint_path)
            else:
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    direction, remapped = load_state_dict_compat(self.model, ckpt["model_state_dict"], strict=True)
                else:
                    direction, remapped = load_state_dict_compat(self.model, ckpt, strict=True)
                if remapped > 0:
                    print(f"Checkpoint key remap applied: {direction} ({remapped} tensors)")

        h, w = int(input_size[0]), int(input_size[1])

        if max_input_side is None:
            max_input_side = 1024

        if self.name in {"ipc4", "ipc4-full", "ipc4-tiny", "yolov4", "yolov4-full", "yolov4-tiny", "yolov4-tiny-custom"}:
            if export_segmentation:
                if self.task != "instance_segmentation":
                    raise ValueError("export_segmentation=True requires task='instance_segmentation'.")
                export_model = IPC8SegONNXExportWrapper(
                    model=self.model,
                    num_classes=self.num_classes,
                    reg_max=self.reg_max,
                    strides=self.strides,
                    normalize_boxes=normalize_boxes,
                    normalize_input=True,
                    max_input_side=max_input_side,
                    resize_to_max_side=False,
                    pad_to_stride=False,
                    input_size=(h, w),
                ).eval()
                output_names = build_output_names(model_output_name, export_segmentation=True)
            else:
                export_model = IPC4ONNXExportWrapper(
                    model=self.model,
                    normalize_boxes=normalize_boxes,
                    normalize_input=True,
                ).eval()
                output_names = build_output_names(model_output_name, export_segmentation=False)

            return export_onnx_model(
                export_model=export_model,
                onnx_path=onnx_path,
                input_size=(h, w),
                opset_version=opset_version,
                dynamic_axes=dynamic_axes,
                output_names=output_names,
                model_output_name=model_output_name,
                embed_metadata_in_onnx=embed_metadata_in_onnx,
                write_metadata_json=write_metadata_json,
                metadata_json_path=metadata_json_path,
                gst_tensor_id=gst_tensor_id,
            )

        if export_segmentation:
            if self.task != "instance_segmentation":
                raise ValueError("export_segmentation=True requires task='instance_segmentation'.")
            export_model = IPC8SegONNXExportWrapper(
                model=self.model,
                num_classes=self.num_classes,
                reg_max=self.reg_max,
                strides=self.strides,
                normalize_boxes=normalize_boxes,
                normalize_input=True,
                max_input_side=max_input_side,
                resize_to_max_side=False,
                pad_to_stride=False,
                input_size=(h, w),
            ).eval()
        else:
            # Detection-only ONNX export (default). This stays compatible with existing
            # detection pipelines even when the model is trained for instance segmentation.
            # Set export_segmentation=True to include mask outputs.
            export_model = IPC8ONNXExportWrapper(
                model=self.model,
                num_classes=self.num_classes,
                reg_max=self.reg_max,
                strides=self.strides,
                normalize_boxes=normalize_boxes,
                normalize_input=True,
                max_input_side=max_input_side,
                resize_to_max_side=False,
                pad_to_stride=False,
                input_size=(h, w),
            ).eval()
        output_names = build_output_names(model_output_name, export_segmentation=export_segmentation)
        return export_onnx_model(
            export_model=export_model,
            onnx_path=onnx_path,
            input_size=(h, w),
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            output_names=output_names,
            model_output_name=model_output_name,
            embed_metadata_in_onnx=embed_metadata_in_onnx,
            write_metadata_json=write_metadata_json,
            metadata_json_path=metadata_json_path,
            gst_tensor_id=gst_tensor_id,
        )


class NMediaPipe(INetwork):
    def __init__(self, name: str, 
                 task: str, 
                 model_path: str,
                 output_segmentation_masks: bool=False,
                 num_poses: int=100,
                 min_pose_detection_confidence: float=0.25,
                 min_tracking_confidence: float=0.25,
                 detection_max_results: int=5,
                 detection_score_threshold: float=0.5,
                 class_names: List=None,
                 use_gpu: bool = False,
                 use_sahi: bool = False,
                 ):
        self.name = name
        self.task = task
        self.model_path = model_path

        self.baseOptions = mp.tasks.BaseOptions
        self.visionRunningMode = mp.tasks.vision.RunningMode
        # Pose estimation arguments
        self.output_segmentation_masks = output_segmentation_masks
        self.num_poses = num_poses
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        # Object detection arguments
        self.detection_max_results = detection_max_results
        self.detection_score_threshold = detection_score_threshold
        self.class_names = class_names
        self.use_gpu = bool(use_gpu)
        self.use_sahi = bool(use_sahi)

        pass

    def build(self):
        if self.task == "pose_estimation":
            self.model = self.__build_pose_estimation()
        elif self.task == "detection":
            self.model = self.__build_detection()
        else:
            raise NotImplemented('The task have not been implemented yet')
    
    def build_predictor(self, max_input_size: int = 1024):
        if self.task == 'pose_estimation':
            self.predictor = self.__build_pose_estimation_predictor()
        elif self.task == 'detection':
            self.predictor = self.__build_detection_predictor(class_names=self.class_names, max_input_size=max_input_size)
        return self.predictor

    def export_onnx(self, onnx_path: str, checkpoint_path: str, input_size: tuple, opset_version: int, dynamic_axes: bool, normalize_boxes: bool):
        raise NotImplementedError(
            "MediaPipe ONNX export is not supported. Use the .tflite model directly in C++."
        )

    def __build_pose_estimation(self):
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        self.options = PoseLandmarkerOptions(
            base_options=self._base_options(),
            output_segmentation_masks=self.output_segmentation_masks,
            running_mode=self.visionRunningMode.IMAGE,
            num_poses = self.num_poses,
            min_pose_detection_confidence = self.min_pose_detection_confidence,
            min_tracking_confidence = self.min_tracking_confidence,
        )
        return PoseLandmarker.create_from_options(self.options)
    
    def __build_pose_estimation_predictor(self):
        self.predictor = self.model
        return self.predictor

    def __build_detection(self):
        ObjectDetector = mp.tasks.vision.ObjectDetector
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions

        self.options = ObjectDetectorOptions(
            base_options=self._base_options(),
            max_results=self.detection_max_results,
            score_threshold=self.detection_score_threshold,
            running_mode=self.visionRunningMode.IMAGE
        )
        return ObjectDetector.create_from_options(self.options)
    
    def __build_detection_predictor(self, class_names: Optional[List[str]] = None, max_input_size: int = 1024):
        sahi_config = SAHIConfig(max_input_size=max_input_size, class_agnostic_nms=True, overlap=0.2, nms_threshold=0.4) if self.use_sahi else None
        return MediaPipeDetectionPredictor(self.model, class_names=class_names, sahi_config=sahi_config)

    def _base_options(self):
        if self.use_gpu:
            return self.baseOptions(
                model_asset_path=self.model_path,
                delegate=mp.tasks.BaseOptions.Delegate.GPU,
            )
        return self.baseOptions(model_asset_path=self.model_path)
