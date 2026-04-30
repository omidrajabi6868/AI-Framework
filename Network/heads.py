from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Union

import torch
from torch import nn


TaskHeadOutput = Union[List[torch.Tensor], Dict[str, Union[List[torch.Tensor], torch.Tensor]]]


class ITaskHead(nn.Module, ABC):
    """Interface for task-aware YOLO heads."""

    def __init__(self, task: str):
        super().__init__()
        self.task = str(task).lower()

    @abstractmethod
    def forward(self, feats: Sequence[torch.Tensor]) -> TaskHeadOutput:
        pass


class Proto(nn.Module):
    """Prototype mask branch used by instance segmentation heads."""

    def __init__(self, c1: int, c_mid: int = 256, num_protos: int = 32):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c_mid, 3, 1, 1)
        self.cv2 = nn.Conv2d(c_mid, c_mid, 3, 1, 1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv3 = nn.Conv2d(c_mid, num_protos, 1, 1, 0)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.cv1(x))
        x = self.up(x)
        x = self.act(self.cv2(x))
        return self.cv3(x)


class DetectionTaskHead(ITaskHead):
    """Reusable detection head supporting coupled or decoupled branches."""

    def __init__(
        self,
        ch: Sequence[int],
        nc: int,
        reg_max: int = 16,
        decoupled: bool = False,
    ):
        super().__init__(task="detection")
        self.nc = int(nc)
        self.reg_max = int(reg_max)
        self.no = self.nc + 4 * self.reg_max
        self.decoupled = bool(decoupled)

        if self.decoupled:
            self.cv_cls = nn.ModuleList(nn.Conv2d(c, self.nc, 1) for c in ch)
            self.cv_reg = nn.ModuleList(nn.Conv2d(c, 4 * self.reg_max, 1) for c in ch)
        else:
            self.cv = nn.ModuleList(nn.Conv2d(c, self.no, 1) for c in ch)

    def forward_detection(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if self.decoupled:
            out: List[torch.Tensor] = []
            for i, feat in enumerate(feats):
                cls = self.cv_cls[i](feat)
                reg = self.cv_reg[i](feat)
                out.append(torch.cat([reg, cls], dim=1))
            return out
        return [self.cv[i](feats[i]) for i in range(len(feats))]

    def forward(self, feats: Sequence[torch.Tensor]) -> TaskHeadOutput:
        return self.forward_detection(feats)


class InstanceSegmentationTaskHead(DetectionTaskHead):
    """Detection + mask prototype + mask coefficients head."""

    def __init__(
        self,
        ch: Sequence[int],
        nc: int,
        reg_max: int = 16,
        num_masks: int = 32,
        num_protos: int = 32,
        decoupled: bool = False,
    ):
        super().__init__(ch=ch, nc=nc, reg_max=reg_max, decoupled=decoupled)
        self.task = "instance_segmentation"
        self.num_masks = int(num_masks)
        self.num_protos = int(num_protos)
        self.cv_mask = nn.ModuleList(nn.Conv2d(c, self.num_masks, 1) for c in ch)
        proto_mid = max(64, ch[0] // 2)
        self.proto = Proto(c1=ch[0], c_mid=proto_mid, num_protos=self.num_protos)

    def forward(self, feats: Sequence[torch.Tensor]) -> TaskHeadOutput:
        det_out = self.forward_detection(feats)
        mask_coeffs = [self.cv_mask[i](feats[i]) for i in range(len(feats))]
        proto = self.proto(feats[0])
        return {
            "det": det_out,
            "mask_coeffs": mask_coeffs,
            "proto": proto,
        }


def build_task_head(
    task: str,
    ch: Sequence[int],
    nc: int,
    reg_max: int = 16,
    decoupled: bool = False,
    num_masks: int = 32,
    num_protos: int = 32,
) -> ITaskHead:
    task_name = str(task).lower()
    if task_name == "detection":
        return DetectionTaskHead(ch=ch, nc=nc, reg_max=reg_max, decoupled=decoupled)
    if task_name == "instance_segmentation":
        return InstanceSegmentationTaskHead(
            ch=ch,
            nc=nc,
            reg_max=reg_max,
            num_masks=num_masks,
            num_protos=num_protos,
            decoupled=decoupled,
        )
    raise ValueError(f"Unsupported task for task head: {task}")
