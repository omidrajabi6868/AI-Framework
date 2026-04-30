import json
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .heads import build_task_head
from .SAHI import SAHIWrapper, SAHIConfig

# Static YOLOv4 definitions (exported from official cfg files).
IPC4_BLOCKS_FULL = json.loads(r'''[{"type":"net","batch":"64","subdivisions":"8","width":"608","height":"608","channels":"3","momentum":"0.949","decay":"0.0005","angle":"0","saturation":"1.5","exposure":"1.5","hue":".1","learning_rate":"0.0013","burn_in":"1000","max_batches":"500500","policy":"steps","steps":"400000,450000","scales":".1,.1","mosaic":"1"},{"type":"convolutional","batch_normalize":"1","filters":"32","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"3","stride":"2","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"route","layers":"-2"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"32","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"route","layers":"-1,-7"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"3","stride":"2","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"route","layers":"-2"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"route","layers":"-1,-10"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"3","stride":"2","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"route","layers":"-2"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"route","layers":"-1,-28"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"3","stride":"2","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"route","layers":"-2"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"route","layers":"-1,-28"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"1024","size":"3","stride":"2","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"route","layers":"-2"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"3","stride":"1","pad":"1","activation":"mish"},{"type":"shortcut","from":"-3","activation":"linear"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"route","layers":"-1,-16"},{"type":"convolutional","batch_normalize":"1","filters":"1024","size":"1","stride":"1","pad":"1","activation":"mish"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","size":"3","stride":"1","pad":"1","filters":"1024","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"maxpool","stride":"1","size":"5"},{"type":"route","layers":"-2"},{"type":"maxpool","stride":"1","size":"9"},{"type":"route","layers":"-4"},{"type":"maxpool","stride":"1","size":"13"},{"type":"route","layers":"-1,-3,-5,-6"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","size":"3","stride":"1","pad":"1","filters":"1024","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"upsample","stride":"2"},{"type":"route","layers":"85"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"route","layers":"-1, -3"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","size":"3","stride":"1","pad":"1","filters":"512","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","size":"3","stride":"1","pad":"1","filters":"512","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"upsample","stride":"2"},{"type":"route","layers":"54"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"route","layers":"-1, -3"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","size":"3","stride":"1","pad":"1","filters":"256","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","size":"3","stride":"1","pad":"1","filters":"256","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","size":"3","stride":"1","pad":"1","filters":"256","activation":"leaky"},{"type":"convolutional","size":"1","stride":"1","pad":"1","filters":"255","activation":"linear"},{"type":"yolo","mask":"0,1,2","anchors":"12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401","classes":"80","num":"9","jitter":".3","ignore_thresh":".7","truth_thresh":"1","scale_x_y":"1.2","iou_thresh":"0.213","cls_normalizer":"1.0","iou_normalizer":"0.07","iou_loss":"ciou","nms_kind":"greedynms","beta_nms":"0.6","max_delta":"5"},{"type":"route","layers":"-4"},{"type":"convolutional","batch_normalize":"1","size":"3","stride":"2","pad":"1","filters":"256","activation":"leaky"},{"type":"route","layers":"-1, -16"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","size":"3","stride":"1","pad":"1","filters":"512","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","size":"3","stride":"1","pad":"1","filters":"512","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","size":"3","stride":"1","pad":"1","filters":"512","activation":"leaky"},{"type":"convolutional","size":"1","stride":"1","pad":"1","filters":"255","activation":"linear"},{"type":"yolo","mask":"3,4,5","anchors":"12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401","classes":"80","num":"9","jitter":".3","ignore_thresh":".7","truth_thresh":"1","scale_x_y":"1.1","iou_thresh":"0.213","cls_normalizer":"1.0","iou_normalizer":"0.07","iou_loss":"ciou","nms_kind":"greedynms","beta_nms":"0.6","max_delta":"5"},{"type":"route","layers":"-4"},{"type":"convolutional","batch_normalize":"1","size":"3","stride":"2","pad":"1","filters":"512","activation":"leaky"},{"type":"route","layers":"-1, -37"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","size":"3","stride":"1","pad":"1","filters":"1024","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","size":"3","stride":"1","pad":"1","filters":"1024","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","size":"3","stride":"1","pad":"1","filters":"1024","activation":"leaky"},{"type":"convolutional","size":"1","stride":"1","pad":"1","filters":"255","activation":"linear"},{"type":"yolo","mask":"6,7,8","anchors":"12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401","classes":"80","num":"9","jitter":".3","ignore_thresh":".7","truth_thresh":"1","random":"1","scale_x_y":"1.05","iou_thresh":"0.213","cls_normalizer":"1.0","iou_normalizer":"0.07","iou_loss":"ciou","nms_kind":"greedynms","beta_nms":"0.6","max_delta":"5"}]''')
IPC4_BLOCKS_TINY = json.loads(r'''[{"type":"net","batch":"64","subdivisions":"1","width":"416","height":"416","channels":"3","momentum":"0.9","decay":"0.0005","angle":"0","saturation":"1.5","exposure":"1.5","hue":".1","learning_rate":"0.00261","burn_in":"1000","max_batches":"500200","policy":"steps","steps":"400000,450000","scales":".1,.1"},{"type":"convolutional","batch_normalize":"1","filters":"32","size":"3","stride":"2","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"3","stride":"2","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"3","stride":"1","pad":"1","activation":"leaky"},{"type":"route","layers":"-1","groups":"2","group_id":"1"},{"type":"convolutional","batch_normalize":"1","filters":"32","size":"3","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"32","size":"3","stride":"1","pad":"1","activation":"leaky"},{"type":"route","layers":"-1,-2"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"route","layers":"-6,-1"},{"type":"maxpool","size":"2","stride":"2"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"3","stride":"1","pad":"1","activation":"leaky"},{"type":"route","layers":"-1","groups":"2","group_id":"1"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"3","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"64","size":"3","stride":"1","pad":"1","activation":"leaky"},{"type":"route","layers":"-1,-2"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"route","layers":"-6,-1"},{"type":"maxpool","size":"2","stride":"2"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"3","stride":"1","pad":"1","activation":"leaky"},{"type":"route","layers":"-1","groups":"2","group_id":"1"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"3","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"3","stride":"1","pad":"1","activation":"leaky"},{"type":"route","layers":"-1,-2"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"route","layers":"-6,-1"},{"type":"maxpool","size":"2","stride":"2"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"3","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","batch_normalize":"1","filters":"512","size":"3","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","size":"1","stride":"1","pad":"1","filters":"255","activation":"linear"},{"type":"yolo","mask":"3,4,5","anchors":"10,14,  23,27,  37,58,  81,82,  135,169,  344,319","classes":"80","num":"6","jitter":".3","scale_x_y":"1.05","cls_normalizer":"1.0","iou_normalizer":"0.07","iou_loss":"ciou","ignore_thresh":".7","truth_thresh":"1","random":"0","resize":"1.5","nms_kind":"greedynms","beta_nms":"0.6"},{"type":"route","layers":"-4"},{"type":"convolutional","batch_normalize":"1","filters":"128","size":"1","stride":"1","pad":"1","activation":"leaky"},{"type":"upsample","stride":"2"},{"type":"route","layers":"-1, 23"},{"type":"convolutional","batch_normalize":"1","filters":"256","size":"3","stride":"1","pad":"1","activation":"leaky"},{"type":"convolutional","size":"1","stride":"1","pad":"1","filters":"255","activation":"linear"},{"type":"yolo","mask":"0,1,2","anchors":"10,14,  23,27,  37,58,  81,82,  135,169,  344,319","classes":"80","num":"6","jitter":".3","scale_x_y":"1.05","cls_normalizer":"1.0","iou_normalizer":"0.07","iou_loss":"ciou","ignore_thresh":".7","truth_thresh":"1","random":"0","resize":"1.5","nms_kind":"greedynms","beta_nms":"0.6"}]''')

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class YOLOLayer(nn.Module):
    def __init__(self, anchors: List[Tuple[float, float]], mask: List[int], num_classes: int):
        super().__init__()
        self.num_classes = int(num_classes)
        self.anchors = [(float(a), float(b)) for a, b in anchors]
        self.mask = [int(m) for m in mask]

    def forward(self, x):
        return x


def _create_modules(blocks: List[dict]):
    module_list = nn.ModuleList()
    output_filters: List[int] = []
    yolo_layers: List[YOLOLayer] = []
    yolo_in_indices: List[int] = []

    prev_filters = int(blocks[0].get("channels", 3))

    for i, block in enumerate(blocks[1:], start=1):
        mtype = block["type"]
        if mtype == "convolutional":
            bn = int(block.get("batch_normalize", 0))
            filters = int(block["filters"])
            k = int(block["size"])
            s = int(block.get("stride", 1))
            pad = int(block.get("pad", 0))
            p = (k - 1) // 2 if pad else 0
            conv = nn.Conv2d(prev_filters, filters, k, s, p, bias=(bn == 0))
            modules = [conv]
            if bn:
                modules.append(nn.BatchNorm2d(filters))
            act = block.get("activation", "linear")
            if act == "leaky":
                modules.append(nn.LeakyReLU(0.1, inplace=True))
            elif act == "mish":
                modules.append(Mish())
            elif act == "relu":
                modules.append(nn.ReLU(inplace=True))
            elif act == "linear":
                pass
            else:
                raise ValueError(f"Unsupported activation: {act}")
            module_list.append(nn.Sequential(*modules))
            prev_filters = filters
            output_filters.append(prev_filters)
            continue

        if mtype == "maxpool":
            size = int(block["size"])
            stride = int(block["stride"])
            if stride == 1 and size == 2:
                module_list.append(nn.Sequential(
                    nn.ZeroPad2d((0, 1, 0, 1)),
                    nn.MaxPool2d(kernel_size=size, stride=stride, padding=0),
                ))
            else:
                module_list.append(nn.MaxPool2d(kernel_size=size, stride=stride, padding=(size - 1) // 2))
            output_filters.append(prev_filters)
            continue

        if mtype == "upsample":
            stride = int(block.get("stride", 2))
            module_list.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            output_filters.append(prev_filters)
            continue

        if mtype == "route":
            layers = [int(x) for x in block["layers"].split(",")]
            groups = int(block.get("groups", 1))
            module_list.append(nn.Identity())
            if len(layers) == 1:
                filters = output_filters[layers[0]]
            else:
                filters = sum(output_filters[l] for l in layers)
            if groups > 1:
                filters = filters // groups
            prev_filters = filters
            output_filters.append(prev_filters)
            continue

        if mtype == "shortcut":
            module_list.append(nn.Identity())
            output_filters.append(prev_filters)
            continue

        if mtype == "yolo":
            mask = [int(x) for x in block["mask"].split(",")]
            anchors = [float(x) for x in block["anchors"].split(",")]
            anchors = list(zip(anchors[0::2], anchors[1::2]))
            num_classes = int(block["classes"])
            yolo = YOLOLayer(anchors=anchors, mask=mask, num_classes=num_classes)
            yolo_layers.append(yolo)
            yolo_in_indices.append(len(output_filters) - 1)
            module_list.append(yolo)
            output_filters.append(prev_filters)
            continue

        if mtype == "net":
            module_list.append(nn.Identity())
            output_filters.append(prev_filters)
            continue

        raise ValueError(f"Unsupported layer type: {mtype}")

    return module_list, output_filters, yolo_layers, yolo_in_indices


class YOLOIPC4(nn.Module):
    def __init__(
        self,
        size: str,
        num_classes: int,
        task: str = "detection",
        num_masks: int = 32,
        num_protos: int = 32,
        reg_max: int = 16,
    ):
        super().__init__()
        self.size = str(size).lower()
        self.num_classes = int(num_classes)
        self.task = str(task).lower()
        self.num_masks = int(num_masks)
        self.num_protos = int(num_protos)
        self.reg_max = int(reg_max)

        if self.size in {"full", "l", "large"}:
            self.blocks = IPC4_BLOCKS_FULL
        elif self.size in {"tiny", "t", "s"}:
            self.blocks = IPC4_BLOCKS_TINY
        else:
            raise ValueError(f"Unsupported IPC4 size: {self.size}")

        net = self.blocks[0]
        self.net_w = int(net.get("width", 608))
        self.net_h = int(net.get("height", 608))

        self.module_list, self.output_filters, self.yolo_layers, self.yolo_in_indices = _create_modules(self.blocks)
        for yl in self.yolo_layers:
            if yl.num_classes != self.num_classes:
                raise ValueError(
                    f"IPC4 cfg classes ({yl.num_classes}) do not match requested num_classes ({self.num_classes})."
                )

        if self.task == "instance_segmentation":
            ch = [self.output_filters[i] for i in self.yolo_in_indices]
            self.task_head = build_task_head(
                task="instance_segmentation",
                ch=ch,
                nc=self.num_classes,
                reg_max=self.reg_max,
                num_masks=self.num_masks,
                num_protos=self.num_protos,
            )
        else:
            self.task_head = None

    def load_darknet_weights(self, weights_path: str, cutoff: Optional[int] = None):
        with open(weights_path, "rb") as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        weights = torch.from_numpy(weights)

        ptr = 0
        cutoff = cutoff if cutoff is not None else len(self.module_list)

        for i, block in enumerate(self.blocks[1:1 + cutoff], start=0):
            if block["type"] != "convolutional":
                continue
            module = self.module_list[i]
            conv = None
            bn = None
            if isinstance(module, nn.Sequential):
                for m in module:
                    if isinstance(m, nn.Conv2d):
                        conv = m
                    elif isinstance(m, nn.BatchNorm2d):
                        bn = m

            if conv is None:
                continue

            if bn is not None:
                num_b = bn.bias.numel()
                bn_bias = weights[ptr:ptr + num_b].view_as(bn.bias)
                ptr += num_b
                bn_weight = weights[ptr:ptr + num_b].view_as(bn.weight)
                ptr += num_b
                bn_running_mean = weights[ptr:ptr + num_b].view_as(bn.running_mean)
                ptr += num_b
                bn_running_var = weights[ptr:ptr + num_b].view_as(bn.running_var)
                ptr += num_b
                bn.bias.data.copy_(bn_bias)
                bn.weight.data.copy_(bn_weight)
                bn.running_mean.data.copy_(bn_running_mean)
                bn.running_var.data.copy_(bn_running_var)
            else:
                num_b = conv.bias.numel()
                conv_bias = weights[ptr:ptr + num_b].view_as(conv.bias)
                ptr += num_b
                conv.bias.data.copy_(conv_bias)

            num_w = conv.weight.numel()
            conv_weight = weights[ptr:ptr + num_w].view_as(conv.weight)
            ptr += num_w
            conv.weight.data.copy_(conv_weight)

        return ptr

    def forward(self, x):
        outputs: List[torch.Tensor] = []
        yolo_features: List[torch.Tensor] = []
        yolo_outs: List[torch.Tensor] = []

        for i, block in enumerate(self.blocks[1:], start=0):
            mtype = block["type"]
            module = self.module_list[i]

            if mtype in {"convolutional", "maxpool", "upsample"}:
                x = module(x)
            elif mtype == "route":
                layers = [int(l) for l in block["layers"].split(",")]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                else:
                    x = torch.cat([outputs[l] for l in layers], dim=1)
                groups = int(block.get("groups", 1))
                if groups > 1:
                    group_id = int(block.get("group_id", 0))
                    channels = x.shape[1] // groups
                    x = x[:, group_id * channels:(group_id + 1) * channels]
            elif mtype == "shortcut":
                from_idx = int(block["from"])
                x = outputs[-1] + outputs[from_idx]
            elif mtype == "yolo":
                yolo_features.append(outputs[-1])
                yolo_outs.append(module(outputs[-1]))
                x = outputs[-1]
            else:
                raise ValueError(f"Unsupported layer in forward: {mtype}")

            outputs.append(x)

        if self.task == "instance_segmentation":
            return self.task_head(yolo_features)
        return yolo_outs


class IPC4Predictor:
    def __init__(
        self,
        model: YOLOIPC4,
        conf_thres: float = 0.25,
        iou_thres: float = 0.5,
        max_det: int = 300,
        max_input_size: int = 1024,
        device: str = "cpu",
        sahi_config: Optional[SAHIConfig] = None,
    ):
        self.model = model.to(device).eval()
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.max_det = int(max_det)
        self.max_input_size = int(max_input_size)
        self.device = device
        self.sahi = None

        if sahi_config is not None:
            self.sahi = self._build_sahi_wrapper(sahi_config)

    def _to_nchw_float(self, image_input):
        if isinstance(image_input, torch.Tensor):
            t = image_input
        else:
            t = torch.from_numpy(image_input)
        if t.ndim == 4:
            image = t
        elif t.ndim == 3:
            if t.shape[0] in (1, 3):
                image = t.unsqueeze(0)
            else:
                image = t.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError("Expected image input with 3 or 4 dimensions.")
        image = image.to(self.device).float()
        if image.max() > 1.0:
            image = image / 255.0
        return image

    def _resize_to_max_side(self, image: torch.Tensor, max_side: int = 10):
        _, _, orig_h, orig_w = image.shape
        largest_side = max(orig_h, orig_w)
        if largest_side <= 0:
            raise ValueError("Invalid image size for resizing.")
        scale = float(max_side) / float(largest_side)
        new_h = max(1, int(round(orig_h * scale)))
        new_w = max(1, int(round(orig_w * scale)))
        if new_h != orig_h or new_w != orig_w:
            image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return image, (orig_h, orig_w), scale

    def _pad_to_stride(self, image: torch.Tensor, stride: int = 32):
        _, _, h, w = image.shape
        pad_h = (stride - (h % stride)) % stride
        pad_w = (stride - (w % stride)) % stride
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        if pad_h == 0 and pad_w == 0:
            return image, (h, w), (left, top)
        image = F.pad(image, (left, right, top, bottom), value=114.0 / 255.0)
        return image, (h, w), (left, top)

    def _preprocess(self, image_input):
        image = self._to_nchw_float(image_input)
        image, (orig_h, orig_w), resize_scale = self._resize_to_max_side(image, max_side=self.max_input_size)
        image, _, (pad_left, pad_top) = self._pad_to_stride(image, stride=32)
        meta = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "resize_scale": resize_scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
            "in_h": int(image.shape[2]),
            "in_w": int(image.shape[3]),
        }
        return image, meta

    @torch.no_grad()
    def __call__(self, image_input):
        if self.sahi is not None:
            return self.sahi(image_input)
        image, meta = self._preprocess(image_input)
        preds = self.model(image)
        return self._postprocess(preds, meta)

    def _decode(self, outputs: List[torch.Tensor], meta):
        device = outputs[0].device
        orig_h = meta["orig_h"]
        orig_w = meta["orig_w"]
        resize_scale = meta["resize_scale"]
        pad_left = meta["pad_left"]
        pad_top = meta["pad_top"]

        boxes_all = []
        scores_all = []

        for out, yolo in zip(outputs, self.model.yolo_layers):
            b, c, h, w = out.shape
            na = len(yolo.mask)
            out = out.view(b, na, yolo.num_classes + 5, h, w).permute(0, 1, 3, 4, 2).contiguous()

            xy = torch.sigmoid(out[..., 0:2])
            wh = out[..., 2:4]
            obj = torch.sigmoid(out[..., 4:5])
            cls = torch.sigmoid(out[..., 5:])

            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, h, w, 2).float()

            anchors = torch.tensor([yolo.anchors[i] for i in yolo.mask], device=device).view(1, na, 1, 1, 2)
            scale = torch.tensor(
                [meta["in_w"] / float(self.model.net_w), meta["in_h"] / float(self.model.net_h)],
                device=device,
            ).view(1, 1, 1, 1, 2)
            anchors = anchors * scale
            stride = torch.tensor([meta["in_w"] / w, meta["in_h"] / h], device=device).view(1, 1, 1, 1, 2)

            box_xy = (xy + grid) * stride
            box_wh = torch.exp(wh) * anchors
            x1y1 = box_xy - box_wh / 2
            x2y2 = box_xy + box_wh / 2
            boxes = torch.cat([x1y1, x2y2], dim=-1).view(b, -1, 4)

            scores = (obj * cls).view(b, -1, yolo.num_classes)
            boxes_all.append(boxes)
            scores_all.append(scores)

        boxes = torch.cat(boxes_all, dim=1)
        scores = torch.cat(scores_all, dim=1)

        boxes[:, :, [0, 2]] = (boxes[:, :, [0, 2]] - float(pad_left)) / float(resize_scale)
        boxes[:, :, [1, 3]] = (boxes[:, :, [1, 3]] - float(pad_top)) / float(resize_scale)
        boxes[:, :, [0, 2]] = boxes[:, :, [0, 2]].clamp(0.0, float(max(0, orig_w - 1)))
        boxes[:, :, [1, 3]] = boxes[:, :, [1, 3]].clamp(0.0, float(max(0, orig_h - 1)))

        results = []
        for b in range(boxes.shape[0]):
            b_boxes = boxes[b]
            b_scores = scores[b]
            conf, labels = b_scores.max(dim=1)
            keep = conf > self.conf_thres
            b_boxes = b_boxes[keep]
            conf = conf[keep]
            labels = labels[keep]
            if b_boxes.numel() == 0:
                results.append({"boxes": b_boxes, "scores": conf, "classes": labels})
                continue
            nms_keep = torchvision.ops.batched_nms(b_boxes, conf, labels, self.iou_thres)[: self.max_det]
            results.append({
                "boxes": b_boxes[nms_keep],
                "scores": conf[nms_keep],
                "classes": labels[nms_keep],
                "box_format": "xyxy",
                "normalized": False,
                "image_size": (orig_h, orig_w),
            })
        return results[0] if len(results) == 1 else results

    def _postprocess(self, outputs, meta):
        return self._decode(outputs, meta)

    def _build_sahi_wrapper(self, sahi_config: SAHIConfig) -> SAHIWrapper:
        def preprocess_fn(image_slice):
            return self._preprocess(image_slice)

        def postprocess_fn(raw_output, meta):
            return self._postprocess(raw_output, meta)

        return SAHIWrapper(
            model_fn=self.model,
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


