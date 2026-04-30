from typing import Dict, Tuple

import torch

def make_anchors(feats, strides, device):
    anchor_points = []
    stride_tensor = []

    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape

        sy, sx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij"
        )

        points = torch.stack(
            (sx + 0.5, sy + 0.5), dim=-1
        ).reshape(-1, 2).float()

        anchor_points.append(points)

        stride_tensor.append(
            torch.full(
                (points.shape[0], 1),
                float(stride),
                device=device
            )
        )

    return (
        torch.cat(anchor_points, dim=0),
        torch.cat(stride_tensor, dim=0),
    )

def decode_boxes(pred_dist, anchor_points, stride_tensor, reg_max):
    """
    pred_dist: [N, 4*reg_max]
    anchor_points: [N,2]  (in grid coords)
    stride_tensor: [N] or [N,1]
    """
    N = pred_dist.shape[0]

    pred_dist = pred_dist.view(N, 4, reg_max)
    pred_dist = pred_dist.softmax(dim=2)

    proj = torch.arange(reg_max, device=pred_dist.device, dtype=pred_dist.dtype)
    dist = pred_dist @ proj  # [N,4]

    x1y1 = anchor_points - dist[:, :2]
    x2y2 = anchor_points + dist[:, 2:]

    boxes = torch.cat([x1y1, x2y2], dim=1)

    # critical fix: force stride to [N,1]
    if stride_tensor.ndim == 1:
        stride_tensor = stride_tensor.unsqueeze(1)

    return boxes * stride_tensor

def dist2bbox(dist, anchor_points, stride):
    """
    dist: [N,4] or [B,N,4]   (l,t,r,b) in feature-map units
    anchor_points: [N,2] or [B,N,2]  (cx,cy) in feature-map units
    stride: scalar or [N,1]
    returns: boxes in xyxy image coords
    """
    x1 = anchor_points[..., 0] - dist[..., 0]
    y1 = anchor_points[..., 1] - dist[..., 1]
    x2 = anchor_points[..., 0] + dist[..., 2]
    y2 = anchor_points[..., 1] + dist[..., 3]

    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return boxes * stride

def box2dist(anchor_points, gt_boxes, stride, reg_max):
    """
    anchor_points: [N,2] (cx,cy) in feature-map units
    gt_boxes: [N,4] (x1,y1,x2,y2) in image coords
    stride: [N,1] or scalar
    returns: [N,4] distances in feature-map units
    """
    device = gt_boxes.device

    if not torch.is_tensor(stride):
        stride = torch.tensor(stride, device=device)

    if stride.ndim == 1:
        stride = stride.unsqueeze(1)

    # convert GT boxes to feature-map space
    x1y1 = gt_boxes[:, :2] / stride
    x2y2 = gt_boxes[:, 2:] / stride

    # L, T, R, B distances
    dist = torch.cat(
        (
            anchor_points - x1y1,
            x2y2 - anchor_points
        ),
        dim=1
    )

    eps = 1e-6
    return dist.clamp(min=0.0, max=reg_max - eps)
    
def bbox_ciou(pred, target, eps=1e-7):
    """
    pred, target: [..., 4] in (x1, y1, x2, y2), image space
    returns: CIoU in [-1, 1]
    """
    px1, py1, px2, py2 = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    tx1, ty1, tx2, ty2 = target[..., 0], target[..., 1], target[..., 2], target[..., 3]

    # widths & heights
    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)

    # centers
    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5
    tcx = (tx1 + tx2) * 0.5
    tcy = (ty1 + ty2) * 0.5

    # intersection
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)

    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    area_p = pw * ph
    area_t = tw * th
    union = area_p + area_t - inter + eps

    iou = inter / union

    # enclosing box
    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)
    ex2 = torch.max(px2, tx2)
    ey2 = torch.max(py2, ty2)

    cw = (ex2 - ex1).clamp(min=eps)
    ch = (ey2 - ey1).clamp(min=eps)

    # center distance
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
    c2 = cw ** 2 + ch ** 2 + eps

    # aspect ratio penalty
    v = (4 / (torch.pi ** 2)) * (
        torch.atan(tw / th) - torch.atan(pw / ph)
    ) ** 2

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2) - alpha * v
    return ciou


def remap_task_head_state_dict_prefixes(
    state_dict: Dict[str, torch.Tensor],
    direction: str = "auto",
) -> Tuple[Dict[str, torch.Tensor], str, int]:
    """
    Remap checkpoint keys between legacy `detect.*` and current `task_head.*`.
    direction:
      - "auto"
      - "detect_to_task_head"
      - "task_head_to_detect"
    """
    direction = str(direction).lower()
    ckpt_has_task_head = any(k.startswith("task_head.") for k in state_dict.keys())
    ckpt_has_detect = any(k.startswith("detect.") for k in state_dict.keys())

    if direction == "auto":
        if ckpt_has_detect and not ckpt_has_task_head:
            direction = "detect_to_task_head"
        elif ckpt_has_task_head and not ckpt_has_detect:
            direction = "task_head_to_detect"
        else:
            return state_dict, "none", 0

    if direction not in {"detect_to_task_head", "task_head_to_detect"}:
        raise ValueError(
            f"Unsupported remap direction: {direction}. "
            "Use one of: auto, detect_to_task_head, task_head_to_detect."
        )

    if direction == "detect_to_task_head":
        src_prefix = "detect."
        dst_prefix = "task_head."
    else:
        src_prefix = "task_head."
        dst_prefix = "detect."

    out: Dict[str, torch.Tensor] = {}
    remapped = 0
    for k, v in state_dict.items():
        if k.startswith(src_prefix):
            out[dst_prefix + k[len(src_prefix):]] = v
            remapped += 1
        else:
            out[k] = v

    return out, direction, remapped


def remap_task_head_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> Tuple[Dict[str, torch.Tensor], str, int]:
    """
    Remap checkpoint keys between legacy `detect.*` and current `task_head.*`.
    Returns: (possibly remapped state_dict, mapping_direction, num_remapped)
    """
    model_keys = set(model.state_dict().keys())
    model_uses_task_head = any(k.startswith("task_head.") for k in model_keys)
    model_uses_detect = any(k.startswith("detect.") for k in model_keys)

    ckpt_has_task_head = any(k.startswith("task_head.") for k in state_dict.keys())
    ckpt_has_detect = any(k.startswith("detect.") for k in state_dict.keys())

    if model_uses_task_head and ckpt_has_detect and not ckpt_has_task_head:
        return remap_task_head_state_dict_prefixes(state_dict, direction="detect_to_task_head")

    if model_uses_detect and ckpt_has_task_head and not ckpt_has_detect:
        return remap_task_head_state_dict_prefixes(state_dict, direction="task_head_to_detect")

    return state_dict, "none", 0


def load_state_dict_compat(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> Tuple[str, int]:
    remapped_state, direction, remapped = remap_task_head_state_dict_keys(state_dict, model)
    model.load_state_dict(remapped_state, strict=strict)
    return direction, remapped
