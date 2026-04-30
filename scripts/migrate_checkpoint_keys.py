#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict

import torch

# Allow running as: python scripts/migrate_checkpoint_keys.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Network.utils import remap_task_head_state_dict_prefixes


def _load_checkpoint(path: Path):
    return torch.load(str(path), map_location="cpu")


def _extract_state_dict(ckpt) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")


def _replace_state_dict(ckpt, new_state_dict: Dict[str, torch.Tensor]):
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        out = dict(ckpt)
        out["model_state_dict"] = new_state_dict
        return out
    return new_state_dict


def _default_output_path(path: Path, direction: str) -> Path:
    suffix = path.suffix if path.suffix else ".pt"
    stem = path.name[:-len(suffix)] if suffix else path.name
    return path.with_name(f"{stem}.{direction}{suffix}")


def migrate_checkpoint(path: Path, direction: str, inplace: bool) -> None:
    ckpt = _load_checkpoint(path)
    state_dict = _extract_state_dict(ckpt)
    remapped_sd, final_direction, remapped = remap_task_head_state_dict_prefixes(state_dict, direction=direction)

    if remapped == 0:
        print(f"[SKIP] {path}: no key remap needed (direction={final_direction})")
        return

    out_ckpt = _replace_state_dict(ckpt, remapped_sd)
    out_path = path if inplace else _default_output_path(path, final_direction)
    torch.save(out_ckpt, str(out_path))
    print(f"[OK] {path} -> {out_path} | {final_direction} | remapped={remapped}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate checkpoint keys between detect.* and task_head.* prefixes."
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        help="Checkpoint file paths (*.pt) to migrate.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="auto",
        choices=["auto", "detect_to_task_head", "task_head_to_detect"],
        help="Remap direction. Default: auto.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite original checkpoint(s).",
    )
    args = parser.parse_args()

    for ckpt_arg in args.checkpoints:
        path = Path(ckpt_arg)
        if not path.exists():
            print(f"[MISS] {path}: file not found")
            continue
        try:
            migrate_checkpoint(path, direction=args.direction, inplace=bool(args.inplace))
        except Exception as exc:
            print(f"[ERR] {path}: {exc}")


if __name__ == "__main__":
    main()
