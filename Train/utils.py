import torch
from Network.utils import load_state_dict_compat

def load_pretrained_yolo(model, weight_path):
    ckpt = torch.load(weight_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Load only matching tensors (safe when class counts differ).
    model_state = model.state_dict()
    filtered = {}
    skipped = 0
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped += 1

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    _, remapped = load_state_dict_compat(model, filtered, strict=False)

    print(f"Loaded tensors: {len(filtered)}/{len(model_state)}")
    print(f"Skipped tensors (shape/key mismatch): {skipped}")
    if remapped > 0:
        print(f"Checkpoint key remap applied: {remapped} tensors")
    if missing:
        print(f"Missing in checkpoint: {len(missing)}")
    if unexpected:
        print(f"Unexpected in checkpoint: {len(unexpected)}")

    return model
