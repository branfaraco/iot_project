# shared/utils/mask.py
import os
import h5py
import numpy as np
import torch

def load_mask(data_root: str, device: torch.device) -> torch.Tensor:
    mask_path = os.path.join(data_root, "BERLIN_static.h5")
    with h5py.File(mask_path, "r") as f:
        static = f["array"][()]  # (C, H, W)
    mask_np = (static[0] > 0).astype(np.float32)  # (H, W)
    mask = torch.from_numpy(mask_np).to(device)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
