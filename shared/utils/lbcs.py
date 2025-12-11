
from __future__ import annotations

import json
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


def load_lbcs_onehot(path: str, H_target: int, W_target: int) -> torch.Tensor:
    """Load an LBCS matrix and return a one-hot encoded tensor.
    """
    # Define known LBCS codes.  If the dataset contains codes not in
    # this list, add them here in the desired order.
    LBCS_CODES: List[str] = ["1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000", "9000"]
    code_to_index = {c: i for i, c in enumerate(LBCS_CODES)}

    # Load matrix from JSON and convert to numpy array
    with open(path, "r") as f:
        M = np.array(json.load(f))  # shape (R, C)

    R, C = M.shape
    n_classes = len(LBCS_CODES)

    # Initialise one-hot array (n_classes, R, C)
    onehot = np.zeros((n_classes, R, C), dtype=np.float32)
    for r in range(R):
        for c in range(C):
            code = M[r, c]
            try:
                idx = code_to_index[code]
            except KeyError:
                raise KeyError(f"Unknown LBCS code '{code}' in matrix at position {(r, c)}")
            onehot[idx, r, c] = 1.0

    # Convert to tensor and add batch dimension for interpolation
    oh = torch.from_numpy(onehot).unsqueeze(0)  # (1, n_classes, R, C)
    # Upsample to target resolution using nearest neighbour
    oh_big = F.interpolate(oh, size=(H_target, W_target), mode="nearest")[0]
    return oh_big  # (n_classes, H_target, W_target)

