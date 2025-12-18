
from __future__ import annotations

from typing import Union

import numpy as np
import torch


def _flatten_history(x: np.ndarray) -> np.ndarray:
    """Flatten a history window along the channel dimension.

    Parameters
    ----------
    x : np.ndarray
        Either a 4D array of shape ``(T, H, W, C)`` or a 3D array
        ``(T*C, H, W)``.  In the former case the frames are stacked
        over ``T``; in the latter case the history has already been
        flattened.

    Returns
    -------
    np.ndarray
        A 3D array of shape ``(T*C, H, W)``.

    Raises
    ------
    ValueError
        If ``x`` does not have 3 or 4 dimensions.
    """
    if x.ndim == 4:
        # x: (T, H, W, C) → (T, C, H, W)
        T, H, W, C = x.shape
        x = x.transpose(0, 3, 1, 2)
        # reshape to (T*C, H, W)
        return x.reshape(T * C, H, W)
    elif x.ndim == 3:
        # Already flattened (C, H, W) or (T*C, H, W)
        return x
    else:
        raise ValueError(
            f"Expected input of dimension 3 or 4, got shape {x.shape}"
        )


def prepare_inputs_raw(x_raw: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Prepare raw history inputs for the baseline model.

    This function converts a raw history window into a PyTorch tensor
    suitable for the baseline model.  The input history must be a
    ``numpy`` array of shape ``(T, H, W, C)`` or ``(T*C, H, W)``.  The
    output tensor has shape ``(1, T*C, H, W)`` and ``dtype=float32``.

    Parameters
    ----------
    x_raw : np.ndarray or torch.Tensor
        The history window.  If a tensor is provided it will be
        converted to a NumPy array on the CPU.

    Returns
    -------
    torch.Tensor
        A 4D tensor ready to be fed into the baseline model.
    """
    if isinstance(x_raw, torch.Tensor):
        x_raw_np = x_raw.cpu().numpy()
    else:
        x_raw_np = np.asarray(x_raw)
    flat = _flatten_history(x_raw_np).astype(np.float32)
    return torch.tensor(flat, dtype=torch.float32).unsqueeze(0)


def prepare_inputs_enriched(x_enriched: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Prepare inputs for the FiLM conditioned model.

    Given a concatenated traffic and land use history this function
    produces a PyTorch tensor of shape ``(1, C_total, H, W)``.  The
    input must be a ``numpy`` array of shape ``(T*(C+L), H, W)`` or a
    4D array ``(T, H, W, C_total)``.  It is converted to ``float32``
    and a batch dimension is added.

    Parameters
    ----------
    x_enriched : np.ndarray or torch.Tensor
        The enriched history window.  If a tensor is provided it will
        be moved to the CPU before conversion.

    Returns
    -------
    torch.Tensor
        A 4D tensor ready for the enriched model.
    """
    if isinstance(x_enriched, torch.Tensor):
        x_enr_np = x_enriched.cpu().numpy()
    else:
        x_enr_np = np.asarray(x_enriched)
    flat = _flatten_history(x_enr_np).astype(np.float32)
    return torch.tensor(flat, dtype=torch.float32).unsqueeze(0)



def preprocess_frame(arr: np.ndarray) -> np.ndarray:
    """Crop a single frame to the region of interest and normalise to [0,1].

    The shared ``data_reduction`` module defines the row and column indices
    for the crop.  This function assumes the frame has shape ``(H, W, C)``.

    Parameters
    ----------
    arr : np.ndarray
        Input frame with dimensions ``(H, W, C)``.

    Returns
    -------
    np.ndarray
        Cropped and normalised frame of shape ``(H_roi, W_roi, C)``.

    Raises
    ------
    ValueError
        If the array does not have 3 dimensions or is too small for the crop.
    """
    from shared.utils.data_reduction import r0, r1, c0, c1  # type: ignore
    if arr.ndim != 3:
        raise ValueError(f"Expected frame with 3 dims (H,W,C), got shape {arr.shape}")
    H, W, C = arr.shape
    if r1 > H or c1 > W:
        raise ValueError(
            f"Frame of shape {arr.shape} is too small for crop r0={r0}, r1={r1}, c0={c0}, c1={c1}"
        )
    cropped = arr[r0:r1, c0:c1, :]
    return cropped.astype(np.float32) / 255.0