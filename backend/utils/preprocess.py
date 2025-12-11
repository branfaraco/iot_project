"""
Preprocessing functions for history windows
-----------------------------------------

The models used in this project expect their inputs in the format
``(batch, channels, height, width)``.  For traffic prediction the
channels dimension is constructed by concatenating multiple past
frames along the channel axis.  Each raw traffic frame has shape
``(H, W, C)`` where ``C`` is the number of raw traffic channels (8 in
the provided models).  A history window of ``HISTORY_STEPS`` past
frames therefore yields a tensor of shape ``(HISTORY_STEPS * C, H, W)``.

When using the FiLM conditioned model, additional channels
representing land use (LBCS) codes are concatenated to the traffic
channels.  The LBCS codes are one‑hot encoded and upsampled to
``(LBCS_CLASSES, H, W)``.  To incorporate land use information over
time the one‑hot channels are repeated ``HISTORY_STEPS`` times and
concatenated with the traffic history to yield a tensor of shape
``((HISTORY_STEPS * (C + LBCS_CLASSES)), H, W)``.

The functions in this module perform the reshaping and conversion
from ``numpy`` arrays to PyTorch tensors.  They add a leading batch
dimension so that the result can be passed directly to the models.
"""

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


__all__ = [
    "prepare_inputs_raw",
    "prepare_inputs_enriched",
]