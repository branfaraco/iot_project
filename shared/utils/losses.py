import torch
import torch.nn as nn

class MaskedMAEFocalLoss(nn.Module):
    """Masked MAE-Focal loss with built-in masking and scalar reduction.

    The loss combines mean absolute error and a focal component to
    emphasise larger errors.  A binary mask is applied to ignore
    invalid regions (e.g., non-road cells).  The mask is broadcast
    over batch and channel dimensions during the forward pass.
    """

    def __init__(self, mask: torch.Tensor, beta: float = 0.2, gamma: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.register_buffer("mask", mask)  # ensures device sync, saves in state_dict

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, F, H, W)
        err = torch.abs(pred - target)  # (B, F, H, W)
        w = torch.sigmoid(self.beta * err) ** self.gamma
        raw = w * err  # (B, F, H, W)
        valid = self.mask.expand_as(raw)  # (B, F, H, W)
        loss = (raw * valid).sum() / (valid.sum() + 1e-8)
        return loss