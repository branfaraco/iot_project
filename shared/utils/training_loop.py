# shared/utils/training_loop.py
from typing import Callable, Tuple, List
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

def run_epoch_generic(
    model: nn.Module,
    loader: DataLoader,
    forward_fn: Callable,  # takes (model, batch, device) -> pred
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_mode: bool,
    verbose: bool,
) -> Tuple[float, List[float]]:
    model.train(mode=train_mode)
    running_loss = 0.0
    total_samples = 0
    batch_losses: List[float] = []

    for batch_idx, batch in enumerate(loader):
        with torch.set_grad_enabled(train_mode):
            pred, target, bs = forward_fn(model, batch, device)
            loss = criterion(pred, target)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * bs
        total_samples += bs
        batch_losses.append(loss.item())

        if verbose and train_mode and batch_idx % 50 == 0:
            print(f"  batch {batch_idx}, loss {loss.item():.4f}")

    epoch_loss = running_loss / total_samples
    return epoch_loss, batch_losses
