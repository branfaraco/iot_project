from __future__ import annotations

import os
import glob
import json
import h5py
from datetime import datetime, timedelta
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from shared.models.unet_film import UNet2D_FiLM
from shared.utils.lbcs import load_lbcs_onehot
from shared.utils.indexing import build_time_index
from shared.utils.weather_encoder import WeatherEncoder
from shared.utils.losses import MaskedMAEFocalLoss
from shared.utils.mask import load_mask



# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to data and models.  Adjust these paths if your dataset or
# repository lives elsewhere.
REPO_ROOT = r"C:\Users\user\UPM\Imperial-4año\IoT\Github"
DATA_ROOT = os.path.join(REPO_ROOT, "hugging_face", "BERLIN_reduced")
WEATHER_ROOT = os.path.join(REPO_ROOT, "hugging_face", "weather_berlin-tempel", "cleaned")
MODELS_ROOT = os.path.join(REPO_ROOT, "models")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_LBCS = os.path.join(DATA_ROOT, "grid_lbcs-2", "lbcs_matrix.json")


class TrafficDatasetEnriched(Dataset):
    """Traffic dataset that includes LBCS and weather information.

    Each sample returned by this dataset consists of:

    * A tensor ``x`` of shape ``(S*(C+9), H, W)`` where ``S`` is the
      number of history steps, ``C`` is the number of traffic channels
      (8 in this dataset), and 9 corresponds to the number of LBCS
      classes.  The traffic channels for each historical frame are
      concatenated along the channel dimension, followed by the LBCS
      channels repeated for each history step.

    * A tensor ``y`` of shape ``(F, H, W)`` representing the summed
      traffic volumes across four volume channels for ``F`` future
      steps.

    * A weather vector ``w`` of shape ``(D,)`` where ``D`` is the
      number of weather variables loaded by the ``WeatherEncoder``.
      This vector is normalised using training statistics and looked
      up via timestamp alignment.

    Parameters
    ----------
    h5_paths : list[str]
        List of HDF5 file paths containing traffic arrays.  Each file
        name should begin with a date in ``YYYY-MM-DD`` format which
        is used to compute timestamps for weather lookups.

    lbcs_path : str
        Path to the JSON file containing the coarse LBCS matrix.

    weather_encoder : WeatherEncoder
        Instance of ``WeatherEncoder`` to produce weather vectors.

    history_steps : int, default 12
        Number of past frames to include in the input ``x``.

    future_steps : int, default 4
        Number of future frames to predict.  The target ``y`` will
        have this many channels.

    dataset_key : str, default "array"
        Name of the dataset inside the HDF5 files containing the
        traffic tensor.  For the Berlin dataset this is "array".

    step_minutes : int, default 5
        Temporal resolution of traffic frames in minutes.  Used to
        compute timestamps when looking up weather data.
    """

    def __init__(
        self,
        h5_paths: List[str],
        lbcs_path: str,
        weather_encoder: WeatherEncoder,
        history_steps: int = 12,
        future_steps: int = 4,
        dataset_key: str = "array",
        step_minutes: int = 5,
    ) -> None:
        super().__init__()
        self.h5_paths = list(h5_paths)
        self.S = history_steps
        self.F = future_steps
        self.dataset_key = dataset_key
        self.weather_encoder = weather_encoder
        self.step_minutes = step_minutes

        # Precompute index of (file_idx, t0) pairs
        self.index = build_time_index(
            self.h5_paths,
            history_steps=self.S,
            future_steps=self.F,
            dataset_key=self.dataset_key,
        )

        if not self.index:
            raise ValueError("No valid temporal windows found in the provided HDF5 files.")

        # Load one file to determine spatial dimensions and number of channels
        sample_h5 = self.h5_paths[0]
        with h5py.File(sample_h5, "r") as f:
            sample_shape = f[self.dataset_key].shape  # (T, H, W, C)
            _, H, W, C = sample_shape

        # Load LBCS as one-hot tensor and upsample to traffic resolution
        lbcs = load_lbcs_onehot(lbcs_path, H, W)  # (9, H, W)
        # Repeat LBCS for each history step and reshape for concatenation
        self.lbcs_rep = lbcs.unsqueeze(0).repeat(self.S, 1, 1, 1).reshape(self.S * lbcs.shape[0], H, W)

        # Define volume channels to sum for the target
        self.vol_channels = [0, 2, 4, 6]

    def __len__(self) -> int:
        return len(self.index)

    def _timestamp_for(self, filename: str, t0: int) -> datetime:
        """Compute a timestamp for a given file and start index.

        The filename is expected to start with a date in ISO format
        (YYYY-MM-DD).  The returned datetime combines the date with
        an offset in minutes proportional to ``t0`` and the
        ``step_minutes`` parameter.
        """
        date_str = os.path.basename(filename).split("_")[0]
        try:
            base_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Filename '{filename}' does not begin with a valid date.") from exc
        return base_date + timedelta(minutes=self.step_minutes * t0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        file_idx, t0 = self.index[idx]
        path = self.h5_paths[file_idx]
        # Load the window of frames from HDF5
        with h5py.File(path, "r") as f:
            window = f[self.dataset_key][t0 : t0 + self.S + self.F]  # (S+F, H, W, C)
        arr = torch.from_numpy(window).float()  # (T, H, W, C)
        arr = arr.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)

        # Split into history and future
        x_hist = arr[: self.S]  # (S, C, H, W)
        y_full = arr[self.S : self.S + self.F]  # (F, C, H, W)

        # Flatten history traffic channels
        S, C, H, W = x_hist.shape
        x_traf = x_hist.reshape(S * C, H, W)  # (S*C, H, W)

        # Concatenate LBCS repeated channels
        x = torch.cat([x_traf, self.lbcs_rep], dim=0)  # (S*(C+9), H, W)

        # Compute the target by summing volume channels
        # y_vol shape: (F, H, W)
        y_vol = y_full[:, self.vol_channels].sum(dim=1)

        # Look up weather vector using the timestamp
        ts = self._timestamp_for(path, t0)
        # WeatherEncoder accepts pandas.Timestamp or datetime
        weather_vec_np = self.weather_encoder.encode_timestamp(ts)
        weather_vec = torch.from_numpy(weather_vec_np)  # (D,)
        return x, y_vol, weather_vec








def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_mode: bool,
    verbose: bool,
) -> Tuple[float, List[float]]:
    """Run one epoch of training or validation.

    Parameters
    ----------
    model : nn.Module
        The neural network to train or evaluate.
    loader : DataLoader
        DataLoader yielding ``(x, y, w)`` tuples.
    criterion : nn.Module
        Loss function returning a scalar per batch.
    optimizer : torch.optim.Optimizer
        Optimiser used during training.  Ignored in evaluation mode.
    device : torch.device
        Device on which to perform computation.
    train_mode : bool
        If ``True``, gradients are computed and parameters are
        updated.  Otherwise the model is set to eval mode and no
        optimisation occurs.
    verbose : bool
        If ``True``, prints progress every 50 batches during training.

    Returns
    -------
    Tuple[float, List[float]]
        Mean loss over the epoch and a list of per-batch losses.
    """
    model.train(mode=train_mode)
    running_loss = 0.0
    total_samples = 0
    batch_losses: List[float] = []
    for batch_idx, (x, y, w) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        w = w.to(device)
        with torch.set_grad_enabled(train_mode):
            y_hat = model(x, w)  # (B, F, H, W)
            loss = criterion(y_hat, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        bs = x.size(0)
        running_loss += loss.item() * bs
        total_samples += bs
        batch_losses.append(loss.item())
        if verbose and train_mode and batch_idx % 50 == 0:
            print(f"  batch {batch_idx}, loss {loss.item():.4f}")
    epoch_loss = running_loss / total_samples
    return epoch_loss, batch_losses


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int = 5,
    min_delta: float = 1e-4,
    verbose: bool = True,
    save_dir: str = "models",
    model_name: str = "checkpoint"
) -> dict[str, List]:
    """Train a model with early stopping based on validation loss.

    Returns a history dictionary containing epoch and batch losses for
    both training and validation.
    """
    history = {
        "train_epoch_loss": [],
        "val_epoch_loss": [],
        "train_batch_loss": [],
        "val_batch_loss": [],
    }
    best_val = float("inf")
    patience_counter = 0
    try: 
        for epoch in range(num_epochs):
            if verbose:
                print(f"\n----- Epoch {epoch+1}/{num_epochs} -----")
            # Training
            train_loss, train_batches = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                train_mode=True,
                verbose=verbose,
            )
            # Validation
            val_loss, val_batches = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer,
                device,
                train_mode=False,
                verbose=verbose,
            )
            # Record history
            history["train_epoch_loss"].append(train_loss)
            history["train_batch_loss"].append(train_batches)
            history["val_epoch_loss"].append(val_loss)
            history["val_batch_loss"].append(val_batches)

            # ---- SAVE AFTER EVERY EPOCH ----
            epoch_path = os.path.join(save_dir, f"{model_name}_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), epoch_path)


            if verbose:
                print(f"train: {train_loss:.4f}   val: {val_loss:.4f}")
            # Early stopping
            if val_loss < best_val - min_delta:
                best_val = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print("Early stopping triggered.")
                    break
    except KeyboardInterrupt:
        # Save safe interrupt checkpoint
        interrupt_path = os.path.join(save_dir, f"{model_name}_interrupt.pth")
        torch.save(model.state_dict(), interrupt_path)
        print(f"\nTraining interrupted. Model saved at: {interrupt_path}")
        return history
    return history


def main() -> None:
    """Entry point for training the enriched model."""
    # Load split file
    split_file = os.path.join(DATA_ROOT, "splits", "splits.json")
    with open(split_file, "r") as f:
        splits = json.load(f)
    train_names = splits.get("train", [])
    val_names = splits.get("val", [])
    train_paths = [os.path.join(DATA_ROOT, "data", fn) for fn in train_names]
    val_paths = [os.path.join(DATA_ROOT, "data", fn) for fn in val_names]
    history_steps = 12
    future_steps = 4
    # Instantiate WeatherEncoder (loads train/val/test subsets)
    weather_encoder = WeatherEncoder(WEATHER_ROOT)
    # Create datasets and loaders
    train_ds = TrafficDatasetEnriched(
        train_paths,
        lbcs_path=PATH_LBCS,
        weather_encoder=weather_encoder,
        history_steps=history_steps,
        future_steps=future_steps,
    )
    val_ds = TrafficDatasetEnriched(
        val_paths,
        lbcs_path=PATH_LBCS,
        weather_encoder=weather_encoder,
        history_steps=history_steps,
        future_steps=future_steps,
    )
    batch_size = 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Determine model dimensions
    x_sample, y_sample, w_sample = train_ds[0]
    in_channels = x_sample.shape[0]  # S*(C+9)
    out_channels = y_sample.shape[0]  # F
    weather_dim = w_sample.shape[0]   # number of weather features
    # Initialise model, loss and optimiser
    model = UNet2D_FiLM(in_channels, out_channels, weather_dim, base_ch=16).to(device)
    mask = load_mask(DATA_ROOT, device)
    criterion = MaskedMAEFocalLoss(mask=mask, beta=0.2, gamma=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 5

    # Save model checkpoint
    model_name = "U-net_enriched-0"
    # Train
    history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs,
        patience=5,
        min_delta=1e-4,
        verbose=True,
        save_dir=MODELS_ROOT,
        model_name=model_name
    )

    os.makedirs(MODELS_ROOT, exist_ok=True)
    checkpoint_path = os.path.join(MODELS_ROOT, f"{model_name}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved enriched model checkpoint to {checkpoint_path}")
    # Save training history
    history_folder = os.path.join(SCRIPT_DIR, "history")
    os.makedirs(history_folder, exist_ok=True)
    history_path = os.path.join(history_folder, f"history-{model_name}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()