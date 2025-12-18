import os
import glob
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


from shared.models.unet_base import UNet2D
from shared.utils.indexing import build_time_index
from shared.utils.losses import MaskedMAEFocalLoss
from shared.utils.mask import load_mask

from dotenv import load_dotenv
import os

load_dotenv()

REPO_ROOT = os.environ["REPO_ROOT"]
DATA_ROOT = os.environ["DATA_ROOT"]
MODELS_ROOT = os.environ["MODEL_PARAMETERS_DIR"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class TrafficDataset(Dataset):
    """
    Lazy-loading version:
    - Does NOT load full days into RAM.
    - Only stores paths + index of (file, t0).
    - Each __getitem__ loads (S+F) frames from HDF5 on demand.
    """

    def __init__(self, h5_paths, history_steps=12, future_steps=6, dataset_key="array"):
        self.h5_paths = list(h5_paths)
        self.S = history_steps
        self.F = future_steps
        self.dataset_key = dataset_key

        self.index = build_time_index(
                self.h5_paths,
                history_steps=self.S,
                future_steps=self.F,
                dataset_key=self.dataset_key,
            )

        if not self.index:
            raise ValueError("No valid temporal windows found.")

        print(f"Lazy Dataset: {len(self.index)} samples from {len(self.h5_paths)} files (lazy loading).")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, t0 = self.index[idx]
        path = self.h5_paths[file_idx]

        # Open the file and load ONLY the slice needed
        with h5py.File(path, "r") as f:
            dset = f[self.dataset_key]
            # read window (S + F frames)
            window = dset[t0:t0 + self.S + self.F]  # (T, H, W, 8), uint8

        # Convert to float32 only now (NOT earlier)
        arr = torch.from_numpy(window).float()     # (T, H, W, C)
        arr = arr.permute(0, 3, 1, 2).contiguous() # (T, C, H, W)

        # Split into history + future
        x_full = arr[: self.S]              # (S, 8, H, W)
        y_full = arr[self.S: self.S+self.F] # (F, 8, H, W)

        # Flatten history channels
        S, C, H, W = x_full.shape
        x = x_full.reshape(S * C, H, W)     # (S*8, H, W)

        # Aggregate 4 volume channels for target
        vol_channels = [0, 2, 4, 6]
        y_vol = y_full[:, vol_channels].sum(dim=1)  # (F, H, W)

        return x, y_vol




def run_epoch(model, loader, criterion, optimizer, device, train_mode, verbose):
    model.train(mode=train_mode)

    running_loss = 0.0
    total_samples = 0
    batch_losses = []

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train_mode):
            y_hat = model(x)              # (B, F, H, W)
            loss = criterion(y_hat, y)    # escalar

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
        save_dir: str = "models",
    model_name: str = "checkpoint"
    ):
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

            # ------- TRAIN -------
            train_loss, train_batches = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                train_mode=True,
                verbose=verbose
            )

            # ------- VALIDATION -------
            val_loss, val_batches = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                train_mode=False,
                verbose=verbose
            )

            # Log
            history["train_epoch_loss"].append(train_loss)
            history["train_batch_loss"].append(train_batches)
            history["val_epoch_loss"].append(val_loss)
            history["val_batch_loss"].append(val_batches)

            if verbose:
                print(f"train: {train_loss:.4f}   val: {val_loss:.4f}")

            # ------- EARLY STOP -------
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


def main():
    # load train and val
    
    SPLIT_FILE = os.path.join(DATA_ROOT, "splits", "splits.json")
    all_h5_files = sorted(glob.glob(os.path.join(DATA_ROOT, "data", "*.h5")))
    
    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)
    
    train_names = splits["train"]
    val_names   = splits["val"]

    train_paths = [os.path.join(DATA_ROOT, "data", fn) for fn in train_names]
    val_paths   = [os.path.join(DATA_ROOT,"data", fn) for fn in val_names]

    history_steps = 12
    future_steps  = 4

    # datasets and dataloaders
    train_ds = TrafficDataset(
        train_paths,
        history_steps=history_steps,
        future_steps=future_steps
    )

    val_ds = TrafficDataset(
        val_paths,
        history_steps=history_steps,
        future_steps=future_steps
    ) 
    batch_size  = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    ) 

    # model def
    x_sample, y_sample = train_ds[0]
    in_channels  = x_sample.shape[0]   # S * 8
    out_channels = y_sample.shape[0]   # F


    model = UNet2D(in_channels, out_channels, base_ch=16).to(device)
    mask = load_mask(DATA_ROOT, device)
    criterion = MaskedMAEFocalLoss(mask=mask, beta=0.2, gamma=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 5 # each epoch is 10 min

    model_name = "U-net_base-0"
    # training
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
    save_dir=MODELS_ROOT,
    model_name=model_name,
    verbose=True
)
    
    checkpoint_path = os.path.join(REPO_ROOT, "models", f"{model_name}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model checkpoint to {checkpoint_path}")
    
    history_folder = os.path.join(SCRIPT_DIR, "history")
    os.makedirs(history_folder, exist_ok=True)
    history_path = os.path.join(history_folder, f"history-{model_name}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved training history to {history_path}")



if __name__ == "__main__":
    main()