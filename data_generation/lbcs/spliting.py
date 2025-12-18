import os
import glob
import json
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]

data_dir = os.path.join(DATA_ROOT, "BERLIN", "BERLIN", "training")

def extract_date(path):
    """Extract YYYY-MM-DD from filename."""
    fname = os.path.basename(path)
    date_str = fname.split("_")[0]
    return datetime.strptime(date_str, "%Y-%m-%d")

# Load all HDF5 files
all_paths = sorted(glob.glob(os.path.join(data_dir, "*_8ch.h5")))
all_dates = [extract_date(p) for p in all_paths]

# Last available date
last_date = max(all_dates)

# We want: 30 days train + 7 days val + 7 days test = 44 days
WINDOW_DAYS = 44
cutoff = last_date - timedelta(days=WINDOW_DAYS)

# Select last 44 days
paths_44 = [(p, d) for p, d in zip(all_paths, all_dates) if d > cutoff]
paths_44_sorted = sorted(paths_44, key=lambda x: x[1])

# Split
train_paths = [p for p, d in paths_44_sorted[:30]]
val_paths   = [p for p, d in paths_44_sorted[30:37]]
test_paths  = [p for p, d in paths_44_sorted[37:44]]

print("Train:", len(train_paths))
print("Val:", len(val_paths))
print("Test:", len(test_paths))

# Create JSON dictionary
splits = {
    "train": [os.path.basename(p) for p in train_paths],
    "val":   [os.path.basename(p) for p in val_paths],
    "test":  [os.path.basename(p) for p in test_paths],
}

# Save to file
out_path = os.path.join(DATA_ROOT, "BERLIN_reduced", 'splits', "splits.json")
with open(out_path, "w") as f:
    json.dump(splits, f, indent=2)

print(f"Saved new split to {out_path}")
