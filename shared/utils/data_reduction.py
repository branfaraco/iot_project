import os, glob, h5py, numpy as np, warnings
import json

from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]

# ----- CONFIG -----
data_dir = os.path.join(DATA_ROOT, "BERLIN", "BERLIN", "training")
out_dir  = os.path.join(DATA_ROOT, "BERLIN_reduced", "data")
test_raw_dir = os.path.join(DATA_ROOT, "BERLIN_reduced", "test")

os.makedirs(out_dir, exist_ok=True)
os.makedirs(test_raw_dir, exist_ok=True)

split_file = os.path.join(DATA_ROOT, "BERLIN_reduced", "splits", "splits.json")

with open(split_file, "r") as f:
    splits = json.load(f)

train_files = splits.get("train", [])
val_files   = splits.get("val", [])
test_files  = splits.get("test", [])

files_to_process = train_files + val_files + test_files


T_START, T_END      = 100, 220      # temporal ROI



def temporal_crop(arr, t0, t1):
    return arr[t0:t1]                       # (T', H, W, C)

def spatial_crop(arr, r0, r1, c0, c1):
    return arr[:, r0:r1, c0:c1, :]          # (T', H_roi, W_roi, C)

def normalize(arr):
    return arr.astype(np.float32) / 255.0


# ----- global grid definition -----
H, W = 495, 436  # rows, cols

min_lon, max_lon = 13.189, 13.625   # west, east
min_lat, max_lat = 52.359, 52.854   # south, north

# cell size (assuming regular grid)
lon_step = (max_lon - min_lon) / W
lat_step = (max_lat - min_lat) / H

# ----- sub-grid indices ----- 

r0, r1 = 10, 274   # row range [128, 200)
c0, c1 = 0, 432 # col range [128, 200)

# Multiples of 8
m8 = lambda x: (x+7)//8*8
r0,r1,c0,c1 = map(m8, (r0,r1,c0,c1))


R_H = r1 - r0
R_W = c1 - c0

# If row 0 = south edge (increasing row → increasing latitude):
sub_min_lat = min_lat + r0 * lat_step
sub_max_lat = min_lat + r1 * lat_step
sub_min_lon = min_lon + c0 * lon_step
sub_max_lon = min_lon + c1 * lon_step





def process_all_files():
    for fname in files_to_process:
        in_path  = os.path.join(data_dir, fname)
        out_path = os.path.join(out_dir, fname)

        if not os.path.exists(in_path):
            print("WARNING: Missing file:", in_path)
            continue

        print("Processing:", fname)

        # read full, unprocessed array
        with h5py.File(in_path, "r") as f:
            arr = f["array"][()]                 # (T, H, W, 8)

        arr_proc = temporal_crop(arr, T_START, T_END) # temporal crop donne to everyone
        # if this is a test file, save an unprocessed copy
        if fname in test_files:
            raw_out_path = os.path.join(test_raw_dir, fname)
            os.makedirs(os.path.dirname(raw_out_path), exist_ok=True)
            with h5py.File(raw_out_path, "w") as f_raw:
                f_raw.create_dataset("array", data=arr_proc,
                                     compression="gzip", compression_opts=4)

        # now do preprocessing for the main reduced dataset
        arr_proc = spatial_crop(arr_proc, r0, r1, c0, c1)
        arr_proc = normalize(arr_proc)

        with h5py.File(out_path, "w") as f_out:
           f_out.create_dataset("array", data=arr_proc,
                                 compression="gzip", compression_opts=4)

    print("Done. Reduced files saved in:", out_dir)
    print("Unprocessed test files saved in:", test_raw_dir)

    # clean up extra processed files
    existing_reduced = glob.glob(os.path.join(out_dir, "*.h5"))
    files_to_keep = {os.path.join(out_dir, f) for f in files_to_process}

    for path in existing_reduced:
        if path not in files_to_keep:
            print("Removing from data/:", os.path.basename(path))
            os.remove(path)

    # clean up extra raw test files
    existing_test_raw = glob.glob(os.path.join(test_raw_dir, "*.h5"))
    test_files_to_keep = {os.path.join(test_raw_dir, f) for f in test_files}

    for path in existing_test_raw:
        if path not in test_files_to_keep:
            print("Removing from test/:", os.path.basename(path))
            os.remove(path)

def save_metadata():
    rows = r1 - r0
    cols = c1 - c0

    metadata = {
        "rows": rows,
        "cols": cols,
        "bbox": [
            [sub_min_lat, sub_min_lon],   # southwest
            [sub_max_lat, sub_max_lon],   # northeast
        ],
    }

    meta_path = os.path.join(DATA_ROOT, "BERLIN_reduced", "metadata.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved metadata to:", meta_path)



def reduce_static_mask():
    mask_path = os.path.join(DATA_ROOT, "BERLIN", "BERLIN", "BERLIN_static.h5")
    out_path  = os.path.join(DATA_ROOT, "BERLIN_reduced", "BERLIN_static.h5")

    if not os.path.exists(mask_path):
        print("WARNING: Missing static mask file:", mask_path)
        return

    with h5py.File(mask_path, "r") as f:
        static = f["array"][()]  # expected (C, H, W)

    static_crop = static[:, r0:r1, c0:c1]  # (C, H_roi, W_roi)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("array", data=static_crop,
                         compression="gzip", compression_opts=4)

    print("Saved reduced static mask:", out_path, "shape:", static_crop.shape)


if __name__ == "__main__":
    save_metadata()
    reduce_static_mask()
    process_all_files()   
