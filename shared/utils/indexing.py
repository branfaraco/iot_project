import h5py

def build_time_index(h5_paths, history_steps, future_steps, dataset_key="array"):
    index = []
    for fi, path in enumerate(h5_paths):
        with h5py.File(path, "r") as f:
            dset = f[dataset_key]
            T = dset.shape[0]
        max_start = T - (history_steps + future_steps) + 1
        if max_start <= 0:
            continue
        for t0 in range(max_start):
            index.append((fi, t0))
    if not index:
        raise ValueError("No valid temporal windows found.")
    return index
