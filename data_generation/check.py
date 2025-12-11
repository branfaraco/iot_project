import os, glob

DATA_ROOT = r"C:\Users\user\UPM\Imperial-4año\IoT\Github\hugging_face"
data_dir = os.path.join(DATA_ROOT, "BERLIN", "BERLIN", "training")

print("data_dir =", data_dir)
all_files = sorted(glob.glob(os.path.join(data_dir, "*_8ch.h5")))
print("Found", len(all_files), "files in training")
print("First 5:", [os.path.basename(p) for p in all_files[:5]])
