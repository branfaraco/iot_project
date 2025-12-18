from huggingface_hub import snapshot_download

repo_id = "bflc/IoT-Traffic5cast"  # exactly as in the URL

# token = ...


local_dir = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    token=token,               
    local_dir="hugging_face",  
    local_dir_use_symlinks=False
)

print(f"Dataset downloaded to: {local_dir}")