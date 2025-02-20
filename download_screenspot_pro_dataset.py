from huggingface_hub import hf_hub_download, HfApi
import os
from tqdm import tqdm

repo_id = 'likaixin/ScreenSpot-Pro'  # Confirm the correct repo ID
output_dir = "screenspot_dataset"
token = os.getenv("HF_TOKEN")  # Or set token directly: token = "your_token_here"

api = HfApi(token=token)

try:
    files_info = api.list_repo_files(repo_id, repo_type="dataset")  # Change repo_type if needed
except Exception as e:
    print(f"Error fetching file list: {e}")
    exit(1)

os.makedirs(output_dir, exist_ok=True)

for file_path in tqdm(files_info, desc="Downloading files"):
    if file_path.startswith('.git') or '__pycache__' in file_path:
        continue
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
    except Exception as e:
        print(f"Error downloading {file_path}: {e}")

print("Dataset download complete.")
