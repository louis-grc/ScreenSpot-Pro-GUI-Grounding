from huggingface_hub import hf_hub_download
import os
import json
import shutil
from tqdm import tqdm

# Download annotation files
repo_id = 'likaixin/ScreenSpot-Pro'  # Adjust this to the correct repo
output_dir = "screenspot_dataset"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get the list of files in the repo
from huggingface_hub import HfApi

api = HfApi()
files_info = api.list_repo_files(repo_id)

# Download all files
for file_path in tqdm(files_info, desc="Downloading files"):
    try:
        # Skip .git files or any other unwanted files
        if file_path.startswith('.git') or '__pycache__' in file_path:
            continue

        # Create the directory structure
        local_path = os.path.join(output_dir, file_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download the file
        hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
    except Exception as e:
        print(f"Error downloading {file_path}: {e}")

print('Dataset download complete')