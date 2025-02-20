import os
import time
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.auto import tqdm

# Get authentication token from environment
token = os.environ.get("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN environment variable not found. Please set it before running.")

# Dataset information
repo_id = 'likaixin/ScreenSpot-Pro'
output_dir = "screenspot_dataset"
os.makedirs(output_dir, exist_ok=True)

print(f"Starting download of {repo_id} dataset...")
print(f"Using authentication token: {token[:4]}...{token[-4:] if len(token) > 8 else ''}")

# Maximum retry attempts
max_retries = 5


def download_with_retry():
    """Attempt to download the entire dataset with retries"""
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt + 1}/{max_retries}")
            snapshot = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=output_dir,
                token=token,
                max_workers=1,  # Reduce parallel downloads to avoid connection issues
                etag_timeout=30,
                ignore_patterns=[".*", "__pycache__", "*.pyc", ".git*"],
            )
            print(f"Successfully downloaded dataset to: {snapshot}")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
    return False


# Try to download the entire dataset
if not download_with_retry():
    print("Full dataset download failed. Attempting to download individual files...")

    # Try to download individual important files
    important_files = [
        "metadata.json",
        "annotations.json",
        "dataset_info.json",
        "README.md",
        "config.json"
    ]

    for file in important_files:
        for attempt in range(3):
            try:
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    repo_type="dataset",
                    local_dir=output_dir,
                    token=token,
                    resume_download=True
                )
                print(f"Downloaded {file} to {path}")
                break
            except Exception as e:
                print(f"Failed to download {file}: {e}")
                time.sleep(5)

print("Download process completed.")