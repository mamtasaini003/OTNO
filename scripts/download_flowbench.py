import os
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "BGLab/FlowBench"
REPO_TYPE = "dataset"
LOCAL_DIR = "/media/HDD/mamta_backup/datasets/topos/flowbench/"
INCLUDE_PATTERN = "LDC_NS_2D/512x512/"

def download_flowbench():
    files = list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    target_files = [f for f in files if INCLUDE_PATTERN in f]
    
    for file_path in target_files:
        print(f"Downloading {file_path}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=file_path,
            repo_type=REPO_TYPE,
            local_dir=LOCAL_DIR
        )

if __name__ == "__main__":
    download_flowbench()
