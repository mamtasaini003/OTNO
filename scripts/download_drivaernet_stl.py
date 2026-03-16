import requests
import os
import sys

# Dataset persistent ID
PERSISTENT_ID = "doi:10.7910/DVN/OYU2FG"
BASE_URL = "https://dataverse.harvard.edu/api/access/datafile/"
TARGET_DIR = "/media/HDD/mamta_backup/datasets/topos/drivaernet/stl/"

# File manifest (ID, Name, Size) - Fastback Subset (~83 GB)
FILES = [
    (10816695, "F_D_WM_WW_1.zip", 10428001802),
    (10816697, "F_D_WM_WW_2.zip", 10428001802),
    (10816696, "F_D_WM_WW_3.zip", 10428001802),
    (10816694, "F_D_WM_WW_4.zip", 10428001802),
    (10816692, "F_D_WM_WW_5.zip", 10428001802),
    (10816704, "F_D_WM_WW_6.zip", 10428001802),
    (10816693, "F_D_WM_WW_7.zip", 10428001802),
    (10816705, "F_D_WM_WW_8.zip", 9677185514),
]

def download_file(file_id, filename, expected_size):
    target_path = os.path.join(TARGET_DIR, filename)
    
    # Check if already exists and size matches
    if os.path.exists(target_path):
        actual_size = os.path.getsize(target_path)
        if actual_size == expected_size:
            print(f"Skipping {filename}, already exists and size matches.")
            return True
        else:
            print(f"File {filename} exists but size mismatch ({actual_size} vs {expected_size}). Resuming/Restarting.")

    print(f"Downloading {filename} ({expected_size / 1e9:.2f} GB)...")
    url = f"{BASE_URL}{file_id}"
    
    # Use wget or curl for better large file handling and progress
    cmd = f"wget -c -O {target_path} {url}"
    res = os.system(cmd)
    return res == 0

if __name__ == "__main__":
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        
    for fid, name, size in FILES:
        success = download_file(fid, name, size)
        if not success:
            print(f"Failed to download {name}. Skipping to next.")

    print("Download process complete.")
