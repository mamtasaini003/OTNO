import os
import sys
import torch
import numpy as np

# Add path for modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train_abc_otno import ABCDataset

def test():
    abc_dir = "/media/HDD/mamta_backup/datasets/otno/car-pressure-data/data"
    if not os.path.exists(abc_dir):
        print(f"Error: {abc_dir} not found")
        return
        
    ds = ABCDataset(abc_dir, abc_dir, n_train=10, n_test=5, split='train')
    if len(ds) == 0:
        print("Error: Dataset empty")
        return
        
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    print(f"Points shape: {sample['points'].shape}")
    print(f"Pressure shape: {sample['pressure'].shape}")
    print(f"OT Encoder shape: {sample['idx_encoder'].shape}")
    print(f"Success!")

if __name__ == "__main__":
    test()
