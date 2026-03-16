import os
import sys
from pathlib import Path
sys.path.append('/home/mamta/work/neuraloperator/')
from neuralop.data.datasets.car_cfd_dataset import CarCFDDataset

# Define the root directory for the dataset
root_dir = Path('/home/mamta/data/car-pressure-data/')

# Ensure the directory exists
root_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading ShapeNet-Car dataset to {root_dir}...")

# Initialize the dataset with download=True
# n_train and n_test are set to 1 just to trigger the download/initialization
# The actual script will load what's available
dataset = CarCFDDataset(root_dir=root_dir, n_train=1, n_test=1, download=True)

print("Download complete or dataset already present.")
