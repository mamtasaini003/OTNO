# OTNO Project: Datasets Summary

This document provides a comprehensive overview of the datasets used in the OTNO project, their storage locations on this workstation, and their intended usage in the codebase.

## 1. ShapeNet-Car

- **Location**: `/media/HDD/mamta_backup/datasets/topos/shapenet/`
- **Type**: 3D Triangle Meshes (`.ply`), Pressure Fields (`.npy`), and Processed PyTorch Data (`.pt`).
- **Content**: 
    - **Raw Data**: 611 car models with corresponding pressure values at mesh vertices, derived from CFD simulations.
    - **Processed Data**: `torus_OTmean_geomloss_expand2.0.pt`. This file contains the optimal transport mappings between the car geometries and a canonical torus latent space, along with normals and pressures.
- **Usage**: Used for training the core OT-FNO model to predict surface pressure fields from 3D car geometry.
- **Key Scripts**: 
    - `shapenet/topos/ot_datamodule.py`: Processing and OT mapping.
    - `shapenet/topos/ot_train.py`: Main training script.

---

## 2. FlowBench (LDC_NS_2D)

- **Location**: `/media/HDD/mamta_backup/datasets/topos/flowbench/`
- **Type**: 2D Numpy Arrays (`.npz`).
- **Content**: 
    - **Resolution**: 512x512 pixels.
    - **Geometry Variations**: `harmonics`, `nurbs`, and `skelneton` based lid-driven cavity walls.
    - **Fields**: Contains Signed Distance Functions (SDF) of the boundaries and the resulting steady-state velocity (x, y) and pressure fields.
- **Usage**: Benchmarking the performance of Optimal Transport Neural Operators on 2D steady-state Navier-Stokes equations with complex interior boundaries.
- **Key Scripts**: 
    - `flowbench/topos/ot_datamodule.py`: Data loading and 2D OT mapping.
    - `flowbench/topos/ot_train.py`: Training for flow field prediction.

---

## 3. DrivAerNet

- **Location**: `/media/HDD/mamta_backup/datasets/topos/drivaernet/`
- **Type**: CSV (`.csv`) for scalars, PyTorch (`.pt`) for pre-defined splits.
- **Content**: 
    - **Aerodynamic Coefficients**: Drag coefficient (Cd) values for thousands of car designs.
    - **Splits**: Train, validation, and test design ID lists.
    - **Note**: Raw STL meshes (85GB+) are currently external (Harvard Dataverse/Globus) and not stored locally due to storage constraints.
- **Usage**: Training regression models to predict high-level aerodynamic performance metrics (like Cd) directly from geometry.
- **Key Scripts**: 
    - `drivaernet/topos/ot_train_Cd.py`: Training the Cd prediction model (`TransportFNOCd`).

---

## Summary Table

| Dataset | Storage Path | Primary Format | Key Usage |
| :--- | :--- | :--- | :--- |
| **ShapeNet-Car** | `.../datasets/topos/shapenet/` | `.pt` (Processed) | 3D Pressure Prediction |
| **FlowBench** | `.../datasets/topos/flowbench/` | `.npz` | 2D Navier-Stokes CFD |
| **DrivAerNet** | `.../datasets/topos/drivaernet/` | `.csv`, `.pt` | Drag Coefficient (Cd) Prediction |
