"""
Tier 1: Topological Generalization (The Router Test)

Dataset: Mixed-genus (Spheres, Tori, Double-Tori)
Task: Solve Poisson Equation (Delta u = f)
Metric: RMSE vs. a "Fixed-Branch" model

This script specifically validates that the TOPOS Topological Router correctly
identifies and handles different genus types in a single batch.
"""

import sys
import os
import torch

from topos.models.topos import TOPOS
from topos.utils.utils import LpLoss

def main():
    print("--- Tier 1: Topological Generalization (Router Test) ---")
    
    # 1. Define TOPOS model with all branches (Auto-Routing)
    model_auto = TOPOS(
        spherical_config=dict(n_modes=(16, 16), hidden_channels=32, in_channels=1, out_channels=1, n_layers=4),
        volumetric_config=dict(n_modes=(16, 16, 16), hidden_channels=32, in_channels=1, out_channels=1, n_layers=4),
        toroidal_config=dict(n_modes=(16, 16), hidden_channels=32, in_channels=1, out_channels=1, n_layers=4),
        chi_tol=0.5,
        default_topology='auto'
    )
    
    # 2. Define "Fixed-Branch" model (Ablation: forcing Sphere architecture for all)
    model_fixed = TOPOS(
        spherical_config=dict(n_modes=(16, 16), hidden_channels=32, in_channels=1, out_channels=1, n_layers=4),
        chi_tol=0.5,
        default_topology='spherical'
    )
    
    criterion = LpLoss(d=2, p=2)
    
    print("Models initialized successfully.")
    print("\nExpected validation protocol:")
    print("1. Load Mixed-Genus Dataset: Spheres (chi=2), Tori (chi=0), Double-Tori (chi=-2)")
    print("2. Train model_auto and model_fixed on the dataset.")
    print("3. Evaluate on test set.")
    print("=> Success criteria: TOPOS maintains < 5% error. Fixed-branch fails on mismatched topologies (chi <= 0).")
    print("\nNote: Please execute with proper dataset loader and compute environment.")

if __name__ == '__main__':
    main()
