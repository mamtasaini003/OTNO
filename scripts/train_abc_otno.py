"""
OTNO Training Script for ABC / ShapeNet Car Pressure Dataset.

Aligned with the reference implementation in experiments/shapenet/topos/ot_train.py
and the OTNO paper specifications:
  - Torus latent space (best per ablation Table 6: 6.70% vs 7.09% sphere)
  - 9-channel input: transports(3) + torus_coords(3) + normal_cross(3)
  - n_modes=(32,32), hidden=120, group_norm, tucker factorization
  - Loss on NORMALIZED space during training; decode only for test
  - StepLR(step_size=50, gamma=0.5)
  - LpLoss(size_average=False) summed, divided by n_train/n_test
"""

import os
import sys
import torch
import torch.nn as nn
import random
import yaml
import argparse
import numpy as np
from timeit import default_timer
from torch.utils.data import DataLoader, Dataset

# Add path for model modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from topos.models.fno_spherical import ToroidalTransportFNO
from topos.utils import LpLoss, UnitGaussianNormalizer, DictDataset, DictDatasetWithConstant


def set_seed(seed):
    """Seed all components for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for convolutions (Note: may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[*] Seed set to: {seed}")


# ---- Torus Geometry Utilities (matching reference ot_train.py) ----

def create_torus_grid(n_s_sqrt, R=1.5, r=1.0):
    """Create a torus mesh in 3D from a 2D parameter grid."""
    theta = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    phi   = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')

    x = (R + r * torch.cos(theta)) * torch.cos(phi)
    y = (R + r * torch.cos(theta)) * torch.sin(phi)
    z = r * torch.sin(theta)

    return torch.stack((x, y, z), dim=-1)  # (n_s_sqrt, n_s_sqrt, 3)


def compute_torus_normals(n_s_sqrt, R=1.5, r=1.0):
    """Compute analytical normals on the torus surface."""
    theta = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    phi   = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')

    # Partial derivatives dP/dtheta and dP/dphi
    dx_dtheta = -r * torch.sin(theta) * torch.cos(phi)
    dy_dtheta = -r * torch.sin(theta) * torch.sin(phi)
    dz_dtheta =  r * torch.cos(theta)

    dx_dphi = -(R + r * torch.cos(theta)) * torch.sin(phi)
    dy_dphi =  (R + r * torch.cos(theta)) * torch.cos(phi)
    dz_dphi =  torch.zeros_like(dx_dphi)

    # Cross product for normals
    nx = dy_dtheta * dz_dphi - dz_dtheta * dy_dphi
    ny = dz_dtheta * dx_dphi - dx_dtheta * dz_dphi
    nz = dx_dtheta * dy_dphi - dy_dtheta * dx_dphi

    normals = torch.stack((nx, ny, nz), dim=-1)
    norm = torch.linalg.norm(normals, dim=2, keepdim=True)
    normals = normals / norm

    return normals  # (n_s_sqrt, n_s_sqrt, 3)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="OTNO Trainer for ABC/ShapeNet Car Pressure Dataset")
    parser.add_argument('--config', type=str, default='configs/otno.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config['training']['n_epochs'] = args.epochs
    
    # Set seed
    set_seed(config['training'].get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Configuration (aligned with reference ot_train.py) ----
    expand_factor = config['dataset'].get('expand_factor', 2.0)
    n_train = config['dataset']['train_samples']
    n_test  = config['dataset']['test_samples']
    epochs  = config['training']['n_epochs']

    # Paper uses n_t = 3586 physical mesh vertices for ShapeNet
    # n_s_sqrt = int(sqrt(expand_factor) * ceil(sqrt(n_t)))
    # For expand_factor=2.0, n_t=3586 → n_s_sqrt = 84

    # ---- Load precomputed OT data (torus, "Mean" strategy) ----
    data_path = config['dataset'].get('ot_data_path', None)
    if data_path is None:
        # Infer from dataset path
        base_dir = os.path.dirname(config['dataset']['path'])
        data_path = os.path.join(base_dir, f'torus_OTmean_geomloss_expand{expand_factor}.pt')

    print(f"[*] Loading precomputed OT data from: {data_path}")
    data = torch.load(data_path, weights_only=False)

    n_s_sqrt = data['transports'].shape[2]  # 84 for expand_factor=2.0
    print(f"[*] Data loaded: {data['transports'].shape[0]} samples, "
          f"latent grid {n_s_sqrt}×{n_s_sqrt}, "
          f"{data['points'].shape[1]} physical vertices")

    # ---- Split train/test ----
    train_transports      = data['transports'][:n_train]
    train_normals         = data['normals'][:n_train]
    train_pressures       = data['pressures'][:n_train]
    train_points          = data['points'][:n_train]
    train_indices_encoder = data['indices_encoder'][:n_train]
    train_indices_decoder = data['indices_decoder'][:n_train]

    test_transports       = data['transports'][n_train:n_train+n_test]
    test_normals          = data['normals'][n_train:n_train+n_test]
    test_pressures        = data['pressures'][n_train:n_train+n_test]
    test_points           = data['points'][n_train:n_train+n_test]
    test_indices_encoder  = data['indices_encoder'][n_train:n_train+n_test]
    test_indices_decoder  = data['indices_decoder'][n_train:n_train+n_test]

    # ---- Normalization (matching reference: normalize BEFORE training) ----
    # pressure: normalize over [samples, vertices] → per-sample scalar stats
    pressure_encoder = UnitGaussianNormalizer(train_pressures, reduce_dim=[0, 1])
    # transport: normalize over [samples, H, W] → per-channel stats
    transport_encoder = UnitGaussianNormalizer(train_transports, reduce_dim=[0, 2, 3])

    # Normalize training data (in-place on cloned data)
    train_pressures = pressure_encoder.encode(train_pressures.clone())
    train_transports = transport_encoder.encode(train_transports.clone())

    # Normalize test transports (test pressures stay raw for evaluation)
    test_transports = transport_encoder.encode(test_transports.clone())

    pressure_encoder.to(device)

    print(f"[*] Normalizers initialized. Pressure mean={pressure_encoder.mean.item():.4f}, "
          f"std={pressure_encoder.std.item():.4f}")

    # ---- Build constant data: torus grid + torus normals ----
    R, r_torus = 1.5, 1.0
    pos_embed = create_torus_grid(n_s_sqrt, R, r_torus)         # (n_s_sqrt, n_s_sqrt, 3)
    torus_normals = compute_torus_normals(n_s_sqrt, R, r_torus)  # (n_s_sqrt, n_s_sqrt, 3)

    # ---- Datasets ----
    train_dict = {
        'transports': train_transports,
        'pressures': train_pressures,
        'points': train_points,
        'normals': train_normals,
        'indices_encoder': train_indices_encoder,
        'indices_decoder': train_indices_decoder,
    }
    test_dict = {
        'transports': test_transports,
        'pressures': test_pressures,
        'points': test_points,
        'normals': test_normals,
        'indices_encoder': test_indices_encoder,
        'indices_decoder': test_indices_decoder,
    }

    train_dataset = DictDatasetWithConstant(train_dict, {'pos': pos_embed, 'nor': torus_normals})
    test_dataset  = DictDatasetWithConstant(test_dict,  {'pos': pos_embed, 'nor': torus_normals})

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False)

    # ---- Model: Toroidal TransportFNO (matching reference exactly) ----
    # Paper input: T_j = (Xi_j, M_j, H_j × N_j(E)) → 9 channels
    # Reference concatenates: transports(3) + torus_pos(3) + normal_cross(3) = 9ch
    # But in_channels in config may say 9, which is correct for the 9-channel input
    in_channels = 9  # transports(3) + torus_coords(3) + cross_normals(3)

    model = ToroidalTransportFNO(
        n_modes=tuple(config['model']['n_modes']),
        hidden_channels=config['model']['hidden_channels'],
        in_channels=in_channels,
        out_channels=config['model']['out_channels'],
        n_layers=config['model']['n_layers'],
        use_mlp=config['model']['use_mlp'],
        mlp=config['model']['mlp'],
        norm=config['model'].get('norm', None),
        domain_padding=config['model'].get('domain_padding', None),
        factorization=config['model'].get('factorization', 'tucker'),
        rank=config['model'].get('rank', 1.0),
    ).to(device)

    n_params = sum(p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters())
    print(f"[*] Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['step_size'],
        gamma=config['training']['gamma'],
    )
    # Reference uses size_average=False: loss summed over batch, divided by n_train/n_test
    myloss = LpLoss(size_average=False)

    # ---- Training Loop (matching reference ot_train.py exactly) ----
    print(f"[*] Training OTNO (Torus) on {n_train} samples for {epochs} epochs...")
    for ep in range(epochs):
        t = default_timer()
        train_l2 = 0.0
        model.train()

        for batch_data in train_loader:
            optimizer.zero_grad()

            # Normalized transport images: (B, 3, n_s_sqrt, n_s_sqrt)
            transports = batch_data['transports'].to(device)
            # Normalized pressures: (B, n_vertices)
            pressures = batch_data['pressures'].to(device)
            # Physical mesh normals: (n_vertices, 3) — take first in batch
            normals = batch_data['normals'][0].to(device)
            indices_encoder = batch_data['indices_encoder'][0].to(dtype=torch.long, device=device)
            indices_decoder = batch_data['indices_decoder'][0].to(dtype=torch.long, device=device)

            # ---- Build 9-channel input (per paper Algorithm 1, Step 3) ----

            # 1. Normal cross-product feature:
            #    Physical normals indexed by encoder → mapped to torus grid
            #    Cross with torus analytical normals → captures deformation
            mapped_normals = normals[indices_encoder]  # (n_s_sqrt^2, 3)
            torus_norms_flat = batch_data['nor'].reshape(-1, 3).to(device)  # (n_s_sqrt^2, 3)
            normal_features = torch.cross(mapped_normals, torus_norms_flat, dim=1)  # (n_s_sqrt^2, 3)
            normal_features = normal_features.reshape(n_s_sqrt, n_s_sqrt, 3).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

            # 2. Torus coordinate embedding: (1, 3, H, W)
            pos_features = batch_data['pos'].permute(0, 3, 1, 2).to(device)

            # 3. Concatenate: transports(3) + torus_coords(3) + normal_cross(3) = 9ch
            transports = torch.cat((transports, pos_features, normal_features), dim=1)

            # Forward pass
            out = model(transports, indices_decoder)

            # Loss on NORMALIZED space (matching reference exactly)
            loss = myloss(out, pressures)
            loss.backward()
            optimizer.step()

            train_l2 += loss.item()

        scheduler.step()

        # ---- Test evaluation ----
        test_l2 = 0.0
        model.eval()
        with torch.no_grad():
            for batch_data in test_loader:
                transports = batch_data['transports'].to(device)
                pressures = batch_data['pressures'].to(device)
                normals = batch_data['normals'][0].to(device)
                indices_encoder = batch_data['indices_encoder'][0].to(dtype=torch.long, device=device)
                indices_decoder = batch_data['indices_decoder'][0].to(dtype=torch.long, device=device)

                mapped_normals = normals[indices_encoder]
                torus_norms_flat = batch_data['nor'].reshape(-1, 3).to(device)
                normal_features = torch.cross(mapped_normals, torus_norms_flat, dim=1)
                normal_features = normal_features.reshape(n_s_sqrt, n_s_sqrt, 3).permute(2, 0, 1).unsqueeze(0)

                pos_features = batch_data['pos'].permute(0, 3, 1, 2).to(device)
                transports = torch.cat((transports, pos_features, normal_features), dim=1)

                out = model(transports, indices_decoder)
                # Decode to physical space for test metric
                out = pressure_encoder.decode(out.clone())

                test_l2 += myloss(out, pressures).item()

        train_l2 /= n_train
        test_l2 /= n_test

        elapsed = default_timer() - t
        print(f"Epoch {ep+1}/{epochs}, Train L2: {train_l2:.6f}, Test L2: {test_l2:.6f}, Time: {elapsed:.2f}s")

    print(f"[*] Training complete.")


if __name__ == "__main__":
    main()
