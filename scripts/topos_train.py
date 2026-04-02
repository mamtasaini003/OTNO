"""
TOPOS Training Script — Unified trainer for the TOPOS architecture.

Supports all three topology branches (spherical, toroidal, volumetric)
and both supervised L2 and physics-informed losses.

Usage examples:
  # ShapeNet with spherical (default OTNO behavior)
  python topos_train.py --dataset shapenet --topology spherical

  # ShapeNet with auto-routing (Euler characteristic based)
  python topos_train.py --dataset shapenet --topology auto

  # Flowbench with volumetric branch
  python topos_train.py --dataset flowbench --topology volumetric \
      --resolution 128 --group_name nurbs --latent_shape square --expand_factor 2

  # With physics-informed loss
  python topos_train.py --dataset shapenet --topology spherical --lambda_pde 0.01
"""

import argparse
import copy
import csv
import os
import sys
import warnings
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from timeit import default_timer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from topos.utils import (
    LpLoss, count_model_params, UnitGaussianNormalizer,
    DictDataset, DictDatasetWithConstant, CombinedLoss,
)
from topos.models.topos import TOPOS


# ============================================================
# Routing helpers
# ============================================================

def _first_scalar(value):
    """Extract a Python scalar from common batch container types."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return value.reshape(-1)[0].item()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return _first_scalar(value[0])
    return value


def _first_string(value):
    scalar = _first_scalar(value)
    if scalar is None:
        return None
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    if isinstance(scalar, str):
        return scalar
    return str(scalar)


def _extract_meta_value(batch_data, key):
    meta = batch_data.get("meta")
    if isinstance(meta, dict):
        return meta.get(key)
    return None


def resolve_routing_for_batch(args, batch_data, default_chi=None, default_topology="spherical"):
    """Resolve topology and chi for a batch.

    If `args.topology != "auto"`, honors the explicit topology.
    If `args.topology == "auto"`, prefers chi from batch data/meta and
    falls back to explicit per-sample topology labels when provided.
    """
    if args.topology != "auto":
        return args.topology, None

    chi_sources = [
        batch_data.get("chi"),
        batch_data.get("euler_chi"),
        _extract_meta_value(batch_data, "euler_chi"),
        _extract_meta_value(batch_data, "chi"),
    ]
    for candidate in chi_sources:
        chi = _first_scalar(candidate)
        if chi is not None:
            return "auto", float(chi)

    if default_chi is not None:
        return "auto", float(default_chi)

    topology_sources = [
        batch_data.get("topology"),
        _extract_meta_value(batch_data, "topology"),
    ]
    for candidate in topology_sources:
        topology = _first_string(candidate)
        if topology in {"spherical", "toroidal", "volumetric", "graph"}:
            return topology, None

    warnings.warn(
        "[TOPOS WARNING] --topology auto requested, but no per-sample chi/topology found. "
        f"Falling back to topology='{default_topology}'."
    )
    return default_topology, None


# ============================================================
# Grid helpers (from existing OTNO code)
# ============================================================

def create_torus_grid(n_s_sqrt, R=1.5, r=1.0):
    """Create a torus grid for the toroidal/spherical latent domain."""
    theta = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    phi = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')
    x = (R + r * torch.cos(theta)) * torch.cos(phi)
    y = (R + r * torch.cos(theta)) * torch.sin(phi)
    z = r * torch.sin(theta)
    return torch.stack((x, y, z), axis=-1)


def compute_torus_normals(n_s_sqrt, R=1.5, r=1.0):
    """Compute normals on a torus grid."""
    theta = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    phi = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')
    dx_dtheta = -r * torch.sin(theta) * torch.cos(phi)
    dy_dtheta = -r * torch.sin(theta) * torch.sin(phi)
    dz_dtheta = r * torch.cos(theta)
    dx_dphi = -(R + r * torch.cos(theta)) * torch.sin(phi)
    dy_dphi = (R + r * torch.cos(theta)) * torch.cos(phi)
    dz_dphi = torch.zeros_like(dx_dphi)
    nx = dy_dtheta * dz_dphi - dz_dtheta * dy_dphi
    ny = dz_dtheta * dx_dphi - dx_dtheta * dz_dphi
    nz = dx_dtheta * dy_dphi - dy_dtheta * dx_dphi
    normals = torch.stack((nx, ny, nz), axis=-1)
    norm = torch.linalg.norm(normals, dim=2, keepdim=True)
    normals = normals / norm
    return normals


def square_grid(n):
    """Create a square latent grid on [0,1]²."""
    x = torch.linspace(0, 1, n)
    y = torch.linspace(0, 1, n)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    return torch.stack([grid_x, grid_y]).permute(1, 2, 0)


# ============================================================
# Default model configs
# ============================================================

def get_default_spherical_config(in_channels=9, out_channels=1):
    """Default config for the spherical/toroidal 2D FNO branch."""
    return dict(
        n_modes=(32, 32),
        hidden_channels=120,
        in_channels=in_channels,
        out_channels=out_channels,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        norm='group_norm',
        use_mlp=True,
        mlp={'expansion': 1.0, 'dropout': 0},
        domain_padding=0.125,
        factorization='tucker',
        rank=0.4,
    )


def get_default_volumetric_config(in_channels=7, out_channels=3):
    """Default config for the volumetric 3D FNO branch."""
    return dict(
        n_modes=(16, 16, 16),
        hidden_channels=64,
        in_channels=in_channels,
        out_channels=out_channels,
        lifting_channels=128,
        projection_channels=128,
        n_layers=4,
        norm='group_norm',
        use_mlp=True,
        mlp={'expansion': 1.0, 'dropout': 0},
        domain_padding=0.125,
        factorization='tucker',
        rank=0.4,
    )


# ============================================================
# Data loaders
# ============================================================

def load_shapenet_data(args):
    """Load ShapeNet dataset (torus OT data) — existing OTNO format."""
    expand_factor = 2.0
    n_t = 3586
    n_s_sqrt = int(np.sqrt(expand_factor) * np.ceil(np.sqrt(n_t)))
    R, r = 1.5, 1.0
    n_train, n_test = 500, 111

    if args.shapenet_data_path:
        data_path = Path(args.shapenet_data_path)
    elif args.data_root:
        data_path = Path(args.data_root) / "shapenet" / f"torus_OTmean_geomloss_expand{expand_factor}.pt"
    else:
        raise ValueError(
            "ShapeNet data path not set. Provide --shapenet_data_path or --data_root "
            "(or env TOPOS_SHAPENET_DATA / TOPOS_DATA_ROOT)."
        )

    data = torch.load(str(data_path))
    print(f"Loaded: {data_path}")

    train_transports = data['transports'][:n_train]
    train_normals = data['normals'][:n_train]
    train_pressures = data['pressures'][:n_train]
    train_points = data['points'][:n_train]
    train_indices_encoder = data['indices_encoder'][:n_train]
    train_indices_decoder = data['indices_decoder'][:n_train]

    pressure_encoder = UnitGaussianNormalizer(train_pressures, reduce_dim=[0, 1])
    transport_encoder = UnitGaussianNormalizer(train_transports, reduce_dim=[0, 2, 3])

    train_pressures = pressure_encoder.encode(train_pressures)
    train_transports = transport_encoder.encode(train_transports)

    test_transports = transport_encoder.encode(data['transports'][n_train:])
    test_normals = data['normals'][n_train:]
    test_pressures = data['pressures'][n_train:]
    test_points = data['points'][n_train:]
    test_indices_encoder = data['indices_encoder'][n_train:]
    test_indices_decoder = data['indices_decoder'][n_train:]

    train_dict = {
        'transports': train_transports, 'pressures': train_pressures,
        'points': train_points, 'normals': train_normals,
        'indices_encoder': train_indices_encoder,
        'indices_decoder': train_indices_decoder,
    }
    test_dict = {
        'transports': test_transports, 'pressures': test_pressures,
        'points': test_points, 'normals': test_normals,
        'indices_encoder': test_indices_encoder,
        'indices_decoder': test_indices_decoder,
    }

    pos_embed = create_torus_grid(n_s_sqrt, R, r)
    torus_normals = compute_torus_normals(n_s_sqrt, R, r)

    train_dataset = DictDatasetWithConstant(train_dict, {'pos': pos_embed, 'nor': torus_normals})
    test_dataset = DictDatasetWithConstant(test_dict, {'pos': pos_embed, 'nor': torus_normals})

    return (train_dataset, test_dataset, n_train, n_test,
            n_s_sqrt, pressure_encoder, transport_encoder)


def load_flowbench_data(args):
    """Load FlowBench dataset — existing OTNO format."""
    n_train, n_test = 800, 200
    resolution = args.resolution
    group_name = args.group_name
    expand_factor = args.expand_factor
    latent_res = int(np.sqrt(resolution * resolution * expand_factor))

    if args.flowbench_root:
        flowbench_root = Path(args.flowbench_root)
    elif args.data_root:
        flowbench_root = Path(args.data_root) / "flowbench" / "LDC_NS_2D"
    else:
        raise ValueError(
            "FlowBench root not set. Provide --flowbench_root or --data_root "
            "(or env TOPOS_FLOWBENCH_ROOT / TOPOS_DATA_ROOT)."
        )

    data_path = (
        flowbench_root
        / f"{resolution}x{resolution}"
        / "ot-data"
        / f"LDC_NS_2D_boundary_{resolution}_{group_name}_{args.latent_shape}_expand{expand_factor}_reg1e-6_combined.pt"
    )
    data = torch.load(str(data_path))
    print(f"Loaded: {data_path}")

    train_inputs = data['inputs'][:n_train]
    train_outputs = data['outs'][:n_train]
    train_indices_decoder = data['indices_decoder'][:n_train]

    cat_train_outputs = torch.cat([train_outputs[i] for i in range(n_train)], dim=0)
    cat_train_inputs = torch.cat([train_inputs[i].reshape(-1, 7) for i in range(n_train)], dim=0)
    output_encoder = UnitGaussianNormalizer(cat_train_outputs)
    input_encoder = UnitGaussianNormalizer(cat_train_inputs)

    train_outputs = [output_encoder.encode(train_outputs[i]) for i in range(n_train)]
    train_inputs = [
        input_encoder.encode(train_inputs[i].reshape(-1, 7)).reshape(
            train_inputs[i].shape[0], train_inputs[i].shape[1], 7
        ) for i in range(n_train)
    ]

    test_inputs = data['inputs'][n_train:]
    test_outputs = data['outs'][n_train:]
    test_indices_decoder = data['indices_decoder'][n_train:]
    test_inputs = [
        input_encoder.encode(test_inputs[i].reshape(-1, 7)).reshape(
            test_inputs[i].shape[0], test_inputs[i].shape[1], 7
        ) for i in range(n_test)
    ]

    train_dict = {'inputs': train_inputs, 'outputs': train_outputs, 'indices_decoder': train_indices_decoder}
    test_dict = {'inputs': test_inputs, 'outputs': test_outputs, 'indices_decoder': test_indices_decoder}

    train_dataset = DictDataset(train_dict)
    test_dataset = DictDataset(test_dict)

    return train_dataset, test_dataset, n_train, n_test, output_encoder, input_encoder


# ============================================================
# Training loop
# ============================================================

def train_shapenet(args):
    """Training loop for ShapeNet (surface pressure prediction)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    (train_dataset, test_dataset, n_train, n_test,
     n_s_sqrt, pressure_encoder, transport_encoder) = load_shapenet_data(args)

    pressure_encoder.to(device)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Build TOPOS model
    spherical_config = get_default_spherical_config(in_channels=9, out_channels=1)
    volumetric_config = get_default_volumetric_config() if args.topology in ('auto', 'volumetric') else None

    model = TOPOS(
        spherical_config=spherical_config,
        toroidal_config=copy.deepcopy(spherical_config) if args.topology in ('auto', 'toroidal') else None,
        volumetric_config=volumetric_config,
        default_topology=args.topology,
    )
    print(f"TOPOS model:\n{model}")
    print(f"Parameters: {count_model_params(model)}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    data_loss = LpLoss(size_average=False)
    loss_fn = CombinedLoss(data_loss, lambda_pde=args.lambda_pde)

    train_losses, test_losses, epoch_times = [], [], []

    for ep in range(args.epochs):
        t = default_timer()
        train_l2, test_l2 = 0.0, 0.0
        model.train()

        for batch_data in train_loader:
            optimizer.zero_grad()

            transports = batch_data['transports'].to(device)
            pressures = batch_data['pressures'].to(device)
            normals = batch_data['normals'][0].to(device)
            indices_encoder = batch_data['indices_encoder'][0].to(dtype=torch.long, device=device)
            indices_decoder = batch_data['indices_decoder'][0].to(dtype=torch.long, device=device)

            # Compute normal features
            normals = normals[indices_encoder]
            torus_normals = batch_data['nor'].reshape(-1, 3).to(device)
            normal_features = torch.cross(normals, torus_normals, dim=1).reshape(
                n_s_sqrt, n_s_sqrt, 3
            ).permute(2, 0, 1).unsqueeze(0)

            # Concatenate input features
            transports = torch.cat(
                (transports, batch_data['pos'].permute(0, 3, 1, 2).to(device), normal_features),
                dim=1,
            )

            # Forward through TOPOS
            topology, chi = resolve_routing_for_batch(
                args,
                batch_data,
                default_chi=2.0,
                default_topology="spherical",
            )
            out = model(transports, indices_decoder, topology=topology, chi=chi)

            loss = loss_fn(out, pressures)
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            for batch_data in test_loader:
                transports = batch_data['transports'].to(device)
                pressures = batch_data['pressures'].to(device)
                normals = batch_data['normals'][0].to(device)
                indices_encoder = batch_data['indices_encoder'][0].to(dtype=torch.long, device=device)
                indices_decoder = batch_data['indices_decoder'][0].to(dtype=torch.long, device=device)

                normals = normals[indices_encoder]
                torus_normals = batch_data['nor'].reshape(-1, 3).to(device)
                normal_features = torch.cross(normals, torus_normals, dim=1).reshape(
                    n_s_sqrt, n_s_sqrt, 3
                ).permute(2, 0, 1).unsqueeze(0)

                transports = torch.cat(
                    (transports, batch_data['pos'].permute(0, 3, 1, 2).to(device), normal_features),
                    dim=1,
                )

                topology, chi = resolve_routing_for_batch(
                    args,
                    batch_data,
                    default_chi=2.0,
                    default_topology="spherical",
                )
                out = model(transports, indices_decoder, topology=topology, chi=chi)
                out = pressure_encoder.decode(out)

                test_l2 += data_loss(out, pressures).item()

        train_l2 /= n_train
        test_l2 /= n_test

        epoch_time = default_timer() - t
        epoch_times.append(epoch_time)
        train_losses.append(train_l2)
        test_losses.append(test_l2)

        print(f"Epoch {ep+1}/{args.epochs}, Time: {epoch_time:.2f}s, "
              f"Train: {train_l2:.4f}, Test: {test_l2:.4f}")

    # Save results
    save_results(args, model, train_losses, test_losses, epoch_times)
    print(f"Total time: {sum(epoch_times):.2f}s")


def train_flowbench(args):
    """Training loop for FlowBench (2D fluid fields)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    (train_dataset, test_dataset, n_train, n_test,
     output_encoder, input_encoder) = load_flowbench_data(args)

    output_encoder.to(device)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Build TOPOS model
    spherical_config = get_default_spherical_config(in_channels=7, out_channels=3)
    spherical_config['hidden_channels'] = 64  # Flowbench uses smaller model
    volumetric_config = get_default_volumetric_config() if args.topology in ('auto', 'volumetric') else None

    model = TOPOS(
        spherical_config=spherical_config,
        toroidal_config=copy.deepcopy(spherical_config) if args.topology in ('auto', 'toroidal') else None,
        volumetric_config=volumetric_config,
        default_topology=args.topology,
    )
    print(f"TOPOS model:\n{model}")
    print(f"Parameters: {count_model_params(model)}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    data_loss = LpLoss(d=3, size_average=False)
    loss_fn = CombinedLoss(data_loss, lambda_pde=args.lambda_pde)

    train_losses, test_losses, epoch_times = [], [], []

    for ep in range(args.epochs):
        t1 = default_timer()
        train_l2, test_l2 = 0.0, 0.0
        model.train()

        for batch_data in train_loader:
            optimizer.zero_grad()

            inp = batch_data['inputs'].to(device)
            output = batch_data['outputs'].to(device)
            indices_decoder = batch_data['indices_decoder'][0].to(dtype=torch.long, device=device)

            topology, chi = resolve_routing_for_batch(
                args,
                batch_data,
                default_chi=0.0,
                default_topology="toroidal",
            )
            predict = model(
                inp.permute(0, 3, 1, 2).to(dtype=torch.float32, device=device),
                indices_decoder,
                topology=topology,
                chi=chi,
            )

            loss = loss_fn(output, predict)
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            for batch_data in test_loader:
                inp = batch_data['inputs'].to(device)
                output = batch_data['outputs'][0].to(device)
                indices_decoder = batch_data['indices_decoder'][0].to(dtype=torch.long, device=device)

                topology, chi = resolve_routing_for_batch(
                    args,
                    batch_data,
                    default_chi=0.0,
                    default_topology="toroidal",
                )
                predict = model(
                    inp.permute(0, 3, 1, 2).to(dtype=torch.float32, device=device),
                    indices_decoder,
                    topology=topology,
                    chi=chi,
                )
                predict = output_encoder.decode(predict)
                loss = data_loss(output.unsqueeze(0), predict)
                test_l2 += loss.item()

        train_l2 /= n_train
        test_l2 /= n_test

        epoch_time = default_timer() - t1
        epoch_times.append(epoch_time)
        train_losses.append(train_l2)
        test_losses.append(test_l2)

        print(f"Epoch {ep+1}/{args.epochs}, Time: {epoch_time:.4f}s, "
              f"Train: {train_l2:.4f}, Test: {test_l2:.4f}")

    save_results(args, model, train_losses, test_losses, epoch_times)
    print(f"Total Training Time: {sum(epoch_times):.4f}s")


def save_results(args, model, train_losses, test_losses, epoch_times):
    """Save training results to CSV and model weights to .pt."""
    os.makedirs('results', exist_ok=True)
    file_name = f'results/topos_{args.dataset}_{args.topology}.csv'
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Epoch Time', 'Total Time'])
        for ep in range(len(train_losses)):
            writer.writerow([
                ep + 1, train_losses[ep], test_losses[ep],
                epoch_times[ep], sum(epoch_times[:ep + 1])
            ])
    print(f"Results saved to {file_name}")

    model_path = f'results/topos_{args.dataset}_{args.topology}_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TOPOS Training Script")

    # Core
    parser.add_argument('--dataset', type=str, required=True, choices=['shapenet', 'flowbench'],
                        help="Dataset to train on.")
    parser.add_argument('--topology', type=str, default='spherical',
                        choices=['spherical', 'toroidal', 'volumetric', 'auto'],
                        help="Topology routing mode.")
    parser.add_argument('--epochs', type=int, default=151, help="Number of epochs.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay.")

    # Loss
    parser.add_argument('--loss', type=str, default='l2', choices=['l2', 'combined'],
                        help="Loss function type.")
    parser.add_argument('--lambda_pde', type=float, default=0.0,
                        help="PDE residual loss weight (0 = pure supervised).")

    # Flowbench-specific
    parser.add_argument('--resolution', type=int, default=128, choices=[128, 256, 512],
                        help="Resolution (flowbench only).")
    parser.add_argument('--group_name', type=str, default='nurbs',
                        choices=['nurbs', 'harmonics', 'skelneton'],
                        help="Group name (flowbench only).")
    parser.add_argument('--latent_shape', type=str, default='square',
                        choices=['square', 'ring'],
                        help="Latent shape (flowbench only).")
    parser.add_argument('--expand_factor', type=int, default=2, choices=[1, 2, 3, 4],
                        help="Expand factor (flowbench only).")

    # Data locations
    parser.add_argument(
        '--data_root',
        type=str,
        default=os.environ.get('TOPOS_DATA_ROOT'),
        help="Root directory containing dataset families (env: TOPOS_DATA_ROOT).",
    )
    parser.add_argument(
        '--shapenet_data_path',
        type=str,
        default=os.environ.get('TOPOS_SHAPENET_DATA'),
        help="Absolute/relative path to ShapeNet OT tensor file (env: TOPOS_SHAPENET_DATA).",
    )
    parser.add_argument(
        '--flowbench_root',
        type=str,
        default=os.environ.get('TOPOS_FLOWBENCH_ROOT'),
        help="Root of FlowBench LDC_NS_2D directory (env: TOPOS_FLOWBENCH_ROOT).",
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"TOPOS Training — dataset={args.dataset}, topology={args.topology}")

    if args.dataset == 'shapenet':
        train_shapenet(args)
    elif args.dataset == 'flowbench':
        train_flowbench(args)
