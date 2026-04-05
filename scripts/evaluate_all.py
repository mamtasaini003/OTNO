"""
Unified evaluation harness for TOPOS paper.

Evaluates GINO, OTNO, and TOPOS on both the Custom Mixed-Genus dataset
and the Thingi10K dataset, and serialises per-sample results to
``results/eval/<model>_<dataset>.pt`` for downstream plotting scripts.

Usage
-----
    python scripts/evaluate_all.py --config configs/mixed_genus_fair_comparison.yaml \
        --models gino otno topos --datasets mixed_genus --gpus 0

    python scripts/evaluate_all.py --config configs/thingi10k_topos.yaml \
        --models gino otno topos --datasets thingi10k --gpus 0
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topos.data.synthetic_mixed_geometry import (
    CASE_LIBRARY,
    SyntheticGeometryDatasetOTNO,
    SyntheticGeometryDatasetTOPOS,
    compute_torus_normals,
)
from topos.data.mixed_geometry_baselines import SyntheticGeometryDatasetGINO
from topos.models import TOPOS
from topos.models.fno_spherical import ToroidalTransportFNO
from topos.router.topology_check import compute_genus
from topos.utils import LpLoss, UnitGaussianNormalizer, count_model_params, resolve_device

try:
    from topos.data.thingi10k_geometry import (
        Thingi10KToposDataset,
        Thingi10KOtnoDataset,
        Thingi10KGinoDataset,
    )
    HAS_THINGI = True
except Exception:
    HAS_THINGI = False


# ── helpers ──────────────────────────────────────────────────────────

def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _build_transports_topos(batch, device):
    """Build 9-channel transport tensor for TOPOS / OTNO from a batch dict."""
    topology = batch["topology"][0] if "topology" in batch else batch.get("source_topology", ["unknown"])[0]
    if topology == "graph":
        return None, topology

    points = batch["points"][0].to(device)
    normals = batch["normals"][0].to(device)
    idx_encoder = batch["idx_encoder"][0].to(device)
    grid_width = batch["grid_width"][0].item()
    latent_coords = batch["latent_coords"][0].to(device)
    latent_normals = batch["latent_normals"][0].to(device)

    mapped_points = points[idx_encoder]
    mapped_normals = normals[idx_encoder]
    normal_cross = torch.cross(mapped_normals, latent_normals.view(-1, 3), dim=-1)

    is_vol = latent_coords.dim() == 4
    if is_vol:
        vs = (grid_width, grid_width, grid_width, 3)
        ps = (3, 0, 1, 2)
    else:
        vs = (grid_width, grid_width, 3)
        ps = (2, 0, 1)

    mapped_points = mapped_points.view(vs)
    normal_cross = normal_cross.view(vs)
    transports = torch.cat([mapped_points, latent_coords, normal_cross], dim=-1).permute(ps).unsqueeze(0)
    return transports, topology


def _build_transports_otno(batch, device):
    """Build 9-channel transport tensor for the flat OTNO baseline."""
    points = batch["points"][0].to(device)
    normals = batch["normals"][0].to(device)
    idx_encoder = batch["idx_encoder"][0].to(device)
    grid_width = batch["grid_width"][0].item()
    latent_coords = batch["latent_coords"][0].to(device)
    latent_normals = batch["latent_normals"][0].to(device)

    mapped_points = points[idx_encoder].view(grid_width, grid_width, 3)
    mapped_normals = normals[idx_encoder].view(grid_width, grid_width, 3)
    normal_cross = torch.cross(mapped_normals, latent_normals, dim=-1)
    transports = torch.cat([mapped_points, latent_coords, normal_cross], dim=-1).permute(2, 0, 1).unsqueeze(0)
    return transports


def _compute_boundary_mask(points, topology, chi, threshold=0.15):
    """Heuristic boundary mask — vertices near topological features.

    For genus-0 shapes, boundary is defined as points near the
    equatorial belt.  For higher genus or open shapes, we use the
    high-curvature radial region.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = torch.sqrt(x * x + y * y + z * z + 1e-8)

    if topology in ("toroidal",):
        # Ring region near the inner hole of a torus
        R_major = 1.5
        dist_to_ring = torch.abs(torch.sqrt(x * x + y * y) - R_major)
        mask = dist_to_ring < threshold * 2
    elif topology in ("open_surface",):
        # Boundary is literally edges — approximate as extreme z
        z_range = z.max() - z.min()
        mask = (z > z.max() - threshold * z_range) | (z < z.min() + threshold * z_range)
    elif topology in ("high_genus",):
        # High-curvature saddle regions
        curvature_proxy = torch.abs(x * y) + torch.abs(y * z)
        mask = curvature_proxy > curvature_proxy.quantile(0.75)
    else:
        # Spherical — polar caps
        mask = torch.abs(z) > (z.max() - threshold * (z.max() - z.min()))

    return mask


# ── Per-model evaluators ─────────────────────────────────────────────

def evaluate_topos_mixed(config, device):
    """Evaluate TOPOS on the mixed-genus synthetic dataset."""
    cache_dir = config["dataset"].get("cache_dir")
    if cache_dir:
        cache_dir = os.path.join(cache_dir, "synthetic")

    test_ds = SyntheticGeometryDatasetTOPOS(
        cache_dir=cache_dir,
        n_train=config["dataset"]["train_samples"],
        n_test=config["dataset"]["test_samples"],
        split="test",
        expand_factor=config["dataset"]["expand_factor"],
        num_points=config["dataset"].get("num_points", 3586),
        base_seed=config["training"].get("seed", 42),
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # Build model
    mc = config["model"]
    shared = {
        "n_modes": tuple(mc["n_modes"]),
        "hidden_channels": mc["hidden_channels"],
        "in_channels": mc["in_channels"],
        "out_channels": mc["out_channels"],
        "n_layers": mc["n_layers"],
        "use_mlp": mc["use_mlp"],
        "mlp": mc["mlp"],
        "norm": mc.get("norm"),
        "domain_padding": mc.get("domain_padding"),
        "factorization": mc.get("factorization"),
        "rank": mc.get("rank", 1.0),
    }
    vol = shared.copy()
    vol["n_modes"] = (mc["n_modes"][0] // 2, mc["n_modes"][1] // 2, 1)
    graph_cfg = {"in_channels": mc["in_channels"], "out_channels": mc["out_channels"], "hidden_channels": mc["hidden_channels"] // 3}

    model = TOPOS(spherical_config=shared, toroidal_config=shared.copy(), volumetric_config=vol, graph_config=graph_cfg)
    chk_path = os.path.join(config.get("output", {}).get("dir", "results"), "checkpoints", "topos_mixed_genus.pt")
    if os.path.exists(chk_path):
        model.load_state_dict(torch.load(chk_path, map_location="cpu", weights_only=False), strict=False)
    model = model.to(device).eval()

    # Quick normalizer
    pressures, topo_transports = [], defaultdict(list)
    for i, batch in enumerate(DataLoader(test_ds, batch_size=1, shuffle=False)):
        if i >= min(50, len(test_ds)):
            break
        tr, topo = _build_transports_topos(batch, torch.device("cpu"))
        if tr is not None:
            topo_transports[topo].append(tr)
        pressures.append(batch["pressure"][0].unsqueeze(0))
    pressure_norm = UnitGaussianNormalizer(torch.cat(pressures, 0), reduce_dim=[0]).to(device)
    transport_norms = {}
    for topo, samples in topo_transports.items():
        dims = [0, 2, 3, 4] if samples[0].dim() == 5 else [0, 2, 3]
        transport_norms[topo] = UnitGaussianNormalizer(torch.cat(samples, 0), reduce_dim=dims).to(device)

    loss_fn = LpLoss(size_average=False)
    per_sample = []
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    t_start = time.perf_counter()

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            topology = batch["topology"][0]
            latent_topo = batch.get("source_latent_topology", batch.get("topology"))[0]
            chi = float(batch["chi"][0].item())
            target = batch["pressure"][0].to(device)

            if topology == "graph":
                pts = batch["points"][0].to(device)
                nrm = batch["normals"][0].to(device)
                feat = torch.cat([pts, nrm, torch.cross(pts, nrm, dim=1)], dim=1).unsqueeze(0)
                pred = model(points=pts.unsqueeze(0), features=feat, topology=latent_topo, chi=chi)
            else:
                transports, _ = _build_transports_topos(batch, device)
                if topology in transport_norms:
                    transports = transport_norms[topology].encode(transports.clone())
                pred = model(transports=transports, idx_decoder=batch["idx_decoder"][0].to(device), topology=latent_topo, chi=chi)
                pred = pressure_norm.decode(pred.clone())

            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            abs_err = (pred_flat - target_flat).abs()
            rel_l2 = loss_fn(pred.view(1, -1), target.view(1, -1)).item()
            points_cpu = batch["points"][0].cpu()
            boundary_mask = _compute_boundary_mask(points_cpu, topology, chi)

            per_sample.append({
                "idx": idx,
                "topology": topology,
                "chi": chi,
                "genus": compute_genus(chi),
                "rel_l2": rel_l2,
                "abs_error": abs_err.cpu(),
                "pred": pred_flat.cpu(),
                "target": target_flat.cpu(),
                "points": points_cpu,
                "boundary_mask": boundary_mask,
            })

    t_elapsed = time.perf_counter() - t_start
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
    n_params = count_model_params(model)

    return {
        "model": "TOPOS",
        "dataset": "mixed_genus",
        "per_sample": per_sample,
        "timing": {"inference_time_s": t_elapsed, "peak_gpu_mb": peak_mb, "n_params": n_params},
    }


def evaluate_otno_mixed(config, device):
    """Evaluate OTNO on the mixed-genus dataset."""
    cache_dir = config["dataset"].get("cache_dir")
    if cache_dir:
        cache_dir = os.path.join(cache_dir, "synthetic")

    test_ds = SyntheticGeometryDatasetOTNO(
        cache_dir=cache_dir,
        n_train=config["dataset"]["train_samples"],
        n_test=config["dataset"]["test_samples"],
        split="test",
        expand_factor=config["dataset"]["expand_factor"],
        num_points=config["dataset"].get("num_points", 3586),
        base_seed=config["training"].get("seed", 42),
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    mc = config["model"]
    model = ToroidalTransportFNO(
        n_modes=tuple(mc["n_modes"]),
        hidden_channels=mc["hidden_channels"],
        in_channels=9,
        out_channels=mc["out_channels"],
        n_layers=mc["n_layers"],
        use_mlp=mc["use_mlp"],
        mlp=mc["mlp"],
        norm=mc.get("norm"),
        domain_padding=mc.get("domain_padding"),
        factorization=mc.get("factorization", "tucker"),
        rank=mc.get("rank", 1.0),
    )
    chk_path = os.path.join(config.get("output", {}).get("dir", "results"), "checkpoints", "otno_mixed_genus.pt")
    if os.path.exists(chk_path):
        model.load_state_dict(torch.load(chk_path, map_location="cpu", weights_only=False), strict=False)
    model = model.to(device).eval()

    # Normalizers
    pressures, transports_list = [], []
    for i, batch in enumerate(DataLoader(test_ds, batch_size=1, shuffle=False)):
        if i >= 50:
            break
        tr = _build_transports_otno(batch, torch.device("cpu"))
        transports_list.append(tr)
        pressures.append(batch["pressure"][0])
    pressure_norm = UnitGaussianNormalizer(torch.cat(pressures, 0), reduce_dim=[0]).to(device)
    transport_norm = UnitGaussianNormalizer(torch.cat(transports_list, 0), reduce_dim=[0, 2, 3]).to(device)

    loss_fn = LpLoss(size_average=False)
    per_sample = []
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    t_start = time.perf_counter()

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            target = batch["pressure"][0].to(device)
            topology = batch.get("source_topology", ["unknown"])[0]
            chi = float(batch.get("source_chi", [2.0])[0])

            transports = _build_transports_otno(batch, device)
            transports = transport_norm.encode(transports.clone())
            idx_decoder = batch["idx_decoder"][0].to(device)
            pred = model(transports, idx_decoder)
            pred = pressure_norm.decode(pred.clone())

            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            abs_err = (pred_flat - target_flat).abs()
            rel_l2 = loss_fn(pred.view(1, -1), target.view(1, -1)).item()
            points_cpu = batch["points"][0].cpu()
            boundary_mask = _compute_boundary_mask(points_cpu, topology, chi)

            per_sample.append({
                "idx": idx,
                "topology": topology,
                "chi": chi,
                "genus": compute_genus(chi),
                "rel_l2": rel_l2,
                "abs_error": abs_err.cpu(),
                "pred": pred_flat.cpu(),
                "target": target_flat.cpu(),
                "points": points_cpu,
                "boundary_mask": boundary_mask,
            })

    t_elapsed = time.perf_counter() - t_start
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
    n_params = count_model_params(model)

    return {
        "model": "OTNO",
        "dataset": "mixed_genus",
        "per_sample": per_sample,
        "timing": {"inference_time_s": t_elapsed, "peak_gpu_mb": peak_mb, "n_params": n_params},
    }


def evaluate_gino_mixed(config, device):
    """Evaluate GINO on the mixed-genus dataset."""
    from neuralop.models import GINO

    cache_dir = config["dataset"].get("cache_dir")
    if cache_dir:
        cache_dir = os.path.join(cache_dir, "synthetic")

    gc = config.get("gino_official", {})
    latent_grid_size = tuple(gc.get("latent_grid_size", [16, 16, 16]))

    test_ds = SyntheticGeometryDatasetGINO(
        cache_dir=cache_dir,
        n_train=config["dataset"]["train_samples"],
        n_test=config["dataset"]["test_samples"],
        split="test",
        expand_factor=config["dataset"]["expand_factor"],
        num_points=config["dataset"].get("num_points", 3586),
        base_seed=config["training"].get("seed", 42),
        latent_grid_size=latent_grid_size,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    mc = config["model"]
    n_modes = tuple(gc.get("n_modes", [16, 16, 16]))
    hc = gc.get("hidden_channels", mc["hidden_channels"])
    gino_in = gc.get("in_channels", 1)

    model = GINO(
        in_channels=gino_in,
        out_channels=mc["out_channels"],
        latent_feature_channels=gc.get("latent_feature_channels"),
        gno_coord_dim=3,
        in_gno_radius=gc.get("in_gno_radius", 0.2),
        out_gno_radius=gc.get("out_gno_radius", 0.2),
        in_gno_transform_type=gc.get("in_gno_transform_type", "linear"),
        out_gno_transform_type=gc.get("out_gno_transform_type", "linear"),
        in_gno_pos_embed_type=gc.get("in_gno_pos_embed_type", "transformer"),
        out_gno_pos_embed_type=gc.get("out_gno_pos_embed_type", "transformer"),
        fno_in_channels=gino_in,
        fno_n_modes=n_modes,
        fno_hidden_channels=hc,
        fno_n_layers=gc.get("n_layers", mc["n_layers"]),
        gno_embed_channels=gc.get("gno_embed_channels", 32),
        in_gno_channel_mlp_hidden_layers=gc.get("in_gno_channel_mlp_hidden_layers", [80, 80, 80]),
        out_gno_channel_mlp_hidden_layers=gc.get("out_gno_channel_mlp_hidden_layers", [256, 256]),
        gno_use_open3d=False,
        gno_use_torch_scatter=False,
        fno_factorization=gc.get("factorization", mc.get("factorization")),
        fno_rank=gc.get("rank", mc.get("rank", 1.0)),
        fno_norm=gc.get("norm", mc.get("norm")),
        fno_use_channel_mlp=gc.get("use_mlp", mc.get("use_mlp", True)),
        fno_channel_mlp_expansion=gc.get("mlp_expansion", mc.get("mlp", {}).get("expansion", 0.5)),
        fno_channel_mlp_dropout=gc.get("mlp_dropout", mc.get("mlp", {}).get("dropout", 0.0)),
    )
    chk_path = os.path.join(config.get("output", {}).get("dir", "results"), "checkpoints", "gino_mixed_genus.pt")
    if os.path.exists(chk_path):
        model.load_state_dict(torch.load(chk_path, map_location="cpu", weights_only=False), strict=False)
    model = model.to(device).eval()

    # Normalizers
    pressures, features = [], []
    for i, batch in enumerate(DataLoader(test_ds, batch_size=1, shuffle=False)):
        if i >= 50:
            break
        pressures.append(batch["pressure"])
        features.append(batch["features"])
    pressure_norm = UnitGaussianNormalizer(torch.cat(pressures, 0), reduce_dim=[0, 1]).to(device)
    feat_norm = UnitGaussianNormalizer(torch.cat(features, 0), reduce_dim=[0, 1]).to(device)

    loss_fn = LpLoss(size_average=False)
    per_sample = []
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    t_start = time.perf_counter()

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            target = batch["pressure"].to(device)
            topology = batch.get("source_topology", ["unknown"])[0]
            chi = float(batch.get("source_chi", [2.0])[0])

            x = feat_norm.encode(batch["features"].to(device).clone())
            pred = model(
                input_geom=batch["input_geom"].to(device),
                latent_queries=batch["latent_queries"].to(device),
                output_queries=batch["output_queries"].to(device),
                x=x,
            )
            pred = pressure_norm.decode(pred.squeeze(-1).clone())
            target_flat = target.view(-1)
            pred_flat = pred.view(-1)
            abs_err = (pred_flat - target_flat).abs()
            rel_l2 = loss_fn(pred.view(1, -1), target.view(1, -1)).item()
            points_cpu = batch["output_queries"][0].cpu()
            boundary_mask = _compute_boundary_mask(points_cpu, topology, chi)

            per_sample.append({
                "idx": idx,
                "topology": topology,
                "chi": chi,
                "genus": compute_genus(chi),
                "rel_l2": rel_l2,
                "abs_error": abs_err.cpu(),
                "pred": pred_flat.cpu(),
                "target": target_flat.cpu(),
                "points": points_cpu,
                "boundary_mask": boundary_mask,
            })

    t_elapsed = time.perf_counter() - t_start
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
    n_params = count_model_params(model)

    return {
        "model": "GINO",
        "dataset": "mixed_genus",
        "per_sample": per_sample,
        "timing": {"inference_time_s": t_elapsed, "peak_gpu_mb": peak_mb, "n_params": n_params},
    }


# ── Thingi10K evaluators ─────────────────────────────────────────────

def evaluate_topos_thingi(config, device):
    """Evaluate TOPOS on Thingi10K."""
    if not HAS_THINGI:
        print("[!] Thingi10K dataset not available; skipping TOPOS-Thingi10K evaluation.")
        return None

    dc = config["dataset"]
    cache_dir = dc.get("cache_dir")
    if cache_dir:
        cache_dir = os.path.join(cache_dir, "thingi10k")

    test_ds = Thingi10KToposDataset(
        split="test",
        cache_dir=cache_dir,
        train_samples=dc["train_samples"],
        test_samples=dc["test_samples"],
        split_offset=dc.get("split_offset", 0),
        variant=dc.get("variant", "npz"),
        source_cache_dir=dc.get("source_cache_dir"),
        num_points=dc.get("num_points", 3586),
        seed=config["training"].get("seed", 42),
        thingi_filters=dc.get("filters", {}),
        expand_factor=dc.get("expand_factor", 2.0),
        width_2d=dc.get("resolution_2d", 84),
        width_3d=dc.get("resolution_3d", 16),
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    mc = config["model"]
    shared = {
        "n_modes": tuple(mc["n_modes"]),
        "hidden_channels": mc["hidden_channels"],
        "in_channels": mc.get("in_channels", 9),
        "out_channels": mc.get("out_channels", 1),
        "n_layers": mc["n_layers"],
        "use_mlp": mc["use_mlp"],
        "mlp": mc["mlp"],
        "norm": mc.get("norm"),
        "domain_padding": mc.get("domain_padding"),
        "factorization": mc.get("factorization"),
        "rank": mc.get("rank", 1.0),
    }
    vol = shared.copy()
    vol["n_modes"] = tuple(mc.get("volumetric_n_modes", [8, 8, 1]))
    graph_cfg = {"in_channels": mc.get("in_channels", 9), "out_channels": mc.get("out_channels", 1), "hidden_channels": mc["hidden_channels"] // 3}

    model = TOPOS(spherical_config=shared, toroidal_config=shared.copy(), volumetric_config=vol, graph_config=graph_cfg)
    chk_path = os.path.join(config.get("output", {}).get("dir", "results"), "checkpoints", "topos_mixed_genus.pt")
    if os.path.exists(chk_path):
        model.load_state_dict(torch.load(chk_path, map_location="cpu", weights_only=False), strict=False)
    model = model.to(device).eval()

    # Quick normalizers
    pressures, topo_transports = [], defaultdict(list)
    train_ds_norm = Thingi10KToposDataset(
        split="train",
        cache_dir=cache_dir,
        train_samples=dc["train_samples"],
        test_samples=dc["test_samples"],
        split_offset=dc.get("split_offset", 0),
        variant=dc.get("variant", "npz"),
        source_cache_dir=dc.get("source_cache_dir"),
        num_points=dc.get("num_points", 3586),
        seed=config["training"].get("seed", 42),
        thingi_filters=dc.get("filters", {}),
        expand_factor=dc.get("expand_factor", 2.0),
        width_2d=dc.get("resolution_2d", 84),
        width_3d=dc.get("resolution_3d", 16),
    )
    for i, batch in enumerate(DataLoader(train_ds_norm, batch_size=1, shuffle=False)):
        if i >= min(50, len(train_ds_norm)):
            break
        tr, topo = _build_transports_topos(batch, torch.device("cpu"))
        if tr is not None:
            topo_transports[topo].append(tr)
        pressures.append(batch["pressure"][0].unsqueeze(0))
    pressure_norm = UnitGaussianNormalizer(torch.cat(pressures, 0), reduce_dim=[0, 1]).to(device)
    transport_norms = {}
    for topo, samples in topo_transports.items():
        dims = [0, 2, 3, 4] if samples[0].dim() == 5 else [0, 2, 3]
        transport_norms[topo] = UnitGaussianNormalizer(torch.cat(samples, 0), reduce_dim=dims).to(device)

    loss_fn = LpLoss(size_average=False)
    per_sample = []
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    t_start = time.perf_counter()

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            topology = batch["topology"][0]
            latent_topo = batch.get("source_latent_topology", batch.get("topology"))[0]
            chi = float(batch["chi"][0].item())
            target = batch["pressure"].to(device)

            if topology == "graph":
                pts = batch["points"].to(device)
                nrm = batch["normals"].to(device)
                feat = torch.cat([pts, nrm, torch.cross(pts, nrm, dim=-1)], dim=-1).to(device)
                pred = model(points=pts, features=feat, topology=latent_topo, chi=chi)
            else:
                transports, _ = _build_transports_topos(batch, device)
                if topology in transport_norms:
                    transports = transport_norms[topology].encode(transports.clone())
                pred = model(transports=transports, idx_decoder=batch["idx_decoder"][0].to(device), topology=latent_topo, chi=chi)
                pred = pressure_norm.decode(pred.clone())

            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            abs_err = (pred_flat - target_flat).abs()
            rel_l2 = loss_fn(pred.view(1, -1), target.view(1, -1)).item()
            points_cpu = batch["points"][0].cpu()
            boundary_mask = _compute_boundary_mask(points_cpu, topology, chi)

            per_sample.append({
                "idx": idx,
                "topology": topology,
                "chi": chi,
                "genus": compute_genus(chi),
                "rel_l2": rel_l2,
                "abs_error": abs_err.cpu(),
                "pred": pred_flat.cpu(),
                "target": target_flat.cpu(),
                "points": points_cpu,
                "boundary_mask": boundary_mask,
            })

    t_elapsed = time.perf_counter() - t_start
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
    n_params = count_model_params(model)

    return {
        "model": "TOPOS",
        "dataset": "thingi10k",
        "per_sample": per_sample,
        "timing": {"inference_time_s": t_elapsed, "peak_gpu_mb": peak_mb, "n_params": n_params},
    }


def evaluate_otno_thingi(config, device):
    """Evaluate OTNO on Thingi10K."""
    if not HAS_THINGI:
        print("[!] Thingi10K dataset not available; skipping OTNO-Thingi10K evaluation.")
        return None

    dc = config["dataset"]
    cache_dir = dc.get("cache_dir")
    if cache_dir:
        cache_dir = os.path.join(cache_dir, "thingi10k")

    test_ds = Thingi10KOtnoDataset(
        split="test",
        cache_dir=cache_dir,
        train_samples=dc["train_samples"],
        test_samples=dc["test_samples"],
        split_offset=dc.get("split_offset", 0),
        variant=dc.get("variant", "npz"),
        source_cache_dir=dc.get("source_cache_dir"),
        num_points=dc.get("num_points", 3586),
        seed=config["training"].get("seed", 42),
        thingi_filters=dc.get("filters", {}),
        expand_factor=dc.get("expand_factor", 2.0),
        width=dc.get("resolution_2d", 84),
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    mc = config["model"]
    model = ToroidalTransportFNO(
        n_modes=tuple(mc["n_modes"]),
        hidden_channels=mc["hidden_channels"],
        in_channels=mc.get("in_channels", 9),
        out_channels=mc.get("out_channels", 1),
        n_layers=mc["n_layers"],
        use_mlp=mc["use_mlp"],
        mlp=mc["mlp"],
        norm=mc.get("norm"),
        domain_padding=mc.get("domain_padding"),
        factorization=mc.get("factorization", "tucker"),
        rank=mc.get("rank", 1.0),
    )
    chk_path = os.path.join(config.get("output", {}).get("dir", "results"), "checkpoints", "otno_mixed_genus.pt")
    if os.path.exists(chk_path):
        model.load_state_dict(torch.load(chk_path, map_location="cpu", weights_only=False), strict=False)
    model = model.to(device).eval()

    # Normalizers from training split
    train_ds_norm = Thingi10KOtnoDataset(
        split="train",
        cache_dir=cache_dir,
        train_samples=dc["train_samples"],
        test_samples=dc["test_samples"],
        split_offset=dc.get("split_offset", 0),
        variant=dc.get("variant", "npz"),
        source_cache_dir=dc.get("source_cache_dir"),
        num_points=dc.get("num_points", 3586),
        seed=config["training"].get("seed", 42),
        thingi_filters=dc.get("filters", {}),
        expand_factor=dc.get("expand_factor", 2.0),
        width=dc.get("resolution_2d", 84),
    )
    pressures, transports_list = [], []
    for i, batch in enumerate(DataLoader(train_ds_norm, batch_size=1, shuffle=False)):
        if i >= 50:
            break
        tr = _build_transports_otno(batch, torch.device("cpu"))
        transports_list.append(tr)
        pressures.append(batch["pressure"][0])
    pressure_norm = UnitGaussianNormalizer(torch.cat(pressures, 0), reduce_dim=[0]).to(device)
    transport_norm = UnitGaussianNormalizer(torch.cat(transports_list, 0), reduce_dim=[0, 2, 3]).to(device)

    loss_fn = LpLoss(size_average=False)
    per_sample = []
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    t_start = time.perf_counter()

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            target = batch["pressure"][0].to(device)
            topology = batch.get("source_topology", ["unknown"])[0]
            chi = float(batch.get("chi", [2.0])[0])

            transports = _build_transports_otno(batch, device)
            transports = transport_norm.encode(transports.clone())
            pred = model(transports, batch["idx_decoder"][0].to(device))
            pred = pressure_norm.decode(pred.clone())

            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            abs_err = (pred_flat - target_flat).abs()
            rel_l2 = loss_fn(pred.view(1, -1), target.view(1, -1)).item()
            points_cpu = batch["points"][0].cpu()
            boundary_mask = _compute_boundary_mask(points_cpu, topology, chi)

            per_sample.append({
                "idx": idx,
                "topology": topology,
                "chi": chi,
                "genus": compute_genus(chi),
                "rel_l2": rel_l2,
                "abs_error": abs_err.cpu(),
                "pred": pred_flat.cpu(),
                "target": target_flat.cpu(),
                "points": points_cpu,
                "boundary_mask": boundary_mask,
            })

    t_elapsed = time.perf_counter() - t_start
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
    n_params = count_model_params(model)

    return {
        "model": "OTNO",
        "dataset": "thingi10k",
        "per_sample": per_sample,
        "timing": {"inference_time_s": t_elapsed, "peak_gpu_mb": peak_mb, "n_params": n_params},
    }


# ── main ─────────────────────────────────────────────────────────────

EVALUATORS = {
    ("topos", "mixed_genus"):  evaluate_topos_mixed,
    ("otno",  "mixed_genus"):  evaluate_otno_mixed,
    ("gino",  "mixed_genus"):  evaluate_gino_mixed,
    ("topos", "thingi10k"):    evaluate_topos_thingi,
    ("otno",  "thingi10k"):    evaluate_otno_thingi,
}


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation for TOPOS paper plots")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--models", nargs="+", default=["gino", "otno", "topos"])
    parser.add_argument("--datasets", nargs="+", default=["mixed_genus"])
    parser.add_argument("--gpus", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default="results/eval")
    args = parser.parse_args()

    config = _load_config(args.config)
    device, _ = resolve_device(args.gpus)
    os.makedirs(args.out_dir, exist_ok=True)

    for model_name in args.models:
        for dataset in args.datasets:
            key = (model_name.lower(), dataset.lower())
            if key not in EVALUATORS:
                print(f"[!] No evaluator for {key}; skipping.")
                continue
            print(f"\n{'='*60}")
            print(f"  Evaluating {model_name.upper()} on {dataset}")
            print(f"{'='*60}")
            result = EVALUATORS[key](config, device)
            if result is None:
                continue
            out_path = os.path.join(args.out_dir, f"{model_name.lower()}_{dataset.lower()}.pt")
            torch.save(result, out_path)
            n = len(result["per_sample"])
            avg_l2 = np.mean([s["rel_l2"] for s in result["per_sample"]])
            print(f"  ✓ Saved {out_path} ({n} samples, mean rel L²={avg_l2:.4f})")
            print(f"    Timing: {result['timing']['inference_time_s']:.2f}s | "
                  f"Peak GPU: {result['timing']['peak_gpu_mb']:.0f} MB | "
                  f"Params: {result['timing']['n_params']:,}")


if __name__ == "__main__":
    main()
