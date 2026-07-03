import os

import numpy as np
import torch
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from .synthetic_mixed_geometry import SYNTH_SCHEMA_VERSION, SyntheticGeometryDatasetTOPOS


def rescale_per_sample(points):
    pmin = points.min(dim=0, keepdim=True).values
    pmax = points.max(dim=0, keepdim=True).values
    prange = torch.where((pmax - pmin) == 0, torch.ones_like(pmax - pmin), pmax - pmin)
    return 2.0 * (points - pmin) / prange - 1.0


def make_regular_grid(grid_size):
    depth, height, width = tuple(grid_size)
    z = torch.linspace(-1.0, 1.0, steps=depth, dtype=torch.float32)
    y = torch.linspace(-1.0, 1.0, steps=height, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, steps=width, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1)


def estimate_point_normals(points_np, k_neighbors=16):
    tree = cKDTree(points_np)
    _, nn_idx = tree.query(points_np, k=min(k_neighbors + 1, points_np.shape[0]))
    centroid = points_np.mean(axis=0, keepdims=True)
    normals = np.zeros_like(points_np, dtype=np.float32)

    for i in range(points_np.shape[0]):
        nbrs = points_np[nn_idx[i, 1:]]
        centered = nbrs - nbrs.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(centered.shape[0], 1)
        _, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0].astype(np.float32)
        outward = points_np[i] - centroid[0]
        if np.dot(normal, outward) < 0:
            normal = -normal
        normals[i] = normal

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.clip(norms, a_min=1e-6, a_max=None)


def compute_signed_distance_features(surface_points, grid_points):
    surface_np = surface_points.detach().cpu().numpy().astype(np.float32)
    grid_np = grid_points.detach().cpu().numpy().astype(np.float32)

    surface_normals = estimate_point_normals(surface_np)
    tree = cKDTree(surface_np)
    distances, nn_idx = tree.query(grid_np, k=1)

    nearest_points = surface_np[nn_idx]
    nearest_normals = surface_normals[nn_idx]
    offsets = grid_np - nearest_points
    signs = np.sign(np.sum(offsets * nearest_normals, axis=1, keepdims=True))
    signs[signs == 0] = 1.0
    sdf = distances.reshape(-1, 1).astype(np.float32) * signs.astype(np.float32)
    return torch.from_numpy(sdf)


def interpolate_field_to_regular_grid(points, values, grid_points):
    points_np = points.detach().cpu().numpy().astype(np.float32)
    values_np = values.detach().cpu().numpy().astype(np.float32)
    grid_np = grid_points.detach().cpu().numpy().astype(np.float32)

    linear = griddata(points_np, values_np, grid_np, method="linear", fill_value=np.nan)
    nearest = griddata(points_np, values_np, grid_np, method="nearest")
    if np.isscalar(linear):
        linear = np.full_like(nearest, fill_value=linear, dtype=np.float32)
    filled = np.where(np.isnan(linear), nearest, linear).astype(np.float32)
    return torch.from_numpy(filled)


def subsample_branch_points(points, num_branch_points):
    if points.shape[0] == num_branch_points:
        return points
    if points.shape[0] > num_branch_points:
        idx = torch.linspace(0, points.shape[0] - 1, steps=num_branch_points).round().long()
        return points[idx]
    pad_idx = torch.arange(num_branch_points - points.shape[0]) % points.shape[0]
    return torch.cat([points, points[pad_idx]], dim=0)


class SyntheticGeometryDatasetGINO(Dataset):
    def __init__(
        self,
        cache_dir=None,
        n_train=500,
        n_test=111,
        split="train",
        expand_factor=2.0,
        num_points=3586,
        base_seed=42,
        latent_grid_size=(16, 16, 16),
    ):
        self.base = SyntheticGeometryDatasetTOPOS(
            cache_dir=cache_dir,
            n_train=n_train,
            n_test=n_test,
            split=split,
            expand_factor=expand_factor,
            num_points=num_points,
            base_seed=base_seed,
        )
        self.latent_queries = make_regular_grid(tuple(latent_grid_size))
        self.flat_input_grid = self.latent_queries.reshape(-1, self.latent_queries.shape[-1]).clone()
        self.cache_dir = None
        if cache_dir:
            self.cache_dir = os.path.join(cache_dir, "gino_sdf")
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"synthetic_gino_{self.base.split}_{idx:03d}.pt")
            if os.path.exists(cache_path):
                cached = torch.load(cache_path, weights_only=False)
                required = {"input_geom", "features", "output_queries", "latent_queries", "pressure", "source_topology", "schema_version"}
                if required.issubset(cached.keys()) and cached.get("schema_version") == SYNTH_SCHEMA_VERSION:
                    return cached

        sample = self.base[idx]
        points_scaled = rescale_per_sample(sample["points"].float())
        sdf = compute_signed_distance_features(points_scaled, self.flat_input_grid)
        processed = {
            "input_geom": self.flat_input_grid.clone(),
            "features": sdf,
            "output_queries": points_scaled,
            "latent_queries": self.latent_queries.clone(),
            "pressure": sample["pressure"].float(),
            "source_topology": sample["topology"],
            "schema_version": sample["schema_version"],
        }
        if self.cache_dir:
            torch.save(processed, cache_path)
        return processed


class SyntheticGeometryDatasetFNO(Dataset):
    def __init__(
        self,
        cache_dir=None,
        n_train=500,
        n_test=111,
        split="train",
        expand_factor=2.0,
        num_points=3586,
        base_seed=42,
        grid_size=(16, 16, 16),
    ):
        self.base = SyntheticGeometryDatasetTOPOS(
            cache_dir=cache_dir,
            n_train=n_train,
            n_test=n_test,
            split=split,
            expand_factor=expand_factor,
            num_points=num_points,
            base_seed=base_seed,
        )
        self.grid = make_regular_grid(tuple(grid_size))
        self.flat_grid = self.grid.reshape(-1, self.grid.shape[-1]).clone()
        self.cache_dir = None
        if cache_dir:
            self.cache_dir = os.path.join(cache_dir, "fno_mask")
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"synthetic_fno_{self.base.split}_{idx:03d}.pt")
            if os.path.exists(cache_path):
                cached = torch.load(cache_path, weights_only=False)
                required = {"mask", "target_grid", "output_queries", "pressure", "source_topology", "schema_version"}
                if required.issubset(cached.keys()) and cached.get("schema_version") == SYNTH_SCHEMA_VERSION:
                    return cached

        sample = self.base[idx]
        points_scaled = rescale_per_sample(sample["points"].float())
        sdf = compute_signed_distance_features(points_scaled, self.flat_grid)
        mask = (sdf <= 0).float().reshape(*self.grid.shape[:-1], 1).permute(3, 0, 1, 2).contiguous()
        target_grid = interpolate_field_to_regular_grid(points_scaled, sample["pressure"].float(), self.flat_grid)
        target_grid = target_grid.reshape(*self.grid.shape[:-1]).unsqueeze(0).contiguous()
        target_grid = target_grid * mask
        processed = {
            "mask": mask,
            "target_grid": target_grid,
            "output_queries": points_scaled,
            "pressure": sample["pressure"].float(),
            "source_topology": sample["topology"],
            "schema_version": sample["schema_version"],
        }
        if self.cache_dir:
            torch.save(processed, cache_path)
        return processed


class SyntheticGeometryDatasetDeepONet(Dataset):
    def __init__(
        self,
        cache_dir=None,
        n_train=500,
        n_test=111,
        split="train",
        expand_factor=2.0,
        num_points=3586,
        base_seed=42,
        branch_points=256,
    ):
        self.base = SyntheticGeometryDatasetTOPOS(
            cache_dir=cache_dir,
            n_train=n_train,
            n_test=n_test,
            split=split,
            expand_factor=expand_factor,
            num_points=num_points,
            base_seed=base_seed,
        )
        self.branch_points = branch_points
        self.cache_dir = None
        if cache_dir:
            self.cache_dir = os.path.join(cache_dir, "deeponet_points")
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"synthetic_deeponet_{self.base.split}_{idx:03d}.pt")
            if os.path.exists(cache_path):
                cached = torch.load(cache_path, weights_only=False)
                required = {"branch_input", "trunk_input", "pressure", "source_topology", "schema_version"}
                if required.issubset(cached.keys()) and cached.get("schema_version") == SYNTH_SCHEMA_VERSION:
                    return cached

        sample = self.base[idx]
        points_scaled = rescale_per_sample(sample["points"].float())
        branch_points = subsample_branch_points(points_scaled, self.branch_points)
        processed = {
            "branch_input": branch_points.reshape(-1).contiguous(),
            "trunk_input": points_scaled,
            "pressure": sample["pressure"].float(),
            "source_topology": sample["topology"],
            "schema_version": sample["schema_version"],
        }
        if self.cache_dir:
            torch.save(processed, cache_path)
        return processed
