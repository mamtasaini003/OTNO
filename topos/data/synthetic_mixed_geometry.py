import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .ot_mapper_3d import OT3Dto2DMapper


SYNTH_SCHEMA_VERSION = 2
CASE_LIBRARY = [
    {"name": "spherical", "chi": 2.0, "latent_topology": "spherical"},
    {"name": "toroidal", "chi": 0.0, "latent_topology": "toroidal"},
    {"name": "open_surface", "chi": 1.0, "latent_topology": "volumetric"},
    {"name": "high_genus", "chi": -2.0, "latent_topology": "volumetric"},
]


def compute_torus_normals(width, R=1.5, r=1.0):
    theta = torch.linspace(0, 2 * np.pi, width + 1)[:-1]
    phi = torch.linspace(0, 2 * np.pi, width + 1)[:-1]
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")

    dx_dtheta = -r * torch.sin(theta) * torch.cos(phi)
    dy_dtheta = -r * torch.sin(theta) * torch.sin(phi)
    dz_dtheta = r * torch.cos(theta)

    dx_dphi = -(R + r * torch.cos(theta)) * torch.sin(phi)
    dy_dphi = (R + r * torch.cos(theta)) * torch.cos(phi)
    dz_dphi = torch.zeros_like(dx_dphi)

    nx = dy_dtheta * dz_dphi - dz_dtheta * dy_dphi
    ny = dz_dtheta * dx_dphi - dx_dtheta * dz_dphi
    nz = dx_dtheta * dy_dphi - dy_dtheta * dx_dphi

    normals = torch.stack((nx, ny, nz), dim=-1)
    norm = torch.linalg.norm(normals, dim=2, keepdim=True)
    return normals / norm


def sample_or_repeat_points(points, target_n, generator):
    n = points.shape[0]
    if n == target_n:
        return points
    if n > target_n:
        idx = torch.randperm(n, generator=generator, device=points.device)[:target_n]
        return points[idx]
    extra = target_n - n
    idx_extra = torch.randint(0, n, (extra,), generator=generator, device=points.device)
    return torch.cat([points, points[idx_extra]], dim=0)


def synthetic_pressure(points, topology):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    base = torch.sin(2.5 * x) + 0.7 * torch.cos(3.0 * y) + 0.35 * z
    interaction = 0.25 * x * y - 0.12 * y * z
    radial = torch.sqrt(x * x + y * y + z * z + 1e-6)

    if topology == "toroidal":
        return base + interaction + 0.15 * torch.sin(4.0 * radial)
    if topology == "spherical":
        return base + 0.20 * torch.cos(5.0 * radial) - 0.08 * x * z
    return base + 0.18 * torch.sin(2.0 * x * z) + 0.1 * radial


def apply_complex_deformation(points, generator, case_name):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    noise = torch.randn(points.shape, generator=generator, device=points.device) * 0.03
    twist = torch.rand(1, generator=generator, device=points.device).item() * 0.6 + 0.2
    bend = torch.rand(1, generator=generator, device=points.device).item() * 0.5 + 0.1

    if case_name == "spherical":
        r = torch.sqrt(x * x + y * y + z * z + 1e-6)
        points[:, 0] = x + 0.12 * torch.sin(3.0 * y) + 0.08 * r * torch.cos(2.0 * z)
        points[:, 1] = y + 0.10 * torch.sin(2.5 * z) - 0.06 * r * torch.sin(2.0 * x)
        points[:, 2] = z + 0.07 * torch.cos(3.5 * x)
    elif case_name == "toroidal":
        theta = torch.atan2(y, x + 1e-6)
        points[:, 0] = x + twist * 0.08 * torch.sin(4.0 * theta) + 0.06 * torch.cos(2.0 * z)
        points[:, 1] = y + twist * 0.08 * torch.cos(3.0 * theta) - 0.04 * torch.sin(2.0 * z)
        points[:, 2] = z + 0.09 * torch.sin(3.0 * theta) + 0.05 * torch.cos(2.0 * x)
    elif case_name == "open_surface":
        points[:, 0] = x + 0.18 * torch.sin(2.2 * y) + 0.1 * x * z
        points[:, 1] = y + bend * 0.14 * torch.sin(2.0 * x) - 0.08 * z
        points[:, 2] = z + 0.15 * torch.cos(2.8 * y) + 0.04 * x * y
    else:
        points[:, 0] = x + 0.2 * torch.sin(3.3 * y) + 0.12 * torch.cos(2.3 * z)
        points[:, 1] = y + 0.18 * torch.sin(2.7 * x) + 0.10 * torch.sin(2.1 * z)
        points[:, 2] = z + 0.16 * torch.cos(3.1 * x) - 0.08 * x * y

    points += noise
    return points


def _build_base_geometry(case, num_points, expand_factor, sample_seed):
    latent_topology = case["latent_topology"]
    mapper = OT3Dto2DMapper(
        latent_topology=latent_topology,
        expand_factor=expand_factor,
        width=84 if latent_topology != "volumetric" else 16,
    )
    if latent_topology == "toroidal":
        clean_points, _ = mapper._generate_latent_torus(num_points)
    elif latent_topology == "spherical":
        clean_points, _ = mapper._generate_latent_sphere(num_points)
    else:
        clean_points, _ = mapper._generate_latent_volume(num_points)

    g = torch.Generator(device=clean_points.device.type)
    g.manual_seed(sample_seed)
    clean_points = sample_or_repeat_points(clean_points, num_points, g)

    sx = torch.rand(1, generator=g, device=clean_points.device) * 0.4 + 0.8
    sy = torch.rand(1, generator=g, device=clean_points.device) * 0.4 + 0.8
    sz = torch.rand(1, generator=g, device=clean_points.device) * 0.3 + 0.85

    points = clean_points.clone()
    points[:, 0] *= sx
    points[:, 1] *= sy
    points[:, 2] *= sz
    points = apply_complex_deformation(points, g, case["name"]).float()

    normals = points / (torch.linalg.norm(points, dim=-1, keepdim=True) + 1e-6)
    cross = torch.cross(points, normals, dim=1)
    pressure = synthetic_pressure(points, case["name"])
    return mapper, clean_points, points, normals, cross, pressure


def build_point_sample(case, num_points, expand_factor, sample_seed, include_chi=False):
    _, _, points, normals, cross, pressure = _build_base_geometry(
        case=case,
        num_points=num_points,
        expand_factor=expand_factor,
        sample_seed=sample_seed,
    )
    features = [points, normals, cross]
    if include_chi:
        chi_feat = torch.full((points.shape[0], 1), fill_value=case["chi"], dtype=points.dtype, device=points.device)
        features.append(chi_feat)
    return {
        "points": points.cpu(),
        "features": torch.cat(features, dim=1).cpu(),
        "pressure": pressure.cpu(),
        "source_topology": case["name"],
        "source_chi": case["chi"],
        "schema_version": SYNTH_SCHEMA_VERSION,
    }


def build_topos_sample(case, num_points, expand_factor, sample_seed):
    _, _, points, normals, _, pressure = _build_base_geometry(
        case=case,
        num_points=num_points,
        expand_factor=expand_factor,
        sample_seed=sample_seed,
    )
    latent_topology = case["latent_topology"]
    mapper = OT3Dto2DMapper(
        latent_topology=latent_topology,
        expand_factor=expand_factor,
        width=84 if latent_topology != "volumetric" else 16,
    )
    idx_encoder, idx_decoder, grid_width = mapper.get_otno_indices(points, blur=0.01)

    if latent_topology == "toroidal":
        latent_coords, _ = mapper._generate_latent_torus(num_points)
        latent_coords = latent_coords.view(grid_width, grid_width, 3)
        latent_normals = compute_torus_normals(grid_width)
    elif latent_topology == "spherical":
        latent_coords, _ = mapper._generate_latent_sphere(num_points)
        latent_coords = latent_coords.view(grid_width, grid_width, 3)
        latent_normals = latent_coords / (torch.linalg.norm(latent_coords, dim=-1, keepdim=True) + 1e-6)
    else:
        latent_coords, _ = mapper._generate_latent_volume(num_points)
        latent_coords = latent_coords.view(grid_width, grid_width, grid_width, 3)
        latent_normals = latent_coords / (torch.linalg.norm(latent_coords, dim=-1, keepdim=True) + 1e-6)

    return {
        "points": points.cpu(),
        "normals": normals.cpu(),
        "pressure": pressure.cpu(),
        "topology": case["name"],
        "source_latent_topology": latent_topology,
        "chi": case["chi"],
        "idx_encoder": idx_encoder.long().cpu(),
        "idx_decoder": idx_decoder.long().cpu(),
        "latent_coords": latent_coords.cpu(),
        "latent_normals": latent_normals.cpu(),
        "grid_width": grid_width,
        "schema_version": SYNTH_SCHEMA_VERSION,
    }


def build_otno_sample(case, num_points, expand_factor, sample_seed):
    _, _, points, normals, _, pressure = _build_base_geometry(
        case=case,
        num_points=num_points,
        expand_factor=expand_factor,
        sample_seed=sample_seed,
    )
    mapper = OT3Dto2DMapper(latent_topology="toroidal", expand_factor=expand_factor, width=84)
    idx_encoder, idx_decoder, grid_width = mapper.get_otno_indices(points, blur=0.01)
    latent_coords, _ = mapper._generate_latent_torus(num_points)
    latent_coords = latent_coords.view(grid_width, grid_width, 3).float()
    latent_normals = compute_torus_normals(grid_width).float()

    return {
        "points": points.cpu(),
        "normals": normals.cpu(),
        "pressure": pressure.cpu(),
        "idx_encoder": idx_encoder.long().cpu(),
        "idx_decoder": idx_decoder.long().cpu(),
        "latent_coords": latent_coords.cpu(),
        "latent_normals": latent_normals.cpu(),
        "grid_width": grid_width,
        "source_topology": case["name"],
        "source_latent_topology": case["latent_topology"],
        "source_chi": case["chi"],
        "schema_version": SYNTH_SCHEMA_VERSION,
    }


class SyntheticGeometryDatasetTOPOS(Dataset):
    def __init__(
        self,
        cache_dir=None,
        n_train=500,
        n_test=111,
        split="train",
        expand_factor=2.0,
        num_points=3586,
        base_seed=42,
    ):
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.expand_factor = expand_factor
        self.num_points = num_points
        self.base_seed = base_seed
        self.num_samples = n_train if split == "train" else n_test
        self.split = split

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        file_idx = idx if self.split == "train" else idx + 1000
        sample_seed = self.base_seed + file_idx

        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"synthetic_topos_{self.split}_{file_idx:03d}.pt")
            if os.path.exists(cache_path):
                cached = torch.load(cache_path, weights_only=False)
                required = {
                    "points",
                    "normals",
                    "pressure",
                    "topology",
                    "chi",
                    "idx_encoder",
                    "idx_decoder",
                    "latent_coords",
                    "latent_normals",
                    "grid_width",
                    "schema_version",
                }
                if required.issubset(cached.keys()) and cached.get("schema_version") == SYNTH_SCHEMA_VERSION:
                    return cached

        case = CASE_LIBRARY[idx % len(CASE_LIBRARY)]
        out = build_topos_sample(
            case=case,
            num_points=self.num_points,
            expand_factor=self.expand_factor,
            sample_seed=sample_seed,
        )
        if self.cache_dir:
            torch.save(out, cache_path)
        return out


class SyntheticGeometryDatasetOTNO(Dataset):
    def __init__(
        self,
        cache_dir=None,
        n_train=500,
        n_test=111,
        split="train",
        expand_factor=2.0,
        num_points=3586,
        base_seed=42,
    ):
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.expand_factor = expand_factor
        self.num_points = num_points
        self.base_seed = base_seed
        self.num_samples = n_train if split == "train" else n_test
        self.split = split

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        file_idx = idx if self.split == "train" else idx + 1000
        sample_seed = self.base_seed + file_idx

        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"synthetic_otno_{self.split}_{file_idx:03d}.pt")
            if os.path.exists(cache_path):
                cached = torch.load(cache_path, weights_only=False)
                required = {
                    "points",
                    "normals",
                    "pressure",
                    "idx_encoder",
                    "idx_decoder",
                    "latent_coords",
                    "latent_normals",
                    "grid_width",
                    "source_topology",
                    "source_chi",
                    "schema_version",
                }
                if required.issubset(cached.keys()) and cached.get("schema_version") == SYNTH_SCHEMA_VERSION:
                    return cached

        case = CASE_LIBRARY[idx % len(CASE_LIBRARY)]
        out = build_otno_sample(
            case=case,
            num_points=self.num_points,
            expand_factor=self.expand_factor,
            sample_seed=sample_seed,
        )
        if self.cache_dir:
            torch.save(out, cache_path)
        return out


class SharedMixedPointDataset(Dataset):
    def __init__(
        self,
        cache_dir=None,
        cache_prefix="synthetic_points",
        n_train=500,
        n_test=111,
        split="train",
        expand_factor=2.0,
        num_points=3586,
        base_seed=42,
        include_chi=False,
    ):
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_prefix = cache_prefix
        self.num_samples = n_train if split == "train" else n_test
        self.split = split
        self.num_points = num_points
        self.base_seed = base_seed
        self.expand_factor = expand_factor
        self.include_chi = include_chi

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        file_idx = idx if self.split == "train" else idx + 1000
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{self.cache_prefix}_{self.split}_{file_idx:03d}.pt")
            if os.path.exists(cache_path):
                cached = torch.load(cache_path, weights_only=False)
                required = {"points", "features", "pressure", "source_topology", "schema_version"}
                if required.issubset(cached.keys()) and cached.get("schema_version") == SYNTH_SCHEMA_VERSION:
                    return cached

        case = CASE_LIBRARY[idx % len(CASE_LIBRARY)]
        sample_seed = self.base_seed + file_idx
        out = build_point_sample(
            case=case,
            num_points=self.num_points,
            expand_factor=self.expand_factor,
            sample_seed=sample_seed,
            include_chi=self.include_chi,
        )
        if self.cache_dir:
            torch.save(out, cache_path)
        return out
