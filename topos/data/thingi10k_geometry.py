import importlib
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from .mixed_geometry_baselines import (
    compute_signed_distance_features,
    make_regular_grid,
    rescale_per_sample,
)
from .ot_mapper_3d import OT3Dto2DMapper
from .synthetic_mixed_geometry import SYNTH_SCHEMA_VERSION, compute_torus_normals
from topos.router.topology_check import TopologicalRouter


# All topology branch names recognised by TOPOS
_KNOWN_TOPOLOGIES = {"spherical", "toroidal", "volumetric", "graph"}


def parse_geometry_filter(filepath):
    """Parse a support_genus.txt filter file.

    The file must contain two key-value lines (ignoring comments starting with #):
        mode: genus | chi
        values: comma-separated numbers

    Parameters
    ----------
    filepath : str
        Path to the filter file.

    Returns
    -------
    dict
        ``{"mode": "genus"|"chi", "values": set_of_floats}``
    """
    mode = None
    values = None
    with open(filepath, "r") as f:
        for raw_line in f:
            line = raw_line.split("#")[0].strip()  # strip comments
            if not line:
                continue
            if line.lower().startswith("mode:"):
                mode = line.split(":", 1)[1].strip().lower()
            elif line.lower().startswith("values:"):
                raw_vals = line.split(":", 1)[1].strip()
                values = {float(v.strip()) for v in raw_vals.split(",") if v.strip()}
    if mode not in ("genus", "chi"):
        raise ValueError(
            f"Invalid or missing mode in {filepath}. "
            f"Expected 'mode: genus' or 'mode: chi', got: '{mode}'"
        )
    if not values:
        raise ValueError(f"No values found in {filepath}.")
    return {"mode": mode, "values": values}


# Backward-compatible alias
def parse_supported_topologies(filepath):
    """Legacy wrapper - calls :func:`parse_geometry_filter`."""
    return parse_geometry_filter(filepath)


def _import_thingi10k():
    try:
        return importlib.import_module("thingi10k")
    except ImportError as exc:
        raise ImportError(
            "The Thingi10K training scripts require the `thingi10k` package. "
            f"Active python: {sys.executable}. "
            "Install it in this same environment with `python -m pip install thingi10k`. "
            f"Original import error: {exc}"
        ) from exc


def _find_npz_root(source_cache_dir):
    if source_cache_dir is None:
        return None
    candidates = []
    for root, _, files in os.walk(source_cache_dir):
        if any(name.endswith(".npz") for name in files):
            candidates.append(root)
    if not candidates:
        return None
    candidates.sort(key=len)
    return candidates[0]


def _normalize_vertices(vertices):
    vertices = torch.as_tensor(vertices, dtype=torch.float32)
    center = 0.5 * (vertices.max(dim=0).values + vertices.min(dim=0).values)
    extent = (vertices.max(dim=0).values - vertices.min(dim=0).values).max()
    extent = extent.clamp_min(1e-6)
    return (vertices - center) * (2.0 / extent)


def _sanitize_faces(faces):
    faces = torch.as_tensor(faces, dtype=torch.long)
    if faces.ndim != 2 or faces.shape[1] < 3:
        raise ValueError(f"Expected triangular faces, got shape {tuple(faces.shape)}")
    if faces.shape[1] > 3:
        faces = faces[:, :3]
    valid = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[valid]
    if faces.numel() == 0:
        raise ValueError("Mesh has no valid triangular faces after sanitization.")
    return faces


def _unique_edge_count(faces):
    edges = torch.cat(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        dim=0,
    )
    edges = torch.sort(edges, dim=1).values
    return torch.unique(edges, dim=0).shape[0]


def _boundary_edge_count(faces):
    edges = torch.cat(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        dim=0,
    )
    edges = torch.sort(edges, dim=1).values
    _, counts = torch.unique(edges, dim=0, return_counts=True)
    return int((counts == 1).sum().item())


def _vertex_normals(vertices, faces):
    normals = torch.zeros_like(vertices)
    tri = vertices[faces]
    face_normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    for corner in range(3):
        normals.index_add_(0, faces[:, corner], face_normals)
    return normals / (torch.linalg.norm(normals, dim=1, keepdim=True) + 1e-6)


def _sample_or_repeat(points, normals, target_n, seed):
    generator = torch.Generator(device=points.device.type)
    generator.manual_seed(seed)
    n = points.shape[0]
    if n == target_n:
        return points, normals
    if n > target_n:
        idx = torch.randperm(n, generator=generator, device=points.device)[:target_n]
        return points[idx], normals[idx]
    extra = target_n - n
    idx_extra = torch.randint(0, n, (extra,), generator=generator, device=points.device)
    return torch.cat([points, points[idx_extra]], dim=0), torch.cat([normals, normals[idx_extra]], dim=0)


def _synthetic_pressure(points, topology, chi):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    base = torch.sin(2.2 * x) + 0.6 * torch.cos(2.7 * y) + 0.45 * z
    radial = torch.sqrt(x * x + y * y + z * z + 1e-6)
    topo_shift = 0.05 * chi
    if topology == "toroidal":
        return base + 0.18 * torch.sin(3.5 * radial) + 0.12 * x * y + topo_shift
    if topology == "spherical":
        return base + 0.20 * torch.cos(4.0 * radial) - 0.08 * x * z + topo_shift
    if topology == "graph":
        return base + 0.15 * torch.sin(2.0 * x * y * z) + 0.1 * radial
    return base + 0.15 * torch.sin(2.0 * x * z) + 0.08 * y * z + topo_shift


class Thingi10KBaseDataset(Dataset):
    """Base dataset for Thingi10K meshes.

    Parameters
    ----------
    supported_topologies : set of str or None
        If provided, only meshes whose routed topology falls within this set
        are kept.  Use :func:`parse_supported_topologies` to build this from
        a ``support_genus.txt`` file.  When *None* (default), **all**
        geometries are used regardless of topology.
    """

    def __init__(
        self,
        cache_dir=None,
        split="train",
        train_samples=500,
        test_samples=111,
        split_offset=0,
        variant="npz",
        source_cache_dir=None,
        num_points=3586,
        seed=42,
        thingi_filters=None,
        supported_topologies=None,
    ):
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.variant = variant
        self.source_cache_dir = source_cache_dir
        self.num_points = num_points
        self.seed = seed
        self.router = TopologicalRouter(require_watertight=False)
        self.filters = dict(thingi_filters or {})
        self.supported_topologies = supported_topologies
        self.thingi = None
        self.npz_root = _find_npz_root(self.source_cache_dir)
        entries = None

        try:
            self.thingi = _import_thingi10k()
            self.thingi.init(cache_dir=self.source_cache_dir, variant=self.variant)
            entries = list(self.thingi.dataset(**self.filters))
            entries.sort(key=lambda item: int(item.get("file_id", 0)))
        except Exception:
            if self.npz_root is None:
                raise
            entries = self._build_entries_from_npz()

        start = split_offset
        if split == "train":
            self.entries = entries[start:start + train_samples]
        elif split == "test":
            test_start = start + train_samples
            self.entries = entries[test_start:test_start + test_samples]
        else:
            self.entries = entries[start:]
        self.split = split

        # --- Topology filtering ---
        if self.supported_topologies is not None:
            filtered = []
            for i, entry in enumerate(self.entries):
                try:
                    sample = self._load_mesh(i)
                    if sample["topology"] in self.supported_topologies:
                        filtered.append(entry)
                except Exception:
                    pass  # Skip meshes that fail to load
            n_before = len(self.entries)
            self.entries = filtered
            print(
                f"[Thingi10K {split}] Topology filter {self.supported_topologies}: "
                f"kept {len(self.entries)}/{n_before} samples."
            )

    def _build_entries_from_npz(self):
        npz_files = []
        for name in sorted(os.listdir(self.npz_root)):
            if name.endswith(".npz"):
                npz_files.append(os.path.join(self.npz_root, name))

        min_vertices, max_vertices = self.filters.get("num_vertices", (None, None))
        entries = []
        for path in npz_files:
            try:
                with np.load(path) as data:
                    num_vertices = int(data["vertices"].shape[0])
            except Exception:
                continue
            if min_vertices is not None and num_vertices < min_vertices:
                continue
            if max_vertices is not None and num_vertices > max_vertices:
                continue
            file_id = int(os.path.splitext(os.path.basename(path))[0])
            entries.append({
                "file_id": file_id,
                "file_path": path,
                "num_vertices": num_vertices,
            })
        entries.sort(key=lambda item: int(item["file_id"]))
        return entries

    def __len__(self):
        return len(self.entries)

    def _load_mesh(self, idx):
        entry = self.entries[idx]
        if self.thingi is not None:
            vertices, faces = self.thingi.load_file(entry["file_path"])
        else:
            with np.load(entry["file_path"]) as data:
                vertices = data["vertices"]
                faces = data["facets"]
        vertices = _normalize_vertices(vertices)
        faces = _sanitize_faces(faces)
        normals = _vertex_normals(vertices, faces)
        points, normals = _sample_or_repeat(vertices, normals, self.num_points, seed=self.seed + idx)
        V = vertices.shape[0]
        E = _unique_edge_count(faces)
        F = faces.shape[0]
        chi = float(V - E + F)
        boundary_edges = _boundary_edge_count(faces)
        is_closed = bool(entry.get("closed", boundary_edges == 0))
        is_manifold = bool(entry.get("oriented", True) and boundary_edges >= 0)
        topology = self.router.route(chi=chi)
        if not is_closed:
            topology = "volumetric"
        if not is_manifold:
            topology = "graph"
        points = rescale_per_sample(points.float())
        normals = normals.float()
        pressure = _synthetic_pressure(points, topology, chi)
        return {
            "entry": entry,
            "points": points,
            "normals": normals,
            "pressure": pressure.float(),
            "chi": chi,
            "topology": topology,
            "is_closed": is_closed,
            "schema_version": SYNTH_SCHEMA_VERSION,
        }


class Thingi10KGinoDataset(Thingi10KBaseDataset):
    def __init__(self, latent_grid_size=(16, 16, 16), **kwargs):
        super().__init__(**kwargs)
        self.latent_queries = make_regular_grid(tuple(latent_grid_size))
        self.flat_input_grid = self.latent_queries.reshape(-1, self.latent_queries.shape[-1]).clone()

    def __getitem__(self, idx):
        cache_path = None
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"thingi10k_gino_{self.split}_{idx:05d}.pt")
            if os.path.exists(cache_path):
                return torch.load(cache_path, weights_only=False)

        sample = self._load_mesh(idx)
        sdf = compute_signed_distance_features(sample["points"], self.flat_input_grid)
        out = {
            "input_geom": self.flat_input_grid.clone(),
            "features": sdf,
            "output_queries": sample["points"],
            "latent_queries": self.latent_queries.clone(),
            "pressure": sample["pressure"],
            "source_topology": sample["topology"],
            "chi": sample["chi"],
            "schema_version": sample["schema_version"],
        }
        if cache_path:
            torch.save(out, cache_path)
        return out


class Thingi10KOtnoDataset(Thingi10KBaseDataset):
    def __init__(self, expand_factor=2.0, width=84, **kwargs):
        super().__init__(**kwargs)
        self.mapper = OT3Dto2DMapper(latent_topology="toroidal", expand_factor=expand_factor, width=width)

    def __getitem__(self, idx):
        cache_path = None
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"thingi10k_otno_{self.split}_{idx:05d}.pt")
            if os.path.exists(cache_path):
                return torch.load(cache_path, weights_only=False)

        sample = self._load_mesh(idx)
        idx_encoder, idx_decoder, grid_width = self.mapper.get_otno_indices(sample["points"], blur=0.01)
        latent_coords, _ = self.mapper._generate_latent_torus(self.num_points)
        latent_coords = latent_coords.view(grid_width, grid_width, 3).float().cpu()
        latent_normals = compute_torus_normals(grid_width).float().cpu()
        out = {
            "points": sample["points"].cpu(),
            "normals": sample["normals"].cpu(),
            "pressure": sample["pressure"].cpu(),
            "idx_encoder": idx_encoder.long().cpu(),
            "idx_decoder": idx_decoder.long().cpu(),
            "latent_coords": latent_coords,
            "latent_normals": latent_normals,
            "grid_width": grid_width,
            "source_topology": sample["topology"],
            "chi": sample["chi"],
            "schema_version": sample["schema_version"],
        }
        if cache_path:
            torch.save(out, cache_path)
        return out


class Thingi10KToposDataset(Thingi10KBaseDataset):
    def __init__(self, expand_factor=2.0, width_2d=84, width_3d=16, **kwargs):
        super().__init__(**kwargs)
        self.expand_factor = expand_factor
        self.width_2d = width_2d
        self.width_3d = width_3d
        self.mappers = {}

    def _get_mapper(self, topology):
        if topology not in self.mappers:
            width = self.width_2d if topology in ("spherical", "toroidal") else self.width_3d
            self.mappers[topology] = OT3Dto2DMapper(latent_topology=topology, expand_factor=self.expand_factor, width=width)
        return self.mappers[topology]

    def __getitem__(self, idx):
        cache_path = None
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"thingi10k_topos_{self.split}_{idx:05d}.pt")
            if os.path.exists(cache_path):
                return torch.load(cache_path, weights_only=False)

        sample = self._load_mesh(idx)
        topology = sample["topology"]
        if topology == "graph":
            out = {
                "points": sample["points"].cpu(),
                "normals": sample["normals"].cpu(),
                "pressure": sample["pressure"].cpu(),
                "idx_encoder": torch.arange(sample["points"].shape[0], dtype=torch.long),
                "idx_decoder": torch.arange(sample["points"].shape[0], dtype=torch.long),
                "grid_width": 0,
                "topology": topology,
                "chi": sample["chi"],
                "schema_version": sample["schema_version"],
            }
        else:
            mapper = self._get_mapper(topology)
            idx_encoder, idx_decoder, grid_width = mapper.get_otno_indices(sample["points"], blur=0.01)
            if topology == "toroidal":
                latent_coords, _ = mapper._generate_latent_torus(self.num_points)
                latent_coords = latent_coords.view(grid_width, grid_width, 3)
                latent_normals = compute_torus_normals(grid_width)
            elif topology == "spherical":
                latent_coords, _ = mapper._generate_latent_sphere(self.num_points)
                latent_coords = latent_coords.view(grid_width, grid_width, 3)
                latent_normals = latent_coords / (torch.linalg.norm(latent_coords, dim=-1, keepdim=True) + 1e-6)
            else:
                latent_coords, _ = mapper._generate_latent_volume(self.num_points)
                latent_coords = latent_coords.view(grid_width, grid_width, grid_width, 3)
                latent_normals = latent_coords / (torch.linalg.norm(latent_coords, dim=-1, keepdim=True) + 1e-6)

            out = {
                "points": sample["points"].cpu(),
                "normals": sample["normals"].cpu(),
                "pressure": sample["pressure"].cpu(),
                "idx_encoder": idx_encoder.long().cpu(),
                "idx_decoder": idx_decoder.long().cpu(),
                "grid_width": grid_width,
                "topology": topology,
                "chi": sample["chi"],
                "latent_coords": latent_coords.float().cpu(),
                "latent_normals": latent_normals.float().cpu(),
                "schema_version": sample["schema_version"],
            }
        if cache_path:
            torch.save(out, cache_path)
        return out
