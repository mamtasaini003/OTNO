import os
import numpy as np
import torch
import xarray as xr
import netCDF4
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, ConcatDataset


# ─── Benchmark Metadata ────────────────────────────────────────────────────────
# This registry contains both external benchmarks (GAOT) and custom TOPOS benchmarks.

BENCHMARK_METADATA = {
    # ── External: GAOT Benchmark (Elliptic PDEs) ──
    # Elliptic PDEs (Poisson-like) - fixed coordinates
    'Poisson-Gauss': {
        'file': 'Poisson-Gauss.nc',
        'fix_x': True,
        'domain_label': 'square',
        'topology': 'spherical',    # simply-connected planar → genus-0
        'euler_chi': 2,
        'description': 'Poisson equation with Gaussian forcing on unit square',
    },
    'Poisson-C-Sines': {
        'file': 'Poisson-C-Sines.nc',
        'fix_x': True,
        'domain_label': 'complex_2d',
        'topology': 'spherical',    # complex 2D domain, but genus-0
        'euler_chi': 2,
        'description': 'Poisson equation with sinusoidal forcing on complex 2D domains',
    },
    # Bluff body (compressible flow, steady-state) - variable coordinates
    'Circle': {
        'file': 'Circle.nc',
        'fix_x': False,
        'domain_label': 'circle',
        'topology': 'toroidal',     # annular domain around body → genus-1-like
        'euler_chi': 0,
        'description': 'Steady compressible flow around circular body',
    },
    'Cone-F': {
        'file': 'Cone-F.nc',
        'fix_x': False,
        'domain_label': 'cone',
        'topology': 'toroidal',
        'euler_chi': 0,
        'description': 'Steady compressible flow around cone body',
    },
    'Ellipse-1': {
        'file': 'Ellipse-1.nc',
        'fix_x': False,
        'domain_label': 'ellipse',
        'topology': 'toroidal',
        'euler_chi': 0,
        'description': 'Steady compressible flow around elliptical body (variant 1)',
    },
    'Ellipse-2': {
        'file': 'Ellipse-2.nc',
        'fix_x': False,
        'domain_label': 'ellipse',
        'topology': 'toroidal',
        'euler_chi': 0,
        'description': 'Steady compressible flow around elliptical body (variant 2)',
    },
    'Ellipse-3': {
        'file': 'Ellipse-3.nc',
        'fix_x': False,
        'domain_label': 'ellipse',
        'topology': 'toroidal',
        'euler_chi': 0,
        'description': 'Steady compressible flow around elliptical body (variant 3)',
    },
    'Semicircle-F': {
        'file': 'Semicircle-F.nc',
        'fix_x': False,
        'domain_label': 'semicircle',
        'topology': 'toroidal',
        'euler_chi': 0,
        'description': 'Steady compressible flow around semicircular body',
    },
    'Rectangle-S': {
        'file': 'Rectangle-S.nc',
        'fix_x': False,
        'domain_label': 'rectangle',
        'topology': 'toroidal',
        'euler_chi': 0,
        'description': 'Steady compressible flow around rectangular body',
    },
    # ── Tier-2 Resolution Invariance Datasets ──
    'Heat-L-coarse': {
        'file': 'Heat-L-coarse.nc',
        'fix_x': True,
        'domain_label': 'l_shape',
        'topology': 'volumetric',  # Treat generic 2D shapes as volumetric for router test
        'euler_chi': 1,            # L-shape with Dirichlet BC (Euler characteristic for surface with boundary)
        'description': 'Heat equation on L-shaped domain (Coarse: 1024 points)',
    },
    'Heat-L-medium': {
        'file': 'Heat-L-medium.nc',
        'fix_x': True,
        'domain_label': 'l_shape',
        'topology': 'volumetric',
        'euler_chi': 1,
        'description': 'Heat equation on L-shaped domain (Medium: 4096 points)',
    },
    'Heat-L-fine': {
        'file': 'Heat-L-fine.nc',
        'fix_x': True,
        'domain_label': 'l_shape',
        'topology': 'volumetric',
        'euler_chi': 1,
        'description': 'Heat equation on L-shaped domain (Fine: 16384 points)',
    },
    # ── Custom: TOPOS Benchmarks (Resolution Invariance, etc.) ──
    'Heat-L-coarse': {
        'file': 'topos_benchmarks/heat_l_shape/Heat-L-coarse.nc',
        'fix_x': True,
        'domain_label': 'l_shape',
        'topology': 'volumetric',
        'euler_chi': 1,
        'description': 'Heat equation on L-shaped domain (Coarse: 1024 points)',
    },
    'Heat-L-medium': {
        'file': 'topos_benchmarks/heat_l_shape/Heat-L-medium.nc',
        'fix_x': True,
        'domain_label': 'l_shape',
        'topology': 'volumetric',
        'euler_chi': 1,
        'description': 'Heat equation on L-shaped domain (Medium: 4096 points)',
    },
    'Heat-L-fine': {
        'file': 'topos_benchmarks/heat_l_shape/Heat-L-fine.nc',
        'fix_x': True,
        'domain_label': 'l_shape',
        'topology': 'volumetric',
        'euler_chi': 1,
        'description': 'Heat equation on L-shaped domain (Fine: 16384 points)',
    },
    'Heat-Square-coarse': {
        'file': 'topos_benchmarks/heat_square/Heat-Square-coarse.nc',
        'fix_x': True,
        'topology': 'toroidal', # Square is periodic-able or just toroidal
        'euler_chi': 0,
    },
    'Heat-Square-fine': {
        'file': 'topos_benchmarks/heat_square/Heat-Square-fine.nc',
        'fix_x': True,
        'topology': 'toroidal',
        'euler_chi': 0,
    },
    # ── Tier-3 Physical Fidelity (Unsteady Navier-Stokes) ──
    'NS-Sines': {
        'file': 'topos_benchmarks/unsteady_ns/time_dep/NS-Sines.nc',
        'fix_x': True,
        'domain_label': 'complex_2d',
        'topology': 'spherical',  # Simply-connected irregular outer domain
        'euler_chi': 2,
        'is_time_dp': True,
        'description': 'Unsteady 2D Navier-Stokes on irregular domains (Vortex Shedding)',
    },
}


# ─── Dataset Class ────────────────────────────────────────────────────────────

class PDEDataset(Dataset):
    """
    General PyTorch Dataset for NetCDF-based PDE point clouds.

    Each sample returns:
        c:  (num_points, num_c_channels) - input conditions / forcing
        u:  (num_points, num_u_channels) - solution
        x:  (num_points, 2)              - coordinates
        meta: dict with 'topology', 'euler_chi', 'dataset_name'
    """

    def __init__(
        self,
        dataset_name: str,
        base_path: str = '/media/HDD/mamta_backup/datasets',
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        normalize: bool = True,
        seed: int = 42,
        max_samples: Optional[int] = None,
        c_stats: Optional[tuple] = None,
        u_stats: Optional[tuple] = None,
        ot_cache_dir: Optional[str] = None,
    ):
        super().__init__()
        if dataset_name not in BENCHMARK_METADATA:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {list(BENCHMARK_METADATA.keys())}"
            )

        self.dataset_name = dataset_name
        self.meta_info = BENCHMARK_METADATA[dataset_name]
        self.split = split
        self.ot_cache_dir = ot_cache_dir or os.environ.get("TOPOS_OT_CACHE_DIR")

        # Load NetCDF (checks both custom benchmark subdirs and gaot subdir)
        # Try primary path
        filepath = os.path.join(base_path, self.meta_info['file'])
        if not os.path.exists(filepath):
            # Fallback to gaot/ prefix for legacy paths
            filepath = os.path.join(base_path, 'gaot', self.meta_info['file'])
            
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        self.is_time_dp = self.meta_info.get('is_time_dp', False)

        if filepath.endswith('.nc'):
            try:
                # Try netCDF4 first for large unsteady files
                with netCDF4.Dataset(filepath) as ds:
                    u_raw = np.array(ds.variables['u'][:max_samples])
                    x_raw = np.array(ds.variables['x'][:])
                    if 'c' in ds.variables:
                        c_raw = np.array(ds.variables['c'][:max_samples])
                    else:
                        if self.is_time_dp:
                            c_raw = u_raw[:, :1, :, :]
                            u_raw = u_raw[:, 1:, :, :]
                        else:
                            c_raw = u_raw
            except Exception as e:
                # Fallback to xarray
                with xr.open_dataset(filepath) as ds:
                    if max_samples:
                        ds = ds.isel(sample=slice(0, max_samples))
                    u_raw = ds['u'].values
                    x_raw = ds['x'].values
                    if 'c' in ds:
                        c_raw = ds['c'].values
                    else:
                        if self.is_time_dp:
                            c_raw = u_raw[:, :1, :, :]
                            u_raw = u_raw[:, 1:, :, :]
                        else:
                            c_raw = u_raw
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        # Squeeze singleton time dimension ONLY if not time-dependent
        if not self.is_time_dp:
            c_raw = c_raw.squeeze(1) # (N, P, C_ch)
            u_raw = u_raw.squeeze(1) # (N, P, U_ch)
        # Else: (N, T, P, C_ch)

        # Handle coordinates
        if self.meta_info['fix_x']:
            # x: (1, 1, P, 2) → (P, 2) — same coords for all samples
            x_raw = x_raw.squeeze()
            if x_raw.ndim == 1:
                x_raw = x_raw.reshape(-1, 2)
        else:
            # x: (N, 1, P, 2) → (N, P, 2)
            x_raw = x_raw.squeeze(1)

        N = c_raw.shape[0]

        # Split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(N)
        n_train = int(N * train_ratio)
        n_val = int(N * val_ratio)

        if split == 'train':
            idx = indices[:n_train]
        elif split == 'val':
            idx = indices[n_train:n_train + n_val]
        elif split == 'test':
            idx = indices[n_train + n_val:]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        self._abs_indices = idx
        self.c = torch.tensor(c_raw[idx], dtype=torch.float32)
        self.u = torch.tensor(u_raw[idx], dtype=torch.float32)

        if self.meta_info['fix_x']:
            self.x = torch.tensor(x_raw, dtype=torch.float32)  # (P, 2)
            self._fix_x = True
        else:
            self.x = torch.tensor(x_raw[idx], dtype=torch.float32)  # (N_split, P, 2)
            self._fix_x = False

        # Normalize using provided or training statistics
        if normalize:
            if c_stats is not None:
                self.c_mean, self.c_std = c_stats
            else:
                self.c_mean = self.c.mean(dim=(0, 1), keepdim=True)
                self.c_std = self.c.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)

            if u_stats is not None:
                self.u_mean, self.u_std = u_stats
            else:
                self.u_mean = self.u.mean(dim=(0, 1), keepdim=True)
                self.u_std = self.u.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)

            self.c = (self.c - self.c_mean) / self.c_std
            # Note: we normalize u for training, but keep targets for loss? 
            # Usually we normalize both for FNO.
            self.u = (self.u - self.u_mean) / self.u_std

        # ---- OT Mapping Logic ----
        self.ot_indices_encoder = None
        self.ot_indices_decoder = None
        self._load_ot_maps()

    def _load_ot_maps(self, res=64):
        if self.ot_cache_dir is not None:
            ot_cache_dir = Path(self.ot_cache_dir)
        else:
            # Default to repo-local cache: <repo>/scripts/benchmarks/ot_cache
            ot_cache_dir = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "ot_cache"
        ot_path = ot_cache_dir / f"{self.dataset_name}_ot_res{res}.pt"
        if ot_path.exists():
            try:
                ot_data = torch.load(str(ot_path), map_location='cpu')
                # Note: indices in ot_data were for ALL samples in the file.
                enc_all = ot_data['indices_encoder']
                dec_all = ot_data['indices_decoder']
                
                if len(enc_all) == 1:
                    # Shared coordinates Case: Broadcast the single map to all samples
                    self.ot_indices_encoder = [enc_all[0]] * len(self._abs_indices)
                    self.ot_indices_decoder = [dec_all[0]] * len(self._abs_indices)
                else:
                    # Individual coordinates Case: Select the subset for this split
                    # Clip abs_indices if ot_data is smaller (handle potentially missing tail samples)
                    safe_indices = [i for i in self._abs_indices if i < len(enc_all)]
                    if len(safe_indices) < len(self._abs_indices):
                        print(f"  [PDEDataset] Warning: Missing OT maps for {len(self._abs_indices) - len(safe_indices)} samples in {self.dataset_name}")
                        # Fallback: repeat last available map or just subset
                        while len(safe_indices) < len(self._abs_indices):
                            safe_indices.append(safe_indices[-1] if safe_indices else 0)
                            
                    self.ot_indices_encoder = [enc_all[i] for i in safe_indices]
                    self.ot_indices_decoder = [dec_all[i] for i in safe_indices]
                
                print(f"  [PDEDataset] Loaded Dual OT maps for {self.dataset_name} ({len(self.ot_indices_encoder)} samples)")
            except Exception as e:
                print(f"  [PDEDataset] Warning: Failed to load OT maps for {self.dataset_name}: {e}")

    def __len__(self):
        return len(self.c)

    def __getitem__(self, idx):
        c = self.c[idx]           # (T, P, C_ch) or (P, C_ch)
        u = self.u[idx]           # (T, P, U_ch) or (P, U_ch)

        if self._fix_x:
            x = self.x             # (P, 2) — shared
        else:
            x = self.x[idx]       # (P, 2)

        meta = {
            'topology': self.meta_info['topology'],
            'euler_chi': self.meta_info['euler_chi'],
            'dataset_name': self.dataset_name,
        }
        
        batch = {'c': c, 'u': u, 'x': x, 'meta': meta}
        
        if self.ot_indices_decoder is not None:
            batch['indices_decoder'] = self.ot_indices_decoder[idx]
            batch['indices_encoder'] = self.ot_indices_encoder[idx]

        return batch


# ─── Mixed-topology dataset ──────────────────────────────────────────────────

class MixedTopologyDataset(Dataset):
    """
    Combines multiple GAOT datasets with different topologies into one dataset.
    Used for Tier-1 TOPOS router testing: the model must correctly route
    samples from different topologies to the appropriate FNO branch.

    Topology mapping (for TOPOS router):
        - Poisson-Gauss / Poisson-C-Sines → spherical (genus-0, chi=2)
        - Circle / Ellipse / Cone / etc.  → toroidal  (genus-1, chi=0)
    """

    def __init__(
        self,
        dataset_names: List[str],
        base_path: str = '/media/HDD/mamta_backup/datasets/gaot',
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        normalize: bool = True,
        seed: int = 42,
        max_samples_per_dataset: Optional[int] = None,
        ot_cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.datasets = []
        self.dataset_ranges = []   # (start, end) index for each sub-dataset
        self.topology_labels = []  # topology label per sample

        offset = 0
        for name in dataset_names:
            ds = PDEDataset(
                dataset_name=name,
                base_path=base_path,
                split=split,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                normalize=normalize,
                seed=seed,
                ot_cache_dir=ot_cache_dir,
            )
            n = len(ds)
            if max_samples_per_dataset is not None and n > max_samples_per_dataset:
                n = max_samples_per_dataset

            self.datasets.append((ds, n))
            self.dataset_ranges.append((offset, offset + n))
            topo = BENCHMARK_METADATA[name]['topology']
            self.topology_labels.extend([topo] * n)
            offset += n

        self._total_len = offset

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        for (ds, n), (start, end) in zip(self.datasets, self.dataset_ranges):
            if start <= idx < end:
                return ds[idx - start]
        raise IndexError(f"Index {idx} out of range [0, {self._total_len})")


# ─── Custom collate for mixed-topology batches ───────────────────────────────

def mixed_collate_fn(batch):
    """
    Custom collate for MixedTopologyDataset.
    Since different sub-datasets may have different num_points and num_channels,
    we group by topology and return a list-of-dicts rather than stacking.
    
    Returns a dict with:
        'samples': list of individual sample dicts (c, u, x, meta)
        'topologies': list of topology strings
        'euler_chis': list of euler characteristic values
    """
    return {
        'samples': batch,
        'topologies': [s['meta']['topology'] for s in batch],
        'euler_chis': [s['meta']['euler_chi'] for s in batch],
        'dataset_names': [s['meta']['dataset_name'] for s in batch],
    }


# ─── Convenience functions ────────────────────────────────────────────────────

def get_pde_dataloaders(
    dataset_name: str,
    base_path: str = '/media/HDD/mamta_backup/datasets',
    batch_size: int = 8,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    normalize: bool = True,
    seed: int = 42,
    max_samples: Optional[int] = None,
    c_stats: Optional[tuple] = None,
    u_stats: Optional[tuple] = None,
    ot_cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get train/val/test DataLoaders for a single PDE dataset."""
    train_dataset = PDEDataset(
        dataset_name,
        base_path=base_path,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        normalize=normalize,
        seed=seed,
        max_samples=max_samples,
        c_stats=c_stats,
        u_stats=u_stats,
        ot_cache_dir=ot_cache_dir,
    )
    val_dataset = PDEDataset(
        dataset_name,
        base_path=base_path,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        normalize=normalize,
        seed=seed,
        max_samples=max_samples,
        c_stats=c_stats,
        u_stats=u_stats,
        ot_cache_dir=ot_cache_dir,
    )
    test_dataset = PDEDataset(
        dataset_name,
        base_path=base_path,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        normalize=normalize,
        seed=seed,
        max_samples=max_samples,
        c_stats=c_stats,
        u_stats=u_stats,
        ot_cache_dir=ot_cache_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def get_mixed_topology_dataloaders(
    dataset_names: Optional[List[str]] = None,
    base_path: str = '/media/HDD/mamta_backup/datasets',
    batch_size: int = 8,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    normalize: bool = True,
    seed: int = 42,
    max_samples_per_dataset: Optional[int] = None,
    ot_cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train/val/test DataLoaders for a mixed-topology dataset.

    Default dataset_names mixes genus-0 (Poisson) and genus-1 (bluff body) domains:
        - Poisson-Gauss  (spherical / genus-0)
        - Poisson-C-Sines (spherical / genus-0)
        - Circle          (toroidal / genus-1)
        - Ellipse-1       (toroidal / genus-1)
    """
    if dataset_names is None:
        dataset_names = ['Poisson-Gauss', 'Poisson-C-Sines', 'Circle', 'Ellipse-1']

    loaders = {}
    for split in ['train', 'val', 'test']:
        ds = MixedTopologyDataset(
            dataset_names=dataset_names,
            base_path=base_path,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            normalize=normalize,
            seed=seed,
            max_samples_per_dataset=max_samples_per_dataset,
            ot_cache_dir=ot_cache_dir,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=mixed_collate_fn,
        )
    return loaders['train'], loaders['val'], loaders['test']
