# TOPOS Model Architecture Documentation

## Overview

**TOPOS** (Topological Optimal-transport Partitioned Operator Solver) is a unified neural operator pipeline for learning PDE operators on unstructured geometric domains. It uses optimal transport (OT) to map irregular meshes to structured latent spaces, then applies spectral FNO solvers, and decodes back to the original mesh topology.

---

## 1. Component Sources

| Component | File | Source | Purpose |
|-----------|------|--------|---------|
| `TOPOS` | `models/topos.py` | Custom | Main unified model orchestrating 4 stages |
| SphericalTransportFNO | `models/fno_spherical.py` | Extends `neuralop.models.SFNO` | SFNO for genus-0 (sphere) |
| ToroidalTransportFNO | `models/fno_spherical.py` | Extends `neuralop.models.FNO` | 2D periodic FNO for genus-1 (torus) |
| VolumetricFNO | `models/fno_3d_regular.py` | Extends `neuralop.models.FNO` | 3D Cartesian FNO for higher genus/volumes |
| `TopologicalRouter` | `router/topology_check.py` | Custom | Routes to appropriate solver via Euler characteristic |
| `FNO` (base) | `neuralop.models` | External library (neuralop) | Spectral Fourier Neural Operator |
| `SpectralConv` | `neuralop.layers.spectral_convolution` | External library | Spectral convolution layer |
| `ChannelMLP` | `neuralop.layers.channel_mlp` | External library | MLP for channel-wise transformations |

---

## 2. The 4-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TOPOS Pipeline                                     │
└─────────────────────────────────────────────────────────────────────────────┘

  Stage 1          Stage 2              Stage 3              Stage 4
┌──────────┐   ┌──────────┐        ┌──────────┐         ┌──────────┐
│   OT     │──▶│  Router  │───────▶│  Solver  │────────▶│ Decoder  │
│ Encoder  │   │          │        │  (FNO)   │         │          │
└──────────┘   └──────────┘        └──────────┘         └──────────┘
    │              │                   │                    │
    │              │                   │                    │
    │         ┌────┴────┐         ┌────┴────┐              │
    │         │         │         │         │              │
    │         ▼         ▼         ▼         ▼              │
    │    spherical  toroidal  volumetric    │              │
    │                                            │          │
    │         (pre-computed)   (neural network) │          │
    │                                            │          │
    └────────────────────────────────────────────┘
                    Inverse OT (index lookup)
```

### Stage 1: OT Encoder (Pre-computed)
- **Location**: Data pipeline (not in model)
- **Purpose**: Maps irregular mesh → structured latent grid via Optimal Transport
- **Input**: Raw mesh geometry with features (position, normals, etc.)
- **Output**: `transports` tensor on latent grid + `indices_decoder` for inverse mapping

### Stage 2: Router (TopologicalRouter)
- **Location**: `router/topology_check.py`
- **Logic**: Uses **Euler characteristic** to determine topology
  - χ = V − E + F (vertices - edges + faces)
  - χ ≈ 2 → **spherical** (genus 0, closed surface like sphere)
  - χ ≈ 0 → **toroidal** (genus 1, torus-like)
  - else → **volumetric** (higher genus or volumetric domain)
- **Formula**: `genus = (2 - χ) / 2`

### Stage 3: Solver (FNO)
Three branches based on topology:

| Branch | Class | Latent Grid | Use Case |
|--------|-------|-------------|----------|
| spherical | `SphericalTransportFNO` | spherical grid (SFNO) | Genus 0 surfaces |
| toroidal | `ToroidalTransportFNO` | 2D periodic (torus grid) | Genus 1 surfaces |
| volumetric | `VolumetricFNO` | 3D Cartesian [0,1]³ | Higher genus / volumes |

All solvers extend `neuralop.FNO` with:
- **Spectral convolution** via FFT (Fourier Neural Operator)
- **Lifting**: project input channels to hidden dimension
- **FNO blocks**: stack of spectral convolutions
- **Projection**: map hidden → output channels

### Stage 4: Decoder (Integrated in solvers)
- **Method**: Index lookup via `idx_decoder`
- **Process**: 
  1. Reshape output from grid to point cloud
  2. Select points corresponding to original mesh vertices using `transports[idx_decoder]`
  3. Project to output channels via MLP

---

## 3. File Architecture

```
/home/mamta/work/OTNO/
├── models/
│   ├── __init__.py          # Exports TOPOS
│   ├── topos.py             # Main TOPOS class (4-stage orchestrator)
│   ├── fno_spherical.py     # SphericalTransportFNO & ToroidalTransportFNO
│   └── fno_3d_regular.py   # VolumetricFNO (3D Cartesian)
│
├── router/
│   ├── __init__.py
│   └── topology_check.py    # TopologicalRouter, compute_euler_characteristic
│
├── topos_train.py           # Training script for TOPOS
│
├── utils.py                 # Utilities (LpLoss, UnitGaussianNormalizer, etc.)
│
├── tests/
│   └── test_topos_pipeline.py
│
└── topos.txt                # Paper documentation
```

---

## 4. Data Flow (Forward Pass)

```
Input:  (transports, idx_decoder, topology, chi)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Stage 2: Route (if topology="auto")                  │
│  - Compute χ = V - E + F                               │
│  - Determine genus → select solver                    │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Stage 3: Solve in Latent Space                       │
│  - lifting: (C, H, W) → (hidden, H, W)               │
│  - FNO blocks: spectral convolutions                  │
│  - domain_padding/unpad                              │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Stage 4: Decode (inverse OT)                         │
│  - Reshape: grid → point cloud                         │
│  - Index lookup: transports[idx_decoder]              │
│  - Projection MLP: hidden → out_channels              │
└───────────────────────────────────────────────────────┘
        │
        ▼
Output: Predicted field on physical mesh
```

---

## 5. Class Hierarchy

```
nn.Module
├── TOPOS (models/topos.py)
│   ├── TopologicalRouter (router/topology_check.py)
│   ├── SphericalTransportFNO (models/fno_spherical.py)
│   │   └── neuralop.models.SFNO
│   ├── ToroidalTransportFNO (models/fno_spherical.py)
│   │   └── neuralop.models.FNO
│   └── VolumetricFNO (models/fno_3d_regular.py)
│       └── neuralop.models.FNO
```

---

## 6. Configuration Examples

```python
# Spherical branch only (genus 0)
spherical_config = dict(
    n_modes=(32, 32),
    hidden_channels=120,
    in_channels=9,
    out_channels=1,
    n_layers=4,
)

model = TOPOS(spherical_config=spherical_config)

# All branches with auto-routing
model = TOPOS(
    spherical_config=spherical_config,
    volumetric_config=volumetric_config,
    toroidal_config=toroidal_config,
    chi_tol=0.5,
    default_topology="spherical"
)
```

---

## 7. Key Design Decisions

1. **Pre-computed OT**: Stage 1 is done in data pipeline (not learnable) for efficiency
2. **Topology-aware routing**: Single model handles multiple topologies via router
3. **Shared decoder logic**: All solvers use same index-lookup decoding mechanism
4. **Spectral methods**: FNO enables efficient computation on periodic grids
5. **Topology-Specific Models**: Spherical branch uses SFNO, while Toroidal and Volumetric branches use standard FNO.
