# TOPOS: Topological Optimal-transport Partitioned Operator Solver
In this repository, we introduce the **TOPOS** architecture—a unified neural operator framework for learning physics on complex geometric domains. Conventional deep learning requires regular grids or falls back to localized graph methods on unstructured meshes. TOPOS circumvents this by using instance-dependent Optimal Transport (OT) to smoothly deform input geometries into standardized spherical, toroidal, or flat workbenches. 

A topological router ensures the mapping prevents "pinching" or "tearing" by aligning topologies based on their Euler characteristic. Once standard mapped, the framework unleashes powerful spectral Fourier Neural Operators (FNO) optimized on regular structures, delivering state-of-the-art speeds in 3D CFD tasks like Navier-Stokes, ShapeNet-Car CFD predictions, and DrivAerNet aerodynamic benchmarks.

This unified collection combines two complementary lines of work: 
1. **Geometry Operator Learning with Optimal Transport** (ShapeNet, DrivAerNet, Flowbench).
2. **Fourier Neural Operators for PDEs** (Navier-Stokes, Burgers, Darcy Flow).

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/mamtasaini003/OTNO.git
cd OTNO
```

### 2. Set Up a Virtual Environment & Install Required Packages
```bash
# Create and activate environment
conda create -n topos python=3.10
conda activate topos

# Install dependencies (ensure pip is upgraded)
pip install -r requirements.txt

# Install the topos package in editable mode
pip install -e .
```

Alternatively, create an environment from the YAML file:
```bash
conda env create -f env.yml
conda activate topos
pip install -e .
```

## Repository Structure

- `topos/`: Core library containing models, topological routers, and utilities.
- `scripts/`: Training, inference, and plotting scripts.
- `experiments/`: Experiment configurations for datasets like Flowbench, ShapeNet, and DrivAerNet.
- `tests/`: Unit tests and testing suites.

## Run the Project

Run scripts from the `experiments` directory:

### Paper 1: Operator Learning with Optimal Transport
```bash
python experiments/shapenet/topos/ot_train.py
python experiments/drivaernet/topos/ot_train_Cd.py
python experiments/flowbench/topos/ot_train_boundary.py --resolution 512 --expand_factor 3 --group_name nurbs --latent_shape square
python experiments/flowbench/topos/ot_train_fullspace.py --resolution 512 --expand_factor 3 --group_name nurbs --latent_shape square
```

### Paper 2: Spectral Solvers (FNO Benchmarks mapped to TOPOS)
```bash
# Navier-Stokes
python experiments/fno_benchmarks/train_navier_stokes.py
# Burgers Equation
python experiments/fno_benchmarks/train_burgers.py
# Darcy Flow
python experiments/fno_benchmarks/train_darcy.py

## Verification Tiers
A systematic testing suite for the TOPOS framework:

### Tier 1: Topological Generalization (The Router Test)
Validates the **Topological Router**. Tests if the model correctly identifies and directs different genus types (sphere vs torus) to the appropriate branch.
```bash
python experiments/tier1_router_test/train_poisson_mixed_genus.py
```

### Tier 2: Geometric Invariance (The OT Encoder Test)
Validates the **OT Encoder**. Tests resolution invariance by training on coarse meshes (1K points) and testing zero-shot on fine meshes (16K points) on an L-shaped domain.
```bash
# 1. Generate multi-resolution data
python scripts/generate_heat_multiresolution.py

# 2. Run invariance test
python experiments/tier2_geometric_invariance/train_heat_resolution_invariant.py
```
