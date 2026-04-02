# Dataset Summary: Online Mixed-Genus Benchmark

## Purpose
This benchmark is designed to compare `OTNO` vs `TOPOS` fairly on **mixed complex geometries** with varying genus/topology, so the router advantage in TOPOS can be evaluated.

## Scripts
- OTNO baseline data/training: `scripts/train_otno_online.py`
- TOPOS data/training: `scripts/train_topos_online.py`

## Case Library (Shared)
Both scripts generate samples from the same 4-case mixed library:

1. `spherical`  
   - `chi = 2.0`
   - latent topology seed: `spherical`
2. `toroidal`  
   - `chi = 0.0`
   - latent topology seed: `toroidal`
3. `open_surface`  
   - `chi = 1.0`
   - latent topology seed: `volumetric`
4. `high_genus`  
   - `chi = -2.0`
   - latent topology seed: `volumetric`

Samples are assigned by index cycling through this case library.

## Synthetic Geometry Generation
- Base latent geometry is sampled from sphere / torus / volumetric primitives.
- Nonlinear, topology-specific deformations are applied.
- Random perturbation noise is added.
- Point count is fixed per sample via sample/repeat logic.

## Synthetic Target Field
- A topology-conditioned pressure function is used (`synthetic_pressure(...)`).
- This makes the task nontrivial across mixed geometry families.

## Split Protocol
Default split in both scripts:
- Train: `n_train = 500`
- Test: `n_test = 111`

Sample ids:
- Train ids: `0..n_train-1`
- Test ids: `1000..1000+n_test-1` (to avoid overlap with train cache keys)

## Cache Layout
Configured via `dataset.cache_dir` in config, with subfolder:
- `<cache_dir>/synthetic/`

Expected files:
- OTNO cache: `synthetic_otno_train_XXX.pt`, `synthetic_otno_test_XXX.pt`
- TOPOS cache: `synthetic_topos_train_XXX.pt`, `synthetic_topos_test_XXX.pt`

Schema version:
- `schema_version = 2`

If cached schema is stale or missing required keys, samples are regenerated automatically.

## Cached Sample Fields

### OTNO cached sample
- `points`
- `normals`
- `pressure`
- `idx_encoder`
- `idx_decoder`
- `latent_coords` (torus latent for OTNO baseline)
- `latent_normals`
- `grid_width`
- `source_topology`
- `source_latent_topology`
- `source_chi`
- `schema_version`

### TOPOS cached sample
- `points`
- `normals`
- `pressure`
- `topology` (source case label)
- `source_latent_topology`
- `chi`
- `idx_encoder`
- `idx_decoder`
- `latent_coords`
- `latent_normals`
- `grid_width`
- `schema_version`

## Router Usage
In `train_topos_online.py`, model forward uses:
- `topology="auto"`
- `chi=<sample chi>`

So router decisions are part of training/evaluation, and per-epoch routed-branch counts are logged.

## Normalization and Metrics
- Both scripts normalize for training.
- Both report overall train/test L2.
- Both report per-source-topology losses.

## Output Plots
Saved in `results/` by default:

OTNO:
- `results/otno_online_mixed_overall.png`
- `results/otno_online_mixed_per_topology.png`

TOPOS:
- `results/topos_online_mixed_overall.png`
- `results/topos_online_mixed_per_topology.png`

## Config Knobs (Important)
From config (`configs/abc_comparison.yaml` + script defaults):
- `dataset.train_samples`
- `dataset.test_samples`
- `dataset.expand_factor`
- `dataset.num_points` (optional; defaults to `3586` if not provided)
- `dataset.cache_dir`
- `training.seed`

