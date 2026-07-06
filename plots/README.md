# TOPOS - Publication Figures

This directory contains self-contained scripts for generating all five
figures in the TOPOS paper.  Each script reads pre-computed evaluation
data from `results/eval/` and writes PDF + PNG figures to
`results/figures/`.

## Quick Start

```bash
# -- Step 1: Evaluate all models (GPU required) --------------------
# Mixed-genus dataset
python scripts/evaluate_all.py \
    --config configs/mixed_genus_fair_comparison.yaml \
    --models gino otno topos \
    --datasets mixed_genus \
    --gpus 0

# Thingi10K dataset
python scripts/evaluate_all.py \
    --config configs/thingi10k_topos.yaml \
    --models topos otno \
    --datasets thingi10k \
    --gpus 0

# -- Step 2: Generate all figures (CPU only) -----------------------
python plots/generate_all_plots.py
```

## Individual Plots

| # | Script | Description | Dataset |
|---|--------|-------------|---------|
| 1 | `plot_genus_scalability.py` | Genus scalability line plot (error vs genus) | Custom Mixed |
| 2 | `plot_error_heatmaps.py` | 3D qualitative error heatmaps (GT + models grid) | Both |
| 3 | `plot_boundary_vs_global.py` | Global vs Boundary error grouped bars | Both |
| 4 | `plot_robustness_violin.py` | Violin + box-whisker error distributions | Thingi10K |
| 5 | `plot_pareto_efficiency.py` | Pareto efficiency scatter (error vs compute) | Both |

Each script accepts `--eval_dir` and `--out_dir` flags to override defaults.

## Evaluation Data Schema

Each `.pt` file in `results/eval/` is a dict:

```python
{
    "model": "TOPOS",
    "dataset": "mixed_genus",
    "per_sample": [
        {
            "idx": int,
            "topology": str,        # "spherical", "toroidal", "open_surface", "high_genus"
            "chi": float,           # Euler characteristic
            "genus": float,         # computed genus
            "rel_l2": float,        # relative L2 error for this sample
            "abs_error": Tensor,    # per-vertex absolute error
            "pred": Tensor,         # flattened prediction
            "target": Tensor,       # flattened ground truth
            "points": Tensor,       # (N, 3) physical coordinates
            "boundary_mask": Tensor # bool mask for topological boundary vertices
        },
        ...
    ],
    "timing": {
        "inference_time_s": float,
        "peak_gpu_mb": float,
        "n_params": int,
    }
}
```

## Style Configuration

All plots share `plot_config.py` which defines:
- **Publication rcParams** (font sizes, DPI, tick direction, grid)
- **Colour palette** - colour-blind-safe, print-friendly
- **Model markers** and labels (consistent across all figures)
- **Topology ordering** - canonical `[spherical, toroidal, open_surface, high_genus]`

To enable LaTeX rendering, set `text.usetex: True` in `plot_config.py`.

## Directory Structure

```
plots/
+-- README.md                   <- This file
+-- __init__.py
+-- plot_config.py              <- Shared style + palette
+-- plot_genus_scalability.py   <- Fig 1
+-- plot_error_heatmaps.py      <- Fig 2
+-- plot_boundary_vs_global.py  <- Fig 3
+-- plot_robustness_violin.py   <- Fig 4
+-- plot_pareto_efficiency.py   <- Fig 5
+-- generate_all_plots.py       <- Master runner

scripts/
+-- evaluate_all.py             <- Produces results/eval/*.pt

results/
+-- eval/                       <- Raw evaluation data (.pt)
+-- figures/                    <- Generated figures (.pdf, .png)
```
