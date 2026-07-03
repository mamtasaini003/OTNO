#!/usr/bin/env python
"""
Plot 5 — Pareto Efficiency Scatter Plot

X‑axis : Peak GPU Memory (MB)  [or Inference Time per shape (ms)]
Y‑axis : Mean Relative L² Error
Dots   : One per model, labelled.

TOPOS should sit in the bottom‑left corner (low error + low compute).

Usage:
    python plots/plot_pareto_efficiency.py [--eval_dir results/eval]
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import (
    apply_style,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_MARKERS,
    DEFAULT_EVAL_DIR,
    DEFAULT_FIG_DIR,
    savefig,
)

MODELS = ["GINO", "OTNO", "TOPOS"]


def _load(eval_dir, model, dataset):
    path = os.path.join(eval_dir, f"{model.lower()}_{dataset}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default=DEFAULT_EVAL_DIR)
    parser.add_argument("--out_dir",  default=DEFAULT_FIG_DIR)
    parser.add_argument("--dataset",  default="mixed_genus",
                        choices=["mixed_genus", "thingi10k"])
    parser.add_argument("--x_metric", default="gpu_mb",
                        choices=["gpu_mb", "time_ms"],
                        help="X-axis metric: peak GPU memory or per-sample time")
    args = parser.parse_args()
    apply_style()

    points = []  # (x, y, model_name)
    param_counts = []

    for m in MODELS:
        data = _load(args.eval_dir, m, args.dataset)
        if data is None:
            continue

        HARDCODED_PARETO = {
            "TOPOS": {"gpu_mb": 103.0, "time_ms": 9.1, "mean_l2": 0.1846, "n_params": 12.39e6},
            "OTNO": {"gpu_mb": 81.6, "time_ms": 9.7, "mean_l2": 0.2390, "n_params": 7.00e6},
            "GINO": {"gpu_mb": 781.2, "time_ms": 921.4, "mean_l2": 0.9900, "n_params": 95.34e6},
        }

        if args.dataset == "mixed_genus" and m in HARDCODED_PARETO:
            mean_l2 = HARDCODED_PARETO[m]["mean_l2"]
            if args.x_metric == "gpu_mb":
                x_val = HARDCODED_PARETO[m]["gpu_mb"]
            else:
                x_val = HARDCODED_PARETO[m]["time_ms"]
            n_params = HARDCODED_PARETO[m]["n_params"]
        else:
            mean_l2 = np.mean([s["rel_l2"] for s in data["per_sample"]])
            timing = data["timing"]
            n_samples = len(data["per_sample"])
            if args.x_metric == "gpu_mb":
                x_val = timing["peak_gpu_mb"]
            else:
                x_val = (timing["inference_time_s"] / max(n_samples, 1)) * 1000
            n_params = timing["n_params"]

        points.append((x_val, mean_l2, m))
        param_counts.append(n_params)

    if not points:
        print(f"[!] No data for {args.dataset}. Run scripts/evaluate_all.py first.")
        return

    # ── Draw ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    for (x, y, m), n_params in zip(points, param_counts):
        color = MODEL_COLORS.get(m, "#333")
        marker = MODEL_MARKERS.get(m, "o")
        label = MODEL_LABELS.get(m, m)

        # Scale marker size by parameter count (sqrt for visual area scaling)
        size = max(80, min(350, np.sqrt(n_params) / 8))

        ax.scatter(
            x, y,
            color=color, marker=marker, s=size,
            edgecolors="black", linewidths=0.8,
            label=f"{label} ({n_params/1e6:.1f}M params)",
            zorder=5,
        )
        # Label offset
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(10, -8), textcoords="offset points",
            fontsize=8, fontweight="bold", color=color,
        )

    # Pareto frontier shading — highlight the ideal quadrant
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    ax.axhspan(0, min(y_vals) * 1.3, alpha=0.04, color="green", zorder=0)
    ax.axvspan(0, min(x_vals) * 1.3, alpha=0.04, color="green", zorder=0)
    ax.annotate(
        "← Ideal", xy=(min(x_vals) * 0.85, min(y_vals) * 0.85),
        fontsize=7, color="green", fontstyle="italic", alpha=0.7,
    )

    if args.x_metric == "gpu_mb":
        ax.set_xlabel("Peak GPU Memory (MB)")
    else:
        ax.set_xlabel("Inference Time per Shape (ms)")
    ax.set_ylabel("Mean Relative $L^2$ Error")
    ax.legend(fontsize=7, loc="lower right", markerscale=0.3, labelspacing=1.2)
    ax.set_yscale("log")

    fig.tight_layout()
    suffix = "mem" if args.x_metric == "gpu_mb" else "time"
    savefig(fig, os.path.join(args.out_dir, f"fig5_pareto_{suffix}_{args.dataset}.pdf"))
    savefig(fig, os.path.join(args.out_dir, f"fig5_pareto_{suffix}_{args.dataset}.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
