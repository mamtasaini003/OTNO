#!/usr/bin/env python
"""
Plot 2 — Qualitative 3D Error Heatmaps

Multi-row × multi-column grid of 3D scatter renders.
  Rows    : Ground Truth, GINO, OTNO, TOPOS
  Columns : one shape per topology (spherical, toroidal, open, high-genus)

Each panel shows the **absolute error field** on the physical geometry
using a diverging coolwarm colourmap (blue = 0, dark red = large error).

Usage:
    python plots/plot_error_heatmaps.py [--eval_dir results/eval]
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import (
    apply_style,
    MODEL_COLORS,
    MODEL_LABELS,
    TOPOLOGY_ORDER,
    TOPOLOGY_LABELS,
    DEFAULT_EVAL_DIR,
    DEFAULT_FIG_DIR,
    savefig,
)

MODELS = ["GINO", "OTNO", "TOPOS"]
CMAP = "coolwarm"


def _load(eval_dir, model, dataset="mixed_genus"):
    path = os.path.join(eval_dir, f"{model.lower()}_{dataset}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=False)


def _pick_representative(per_sample, topology):
    """Return the sample closest to the median error for the given topology."""
    candidates = [s for s in per_sample if s["topology"] == topology]
    if not candidates:
        return None
    candidates.sort(key=lambda s: s["rel_l2"])
    return candidates[len(candidates) // 2]


def _render_3d(ax, points, values, vmin, vmax, title="", cmap=CMAP):
    """Render a 3D scatter plot on the given axes."""
    pts = np.asarray(points)
    vals = np.asarray(values)
    order = np.argsort(pts[:, 2])
    sc = ax.scatter(
        pts[order, 0], pts[order, 1], pts[order, 2],
        c=vals[order], cmap=cmap, vmin=vmin, vmax=vmax,
        s=4, alpha=0.75, edgecolors="none",
    )
    ax.set_title(title, fontsize=7, pad=2, rotation=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")
    ax.view_init(elev=25, azim=135)
    # Equal aspect
    r = 0.5 * max(pts[:, i].max() - pts[:, i].min() for i in range(3))
    cx = 0.5 * (pts[:, 0].max() + pts[:, 0].min())
    cy = 0.5 * (pts[:, 1].max() + pts[:, 1].min())
    cz = 0.5 * (pts[:, 2].max() + pts[:, 2].min())
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)
    return sc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default=DEFAULT_EVAL_DIR)
    parser.add_argument("--out_dir",  default=DEFAULT_FIG_DIR)
    parser.add_argument("--dataset",  default="mixed_genus",
                        choices=["mixed_genus", "thingi10k"])
    args = parser.parse_args()
    apply_style()

    # Load data
    model_data = {}
    for m in MODELS:
        d = _load(args.eval_dir, m, args.dataset)
        if d is not None:
            model_data[m] = d

    if not model_data:
        print("[!] No evaluation data found. Run scripts/evaluate_all.py first.")
        return

    # Pick representative samples for each topology and filter empty columns
    topos = []
    for topo in TOPOLOGY_ORDER:
        has_data = False
        for m in model_data:
            if _pick_representative(model_data[m]["per_sample"], topo) is not None:
                has_data = True
                break
        if has_data:
            topos.append(topo)
    
    n_cols = len(topos)
    rows_labels = ["Ground Truth"] + [MODEL_LABELS.get(m, m) for m in MODELS]
    n_rows = len(rows_labels)

    fig = plt.figure(figsize=(3.2 * n_cols, 2.4 * n_rows))

    # Use TOPOS data for ground truth row (all models share same dataset)
    ref_model = next(iter(model_data.keys()))
    ref_data = model_data[ref_model]

    # Record target idxs first
    target_idxs = {}
    for topo in topos:
        gt_sample = _pick_representative(ref_data["per_sample"], topo)
        if gt_sample is not None:
            target_idxs[topo] = gt_sample["idx"]

    # Pre-compute global vmax for consistent colorbars
    global_vmax = 0.0
    for m in MODELS:
        if m not in model_data:
            continue
        for topo in topos:
            t_idx = target_idxs.get(topo)
            s = next((item for item in model_data[m]["per_sample"] if item["idx"] == t_idx), None)
            if s is not None:
                global_vmax = max(global_vmax, float(s["abs_error"].max()))
    global_vmax = min(global_vmax, 1.5)  # cap for visual clarity

    for col_idx, topo in enumerate(topos):
        # --- Ground Truth row ---
        gt_sample = _pick_representative(ref_data["per_sample"], topo)
        if gt_sample is None:
            continue
            
        target_idx = gt_sample["idx"]

        ax = fig.add_subplot(n_rows, n_cols, col_idx + 1, projection="3d")
        _render_3d(
            ax, gt_sample["points"], gt_sample["target"],
            vmin=float(gt_sample["target"].min()),
            vmax=float(gt_sample["target"].max()),
            title=TOPOLOGY_LABELS.get(topo, topo) if col_idx < n_cols else "",
            cmap="viridis",
        )
        if col_idx == 0:
            ax.text2D(-0.2, 0.5, "Ground Truth", transform=ax.transAxes, fontsize=8, va="center", ha="right", rotation=0)

        # --- Model rows (absolute error) ---
        for row_offset, m in enumerate(MODELS, start=1):
            if m not in model_data:
                continue
            # Pick the exact same sample based on idx
            s = next((item for item in model_data[m]["per_sample"] if item["idx"] == target_idx), None)
            if s is None:
                continue
            ax = fig.add_subplot(n_rows, n_cols, row_offset * n_cols + col_idx + 1, projection="3d")
            sc = _render_3d(
                ax, s["points"], s["abs_error"],
                vmin=0.0, vmax=global_vmax,
                title="" if row_offset > 0 else TOPOLOGY_LABELS.get(topo, topo),
            )
            if col_idx == 0:
                ax.text2D(-0.2, 0.5, MODEL_LABELS.get(m, m), transform=ax.transAxes, fontsize=8, va="center", ha="right", rotation=0)

    # Shared colourbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.55])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, global_vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Absolute Error")

    fig.subplots_adjust(left=0.15, right=0.90, bottom=0.05, top=0.95, wspace=0.1, hspace=0.15)
    savefig(fig, os.path.join(args.out_dir, f"fig2_error_heatmap_{args.dataset}.pdf"))
    savefig(fig, os.path.join(args.out_dir, f"fig2_error_heatmap_{args.dataset}.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
