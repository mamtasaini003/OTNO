"""
Shared plotting configuration for TOPOS publication figures.

Provides publication-quality rcParams, a consistent colour palette keyed by
model name, and the canonical topology ordering used across every plot.

Usage in any plotting script:

    from plot_config import apply_style, MODEL_COLORS, MODEL_MARKERS, ...
    apply_style()
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────
# 1.  Publication rcParams (Nature / NeurIPS conventions)
# ──────────────────────────────────────────────────────────────────────
_RC = {
    # --- Font ---
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    # --- Layout ---
    "figure.figsize":       (3.5, 2.8),       # single-column default
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
    # --- Lines ---
    "lines.linewidth":      1.5,
    "lines.markersize":     5,
    # --- Axes ---
    "axes.linewidth":       0.8,
    "axes.grid":            True,
    "axes.grid.which":      "major",
    "grid.alpha":           0.25,
    "grid.linewidth":       0.5,
    # --- Ticks ---
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.minor.width":    0.4,
    "ytick.minor.width":    0.4,
    # --- Legend ---
    "legend.frameon":        True,
    "legend.framealpha":     0.85,
    "legend.edgecolor":      "0.7",
    # --- LaTeX ---
    "text.usetex":           False,          # set True when LaTeX is available
    "mathtext.fontset":      "cm",
}


def apply_style():
    """Apply publication rcParams globally."""
    mpl.rcParams.update(_RC)


# ──────────────────────────────────────────────────────────────────────
# 2.  Model colour palette (colour-blind-safe, print-friendly)
# ──────────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "TOPOS":    "#2166AC",   # strong blue
    "OTNO":     "#D6604D",   # subdued red
    "GINO":     "#4DAF4A",   # leaf green
    "DeepONet": "#984EA3",   # purple
    "U-FNO":    "#FF7F00",   # orange
    "FNO":      "#A65628",   # brown
}

MODEL_MARKERS = {
    "TOPOS":    "o",
    "OTNO":     "s",
    "GINO":     "^",
    "DeepONet": "D",
    "U-FNO":    "v",
    "FNO":      "X",
}

MODEL_LABELS = {
    "TOPOS":    "TOPOS (Ours)",
    "OTNO":     "OTNO",
    "GINO":     "GINO",
    "DeepONet": "DeepONet",
    "U-FNO":    "U-FNO",
    "FNO":      "FNO",
}

# ──────────────────────────────────────────────────────────────────────
# 3.  Topology / genus ordering
# ──────────────────────────────────────────────────────────────────────
TOPOLOGY_ORDER = ["spherical", "open_surface", "toroidal", "high_genus"]

TOPOLOGY_LABELS = {
    "spherical":    r"Genus 0 ($\chi\!=\!2$)",
    "toroidal":     r"Genus 1 ($\chi\!=\!0$)",
    "open_surface": r"Open ($\chi\!=\!1$)",
    "high_genus":   r"Genus 2+ ($\chi\!<\!0$)",
}

GENUS_FROM_TOPOLOGY = {
    "spherical":    0,
    "toroidal":     1,
    "open_surface": 0.5,   # disk-like, χ=1
    "high_genus":   2,
}

# ──────────────────────────────────────────────────────────────────────
# 4.  I/O defaults
# ──────────────────────────────────────────────────────────────────────
DEFAULT_EVAL_DIR = "results/eval"
DEFAULT_FIG_DIR  = "results/figures"

# ──────────────────────────────────────────────────────────────────────
# 5.  Utility helpers
# ──────────────────────────────────────────────────────────────────────
DOUBLE_COL_FIGSIZE = (7.0, 3.2)
SINGLE_COL_FIGSIZE = (3.5, 2.8)


def savefig(fig, path, **kw):
    """Save figure and print path for CI / pipeline visibility."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, **kw)
    print(f"[plot] Saved → {path}")
