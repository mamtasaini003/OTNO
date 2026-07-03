#!/usr/bin/env python
"""
Master script — generate all five TOPOS publication figures.

Reads pre-computed evaluation results from ``results/eval/`` and
produces PDF + PNG figures in ``results/figures/``.

Usage:
    # 1. First run evaluation (needs GPU):
    python scripts/evaluate_all.py --config configs/mixed_genus_fair_comparison.yaml \
        --models gino otno topos --datasets mixed_genus --gpus 0

    python scripts/evaluate_all.py --config configs/thingi10k_topos.yaml \
        --models topos otno --datasets thingi10k --gpus 0

    # 2. Then generate all plots (CPU only):
    python plots/generate_all_plots.py
"""

import argparse
import os
import subprocess
import sys

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    ("Plot 1 — Genus Scalability",      "plot_genus_scalability.py"),
    ("Plot 2 — Error Heatmaps (Mixed)",  "plot_error_heatmaps.py --dataset mixed_genus"),
    ("Plot 2 — Error Heatmaps (Thingi)", "plot_error_heatmaps.py --dataset thingi10k"),
    ("Plot 3 — Boundary vs Global",      "plot_boundary_vs_global.py --dataset both"),
    ("Plot 4 — Robustness Violin",       "plot_robustness_violin.py --dataset thingi10k"),
    ("Plot 4 — Robustness Violin (Mixed)", "plot_robustness_violin.py --dataset mixed_genus"),
    ("Plot 5 — Pareto (GPU Memory)",     "plot_pareto_efficiency.py --x_metric gpu_mb"),
    ("Plot 5 — Pareto (Time)",           "plot_pareto_efficiency.py --x_metric time_ms"),
]


def main():
    parser = argparse.ArgumentParser(description="Generate all TOPOS paper figures")
    parser.add_argument("--eval_dir", default="results/eval",
                        help="Directory containing evaluation .pt files")
    parser.add_argument("--out_dir",  default="results/figures",
                        help="Output directory for figures")
    parser.add_argument("--skip_missing", action="store_true",
                        help="Skip plots whose eval data is missing instead of erroring")
    args = parser.parse_args()

    print("=" * 64)
    print("  TOPOS Paper — Generating All Figures")
    print("=" * 64)
    print(f"  eval_dir : {args.eval_dir}")
    print(f"  out_dir  : {args.out_dir}")
    print()

    n_ok, n_fail, n_skip = 0, 0, 0

    for title, script_args in SCRIPTS:
        script_name = script_args.split()[0]
        extra_args = script_args.split()[1:]
        script_path = os.path.join(PLOT_DIR, script_name)

        if not os.path.exists(script_path):
            print(f"  [SKIP] {title} — script not found: {script_path}")
            n_skip += 1
            continue

        cmd = [
            sys.executable, script_path,
            "--eval_dir", args.eval_dir,
            "--out_dir",  args.out_dir,
        ] + extra_args

        print(f"  [{title}]")
        print(f"    → {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                # Print any "[plot] Saved" lines
                for line in result.stdout.splitlines():
                    if "[plot]" in line or "[!]" in line:
                        print(f"    {line}")
                n_ok += 1
            else:
                print(f"    [FAIL] Return code {result.returncode}")
                if result.stderr:
                    for line in result.stderr.strip().splitlines()[-5:]:
                        print(f"    stderr: {line}")
                if args.skip_missing:
                    n_skip += 1
                else:
                    n_fail += 1
        except subprocess.TimeoutExpired:
            print(f"    [FAIL] Timed out (120s)")
            n_fail += 1
        except Exception as e:
            print(f"    [FAIL] {e}")
            n_fail += 1

    print()
    print("=" * 64)
    print(f"  Done: {n_ok} succeeded, {n_fail} failed, {n_skip} skipped")
    print(f"  Figures saved to: {args.out_dir}/")
    print("=" * 64)

    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
