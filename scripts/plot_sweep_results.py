"""
plot_sweep_results.py

The punchline figure: does higher control frequency actually reduce tracking
error, and does that differ by control level? Reads results/sweep_results.csv
(produced by run_stage1_sweep.py) and plots control_hz on the x-axis against
tracking error at a chosen test oscillation frequency, one line per control
level (torque / joint_position / pwm).

This is a different cut than results/sweep_plot.png (which puts test
oscillation frequency on the x-axis, one line per grid point) -- this one
answers "does control frequency help" directly, which is the figure Ian
asked to see.

Usage:
    python scripts/plot_sweep_results.py \
        --summary results/sweep_results.csv \
        --test-hz 1 \
        --output figures/tracking_error_vs_control_hz.png
"""

import argparse
import os
import re

import pandas as pd
import matplotlib.pyplot as plt


def find_track_err_column(df: pd.DataFrame, test_hz: float) -> str:
    """Finds the track_err_hz<value> column closest to the requested test_hz."""
    candidates = []
    for col in df.columns:
        m = re.match(r"track_err_hz([\d.]+)$", col)
        if m:
            candidates.append((float(m.group(1)), col))
    if not candidates:
        raise ValueError(
            f"No track_err_hz* columns found in CSV. Columns present: {list(df.columns)}"
        )
    candidates.sort(key=lambda c: abs(c[0] - test_hz))
    return candidates[0][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default="results/sweep_results.csv")
    parser.add_argument("--test-hz", type=float, default=1.0,
                         help="which test oscillation frequency column to plot "
                              "(picks the closest available track_err_hz* column)")
    parser.add_argument("--output", type=str,
                         default="figures/tracking_error_vs_control_hz.png")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    col = find_track_err_column(df, args.test_hz)
    print(f"[plot_sweep_results] using column '{col}' for test_hz={args.test_hz}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"torque": "o", "joint_position": "s", "pwm": "^"}

    for control_level, group in df.groupby("control_level"):
        group = group.sort_values("control_hz")
        ax.plot(
            group["control_hz"], group[col],
            marker=markers.get(control_level, "o"), label=control_level,
        )

    ax.set_xlabel("Control frequency (Hz)")
    ax.set_ylabel(f"Mean tracking error (m) at test_osc_hz={args.test_hz}")
    ax.set_title("Tracking error vs. control frequency, by control level")
    ax.legend(title="Control level")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"[plot_sweep_results] wrote {args.output}")


if __name__ == "__main__":
    main()
