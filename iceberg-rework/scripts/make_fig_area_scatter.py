"""
make_fig_area_scatter.py: Reference vs predicted iceberg area scatter, one
panel per method, with Pearson r, linear fit, and 1:1 line. The Fisser 2024
Fig. 6 + Fisser 2025 Fig. 10 analog for our six-method optical sweep.

Reads:
  <run>/per_iceberg/eval_per_iceberg.csv  (per-pair rows from Hungarian match)

Writes (via _fig_registry):
  <run>/per_iceberg/fig-archive/<ts>__area_scatter_by_method.png
  <run>/per_iceberg/figures.md (row appended or updated)
  <run>/per_iceberg/area_scatter_fits.csv (per-method slope, intercept, r, n)

Usage:
  python scripts/make_fig_area_scatter.py \
      --run /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/runs/exp_baseline_v1/20260424_185158

Rsync after edit:
  rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/make_fig_area_scatter.py \
      llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from _fig_registry import write as write_fig

DEFAULT_RUN = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/runs/exp_baseline_v1/20260424_185158"

METHOD_ORDER = ["TR", "OT", "UNet", "UNet_TR", "UNet_OT", "UNet_CRF"]
METHOD_COLORS = {
    "TR":       "#9E9E9E",
    "OT":       "#607D8B",
    "UNet":     "#1976D2",
    "UNet_TR":  "#43A047",
    "UNet_OT":  "#F57C00",
    "UNet_CRF": "#D81B60",
}

SZA_ORDER = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]
SZA_PANEL_COLORS = {"sza_lt65": "#1976D2", "sza_65_70": "#43A047",
                    "sza_70_75": "#F57C00", "sza_gt75": "#D81B60"}


def fit_linear(x, y):
    """Return slope, intercept, Pearson r, n_used. Robust to <2 samples."""
    if len(x) < 2:
        return np.nan, np.nan, np.nan, len(x)
    slope, intercept = np.polyfit(x, y, 1)
    r, _ = pearsonr(x, y)
    return float(slope), float(intercept), float(r), int(len(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    args = parser.parse_args()

    pairs_path = os.path.join(args.run, "per_iceberg", "eval_per_iceberg.csv")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(pairs_path)
    df = pd.read_csv(pairs_path)
    required = {"method", "sza_bin", "gt_area_m2", "pred_area_m2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {pairs_path}: {missing}")

    # 1. Compute per-method linear fit + Pearson r
    fit_rows = []
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        slope, intercept, r, n = fit_linear(
            sub["gt_area_m2"].values, sub["pred_area_m2"].values,
        )
        fit_rows.append({
            "method": method, "n": n, "slope": slope,
            "intercept_m2": intercept, "pearson_r": r,
        })

    # 2. Persist fit table next to the figure
    fit_csv = os.path.join(args.run, "per_iceberg", "area_scatter_fits.csv")
    with open(fit_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "n", "slope",
                                                "intercept_m2", "pearson_r"])
        writer.writeheader()
        for row in fit_rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items()})
    print(f"Wrote {fit_csv}")

    # 3. Set shared axis limits across panels for cross-method comparability
    upper = float(np.percentile(
        np.concatenate([df["gt_area_m2"].values, df["pred_area_m2"].values]),
        99.0,
    ))
    upper = max(upper, 1e4)
    axis_max = float(np.ceil(upper / 5e4)) * 5e4

    # 4. Plot 2x3 panel grid, one method per panel
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5),
                              sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for ax, method, fit in zip(axes_flat, METHOD_ORDER, fit_rows):
        sub = df[df["method"] == method]
        # Color points by SZA bin so cross-bin behaviour is visible
        for sza in SZA_ORDER:
            ssub = sub[sub["sza_bin"] == sza]
            if ssub.empty:
                continue
            ax.scatter(ssub["gt_area_m2"], ssub["pred_area_m2"],
                       s=8, alpha=0.35, color=SZA_PANEL_COLORS[sza],
                       edgecolors="none", label=sza.replace("sza_", ""))

        # 1:1 line
        ax.plot([0, axis_max], [0, axis_max],
                color="black", linewidth=0.8, linestyle="--", alpha=0.7)

        # Linear fit line
        if np.isfinite(fit["slope"]):
            xs = np.array([0, axis_max])
            ys = fit["slope"] * xs + fit["intercept_m2"]
            ax.plot(xs, ys, color=METHOD_COLORS[method], linewidth=1.6,
                    alpha=0.9, label="linear fit")

        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3, linestyle="--")
        # Panel title above the axes so it does not collide with the inset
        # stat box in the lower-right.
        ax.set_title(method, fontsize=11, fontweight="bold",
                     color=METHOD_COLORS[method], pad=6)
        ax.text(0.97, 0.04,
                f"n = {fit['n']:,}\nr = {fit['pearson_r']:.3f}\n"
                f"slope = {fit['slope']:.3f}\nintercept = {fit['intercept_m2']:,.0f} m$^2$",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#222",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="#ccc", alpha=0.85))

    # 5. Common axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Reference iceberg area $A_{S2}$ (m$^2$)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Predicted iceberg area $A_{method}$ (m$^2$)")

    # 6. Legend (SZA color key + 1:1 line) once for the figure
    handles = [plt.Line2D([0], [0], marker="o", linestyle="none",
                          markersize=7, alpha=0.6,
                          markerfacecolor=SZA_PANEL_COLORS[s],
                          markeredgecolor="none",
                          label=s.replace("sza_", ""))
               for s in SZA_ORDER]
    handles.append(plt.Line2D([0], [0], color="black",
                              linestyle="--", linewidth=0.8,
                              label="1:1 line"))
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Predicted vs reference iceberg area, per method",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.965])

    archive = write_fig(
        fig,
        slug="area_scatter_by_method",
        caption=(
            "Per-pair predicted vs reference iceberg area, one panel per "
            "method, with the 1:1 line dashed and the linear fit colored "
            "to match the panel title. Points are colored by SZA bin. "
            "Per-panel n, Pearson r, slope, and intercept are inset. "
            "Fisser (2024) Fig. 6 / Fisser (2025) Fig. 10 analog."
        ),
        out_dir=os.path.join(args.run, "per_iceberg"),
    )
    plt.close(fig)
    print(f"Wrote {archive}")

    # 7. Print fit summary to stdout for quick paper text reference
    print("\nPer-method fit summary:")
    print(f"{'method':<10}{'n':>7}{'slope':>10}{'intercept (m^2)':>18}{'r':>10}")
    for row in fit_rows:
        print(f"{row['method']:<10}{row['n']:>7,}"
              f"{row['slope']:>10.3f}{row['intercept_m2']:>18,.0f}"
              f"{row['pearson_r']:>10.3f}")


if __name__ == "__main__":
    main()
