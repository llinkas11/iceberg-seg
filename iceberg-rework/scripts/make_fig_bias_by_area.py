"""
make_fig_bias_by_area.py: Bias and Delta-Bias by reference area bin, one line
per method. The Fisser 2025 Fig 16 analog. Helps reveal the size-dependent
overestimation regime that all threshold methods inherit and that UNet++
variants partially correct.

Bias    = pred_area - gt_area (per-pair, in m^2)
Delta-Bias = method_Bias - TR_Bias (TR is our simplest baseline; analog of
          Fisser CFAR). Negative Delta-Bias means smaller absolute bias than the
          TR baseline at that area bin.

Reads:
  <run>/per_iceberg/eval_per_iceberg.csv

Writes (via _fig_registry):
  <run>/per_iceberg/fig-archive/<ts>__bias_delta_by_area.png
  <run>/per_iceberg/figures.md (row appended or updated)

Usage:
  python scripts/make_fig_bias_by_area.py \
      --run /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/runs/exp_baseline_v1/20260424_185158

Rsync after edit:
  rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/make_fig_bias_by_area.py \
      llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
METHOD_MARKERS = {
    "TR":       "o",
    "OT":       "s",
    "UNet":     "D",
    "UNet_TR":  "^",
    "UNet_OT":  "v",
    "UNet_CRF": "P",
}

# Fisser 2025 area bin lower edges (m^2). Matches their Fig 16 x-axis.
AREA_BIN_EDGES = [1600, 23700, 45800, 67900, 90000, np.inf]
AREA_BIN_LABELS = ["1.6k", "23.7k", "45.8k", "67.9k", ">=90k"]

BASELINE_METHOD = "TR"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN,
                        help="Path to a baseline_v1 run dir; reads <run>/per_iceberg/eval_per_iceberg.csv.")
    args = parser.parse_args()

    # 1. Load per-pair table
    pairs_path = os.path.join(args.run, "per_iceberg", "eval_per_iceberg.csv")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(pairs_path)
    df = pd.read_csv(pairs_path)

    # 2. Validate schema
    required = {"method", "gt_area_m2", "pred_area_m2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {pairs_path}: {missing}")

    # 3. Compute per-pair bias and assign Fisser 2025 area bin
    df = df.copy()
    df["bias_m2"] = df["pred_area_m2"] - df["gt_area_m2"]
    df["area_bin"] = pd.cut(
        df["gt_area_m2"], bins=AREA_BIN_EDGES, labels=AREA_BIN_LABELS,
        right=False, include_lowest=True,
    )

    # 4. Aggregate mean bias per (method, area_bin)
    grouped = df.groupby(["method", "area_bin"], observed=False)["bias_m2"]
    bias_mean = grouped.mean().unstack().reindex(METHOD_ORDER)[AREA_BIN_LABELS]
    bias_p10 = grouped.quantile(0.10).unstack().reindex(METHOD_ORDER)[AREA_BIN_LABELS]
    bias_p90 = grouped.quantile(0.90).unstack().reindex(METHOD_ORDER)[AREA_BIN_LABELS]
    n_pairs = df.groupby(["method", "area_bin"], observed=False).size().unstack().reindex(METHOD_ORDER)[AREA_BIN_LABELS]

    # 5. Compute Delta-Bias relative to TR baseline
    if BASELINE_METHOD not in bias_mean.index:
        raise ValueError(f"baseline method {BASELINE_METHOD} missing from per-pair table")
    delta = bias_mean.subtract(bias_mean.loc[BASELINE_METHOD], axis=1)

    # 6. Two-row plot: top = Bias with P10/P90 band, bottom = Delta-Bias
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8), sharex=True)
    x = np.arange(len(AREA_BIN_LABELS))

    ax_top = axes[0]
    for method in METHOD_ORDER:
        if method not in bias_mean.index:
            continue
        y = bias_mean.loc[method].values
        lo = bias_p10.loc[method].values
        hi = bias_p90.loc[method].values
        ax_top.fill_between(x, lo, hi, color=METHOD_COLORS[method], alpha=0.10)
        ax_top.plot(x, y, marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                    linewidth=1.8, markersize=7, label=method)
    ax_top.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_top.set_ylabel(r"Bias $A_{pred} - A_{ref}$ (m$^2$)")
    ax_top.set_title("Per-pair area bias by reference area bin and method",
                     fontsize=12, fontweight="bold")
    ax_top.grid(True, alpha=0.3, linestyle="--")
    ax_top.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.95)

    ax_bot = axes[1]
    for method in METHOD_ORDER:
        if method == BASELINE_METHOD or method not in delta.index:
            continue
        y = delta.loc[method].values
        ax_bot.plot(x, y, marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                    linewidth=1.8, markersize=7, label=method)
    ax_bot.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6,
                   label=f"{BASELINE_METHOD} (baseline)")
    ax_bot.set_xlabel(r"Reference iceberg area bin (m$^2$, lower edge shown)")
    ax_bot.set_ylabel(r"$\Delta$Bias vs " + BASELINE_METHOD + r" (m$^2$)")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(AREA_BIN_LABELS)
    ax_bot.grid(True, alpha=0.3, linestyle="--")
    ax_bot.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.95)

    # Annotate per-bin pair counts (sum across methods that aren't baseline)
    n_total = n_pairs.sum(axis=0)
    for i, lbl in enumerate(AREA_BIN_LABELS):
        ax_top.text(i, ax_top.get_ylim()[1] * 0.97, f"n={int(n_total[lbl]):,}",
                    ha="center", va="top", fontsize=8, color="#555")

    fig.tight_layout()

    # 7. Route through fig_registry
    out_dir = os.path.join(args.run, "per_iceberg")
    archive = write_fig(
        fig,
        slug="bias_delta_by_area",
        caption=(
            "Top: per-pair bias (predicted minus reference iceberg area) by "
            "reference area bin and method, with the P10-P90 band shaded. "
            "Bottom: Delta-Bias relative to the TR threshold baseline. Area "
            "bin edges (1.6k, 23.7k, 45.8k, 67.9k, 90k m^2) match Fisser "
            "(2025) Fig. 16. Per-bin total n_pairs are annotated."
        ),
        out_dir=out_dir,
    )
    plt.close(fig)
    print(f"Wrote {archive}")


if __name__ == "__main__":
    main()
