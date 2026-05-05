"""
make_fig_bias_by_area.py: Bias and Delta-Bias by reference area bin and SZA
bin. 2 rows (top: per-pair Bias in m^2 with P10-P90 band; bottom: Delta-Bias
relative to the TR baseline) x 4 cols (one column per SZA bin). Companion to
Fig. 7 / Fig. 9 (per-SZA panel layout). Reveals the size-dependent
overestimation regime that all threshold methods inherit and that UNet++
variants partially correct, stratified by SZA bin so SZA-dependent behaviour
is visible without color decoding.

Bias    = pred_area - gt_area (per-pair, in m^2)
Delta-Bias = method_Bias - TR_Bias (TR is the simplest baseline; analog of
          Fisser CFAR). Negative Delta-Bias means smaller absolute bias than
          the TR baseline at that area bin.

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

# Harmonized with Fig. 7 / Fig. 9: cool blue-greys for non-learning (TR, OT);
# learning methods get UNet light orange, UNet_TR pink, UNet_OT red,
# UNet_CRF bright purple so all four are clearly distinguishable.
METHOD_COLORS = {
    "TR":       "#90A4AE",
    "OT":       "#37474F",
    "UNet":     "#FFB74D",
    "UNet_TR":  "#E91E63",
    "UNet_OT":  "#E64A19",
    "UNet_CRF": "#7C4DFF",
}
METHOD_MARKERS = {
    "TR":       "o",
    "OT":       "s",
    "UNet":     "D",
    "UNet_TR":  "^",
    "UNet_OT":  "v",
    "UNet_CRF": "P",
}

SZA_ORDER = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]
SZA_LABELS = {
    "sza_lt65":  r"SZA $<$ 65$^{\circ}$",
    "sza_65_70": r"SZA 65--70$^{\circ}$",
    "sza_70_75": r"SZA 70--75$^{\circ}$",
    "sza_gt75":  r"SZA $>$ 75$^{\circ}$",
}

# Fisser 2025 area bin lower edges (m^2). Matches their Fig 16 x-axis.
AREA_BIN_EDGES = [1600, 23700, 45800, 67900, 90000, np.inf]
AREA_BIN_LABELS = ["1.6k", "23.7k", "45.8k", "67.9k", ">=90k"]

BASELINE_METHOD = "TR"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN,
                        help="Path to a baseline_v1 run dir; reads "
                             "<run>/per_iceberg/eval_per_iceberg.csv.")
    args = parser.parse_args()

    # 1. Load per-pair table
    pairs_path = os.path.join(args.run, "per_iceberg", "eval_per_iceberg.csv")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(pairs_path)
    df = pd.read_csv(pairs_path)

    # 2. Validate schema
    required = {"method", "sza_bin", "gt_area_m2", "pred_area_m2"}
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

    # 4. Aggregate per (method, sza_bin, area_bin)
    grouped = df.groupby(
        ["method", "sza_bin", "area_bin"], observed=False,
    )["bias_m2"]
    bias_mean = grouped.mean().reset_index(name="mean")
    bias_p10 = grouped.quantile(0.10).reset_index(name="p10")
    bias_p90 = grouped.quantile(0.90).reset_index(name="p90")
    n_pairs = (
        df.groupby(["method", "sza_bin", "area_bin"], observed=False)
        .size().reset_index(name="n")
    )
    agg = bias_mean.merge(bias_p10).merge(bias_p90).merge(n_pairs)

    # 5. Compute Delta-Bias per (method, sza_bin, area_bin) relative to TR
    if BASELINE_METHOD not in agg["method"].unique():
        raise ValueError(
            f"baseline method {BASELINE_METHOD} missing from per-pair table",
        )
    baseline = (
        agg[agg["method"] == BASELINE_METHOD]
        .set_index(["sza_bin", "area_bin"])["mean"]
    )
    agg["delta"] = agg.apply(
        lambda r: r["mean"] - baseline.get((r["sza_bin"], r["area_bin"]),
                                           np.nan),
        axis=1,
    )

    # 6. 2x4 grid: rows = (Bias, Delta-Bias), cols = SZA bins. Share y
    # within each row so cross-SZA comparison reads off the same scale.
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5),
                             sharex=True, sharey="row")
    x = np.arange(len(AREA_BIN_LABELS))

    # 6a. Top row: per-pair Bias by area_bin, with P10-P90 band
    for ci, sza in enumerate(SZA_ORDER):
        ax = axes[0, ci]
        for method in METHOD_ORDER:
            sub = (
                agg[(agg["method"] == method) & (agg["sza_bin"] == sza)]
                .set_index("area_bin").reindex(AREA_BIN_LABELS)
            )
            if sub["mean"].isna().all():
                continue
            color = METHOD_COLORS[method]
            ax.fill_between(x, sub["p10"].values, sub["p90"].values,
                            color=color, alpha=0.08)
            ax.plot(x, sub["mean"].values,
                    marker=METHOD_MARKERS[method], color=color,
                    linewidth=1.7, markersize=6.5, label=method)
        ax.axhline(0, color="black", linewidth=0.9, linestyle="-",
                   alpha=0.65)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_title(SZA_LABELS[sza], fontsize=14, fontweight="bold", pad=6)
        ax.tick_params(axis="both", labelsize=11)

    # 6b. Bottom row: Delta-Bias relative to TR baseline (skip TR)
    for ci, sza in enumerate(SZA_ORDER):
        ax = axes[1, ci]
        for method in METHOD_ORDER:
            if method == BASELINE_METHOD:
                continue
            sub = (
                agg[(agg["method"] == method) & (agg["sza_bin"] == sza)]
                .set_index("area_bin").reindex(AREA_BIN_LABELS)
            )
            if sub["delta"].isna().all():
                continue
            color = METHOD_COLORS[method]
            ax.plot(x, sub["delta"].values,
                    marker=METHOD_MARKERS[method], color=color,
                    linewidth=1.7, markersize=6.5, label=method)
        ax.axhline(0, color="black", linewidth=0.9, linestyle="-",
                   alpha=0.65)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(AREA_BIN_LABELS, fontsize=11)
        ax.tick_params(axis="y", labelsize=11)

    # 6c. Per-(SZA, area_bin) total n_pairs annotated near top of bias panels
    for ci, sza in enumerate(SZA_ORDER):
        ax = axes[0, ci]
        n_total = (
            agg[agg["sza_bin"] == sza]
            .groupby("area_bin", observed=False)["n"].sum()
            .reindex(AREA_BIN_LABELS)
        )
        for i, lbl in enumerate(AREA_BIN_LABELS):
            ax.text(i, 0.97, f"n={int(n_total[lbl]):,}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, color="#555")

    # 7. Row labels and shared axes
    axes[0, 0].set_ylabel(r"Bias $A_{pred} - A_{ref}$ (m$^2$)", fontsize=13)
    axes[1, 0].set_ylabel(
        r"$\Delta$Bias vs " + BASELINE_METHOD + r" (m$^2$)", fontsize=13,
    )
    for ax in axes[1, :]:
        ax.set_xlabel(r"Reference area bin (m$^2$, lower edge)", fontsize=13)

    # 8. Figure-level legend (bottom)
    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[m], linewidth=2.0,
                   marker=METHOD_MARKERS[m], markersize=7, label=m)
        for m in METHOD_ORDER
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        "Per-pair area bias and $\\Delta$Bias by reference area bin, per SZA bin",
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.965])

    # 9. Route through fig_registry
    out_dir = os.path.join(args.run, "per_iceberg")
    archive = write_fig(
        fig,
        slug="bias_delta_by_area",
        caption=(
            "Top row: per-pair bias (predicted minus reference iceberg "
            "area) by reference area bin and method, one panel per SZA bin, "
            "with the P10-P90 band shaded. Bottom row: Delta-Bias relative "
            "to the TR threshold baseline, same panel layout. Area bin "
            "edges (1.6k, 23.7k, 45.8k, 67.9k, 90k m^2) match Fisser (2025) "
            "Fig. 16. Color palette matches Fig. 7 (UNet_TR pink). "
            "Per-(SZA, area-bin) total n_pairs annotated near the top of "
            "each bias panel."
        ),
        out_dir=out_dir,
    )
    plt.close(fig)
    print(f"Wrote {archive}")


if __name__ == "__main__":
    main()
