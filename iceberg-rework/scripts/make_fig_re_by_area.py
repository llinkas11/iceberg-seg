"""
make_fig_re_by_area.py: Per-pair relative-error (RE %) by reference-area bin
and SZA bin. 2x2 panel grid, one panel per SZA bin, with all six methods
overlaid as mean line + faint P10-P90 band per panel. Companion to Fig. 7
(area scatter); the per-SZA facet here mirrors that figure's layout so
cross-figure comparisons are immediate. Color palette matches Fig. 7
(UNet_TR pink so the share-a-substring methods stay distinguishable).

Reads:
  <run>/per_iceberg/eval_per_iceberg.csv

Writes (via _fig_registry):
  <run>/per_iceberg/fig-archive/<ts>__re_by_area_bin.png
  <run>/per_iceberg/figures.md (row appended or updated)
  <run>/per_iceberg/re_by_area_bin.csv (per-method per-(sza, area) aggregates)

Usage:
  python scripts/make_fig_re_by_area.py \
      --run /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/runs/exp_baseline_v1/20260424_185158

Rsync after edit:
  rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/make_fig_re_by_area.py \
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

from _fig_registry import write as write_fig

DEFAULT_RUN = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/runs/exp_baseline_v1/20260424_185158"

METHOD_ORDER = ["TR", "OT", "UNet", "UNet_TR", "UNet_OT", "UNet_CRF"]

# Palette harmonized with Fig. 7: cool blue-greys for non-learning
# (TR, OT); learning methods get UNet light orange, UNet_TR pink, UNet_OT
# red, UNet_CRF bright purple so all four are clearly distinguishable.
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

# Fisser 2025 area bin lower edges (m^2). Matches Fig. 12 + 16 x-axis.
AREA_BIN_EDGES = [1600, 23700, 45800, 67900, 90000, np.inf]
AREA_BIN_LABELS = ["1.6k", "23.7k", "45.8k", "67.9k", ">=90k"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN,
                        help="Path to a baseline_v1 run dir; reads "
                             "<run>/per_iceberg/eval_per_iceberg.csv.")
    args = parser.parse_args()

    # 1. Load per-pair CSV
    pairs_path = os.path.join(args.run, "per_iceberg", "eval_per_iceberg.csv")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(pairs_path)
    df = pd.read_csv(pairs_path)
    required = {"method", "sza_bin", "gt_area_m2", "re_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {pairs_path}: {missing}")

    # 2. Bin pairs by reference area; drop NaN re_pct (zero-area GT)
    df = df.dropna(subset=["re_pct"]).copy()
    df["area_bin"] = pd.cut(
        df["gt_area_m2"], bins=AREA_BIN_EDGES, labels=AREA_BIN_LABELS,
        right=False, include_lowest=True,
    )

    # 3. Aggregate mean / P10 / P25 / P75 / P90 per (method, sza_bin, area_bin)
    agg = df.groupby(
        ["method", "sza_bin", "area_bin"], observed=False
    )["re_pct"].agg(
        mean="mean",
        p10=lambda s: float(np.percentile(s, 10)) if len(s) else np.nan,
        p25=lambda s: float(np.percentile(s, 25)) if len(s) else np.nan,
        p75=lambda s: float(np.percentile(s, 75)) if len(s) else np.nan,
        p90=lambda s: float(np.percentile(s, 90)) if len(s) else np.nan,
        n="size",
    ).reset_index()

    # 4. Persist per-method per-(sza, area) aggregates
    csv_path = os.path.join(args.run, "per_iceberg", "re_by_area_bin.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "sza_bin", "area_bin", "n", "mean_re_pct",
            "p10_re_pct", "p25_re_pct", "p75_re_pct", "p90_re_pct",
        ])
        for _, r in agg.iterrows():
            writer.writerow([
                r["method"], r["sza_bin"], r["area_bin"], int(r["n"]),
                f"{r['mean']:.3f}",
                f"{r['p10']:.3f}", f"{r['p25']:.3f}",
                f"{r['p75']:.3f}", f"{r['p90']:.3f}",
            ])
    print(f"Wrote {csv_path}")

    # 5. Plot 2x2 panels, one per SZA bin
    fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()
    x = np.arange(len(AREA_BIN_LABELS))

    for ax, sza in zip(axes_flat, SZA_ORDER):
        for method in METHOD_ORDER:
            sub = (
                agg[(agg["method"] == method) & (agg["sza_bin"] == sza)]
                .set_index("area_bin")
                .reindex(AREA_BIN_LABELS)
            )
            if sub["mean"].isna().all():
                continue
            means = sub["mean"].values
            p10s = sub["p10"].values
            p90s = sub["p90"].values
            color = METHOD_COLORS[method]
            # Faint P10-P90 band per method so it does not dominate
            ax.fill_between(x, p10s, p90s, color=color, alpha=0.05)
            # Mean line + per-method marker
            ax.plot(x, means, marker=METHOD_MARKERS[method], color=color,
                    linewidth=1.8, markersize=6.5, label=method)

        ax.axhline(0, color="black", linewidth=0.9, linestyle="-", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(AREA_BIN_LABELS, fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_title(SZA_LABELS[sza], fontsize=14, fontweight="bold", pad=6)

    # 5a. Common axis labels (only on outer panels)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Reference iceberg area bin (m$^2$, lower edge shown)",
                      fontsize=13)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Per-pair relative error $RE_A$ (%)", fontsize=13)

    # 6. Per-(sza, area_bin) total n_pairs annotated near top of each panel
    # Sums across methods so the annotation is a single value per (sza, area)
    # cell. Uses get_xaxis_transform so x is data, y is axes-fraction.
    for ax, sza in zip(axes_flat, SZA_ORDER):
        n_total = (
            agg[agg["sza_bin"] == sza]
            .groupby("area_bin", observed=False)["n"].sum()
            .reindex(AREA_BIN_LABELS)
        )
        for i, lbl in enumerate(AREA_BIN_LABELS):
            ax.text(i, 0.97, f"n={int(n_total[lbl]):,}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, color="#555")

    # 7. Figure-level legend at bottom
    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[m], linewidth=2.0,
                   marker=METHOD_MARKERS[m], markersize=7, label=m)
        for m in METHOD_ORDER
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        "Per-pair relative error by reference area bin, per SZA bin",
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.965])

    archive = write_fig(
        fig,
        slug="re_by_area_bin",
        caption=(
            "Per-pair relative error in iceberg area as a function of "
            "reference area bin, one panel per SZA bin and one line per "
            "method. Faint shaded band shows the P10-P90 spread per method "
            "per bin; mean is the solid line. Area bin edges (1.6k, 23.7k, "
            "45.8k, 67.9k, 90k m^2) match Fisser (2025) Fig. 12 / 16. "
            "Color palette matches Fig. 7 (UNet_TR pink). Fisser (2024) "
            "Fig. 7 + Fisser (2025) Fig. 12 analog."
        ),
        out_dir=os.path.join(args.run, "per_iceberg"),
    )
    plt.close(fig)
    print(f"Wrote {archive}")

    # 8. Print summary per SZA bin for quick paper text reference
    print("\nPer-method mean RE (%) by (sza, area_bin):")
    for sza in SZA_ORDER:
        print(f"\n  {sza}:")
        sub = agg[agg["sza_bin"] == sza]
        pivot = (
            sub.pivot(index="method", columns="area_bin", values="mean")
            .reindex(METHOD_ORDER)[AREA_BIN_LABELS]
        )
        print(pivot.round(1).to_string())


if __name__ == "__main__":
    main()
