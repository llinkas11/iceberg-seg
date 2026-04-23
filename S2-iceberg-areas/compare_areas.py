"""
compare_areas.py — Compare iceberg areas across detection methods by SZA bin.

Loads GeoPackages from predict_tifs.py (UNet++), threshold_tifs.py, and
otsu_threshold_tifs.py. Computes area statistics per SZA bin and region.

Outputs:
  figures/area_stats.csv       — summary table (all methods)
  figures/area_boxplots.png    — area distributions per SZA bin
  figures/area_ratio.png       — method / UNet++ ratio trend (needs >= 2 bins)

Usage:
  python compare_areas.py \\
      --pred_dir /mnt/research/.../area_comparison \\
      --out_dir  figures

Expected directory structure under pred_dir:
  {region}/{sza_bin}/unet/all_icebergs.gpkg              <- predict_tifs.py
  {region}/{sza_bin}/threshold/all_icebergs_threshold.gpkg <- threshold_tifs.py
  {region}/{sza_bin}/otsu/all_icebergs_otsu.gpkg         <- otsu_threshold_tifs.py
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# SZA bin ordering and display labels
SZA_ORDER = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]
SZA_LABELS = {
    "sza_lt65":  "< 65°",
    "sza_65_70": "65–70°",
    "sza_70_75": "70–75°",
    "sza_gt75":  "> 75°",
}
METHOD_COLORS = {
    "UNet++":    "#1E88E5",
    "Threshold": "#FB8C00",
    "Otsu":      "#43A047",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def collect_data(pred_dir):
    """
    Walk pred_dir/{region}/{sza_bin}/ and load all available method outputs:
      unet/all_icebergs.gpkg              -> UNet++
      threshold/all_icebergs_threshold.gpkg -> Threshold
      otsu/all_icebergs_otsu.gpkg         -> Otsu
    Returns combined DataFrame with columns:
      area_m2, region, sza_bin, method
    Missing method files are silently skipped (partial results are fine).
    """
    records = []
    found_any = False

    if not os.path.isdir(pred_dir):
        print(f"ERROR: pred_dir does not exist: {pred_dir}")
        return pd.DataFrame()

    for region in sorted(os.listdir(pred_dir)):
        region_path = os.path.join(pred_dir, region)
        if not os.path.isdir(region_path):
            continue

        for sza_bin in sorted(os.listdir(region_path)):
            bin_path = os.path.join(region_path, sza_bin)
            if not os.path.isdir(bin_path):
                continue

            for subdir, fname, method in [
                ("unet",      "all_icebergs.gpkg",           "UNet++"),
                ("threshold", "all_icebergs_threshold.gpkg", "Threshold"),
                ("otsu",      "all_icebergs_otsu.gpkg",      "Otsu"),
            ]:
                fpath = os.path.join(bin_path, subdir, fname)
                if not os.path.exists(fpath):
                    continue

                try:
                    gdf      = gpd.read_file(fpath)
                    icebergs = gdf[gdf["class_name"] == "iceberg"]
                    found_any = True
                    for _, row in icebergs.iterrows():
                        records.append({
                            "area_m2": float(row["area_m2"]),
                            "region":  region,
                            "sza_bin": sza_bin,
                            "method":  method,
                        })
                    print(f"  {method:10s}  {region}/{sza_bin}: {len(icebergs)} icebergs")
                except Exception as e:
                    print(f"  WARNING: could not load {fpath}: {e}")

    if not found_any:
        print(f"\nNo GeoPackages found under {pred_dir}.")
        print("Run predict_tifs.py and threshold_tifs.py first.")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def summary_stats(df):
    stats = (
        df.groupby(["region", "sza_bin", "method"])["area_m2"]
        .agg(
            count    = "count",
            mean_m2  = "mean",
            median_m2= "median",
            total_m2 = "sum",
            p25_m2   = lambda x: float(np.percentile(x, 25)),
            p75_m2   = lambda x: float(np.percentile(x, 75)),
        )
        .reset_index()
    )
    # Round to 2 dp
    for col in ["mean_m2", "median_m2", "total_m2", "p25_m2", "p75_m2"]:
        stats[col] = stats[col].round(2)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Box plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplots(df, out_dir):
    # 1. Determine which methods and bins are present
    regions      = sorted(df["region"].unique())
    present_bins = [b for b in SZA_ORDER if b in df["sza_bin"].values]
    methods      = [m for m in ["UNet++", "Threshold", "Otsu"] if m in df["method"].values]
    if not present_bins:
        print("No SZA bins found — skipping box plots.")
        return

    n_methods = len(methods)
    spacing   = n_methods + 1   # gap between SZA bin groups

    fig, axes = plt.subplots(1, len(regions),
                             figsize=(max(6, spacing * len(present_bins)) * len(regions), 6),
                             sharey=False, squeeze=False)
    axes = axes[0]

    for ax, region in zip(axes, regions):
        rdf    = df[df["region"] == region]
        bins   = [b for b in present_bins if b in rdf["sza_bin"].values]

        data_by_method = {m: [] for m in methods}
        pos_by_method  = {m: [] for m in methods}
        x_ticks, x_labels = [], []

        for k, sza_bin in enumerate(bins):
            bdf  = rdf[rdf["sza_bin"] == sza_bin]
            base = k * spacing
            x_ticks.append(base + (n_methods - 1) / 2.0)
            x_labels.append(SZA_LABELS.get(sza_bin, sza_bin))

            for i, method in enumerate(methods):
                vals = bdf[bdf["method"] == method]["area_m2"].values
                if len(vals):
                    data_by_method[method].append(vals)
                    pos_by_method[method].append(base + i)

        def draw_boxes(data, positions, color):
            if not data:
                return
            bp = ax.boxplot(data, positions=positions, widths=0.75,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", linewidth=2))
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.75)

        for method in methods:
            draw_boxes(data_by_method[method], pos_by_method[method],
                       METHOD_COLORS[method])

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.set_xlabel("Solar Zenith Angle bin", fontsize=12)
        ax.set_ylabel("Iceberg area (m2)", fontsize=12)
        ax.set_title(f"{region} fjord", fontsize=13, fontweight="bold")
        ax.legend(handles=[
            Patch(facecolor=METHOD_COLORS[m], alpha=0.75, label=m)
            for m in methods
        ], fontsize=10)

    fig.suptitle("Iceberg area distributions by SZA bin", fontsize=14, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "area_boxplots.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Ratio plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_ratio(stats, out_dir):
    """
    Plot method / UNet++ total-area ratio across SZA bins.
    UNet++ is the reference (ratio = 1 by definition).
    Threshold and Otsu ratios show how much each method over/underestimates
    relative to UNet++. One line per region per non-UNet++ method.
    """
    present_bins = [b for b in SZA_ORDER if b in stats["sza_bin"].values]
    if len(present_bins) < 2:
        print("Only one SZA bin present — skipping ratio plot (need >= 2 bins).")
        return

    # 1. Non-UNet++ methods to compare
    compare_methods = [m for m in ["Threshold", "Otsu"] if m in stats["method"].values]
    if not compare_methods:
        print("No comparison methods found — skipping ratio plot.")
        return

    regions   = sorted(stats["region"].unique())
    # Line style: solid=KQ, dashed=SK; color by method
    r_dashes  = {"KQ": "-",  "SK": "--"}
    r_markers = {"KQ": "o",  "SK": "s"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in compare_methods:
        for region in regions:
            rdf    = stats[stats["region"] == region]
            ratios, x_pos = [], []

            for k, sza_bin in enumerate(present_bins):
                bdf    = rdf[rdf["sza_bin"] == sza_bin]
                u_tot  = bdf[bdf["method"] == "UNet++"]["total_m2"].values
                m_tot  = bdf[bdf["method"] == method]["total_m2"].values
                if len(u_tot) and len(m_tot) and u_tot[0] > 0:
                    ratios.append(m_tot[0] / u_tot[0])
                    x_pos.append(k)

            if not ratios:
                continue

            label = f"{method} / UNet++ ({region})"
            ax.plot(x_pos, ratios,
                    color=METHOD_COLORS.get(method, "gray"),
                    linestyle=r_dashes.get(region, "-"),
                    marker=r_markers.get(region, "o"),
                    linewidth=2, markersize=8, label=label)
            for xi, r in zip(x_pos, ratios):
                ax.annotate(f"{r:.1f}x", (xi, r), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=8)

    ax.set_xticks(range(len(present_bins)))
    ax.set_xticklabels([SZA_LABELS.get(b, b) for b in present_bins], fontsize=11)
    ax.axhline(1.0, color="#1E88E5", linestyle=":", linewidth=1.5, alpha=0.8,
               label="UNet++ (reference)")
    ax.set_xlabel("Solar Zenith Angle bin", fontsize=12)
    ax.set_ylabel("Method total area / UNet++ total area", fontsize=12)
    ax.set_title("Area ratio relative to UNet++ by SZA bin\n"
                 "(ratio > 1: method overestimates; < 1: underestimates)", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "area_ratio.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare iceberg areas across UNet++, Threshold, and Otsu methods by SZA bin"
    )
    parser.add_argument("--pred_dir", required=True,
                        help="Root of area_comparison/ — expects {region}/{sza_bin}/{unet,threshold,otsu}/ subdirs")
    parser.add_argument("--out_dir",  default="figures",
                        help="Output directory for plots and CSV (default: figures/)")
    args = parser.parse_args()

    print(f"\nLoading predictions from: {args.pred_dir}")
    df = collect_data(args.pred_dir)

    if df.empty:
        return

    print(f"\nLoaded {len(df)} iceberg polygons")
    print(f"  Regions  : {sorted(df['region'].unique())}")
    print(f"  SZA bins : {sorted(df['sza_bin'].unique())}")
    print(f"  Methods  : {sorted(df['method'].unique())}")

    os.makedirs(args.out_dir, exist_ok=True)

    stats = summary_stats(df)
    stats_path = os.path.join(args.out_dir, "area_stats.csv")
    stats.to_csv(stats_path, index=False)
    print(f"\nSummary stats → {stats_path}")
    print(stats.to_string(index=False))

    print("\nGenerating plots...")
    plot_boxplots(df, args.out_dir)
    plot_ratio(stats, args.out_dir)

    print(f"\n{'─'*50}")
    print(f"Outputs in: {args.out_dir}/")
    print(f"  area_stats.csv       — per-(region, sza_bin, method) statistics")
    print(f"  area_boxplots.png    — area distributions (box plots, all methods)")
    print(f"  area_ratio.png       — method / UNet++ total area ratio by SZA bin")


if __name__ == "__main__":
    main()
