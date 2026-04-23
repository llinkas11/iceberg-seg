"""
compare_areas.py — Compare iceberg areas across all 6 segmentation methods.

Loads all_icebergs.gpkg for each method from the test-set output directory
(area_comparison/test/{sza_bin}/{METHOD}/all_icebergs.gpkg), computes per-bin
area statistics, and produces summary plots for each pair of methods.

Methods compared:
  TR        — fixed NIR threshold (B08 ≥ 0.22)
  OT        — per-chip Otsu on B08
  UNet      — UNet++ argmax
  UNet_TR   — UNet++ + fixed threshold on P(iceberg)
  UNet_OT   — UNet++ + Otsu on P(iceberg)
  UNet_CRF  — UNet++ + DenseCRF

Usage:
  python compare_areas.py \\
      --test_dir  /mnt/research/.../area_comparison/test \\
      --out_dir   figures/comparison

  # or, if run from the project root with the default layout:
  python compare_areas.py

rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/compare_areas.py smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
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

# ── Constants ─────────────────────────────────────────────────────────────────

SZA_ORDER  = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]
SZA_LABELS = {
    "sza_lt65":  "< 65°",
    "sza_65_70": "65–70°",
    "sza_70_75": "70–75°",
    "sza_gt75":  "> 75°",
}

METHODS = ["TR", "OT", "UNet", "UNet_TR", "UNet_OT", "UNet_CRF"]
METHOD_LABELS = {
    "TR":       "Threshold (NIR ≥ 0.22)",
    "OT":       "Otsu (NIR)",
    "UNet":     "UNet++",
    "UNet_TR":  "UNet++ + Threshold",
    "UNet_OT":  "UNet++ + Otsu",
    "UNet_CRF": "UNet++ + CRF",
}
METHOD_COLORS = {
    "TR":       "#FB8C00",   # orange
    "OT":       "#F4511E",   # deep orange
    "UNet":     "#1E88E5",   # blue
    "UNet_TR":  "#5E35B1",   # deep purple
    "UNet_OT":  "#00897B",   # teal
    "UNet_CRF": "#43A047",   # green
}


# ── Data loading ──────────────────────────────────────────────────────────────

def collect_data(test_dir):
    """
    Walk test_dir/{sza_bin}/{METHOD}/all_icebergs.gpkg and load all iceberg polygons.

    Returns DataFrame with columns: area_m2, sza_bin, method
    """
    records   = []
    found_any = False

    if not os.path.isdir(test_dir):
        print(f"ERROR: test_dir not found: {test_dir}")
        return pd.DataFrame()

    for sza_bin in sorted(os.listdir(test_dir)):
        bin_path = os.path.join(test_dir, sza_bin)
        if not os.path.isdir(bin_path) or sza_bin not in SZA_ORDER:
            continue

        for method in METHODS:
            fpath = os.path.join(bin_path, method, "all_icebergs.gpkg")
            if not os.path.exists(fpath):
                continue

            try:
                gdf = gpd.read_file(fpath)
                if gdf.empty:
                    print(f"  {method:10s}  {sza_bin}: 0 icebergs (empty gpkg)")
                    continue
                found_any = True
                for _, row in gdf.iterrows():
                    records.append({
                        "area_m2": float(row["area_m2"]),
                        "sza_bin": sza_bin,
                        "method":  method,
                    })
                print(f"  {method:10s}  {sza_bin}: {len(gdf)} icebergs")
            except Exception as e:
                print(f"  WARNING: could not load {fpath}: {e}")

    if not found_any:
        print(f"\nNo all_icebergs.gpkg files found under {test_dir}.")
        print("Run run_all_methods.sh for each SZA bin first.")

    return pd.DataFrame(records)


# ── Summary stats ─────────────────────────────────────────────────────────────

def summary_stats(df):
    stats = (
        df.groupby(["sza_bin", "method"])["area_m2"]
        .agg(
            count     = "count",
            mean_m2   = "mean",
            median_m2 = "median",
            total_m2  = "sum",
            p25_m2    = lambda x: float(np.percentile(x, 25)),
            p75_m2    = lambda x: float(np.percentile(x, 75)),
        )
        .reset_index()
    )
    for col in ["mean_m2", "median_m2", "total_m2", "p25_m2", "p75_m2"]:
        stats[col] = stats[col].round(2)
    # enforce SZA bin order
    stats["sza_bin"] = pd.Categorical(stats["sza_bin"], categories=SZA_ORDER, ordered=True)
    stats = stats.sort_values(["sza_bin", "method"]).reset_index(drop=True)
    return stats


# ── Box plots (all methods, one subplot per SZA bin) ─────────────────────────

def plot_boxplots(df, out_dir):
    present_bins    = [b for b in SZA_ORDER if b in df["sza_bin"].values]
    present_methods = [m for m in METHODS if m in df["method"].values]
    if not present_bins:
        print("No SZA bins found — skipping box plots.")
        return

    n_bins    = len(present_bins)
    n_methods = len(present_methods)
    gap       = 1          # gap between bin groups
    w         = 0.7        # box width

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * n_bins * n_methods), 6))

    x_ticks, x_labels = [], []

    for k, sza_bin in enumerate(present_bins):
        bdf  = df[df["sza_bin"] == sza_bin]
        base = k * (n_methods + gap)
        x_ticks.append(base + (n_methods - 1) / 2)
        x_labels.append(SZA_LABELS.get(sza_bin, sza_bin))

        for j, method in enumerate(present_methods):
            vals = bdf[bdf["method"] == method]["area_m2"].values
            if len(vals) == 0:
                continue
            pos = base + j
            bp  = ax.boxplot([vals], positions=[pos], widths=w,
                             patch_artist=True, showfliers=False,
                             medianprops=dict(color="black", linewidth=2))
            for patch in bp["boxes"]:
                patch.set_facecolor(METHOD_COLORS[method])
                patch.set_alpha(0.75)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel("Solar Zenith Angle bin", fontsize=12)
    ax.set_ylabel("Iceberg area (m²)", fontsize=12)
    ax.set_title("Iceberg area distributions by SZA bin and method (test set)",
                 fontsize=13, fontweight="bold")
    ax.legend(handles=[
        Patch(facecolor=METHOD_COLORS[m], alpha=0.75, label=METHOD_LABELS[m])
        for m in present_methods
    ], fontsize=9, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "area_boxplots.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Count bar chart (iceberg counts per bin per method) ──────────────────────

def plot_counts(stats, out_dir):
    present_bins    = [b for b in SZA_ORDER if b in stats["sza_bin"].values]
    present_methods = [m for m in METHODS if m in stats["method"].values]
    if not present_bins:
        return

    n_bins    = len(present_bins)
    n_methods = len(present_methods)
    x         = np.arange(n_bins)
    bar_w     = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(8, 3 * n_bins), 5))

    for j, method in enumerate(present_methods):
        counts = []
        for sza_bin in present_bins:
            row = stats[(stats["sza_bin"] == sza_bin) & (stats["method"] == method)]
            counts.append(int(row["count"].values[0]) if len(row) else 0)
        offset = (j - n_methods / 2 + 0.5) * bar_w
        ax.bar(x + offset, counts, bar_w,
               label=METHOD_LABELS[method],
               color=METHOD_COLORS[method], alpha=0.8, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([SZA_LABELS.get(b, b) for b in present_bins], fontsize=11)
    ax.set_xlabel("Solar Zenith Angle bin", fontsize=12)
    ax.set_ylabel("Iceberg polygon count", fontsize=12)
    ax.set_title("Iceberg detection counts by SZA bin and method (test set)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "area_counts.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Mean area trend (line plot, one line per method) ─────────────────────────

def plot_mean_trend(stats, out_dir):
    present_bins    = [b for b in SZA_ORDER if b in stats["sza_bin"].values]
    present_methods = [m for m in METHODS if m in stats["method"].values]
    if len(present_bins) < 2:
        print("Only one SZA bin present — skipping mean trend plot (need ≥ 2).")
        return

    markers = ["o", "s", "^", "D", "v", "P"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for j, method in enumerate(present_methods):
        means, labels = [], []
        for sza_bin in present_bins:
            row = stats[(stats["sza_bin"] == sza_bin) & (stats["method"] == method)]
            if len(row):
                means.append(float(row["mean_m2"].values[0]))
                labels.append(SZA_LABELS.get(sza_bin, sza_bin))

        if not means:
            continue

        ax.plot(range(len(means)), means,
                color=METHOD_COLORS[method],
                marker=markers[j % len(markers)],
                linewidth=2, markersize=8,
                label=METHOD_LABELS[method])
        for i, v in enumerate(means):
            ax.annotate(f"{v:.0f}", (i, v),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7,
                        color=METHOD_COLORS[method])

    ax.set_xticks(range(len(present_bins)))
    ax.set_xticklabels([SZA_LABELS.get(b, b) for b in present_bins], fontsize=11)
    ax.set_xlabel("Solar Zenith Angle bin", fontsize=12)
    ax.set_ylabel("Mean iceberg area (m²)", fontsize=12)
    ax.set_title("Mean iceberg area by SZA bin and method (test set)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "area_mean_trend.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Ratio plot (each method / UNet, across SZA bins) ─────────────────────────

def plot_ratio(stats, out_dir):
    present_bins    = [b for b in SZA_ORDER if b in stats["sza_bin"].values]
    present_methods = [m for m in METHODS if m in stats["method"].values and m != "UNet"]
    if len(present_bins) < 2:
        print("Only one SZA bin present — skipping ratio plot (need ≥ 2).")
        return
    if "UNet" not in stats["method"].values:
        print("UNet results not found — skipping ratio plot.")
        return

    markers = ["o", "s", "^", "D", "v"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for j, method in enumerate(present_methods):
        ratios, labels = [], []
        for sza_bin in present_bins:
            u_row = stats[(stats["sza_bin"] == sza_bin) & (stats["method"] == "UNet")]
            m_row = stats[(stats["sza_bin"] == sza_bin) & (stats["method"] == method)]
            if len(u_row) and len(m_row):
                u_mean = float(u_row["mean_m2"].values[0])
                m_mean = float(m_row["mean_m2"].values[0])
                if m_mean > 0:
                    ratios.append(u_mean / m_mean)
                    labels.append(SZA_LABELS.get(sza_bin, sza_bin))

        if not ratios:
            continue

        ax.plot(range(len(ratios)), ratios,
                color=METHOD_COLORS[method],
                marker=markers[j % len(markers)],
                linewidth=2, markersize=8,
                label=f"UNet / {METHOD_LABELS[method]}")
        for i, r in enumerate(ratios):
            ax.annotate(f"{r:.2f}", (i, r),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7)

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="Ratio = 1")
    ax.set_xticks(range(len(present_bins)))
    ax.set_xticklabels([SZA_LABELS.get(b, b) for b in present_bins], fontsize=11)
    ax.set_xlabel("Solar Zenith Angle bin", fontsize=12)
    ax.set_ylabel("UNet++ mean area / comparison method mean area", fontsize=12)
    ax.set_title("UNet++ area ratio vs other methods by SZA bin\n"
                 "(ratio > 1 → UNet++ detects larger mean area)", fontsize=13)
    ax.legend(fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "area_ratio.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Summary table print ───────────────────────────────────────────────────────

def print_comparison_table(stats):
    """Print count × mean_m2 table: rows=SZA bin, columns=method."""
    bins    = [b for b in SZA_ORDER if b in stats["sza_bin"].values]
    methods = [m for m in METHODS if m in stats["method"].values]

    col_w = 18
    header = f"{'SZA bin':<14}" + "".join(f"{m:>{col_w}}" for m in methods)
    print("\n" + "─" * len(header))
    print("Iceberg count (n)")
    print(header)
    print("─" * len(header))
    for sza_bin in bins:
        row = f"{SZA_LABELS.get(sza_bin, sza_bin):<14}"
        for m in methods:
            r = stats[(stats["sza_bin"] == sza_bin) & (stats["method"] == m)]
            val = int(r["count"].values[0]) if len(r) else "—"
            row += f"{str(val):>{col_w}}"
        print(row)
    print("─" * len(header))

    print("\n" + "─" * len(header))
    print("Mean iceberg area (m²)")
    print(header)
    print("─" * len(header))
    for sza_bin in bins:
        row = f"{SZA_LABELS.get(sza_bin, sza_bin):<14}"
        for m in methods:
            r = stats[(stats["sza_bin"] == sza_bin) & (stats["method"] == m)]
            val = f"{r['mean_m2'].values[0]:.1f}" if len(r) else "—"
            row += f"{str(val):>{col_w}}"
        print(row)
    print("─" * len(header))

    print("\n" + "─" * len(header))
    print("Total iceberg area (m²)")
    print(header)
    print("─" * len(header))
    for sza_bin in bins:
        row = f"{SZA_LABELS.get(sza_bin, sza_bin):<14}"
        for m in methods:
            r = stats[(stats["sza_bin"] == sza_bin) & (stats["method"] == m)]
            val = f"{r['total_m2'].values[0]:.0f}" if len(r) else "—"
            row += f"{str(val):>{col_w}}"
        print(row)
    print("─" * len(header))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    default_test_dir = os.path.join(
        "/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas",
        "area_comparison", "test"
    )
    parser = argparse.ArgumentParser(
        description="Compare iceberg areas across all 6 methods (test set)"
    )
    parser.add_argument("--test_dir", default=default_test_dir,
        help="Root of area_comparison/test/ — contains {sza_bin}/{METHOD}/all_icebergs.gpkg")
    parser.add_argument("--out_dir",
        default=os.path.join(RESEARCH, "test_outputs"),
        help="Output directory for plots and CSV")
    args = parser.parse_args()

    print(f"\nLoading results from: {args.test_dir}")
    df = collect_data(args.test_dir)

    if df.empty:
        return

    print(f"\nLoaded {len(df)} iceberg polygons total")
    print(f"  SZA bins : {[b for b in SZA_ORDER if b in df['sza_bin'].values]}")
    print(f"  Methods  : {[m for m in METHODS if m in df['method'].values]}")

    os.makedirs(args.out_dir, exist_ok=True)

    stats = summary_stats(df)
    stats_path = os.path.join(args.out_dir, "area_stats.csv")
    stats.to_csv(stats_path, index=False)
    print(f"\nSummary stats → {stats_path}")

    print_comparison_table(stats)

    print("\nGenerating plots...")
    plot_boxplots(df, args.out_dir)
    plot_counts(stats, args.out_dir)
    plot_mean_trend(stats, args.out_dir)
    plot_ratio(stats, args.out_dir)

    print(f"\n{'─'*50}")
    print(f"Outputs in: {args.out_dir}/")
    print(f"  area_stats.csv       — per-(sza_bin, method) statistics")
    print(f"  area_boxplots.png    — area distributions (box plots, all methods)")
    print(f"  area_counts.png      — iceberg detection counts per bin/method")
    print(f"  area_mean_trend.png  — mean area trend across SZA bins")
    print(f"  area_ratio.png       — UNet++ / comparison method area ratio")


if __name__ == "__main__":
    main()
