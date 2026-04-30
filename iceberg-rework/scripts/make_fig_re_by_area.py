"""
make_fig_re_by_area.py: Per-pair relative-error (RE %) by reference-area bin,
one line per method, with shaded IQR and P10-P90 bands. Fisser 2024 Fig. 7 +
Fisser 2025 Fig. 12 analog. Companion to `bias_delta_by_area` (which plots the
absolute Bias in m^2); this figure plots the *relative* error so cross-bin
comparison is scale-free.

Reads:
  <run>/per_iceberg/eval_per_iceberg.csv

Writes (via _fig_registry):
  <run>/per_iceberg/fig-archive/<ts>__re_by_area_bin.png
  <run>/per_iceberg/figures.md (row appended or updated)
  <run>/per_iceberg/re_by_area_bin.csv (per-method per-bin aggregates)

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

# Fisser 2025 area bin lower edges (m^2). Matches Fig. 12 + 16 x-axis.
AREA_BIN_EDGES = [1600, 23700, 45800, 67900, 90000, np.inf]
AREA_BIN_LABELS = ["1.6k", "23.7k", "45.8k", "67.9k", ">=90k"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN,
                        help="Path to a baseline_v1 run dir; reads <run>/per_iceberg/eval_per_iceberg.csv.")
    args = parser.parse_args()

    pairs_path = os.path.join(args.run, "per_iceberg", "eval_per_iceberg.csv")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(pairs_path)
    df = pd.read_csv(pairs_path)
    required = {"method", "gt_area_m2", "re_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {pairs_path}: {missing}")

    # 1. Bin pairs by reference area
    df = df.dropna(subset=["re_pct"]).copy()
    df["area_bin"] = pd.cut(
        df["gt_area_m2"], bins=AREA_BIN_EDGES, labels=AREA_BIN_LABELS,
        right=False, include_lowest=True,
    )

    # 2. Aggregate mean / P10 / P25 / P75 / P90 per (method, area_bin)
    agg = df.groupby(["method", "area_bin"], observed=False)["re_pct"].agg(
        mean="mean",
        p10=lambda s: float(np.percentile(s, 10)) if len(s) else np.nan,
        p25=lambda s: float(np.percentile(s, 25)) if len(s) else np.nan,
        p75=lambda s: float(np.percentile(s, 75)) if len(s) else np.nan,
        p90=lambda s: float(np.percentile(s, 90)) if len(s) else np.nan,
        n="size",
    ).reset_index()

    # 3. Persist per-method per-bin aggregates
    csv_path = os.path.join(args.run, "per_iceberg", "re_by_area_bin.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "area_bin", "n", "mean_re_pct",
                         "p10_re_pct", "p25_re_pct", "p75_re_pct", "p90_re_pct"])
        for _, r in agg.iterrows():
            writer.writerow([r["method"], r["area_bin"], int(r["n"]),
                             f"{r['mean']:.3f}",
                             f"{r['p10']:.3f}", f"{r['p25']:.3f}",
                             f"{r['p75']:.3f}", f"{r['p90']:.3f}"])
    print(f"Wrote {csv_path}")

    # 4. Plot mean line per method, with shaded IQR (P25-P75) and P10-P90
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(AREA_BIN_LABELS))

    for method in METHOD_ORDER:
        sub = agg[agg["method"] == method].set_index("area_bin").reindex(AREA_BIN_LABELS)
        if sub["mean"].isna().all():
            continue
        means = sub["mean"].values
        p10s = sub["p10"].values
        p90s = sub["p90"].values
        # Shaded P10-P90 (very faint) so it does not dominate the chart.
        ax.fill_between(x, p10s, p90s, color=METHOD_COLORS[method], alpha=0.06)
        # Mean line + markers.
        ax.plot(x, means, marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                linewidth=1.8, markersize=7, label=method)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(AREA_BIN_LABELS)
    ax.set_xlabel(r"Reference iceberg area bin (m$^2$, lower edge shown)")
    ax.set_ylabel(r"Per-pair relative error $RE_A$ (%)")
    ax.set_title("Per-pair relative error by reference area bin and method",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.95)

    # Per-bin total n_pairs annotated above the plot
    n_total = agg.groupby("area_bin", observed=False)["n"].sum().reindex(AREA_BIN_LABELS)
    ymax = ax.get_ylim()[1]
    ymin = ax.get_ylim()[0]
    n_y = ymax + (ymax - ymin) * 0.04
    ax.set_ylim(ymin, n_y + (ymax - ymin) * 0.06)
    for i, lbl in enumerate(AREA_BIN_LABELS):
        ax.text(i, n_y, f"n={int(n_total[lbl]):,}",
                ha="center", va="center", fontsize=8, color="#555")

    fig.tight_layout()

    archive = write_fig(
        fig,
        slug="re_by_area_bin",
        caption=(
            "Per-pair relative error in iceberg area as a function of "
            "reference area bin, one line per method. Faint shaded band "
            "shows the P10-P90 spread per method per bin; mean is the solid "
            "line. Area bin edges (1.6k, 23.7k, 45.8k, 67.9k, 90k m^2) match "
            "Fisser (2025) Fig. 12 / 16. Fisser (2024) Fig. 7 + Fisser "
            "(2025) Fig. 12 analog."
        ),
        out_dir=os.path.join(args.run, "per_iceberg"),
    )
    plt.close(fig)
    print(f"Wrote {archive}")

    # 5. Print per-method per-bin mean RE for quick paper text reference
    print("\nPer-method mean RE by area bin (%):")
    pivot = agg.pivot(index="method", columns="area_bin",
                      values="mean").reindex(METHOD_ORDER)[AREA_BIN_LABELS]
    print(pivot.round(1).to_string())


if __name__ == "__main__":
    main()
