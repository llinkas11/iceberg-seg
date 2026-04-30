"""
make_fig_mae_vs_sza.py: Per-pair mean absolute root-length error by SZA bin, one
line per method. The Fisser 2024 Fig 11 analog and the Fisser-comparable
headline figure for our optical 6-method sweep.

Reads:
  <run>/per_iceberg/eval_per_iceberg_summary.csv

Writes (via _fig_registry):
  <run>/per_iceberg/fig-archive/<ts>__mae_rootlen_vs_sza.png
  <run>/per_iceberg/figures.md (row appended or updated)

Usage:
  python scripts/make_fig_mae_vs_sza.py \
      --run /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/runs/exp_baseline_v1/20260424_185158

Rsync after edit:
  rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/make_fig_mae_vs_sza.py \
      llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _fig_registry import write as write_fig

DEFAULT_RUN = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/runs/exp_baseline_v1/20260424_185158"

SZA_ORDER = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]
SZA_LABELS = {"sza_lt65": "< 65", "sza_65_70": "65 to 70",
              "sza_70_75": "70 to 75", "sza_gt75": "> 75"}

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN,
                        help="Path to a baseline_v1 run dir; reads <run>/per_iceberg/eval_per_iceberg_summary.csv.")
    args = parser.parse_args()

    # 1. Load per-pair summary
    summary_path = os.path.join(args.run, "per_iceberg", "eval_per_iceberg_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(summary_path)
    df = pd.read_csv(summary_path)

    # 2. Validate schema
    required = {"method", "sza_bin", "n_pairs", "mean_abs_rootlen_err_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {summary_path}: {missing}")

    # 3. Pivot to (method x sza_bin) with mean RL MAE
    df["sza_bin"] = pd.Categorical(df["sza_bin"], categories=SZA_ORDER, ordered=True)
    pivot = df.pivot(index="method", columns="sza_bin",
                     values="mean_abs_rootlen_err_m").reindex(METHOD_ORDER)
    n_pairs = df.pivot(index="method", columns="sza_bin",
                       values="n_pairs").reindex(METHOD_ORDER)

    # 4. Plot one line per method, X = SZA bin, Y = mean RL MAE
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = list(range(len(SZA_ORDER)))
    for method in METHOD_ORDER:
        if method not in pivot.index:
            continue
        y = pivot.loc[method, SZA_ORDER].values
        ax.plot(x, y, marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                linewidth=2.0, markersize=8, label=method)

    # 5. Axes and labels. Set ylim with headroom so the n_pairs band and
    #    legend do not collide with the highest data marker (e.g. OT lt65
    #    around 22.7 m).
    data_max = float(pivot.values.max())
    ymax = data_max * 1.30
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.4, len(SZA_ORDER) - 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([SZA_LABELS[s] for s in SZA_ORDER])
    ax.set_xlabel("Solar zenith angle bin (degrees)")
    ax.set_ylabel("Mean absolute root-length error (m)")
    ax.set_title("Per-pair root-length MAE by SZA bin and method",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    # 6. Annotate per-bin n on the headroom band, then place legend below
    #    the annotations so neither overlaps the data.
    n_total = n_pairs.sum(axis=0)
    n_y = data_max * 1.22
    for i, sza in enumerate(SZA_ORDER):
        ax.text(i, n_y, f"n={int(n_total[sza]):,}",
                ha="center", va="center", fontsize=8, color="#555")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              fontsize=9, ncol=6, framealpha=0.95, frameon=True)

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    # 7. Route through fig_registry
    out_dir = os.path.join(args.run, "per_iceberg")
    archive = write_fig(
        fig,
        slug="mae_rootlen_vs_sza",
        caption=(
            "Per-pair mean absolute root-length error by SZA bin and method, "
            "from Hungarian-matched reference vs predicted iceberg pairs in "
            "the test split. Fisser 2024 Fig. 11 analog. Per-bin n_pairs are "
            "annotated near the top of each column."
        ),
        out_dir=out_dir,
    )
    plt.close(fig)
    print(f"Wrote {archive}")


if __name__ == "__main__":
    main()
