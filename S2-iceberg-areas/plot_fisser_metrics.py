"""
plot_fisser_metrics.py

Produce Fisser-style comparison figures from fisser_error_metrics.py
outputs. One figure (fig_re.png) reports relative error; a parallel
figure (fig_mae.png) reports MAE in root-length. Both are 3-row x
N-method grids:

  row 1 : RE vs iceberg size (ref root-length buckets) -- Fisser Fig 3a-c
  row 2 : RE vs solar zenith angle (1-deg interp + 5-deg smooth + IQR band)
          -- Fisser Fig 3d
  row 3 : match-rate vs SZA bin (selection-bias disclosure)

Each method that has no data gets a placeholder "not produced" panel.

Inputs:
  {in_dir}/per_pair.csv
  {in_dir}/re_by_size.csv
  {in_dir}/re_by_sza.csv
  {in_dir}/detection_stats.csv

Outputs:
  {out_dir}/fig_re.png
  {out_dir}/fig_mae.png

Usage:
  python plot_fisser_metrics.py \\
      --in_dir  /mnt/research/.../figures/fisser \\
      --out_dir /mnt/research/.../figures/fisser

Rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/plot_fisser_metrics.py \\
      smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_DIR = "/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/figures/fisser"

METHODS      = ["unet", "threshold", "otsu", "densecrf"]
METHOD_LABEL = {"unet": "UNet++", "threshold": "B08 threshold",
                "otsu": "Otsu",  "densecrf": "DenseCRF"}
METHOD_COLOR = {"unet": "#1E88E5", "threshold": "#FB8C00",
                "otsu": "#43A047", "densecrf": "#8E24AA"}
SZA_ORDER    = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]
SZA_LABEL    = {"sza_lt65": "<65 deg", "sza_65_70": "65-70",
                "sza_70_75": "70-75",  "sza_gt75":  ">75"}
BUCKET_ORDER = ["40-80", "80-160", "160-320", "320-640", ">=640"]


# 1. Row-level plotters ------------------------------------------------------

def plot_row_size(ax, method, pair_df, size_df, value_col, ylabel, zero_line):
    """Box-whisker per size bucket, one method."""
    mdf = pair_df[pair_df["method"] == method]
    if len(mdf) == 0:
        ax.text(0.5, 0.5, f"{METHOD_LABEL[method]}\nnot produced",
                ha="center", va="center", fontsize=11, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
        return

    data, labels = [], []
    for b in BUCKET_ORDER:
        lo_hi = {"40-80": (40, 80), "80-160": (80, 160), "160-320": (160, 320),
                 "320-640": (320, 640), ">=640": (640, np.inf)}[b]
        sub = mdf[(mdf["ref_root_length_m"] >= lo_hi[0]) &
                  (mdf["ref_root_length_m"] <  lo_hi[1])]
        if len(sub):
            data.append(sub[value_col].values)
            labels.append(b)
    if not data:
        ax.text(0.5, 0.5, "no pairs", ha="center", va="center")
        return

    bp = ax.boxplot(data, positions=range(len(data)), widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(METHOD_COLOR[method]); patch.set_alpha(0.7)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_xlabel("ref root-length (m)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if zero_line:
        ax.axhline(0, color="k", linestyle=":", linewidth=0.8)
    ax.grid(alpha=0.3, axis="y")


def plot_row_sza(ax, method, sza_df, value_col_prefix, ylabel, zero_line):
    """Line (raw median dots + 5-deg smoothed) with IQR band."""
    mdf = sza_df[sza_df["method"] == method].sort_values("sza_deg")
    if len(mdf) == 0 or mdf[f"{value_col_prefix}_smooth5"].dropna().empty:
        ax.text(0.5, 0.5, f"{METHOD_LABEL[method]}\nnot produced",
                ha="center", va="center", fontsize=11, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
        return

    x = mdf["sza_deg"].values
    y = mdf[f"{value_col_prefix}_smooth5"].values
    p25 = mdf[f"{value_col_prefix.replace('median','p25')}_smooth5"].values \
        if f"{value_col_prefix.replace('median','p25')}_smooth5" in mdf.columns else None
    p75 = mdf[f"{value_col_prefix.replace('median','p75')}_smooth5"].values \
        if f"{value_col_prefix.replace('median','p75')}_smooth5" in mdf.columns else None

    if p25 is not None and p75 is not None:
        finite = np.isfinite(p25) & np.isfinite(p75)
        if finite.any():
            ax.fill_between(x[finite], p25[finite], p75[finite],
                            color=METHOD_COLOR[method], alpha=0.2, label="IQR (p25-p75)")
    ax.plot(x, y, color=METHOD_COLOR[method], linewidth=2, label="5-deg smoothed median")

    raw_col = f"{value_col_prefix}_raw"
    if raw_col in mdf.columns:
        raw = mdf[raw_col].values
        mask = np.isfinite(raw)
        ax.scatter(x[mask], raw[mask], color=METHOD_COLOR[method],
                   s=15, alpha=0.6, label="1-deg median (raw)")
    ax.set_xlabel("solar zenith angle (deg)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if zero_line:
        ax.axhline(0, color="k", linestyle=":", linewidth=0.8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def plot_row_match_rate(ax, method, det_df):
    """Match-rate bar per SZA bin, one method. Regions side by side."""
    mdf = det_df[det_df["method"] == method]
    if len(mdf) == 0:
        ax.text(0.5, 0.5, f"{METHOD_LABEL[method]}\nnot produced",
                ha="center", va="center", fontsize=11, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
        return
    regions = sorted(mdf["region"].unique())
    width   = 0.8 / max(len(regions), 1)
    xs      = np.arange(len(SZA_ORDER))
    for i, region in enumerate(regions):
        rdf = mdf[mdf["region"] == region].set_index("sza_bin")
        vals = [float(rdf.loc[b, "match_rate"]) if b in rdf.index and np.isfinite(rdf.loc[b, "match_rate"]) else 0.0
                for b in SZA_ORDER]
        offset = (i - (len(regions) - 1) / 2.0) * width
        ax.bar(xs + offset, vals, width=width,
               color=METHOD_COLOR[method], alpha=0.5 + 0.5 * i, label=region.upper())
    ax.set_xticks(xs)
    ax.set_xticklabels([SZA_LABEL[b] for b in SZA_ORDER], rotation=30, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("match rate", fontsize=9)
    ax.set_xlabel("SZA bin", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, loc="upper right")


# 2. Figure assembly ---------------------------------------------------------

def make_figure(pair_df, size_df, sza_df, det_df, out_path,
                value_col, ylabel_re, ylabel_re_sza, zero_line_re):
    n_methods = len(METHODS)
    fig, axes = plt.subplots(
        3, n_methods,
        figsize=(4.2 * n_methods, 10),
        squeeze=False,
    )
    for ci, method in enumerate(METHODS):
        plot_row_size(axes[0, ci], method, pair_df, size_df,
                      value_col=value_col, ylabel=ylabel_re,
                      zero_line=zero_line_re)
        axes[0, ci].set_title(METHOD_LABEL[method], fontsize=11, fontweight="bold")

        # Row 2 is RE-vs-SZA regardless of figure flavor (MAE figure still
        # uses RE on row 2 since SZA aggregates are only emitted for RE).
        plot_row_sza(axes[1, ci], method, sza_df,
                     value_col_prefix="RE_median",
                     ylabel=ylabel_re_sza,
                     zero_line=True)

        plot_row_match_rate(axes[2, ci], method, det_df)

    fig.suptitle("Iceberg area error vs size, SZA, match rate (per method)",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# 3. Driver ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot Fisser-style RE and MAE comparisons.")
    parser.add_argument("--in_dir",  default=DEFAULT_DIR,
                        help="Directory containing per_pair.csv, re_by_size.csv, "
                             "re_by_sza.csv, detection_stats.csv.")
    parser.add_argument("--out_dir", default=DEFAULT_DIR,
                        help="Output directory for figures.")
    args = parser.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    pair_df = pd.read_csv(in_dir / "per_pair.csv")         if (in_dir / "per_pair.csv").exists()         else pd.DataFrame()
    size_df = pd.read_csv(in_dir / "re_by_size.csv")       if (in_dir / "re_by_size.csv").exists()       else pd.DataFrame()
    sza_df  = pd.read_csv(in_dir / "re_by_sza.csv")        if (in_dir / "re_by_sza.csv").exists()        else pd.DataFrame()
    det_df  = pd.read_csv(in_dir / "detection_stats.csv")  if (in_dir / "detection_stats.csv").exists()  else pd.DataFrame()

    if pair_df.empty:
        print("per_pair.csv missing or empty; nothing to plot.")
        return

    # 3a. RE figure.
    make_figure(pair_df, size_df, sza_df, det_df,
                out_path=out_dir / "fig_re.png",
                value_col="RE_pct",
                ylabel_re="RE (%)",
                ylabel_re_sza="RE (%)",
                zero_line_re=True)

    # 3b. MAE figure (row 1 shows MAE_rootlen per size bucket; row 2 still RE-vs-SZA).
    make_figure(pair_df, size_df, sza_df, det_df,
                out_path=out_dir / "fig_mae.png",
                value_col="AE_rootlen_m",
                ylabel_re="AE root-length (m)",
                ylabel_re_sza="RE (%)",
                zero_line_re=False)


if __name__ == "__main__":
    main()
