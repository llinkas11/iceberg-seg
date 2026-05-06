#!/usr/bin/env python3
"""A0 vs A1 backbone comparison by SZA bin (UNet method only).

Two side-by-side panels: per-pair root-length MAE (left) and per-pair IoU
(right). Source CSVs: <runs_root>/exp_A{0,1}_*/<latest_ts>/re_eval_v4_clean/
per_iceberg/eval_per_iceberg_summary.csv (UNet rows only).

Usage (from iceberg-rework/):
    python scripts/make_figS03_a0_vs_a1_by_sza.py \\
        --runs_root runs_summaries \\
        --out_dir   /Users/llinkas/.../paper-writing/figures
"""

from __future__ import annotations

import argparse
import os
from glob import glob
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _fig_registry import write as write_fig


# 1. Constants
SZA_ORDER = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]
SZA_LABELS = {
    "sza_lt65": "<65",
    "sza_65_70": "65-70",
    "sza_70_75": "70-75",
    "sza_gt75": ">75",
}


class Backbone(NamedTuple):
    label: str
    exp_dir: str
    description: str
    color: str


BACKBONES = [
    Backbone("A0", "exp_A0_fisser_lt65_original",
             "Fisser preprocessing", "#1f77b4"),
    Backbone("A1", "exp_A1_fisser_lt65_plus_nulls",
             "Fisser + GT-zero chips (original 2nd)", "#d62728"),
    Backbone("A7b", "exp_A7b_a1_size_aug",
             "A1 + size oversample + aug (best higher-SZA)", "#2ca02c"),
]


def latest_summary(runs_root: str, exp_dir_name: str) -> str:
    """Return latest re_eval_v4_clean per_iceberg summary CSV for one exp."""
    pattern = os.path.join(runs_root, exp_dir_name, "*",
                           "re_eval_v4_clean", "per_iceberg",
                           "eval_per_iceberg_summary.csv")
    candidates = sorted(glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No CSV for {exp_dir_name} under {runs_root}")
    return candidates[-1]


def load_one_backbone(csv_path: str) -> pd.DataFrame:
    """Load summary, filter to UNet, reindex by canonical SZA order."""
    df = pd.read_csv(csv_path)
    df = df[df["method"] == "UNet"].set_index("sza_bin")
    return df.reindex(SZA_ORDER)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", required=True,
                        help="Path to iceberg-rework/runs_summaries/")
    parser.add_argument("--out_dir", required=True,
                        help="Figure registry root (paper-writing/figures/)")
    args = parser.parse_args()

    # 2. Load A0 and A1 summaries
    data = {}
    for bb in BACKBONES:
        csv_path = latest_summary(args.runs_root, bb.exp_dir)
        data[bb.label] = load_one_backbone(csv_path)
        print(f"  {bb.label}: {csv_path}")

    # 3. Two-panel figure: MAE on left, IoU on right.
    #    Extra bottom margin so the shared legend does not overlap x labels.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0),
                             constrained_layout=False)
    n_bins = len(SZA_ORDER)
    n_backbones = len(BACKBONES)
    bar_width = 0.8 / n_backbones
    x = np.arange(n_bins)

    panel_specs = [
        ("a", "mean_abs_rootlen_err_m",
         "Mean per-pair root-length MAE (m)",
         "lower is better", axes[0]),
        ("b", "mean_iou",
         "Mean per-pair IoU",
         "higher is better", axes[1]),
    ]

    for panel_label, metric, ylabel, hint, ax in panel_specs:
        for i, bb in enumerate(BACKBONES):
            offset = (i - (n_backbones - 1) / 2) * bar_width
            values = data[bb.label][metric].to_numpy()
            bars = ax.bar(x + offset, values, bar_width,
                          label=f"{bb.label} ({bb.description})",
                          color=bb.color, edgecolor="black", linewidth=0.5)
            # 4. Annotate each bar with its numeric value
            for rect, v in zip(bars, values):
                if np.isnan(v):
                    continue
                txt = f"{v:.2f}" if metric.startswith("mean_abs") else f"{v:.3f}"
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 0.02 * max(values),
                        txt, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([SZA_LABELS[b] for b in SZA_ORDER], fontsize=10)
        ax.set_xlabel("SZA bin (deg)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{ylabel} ({hint})", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        # 4a. Panel label (IGS style: lowercase letter, no period, no parens)
        ax.text(-0.10, 1.02, panel_label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", ha="left", va="bottom")

    # 5. One legend at the bottom shared by both panels, with manual margin
    #    so it sits below the x-axis labels rather than overlapping them.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(BACKBONES),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        "Phase A backbone comparison by SZA bin (UNet, v4_clean test split)",
        fontsize=12, fontweight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.20, wspace=0.22)

    # 6. Register the figure
    archive_path = write_fig(
        fig,
        slug="figs03_a0_vs_a1_by_sza",
        caption=(
            "Phase A backbone comparison. Grouped bars per SZA bin show A0 "
            "(Fisser preprocessing baseline), A1 (Fisser + 29 GT-zero chips, "
            "the original 2nd-best non-A0), and A7b (A1 + size oversample "
            "with augmentation, the new best higher-SZA generaliser across "
            "all 18 Phase A backbones). Left: mean per-pair root-length MAE "
            "(lower is better); right: mean per-pair IoU (higher is better). "
            "UNet method only. A7b beats A1 on every bin including lt65. "
            "Source: iceberg-rework/runs_summaries/exp_A{0,1,7b}_*/"
            "<latest_ts>/re_eval_v4_clean/per_iceberg/"
            "eval_per_iceberg_summary.csv."
        ),
        out_dir=args.out_dir,
        dpi=200,
    )
    plt.close(fig)
    print(f"Saved figure: {archive_path}")


if __name__ == "__main__":
    main()
