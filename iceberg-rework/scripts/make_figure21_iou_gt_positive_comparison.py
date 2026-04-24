#!/usr/bin/env python3
"""Create Figure 21-style GT-positive IoU heatmaps for model comparison."""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _fig_registry import write as write_fig

METHOD_ORDER = ["TR", "OT", "UNet", "UNet_TR", "UNet_OT", "UNet_CRF"]
SZA_ORDER = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]
SZA_LABELS = {
    "sza_lt65": "<65",
    "sza_65_70": "65-70",
    "sza_70_75": "70-75",
    "sza_gt75": ">75",
}


def load_iou_matrix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    pivot = (
        df.pivot(index="method", columns="sza_bin", values="mean_iou")
        .reindex(index=METHOD_ORDER, columns=SZA_ORDER)
    )
    pivot.columns = [SZA_LABELS[col] for col in pivot.columns]
    return pivot


def annotate_heatmap(ax, data: np.ndarray, fmt: str = ".3f") -> None:
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--stage1-summary", required=True)
    parser.add_argument("--out-dir", required=True,
                        help="Directory under which fig-archive/ + figures.md live")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    baseline = load_iou_matrix(args.baseline_summary)
    stage1 = load_iou_matrix(args.stage1_summary)
    delta = stage1 - baseline

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    rows = []
    for method in METHOD_ORDER:
        for sza in baseline.columns:
            rows.append(
                {
                    "method": method,
                    "sza_bin": sza,
                    "baseline_iou": baseline.loc[method, sza],
                    "stage1_iou": stage1.loc[method, sza],
                    "delta_stage1_minus_baseline": delta.loc[method, sza],
                }
            )
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), constrained_layout=True)

    panels = [
        ("Baseline: v3 balanced", baseline, "RdYlGn", 0.0, 1.0, ".3f"),
        ("SZA stage-1 balance", stage1, "RdYlGn", 0.0, 1.0, ".3f"),
        ("Stage-1 minus baseline", delta, "RdBu", -0.20, 0.20, "+.3f"),
    ]

    for ax, (title, frame, cmap, vmin, vmax, fmt) in zip(axes, panels):
        data = frame.to_numpy(dtype=float)
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(frame.columns)))
        ax.set_xticklabels(frame.columns, fontsize=10)
        ax.set_yticks(range(len(frame.index)))
        ax.set_yticklabels(frame.index, fontsize=10)
        ax.set_xlabel("SZA bin")
        annotate_heatmap(ax, data, fmt)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Mean IoU" if "minus" not in title else "IoU delta")

    fig.suptitle(
        "Figure 21. IoU heatmap by method and SZA bin - GT-positive chips only",
        fontsize=14,
        fontweight="bold",
    )
    archive_path = write_fig(
        fig,
        slug="figure21_iou_gt_positive_comparison",
        caption=(
            "IoU heatmap by method (rows) and SZA bin (columns) on "
            "GT-positive test chips, showing baseline IoU, stage-1 IoU, and "
            "the stage-1 minus baseline delta side by side. Stage-1 wins "
            "where the delta panel is red; regressions are blue."
        ),
        out_dir=args.out_dir,
        dpi=200,
    )
    plt.close(fig)

    print(f"Saved figure: {archive_path}")
    print(f"Saved values: {args.out_csv}")


if __name__ == "__main__":
    main()
