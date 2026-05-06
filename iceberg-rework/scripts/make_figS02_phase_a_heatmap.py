#!/usr/bin/env python3
"""Phase A per-SZA-bin x per-experiment heatmap (UNet method only).

Two side-by-side panels: per-pair IoU (left) and per-pair root-length MAE
(right). Source CSVs: <runs_root>/exp_A*/<latest_ts>/re_eval_v4_clean/
per_iceberg/eval_per_iceberg_summary.csv (UNet rows only).

Usage (from iceberg-rework/):
    python scripts/make_figS02_phase_a_heatmap.py \\
        --runs_root runs_summaries \\
        --out_dir   /Users/llinkas/.../paper-writing/figures
"""

from __future__ import annotations

import argparse
import os
import re
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _fig_registry import write as write_fig


# 1. Constants for axis ordering and labels
SZA_ORDER = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]
SZA_LABELS = {
    "sza_lt65": "<65",
    "sza_65_70": "65-70",
    "sza_70_75": "70-75",
    "sza_gt75": ">75",
}
EXP_PATTERN = re.compile(r"^exp_(A\d+[a-z]?)_")


def latest_summary(exp_dir: str) -> str | None:
    """Return the latest re_eval_v4_clean per_iceberg summary CSV, or None."""
    candidates = sorted(
        glob(os.path.join(exp_dir, "*", "re_eval_v4_clean", "per_iceberg",
                          "eval_per_iceberg_summary.csv"))
    )
    return candidates[-1] if candidates else None


def load_phase_a_unet(runs_root: str) -> pd.DataFrame:
    """
    Load every Phase A re_eval per_iceberg summary, filter to UNet rows.

    Returns long-format DataFrame: exp_id (A0, A1, ...), sza_bin, n_pairs,
    mean_iou, mean_abs_rootlen_err_m. Skips experiments that have no
    re_eval_v4_clean dir yet (e.g. A1-anchored variants still training).
    """
    rows = []
    for exp_dir in sorted(glob(os.path.join(runs_root, "exp_A*"))):
        m = EXP_PATTERN.match(os.path.basename(exp_dir))
        if not m:
            continue
        exp_id = m.group(1)
        csv_path = latest_summary(exp_dir)
        if csv_path is None:
            print(f"  skip {exp_id}: no re_eval_v4_clean summary yet")
            continue
        df = pd.read_csv(csv_path)
        df = df[df["method"] == "UNet"].copy()
        df["exp_id"] = exp_id
        rows.append(df[["exp_id", "sza_bin", "n_pairs", "mean_iou",
                        "mean_abs_rootlen_err_m"]])
    if not rows:
        raise RuntimeError(f"No Phase A re_eval CSVs found under {runs_root}")
    return pd.concat(rows, ignore_index=True)


def pivot_metric(df: pd.DataFrame, metric: str, exp_order: list[str]) -> pd.DataFrame:
    """Pivot to: rows=SZA bins (in canonical order), cols=experiments."""
    pivot = (
        df.pivot(index="sza_bin", columns="exp_id", values=metric)
        .reindex(index=SZA_ORDER, columns=exp_order)
    )
    pivot.index = [SZA_LABELS[i] for i in pivot.index]
    return pivot


def annotate_heatmap(ax, data: np.ndarray, cmap, vmin: float, vmax: float,
                     fmt: str = ".3f") -> None:
    """
    Annotate each cell with its value. Text color is chosen from the actual
    cmap color sampled at the cell's normalized position so readability
    holds for both forward (viridis) and reversed (viridis_r) cmaps. The
    original implementation used a hardcoded |v| > 0.6 threshold which
    made every MAE cell white-on-yellow.
    """
    span = max(vmax - vmin, 1e-9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            t = np.clip((v - vmin) / span, 0.0, 1.0)
            r, g, b, _ = cmap(t)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            color = "white" if luminance < 0.5 else "black"
            ax.text(j, i, format(v, fmt), ha="center", va="center",
                    fontsize=8, color=color)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", required=True,
                        help="Path to iceberg-rework/runs_summaries/")
    parser.add_argument("--out_dir", required=True,
                        help="Figure registry root (paper-writing/figures/)")
    args = parser.parse_args()

    # 2. Load + sort experiments naturally (A0, A1, ..., A9, A5a, A6a, ...)
    df = load_phase_a_unet(args.runs_root)
    def _key(e: str) -> tuple[int, int, int]:
        # Group: original (A0..A9, suffix=0), aug=off variants (suffix='a'=1),
        # aug=on variants (suffix='b'=2). Within each group sort by number.
        suffix = e[1:]
        digits = re.match(r"\d+", suffix).group(0)
        rest = suffix[len(digits):]
        group = {"": 0, "a": 1, "b": 2}.get(rest, 99)
        return (group, int(digits), 0)
    exp_order = sorted(df["exp_id"].unique(), key=_key)
    print(f"Experiments: {exp_order}")

    iou = pivot_metric(df, "mean_iou", exp_order)
    mae = pivot_metric(df, "mean_abs_rootlen_err_m", exp_order)

    # 3. Two-panel figure: IoU on left, MAE on right
    n_exp = len(exp_order)
    width = max(10, 1.0 * n_exp + 4)
    fig, axes = plt.subplots(1, 2, figsize=(width, 4.2),
                             constrained_layout=True)

    panels = [
        ("a", axes[0], iou, "viridis", 0.40, 0.75, ".3f",
         "Mean per-pair IoU"),
        ("b", axes[1], mae, "viridis_r", 5, 50, ".1f",
         "Mean per-pair root-length MAE (m)"),
    ]
    for panel_label, ax, frame, cmap_name, vmin, vmax, fmt, title in panels:
        data = frame.to_numpy(dtype=float)
        cmap = plt.get_cmap(cmap_name)
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(frame.columns)))
        ax.set_xticklabels(frame.columns, fontsize=9, rotation=0)
        ax.set_yticks(range(len(frame.index)))
        ax.set_yticklabels(frame.index, fontsize=9)
        ax.set_xlabel("Phase A experiment", fontsize=10)
        ax.set_ylabel("SZA bin (deg)", fontsize=10)
        annotate_heatmap(ax, data, cmap, vmin, vmax, fmt)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)
        # 3a. Panel label (IGS style: lowercase letter, no period, no parens)
        ax.text(-0.07, 1.05, panel_label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", ha="left", va="bottom")

    fig.suptitle(
        f"Phase A per-SZA-bin x per-experiment (UNet only, v4_clean test split, n={len(exp_order)} backbones)",
        fontsize=12, fontweight="bold",
    )

    # 4. Register the figure
    archive_path = write_fig(
        fig,
        slug="figs02_phase_a_heatmap",
        caption=(
            "Phase A re-evaluation on the v4_clean test split. Rows are "
            "SZA bins, columns are Phase A backbones (A0 through A9; "
            "A1-anchored variants A5a..A9a / A7b..A9b appear once their "
            "trainings complete). Left panel: mean per-pair IoU; right "
            "panel: mean per-pair root-length MAE in metres. UNet method "
            "only. Source: iceberg-rework/runs_summaries/exp_A*/<latest_ts>/"
            "re_eval_v4_clean/per_iceberg/eval_per_iceberg_summary.csv."
        ),
        out_dir=args.out_dir,
        dpi=200,
    )
    plt.close(fig)
    print(f"Saved figure: {archive_path}")


if __name__ == "__main__":
    main()
