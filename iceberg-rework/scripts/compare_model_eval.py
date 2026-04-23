#!/usr/bin/env python3
"""Summarize baseline-vs-stage1 model evaluation outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SZA_ORDER = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]
SZA_LABELS = {
    "sza_lt65": "lt65",
    "sza_65_70": "65-70",
    "sza_70_75": "70-75",
    "sza_gt75": "gt75",
}
MODEL_METHODS = ["UNet", "UNet_TR", "UNet_OT", "UNet_CRF"]


def read_summary(eval_dir: str, positive_only: bool) -> pd.DataFrame:
    name = "eval_summary_gt_positive_only.csv" if positive_only else "eval_summary.csv"
    path = os.path.join(eval_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["sza_bin"] = pd.Categorical(df["sza_bin"], categories=SZA_ORDER, ordered=True)
    return df.sort_values(["method", "sza_bin"]).reset_index(drop=True)


def compare_metric(
    baseline: pd.DataFrame,
    stage1: pd.DataFrame,
    baseline_label: str,
    stage1_label: str,
    metric: str,
) -> pd.DataFrame:
    col = f"mean_{metric}"
    left = baseline[["method", "sza_bin", col]].rename(columns={col: baseline_label})
    right = stage1[["method", "sza_bin", col]].rename(columns={col: stage1_label})
    out = left.merge(right, on=["method", "sza_bin"], how="outer")
    out["delta_stage1_minus_baseline"] = out[stage1_label] - out[baseline_label]
    out["sza_bin"] = pd.Categorical(out["sza_bin"], categories=SZA_ORDER, ordered=True)
    return out.sort_values(["method", "sza_bin"]).reset_index(drop=True)


def markdown_table(df: pd.DataFrame, float_cols: list[str]) -> str:
    view = df.copy()
    for col in float_cols:
        if col in view.columns:
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    columns = list(view.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in view.iterrows():
        values = ["" if pd.isna(row[col]) else str(row[col]) for col in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def pivot_model_iou(comp: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = comp[comp["method"].isin(MODEL_METHODS)].copy()
    df["SZA"] = df["sza_bin"].map(SZA_LABELS)
    return df.pivot(index="method", columns="SZA", values=value_col).reindex(MODEL_METHODS)


def plot_delta_heatmap(comp: pd.DataFrame, out_path: str, title: str) -> None:
    pivot = pivot_model_iou(comp, "delta_stage1_minus_baseline")
    data = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    vmax = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 1.0
    vmax = max(vmax, 0.05)
    im = ax.imshow(data, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="IoU delta")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title, fontweight="bold")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def test_set_counts(split_log: str) -> pd.DataFrame:
    log = pd.read_csv(split_log)
    test = log[log["split"].eq("test")].copy()
    test["gt_group"] = np.where(test["n_icebergs"].fillna(0).astype(float) > 0, "GT+", "GT0")
    counts = (
        test.groupby(["sza_bin", "gt_group"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(SZA_ORDER)
        .reset_index()
    )
    counts["total"] = counts.get("GT0", 0) + counts.get("GT+", 0)
    counts["SZA"] = counts["sza_bin"].map(SZA_LABELS)
    return counts[["SZA", "GT0", "GT+", "total"]]


def extract_training_summary(path: str | None) -> list[str]:
    if not path or not os.path.exists(path):
        return []
    keep_prefixes = (
        "- Best validation IoU:",
        "- Final logged epoch:",
        "- Test IoU:",
        "- Test loss:",
        "- Best checkpoint:",
    )
    lines = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(keep_prefixes):
            lines.append(line)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--split_log", required=True)
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--stage1_dir", required=True)
    parser.add_argument("--baseline_label", default="baseline")
    parser.add_argument("--stage1_label", default="stage1")
    parser.add_argument("--training_summary", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    baseline_all = read_summary(args.baseline_dir, positive_only=False)
    stage1_all = read_summary(args.stage1_dir, positive_only=False)
    baseline_pos = read_summary(args.baseline_dir, positive_only=True)
    stage1_pos = read_summary(args.stage1_dir, positive_only=True)

    comp_all = compare_metric(baseline_all, stage1_all, args.baseline_label, args.stage1_label, "iou")
    comp_pos = compare_metric(baseline_pos, stage1_pos, args.baseline_label, args.stage1_label, "iou")

    comp_all.to_csv(os.path.join(args.out_root, "model_iou_comparison_all_chips.csv"), index=False)
    comp_pos.to_csv(os.path.join(args.out_root, "model_iou_comparison_gt_positive_only.csv"), index=False)

    lt65_pos = comp_pos[(comp_pos["sza_bin"] == "sza_lt65") & comp_pos["method"].isin(MODEL_METHODS)].copy()
    lt65_pos.to_csv(os.path.join(args.out_root, "lt65_gt_positive_iou_comparison.csv"), index=False)

    plot_delta_heatmap(
        comp_pos,
        os.path.join(args.out_root, "iou_delta_heatmap_gt_positive_only.png"),
        "Stage-1 minus baseline IoU, GT-positive chips",
    )
    plot_delta_heatmap(
        comp_all,
        os.path.join(args.out_root, "iou_delta_heatmap_all_chips.png"),
        "Stage-1 minus baseline IoU, all chips",
    )

    counts = test_set_counts(args.split_log)
    counts.to_csv(os.path.join(args.out_root, "test_set_gt_counts_by_sza.csv"), index=False)

    model_pos = comp_pos[comp_pos["method"].isin(MODEL_METHODS)].copy()
    model_all = comp_all[comp_all["method"].isin(MODEL_METHODS)].copy()
    model_pos["SZA"] = model_pos["sza_bin"].map(SZA_LABELS)
    model_all["SZA"] = model_all["sza_bin"].map(SZA_LABELS)

    best_lt65 = lt65_pos.sort_values("delta_stage1_minus_baseline", ascending=False).head(1)
    if len(best_lt65):
        best_line = (
            f"Best lt65 GT-positive delta: `{best_lt65.iloc[0]['method']}` "
            f"{best_lt65.iloc[0]['delta_stage1_minus_baseline']:+.4f} IoU "
            f"({args.baseline_label}={best_lt65.iloc[0][args.baseline_label]:.4f}, "
            f"{args.stage1_label}={best_lt65.iloc[0][args.stage1_label]:.4f})."
        )
    else:
        best_line = "No lt65 GT-positive model-method rows were available."

    lines = [
        "# Model Comparison Summary",
        "",
        f"Output root: `{args.out_root}`",
        f"Baseline: `{args.baseline_label}`",
        f"Stage-1 model: `{args.stage1_label}`",
        "",
        "## Test Set",
        markdown_table(counts, []),
        "",
        "## Headline",
        best_line,
        "",
        "## GT-Positive IoU By SZA",
        markdown_table(
            model_pos[["method", "SZA", args.baseline_label, args.stage1_label, "delta_stage1_minus_baseline"]],
            [args.baseline_label, args.stage1_label, "delta_stage1_minus_baseline"],
        ),
        "",
        "## All-Chip IoU By SZA",
        markdown_table(
            model_all[["method", "SZA", args.baseline_label, args.stage1_label, "delta_stage1_minus_baseline"]],
            [args.baseline_label, args.stage1_label, "delta_stage1_minus_baseline"],
        ),
        "",
        "## lt65 GT-Positive Focus",
        markdown_table(
            lt65_pos[["method", args.baseline_label, args.stage1_label, "delta_stage1_minus_baseline"]],
            [args.baseline_label, args.stage1_label, "delta_stage1_minus_baseline"],
        ),
        "",
        "## Stage-1 Training Snapshot",
    ]
    training_lines = extract_training_summary(args.training_summary)
    lines.extend(training_lines if training_lines else ["No training summary was provided."])
    lines.extend(
        [
            "",
            "## Files",
            "- `model_iou_comparison_all_chips.csv`",
            "- `model_iou_comparison_gt_positive_only.csv`",
            "- `lt65_gt_positive_iou_comparison.csv`",
            "- `iou_delta_heatmap_all_chips.png`",
            "- `iou_delta_heatmap_gt_positive_only.png`",
            "- `eval_outputs/baseline_v3_balanced_aug/`",
            "- `eval_outputs/stage1_sza_gt_balance/`",
        ]
    )

    summary_path = os.path.join(args.out_root, "summary_sheet.md")
    Path(summary_path).write_text("\n".join(lines) + "\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
