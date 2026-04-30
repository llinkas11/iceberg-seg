"""
make_figure04_evaluation_schematic.py: render Fig. 4, the per-iceberg
evaluation pipeline. Left panel shows synthetic GT components (gold filled),
predicted components (cyan outline), Hungarian matches with IoU labels,
and one each of unmatched FN / FP. Right panel is the resulting metric
table. Pure diagram; no chip data.

Routes through _fig_registry.write into <out_dir>/fig-archive/.

Usage:
  python scripts/make_figure04_evaluation_schematic.py --out_dir paper-writing/figures
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from _diagram_helpers import PALETTE, setup_axes
from _fig_registry import write as fig_write


# 1. Synthetic component layouts: (cx, cy, w, h)
GT = [
    (18, 70, 11, 8),    # g1
    (38, 78, 9,  7),    # g2
    (62, 60, 13, 9),    # g3
    (28, 35, 10, 7),    # g4 (will be FN)
    (75, 28, 12, 8),    # g5
]
PRED = [
    (20, 68, 10, 7),    # p1 -> g1
    (39, 76, 8,  6),    # p2 -> g2
    (61, 62, 12, 8),    # p3 -> g3
    (76, 30, 11, 7),    # p4 -> g5
    (50, 18, 9,  6),    # p5 (will be FP)
]
# Pairs (gt_idx, pred_idx, IoU)
MATCHES = [
    (0, 0, 0.71),
    (1, 1, 0.62),
    (2, 2, 0.58),
    (4, 3, 0.49),
]
GT_FN = 3   # index in GT list
PRED_FP = 4  # index in PRED list


def main():
    parser = argparse.ArgumentParser(description="Fig. 4 evaluation schematic")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    fig = plt.figure(figsize=(13, 6.5))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.05)

    # 2. Left panel: synthetic chip with GT, pred, matches
    ax = fig.add_subplot(gs[0, 0])
    setup_axes(ax, xlim=(0, 100), ylim=(0, 100))
    ax.set_facecolor("#0e1a2e")  # dark "ocean" background for visual contrast
    rect = mpatches.Rectangle((0, 0), 100, 100, facecolor="#0e1a2e",
                                edgecolor="black", linewidth=1.2)
    ax.add_patch(rect)
    ax.set_title("Predicted vs ground-truth components on one test chip",
                  fontsize=10, color=PALETTE["ink"], pad=8)

    # 2a. Ground-truth filled gold
    for i, (cx, cy, w, h) in enumerate(GT):
        e = mpatches.FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                      boxstyle="round,pad=0.2,rounding_size=2",
                                      facecolor=PALETTE["gold"],
                                      edgecolor=PALETTE["gold"],
                                      linewidth=1.0, alpha=0.85)
        ax.add_patch(e)
        ax.text(cx, cy, f"g{i+1}", ha="center", va="center",
                  fontsize=8, color="black", weight="bold")

    # 2b. Predicted cyan outline
    for j, (cx, cy, w, h) in enumerate(PRED):
        e = mpatches.FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                      boxstyle="round,pad=0.2,rounding_size=2",
                                      facecolor="none",
                                      edgecolor=PALETTE["cyan"],
                                      linewidth=1.6)
        ax.add_patch(e)
        ax.text(cx + w / 2 + 1.5, cy + h / 2 + 0.5, f"p{j+1}",
                  ha="left", va="bottom", fontsize=8,
                  color=PALETTE["cyan"], weight="bold")

    # 2c. Hungarian-match arrows + IoU labels
    # Position the IoU label at a fixed offset above the GT box so it never
    # overlaps a component; use a per-pair offset for readability.
    iou_label_offsets = {0: (4, 7), 1: (5, 6), 2: (6, 7), 3: (5, 7)}
    for k, (gi, pi, iou) in enumerate(MATCHES):
        gx, gy, gw, gh = GT[gi]
        px, py, _, _ = PRED[pi]
        ax.annotate("", xy=(px, py), xytext=(gx, gy),
                      arrowprops=dict(arrowstyle="-", linewidth=1.0,
                                       color="white", linestyle=(0, (3, 2))))
        dx, dy = iou_label_offsets.get(k, (4, 6))
        lx = (gx + px) / 2 + dx
        ly = max(gy, py) + dy
        ax.text(lx, ly, f"IoU {iou:.2f}", ha="center", va="center",
                  fontsize=8, color="white",
                  bbox=dict(facecolor="#0e1a2e", edgecolor="white",
                            linewidth=0.5, pad=1.5))

    # 2d. FN marker
    fn_cx, fn_cy, fn_w, fn_h = GT[GT_FN]
    ax.annotate("FN", xy=(fn_cx, fn_cy - fn_h / 2 - 1.5),
                  xytext=(fn_cx, fn_cy - fn_h / 2 - 7),
                  ha="center", fontsize=10, weight="bold",
                  color=PALETTE["crimson"],
                  arrowprops=dict(arrowstyle="-|>", color=PALETTE["crimson"],
                                   linewidth=1.2, mutation_scale=12))
    # 2e. FP marker
    fp_cx, fp_cy, fp_w, fp_h = PRED[PRED_FP]
    ax.annotate("FP", xy=(fp_cx, fp_cy + fp_h / 2 + 1.5),
                  xytext=(fp_cx, fp_cy + fp_h / 2 + 7),
                  ha="center", fontsize=10, weight="bold",
                  color=PALETTE["crimson"],
                  arrowprops=dict(arrowstyle="-|>", color=PALETTE["crimson"],
                                   linewidth=1.2, mutation_scale=12))

    # 2f. Legend in lower-left, raised above bottom edge so the third entry
    #     (red, unmatched FN/FP) is fully visible. Bottom-most rect bottom
    #     edge sits at y=8, well clear of y=0 even when bbox_inches='tight'
    #     crops aggressively or PDF includegraphics adds a sub-pixel inset.
    legend_items = [
        (PALETTE["gold"], "ground truth"),
        (PALETTE["cyan"], "predicted"),
        (PALETTE["crimson"], "unmatched (FN / FP)"),
    ]
    for k, (c, lab) in enumerate(legend_items):
        y = 20 - k * 4
        rect = mpatches.Rectangle((3, y), 4, 2.5, facecolor=c, edgecolor=c)
        ax.add_patch(rect)
        ax.text(8, y + 1.25, lab, ha="left", va="center",
                  fontsize=9, color="white")

    # 3. Right panel: results table
    ax2 = fig.add_subplot(gs[0, 1])
    setup_axes(ax2, xlim=(0, 100), ylim=(0, 100))
    ax2.set_title("Per-pair metrics (this chip)",
                   fontsize=10, color=PALETTE["ink"], pad=8)
    rows = [
        ("matched pairs",         "4"),
        ("unmatched FN",          "1"),
        ("unmatched FP",          "1"),
        ("match rate",            "80 %"),
        ("precision",             "80 %"),
        ("mean IoU (matched)",    "0.60"),
        ("MAE area (m$^2$)",      "1240"),
        ("MAE root length (m)",   "8.7"),
        ("MSE area (m$^4$)",      "$2.1 \\times 10^7$"),
    ]
    # Header
    header_y = 88
    ax2.text(8, header_y, "metric", ha="left", va="center",
              fontsize=10, weight="bold", color=PALETTE["ink"])
    ax2.text(80, header_y, "value", ha="right", va="center",
              fontsize=10, weight="bold", color=PALETTE["ink"])
    ax2.plot([5, 95], [header_y - 4, header_y - 4],
              color=PALETTE["ink"], linewidth=1.2)

    row_h = 8
    for k, (metric, value) in enumerate(rows):
        y = header_y - 8 - k * row_h
        ax2.text(8, y, metric, ha="left", va="center",
                  fontsize=9, color=PALETTE["ink"])
        ax2.text(80, y, value, ha="right", va="center",
                  fontsize=9, color=PALETTE["ink"], family="monospace")
    # Bottom rule
    bottom_y = header_y - 8 - len(rows) * row_h + 4
    ax2.plot([5, 95], [bottom_y, bottom_y],
              color=PALETTE["ink"], linewidth=1.2)

    fig.suptitle("Figure 4. Per-iceberg evaluation pipeline",
                  fontsize=11, weight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # 4. Route through fig registry
    caption = (
        "Per-iceberg evaluation pipeline. Ground-truth components and "
        "predicted components are matched by Hungarian assignment on "
        "(1 - IoU), keeping pairs with IoU >= 0.30. Unmatched components "
        "count as false negatives or false positives in the detection "
        "table; matched pairs feed mean absolute error on area, mean "
        "absolute error on root length, mean IoU, and the area MSE "
        "reported per (method, SZA bin) cell."
    )
    archive = fig_write(
        fig=fig, slug="fig04_evaluation_schematic",
        caption=caption, out_dir=args.out_dir, dpi=args.dpi,
    )
    plt.close(fig)
    print(f"Figure written: {archive}")


if __name__ == "__main__":
    main()
