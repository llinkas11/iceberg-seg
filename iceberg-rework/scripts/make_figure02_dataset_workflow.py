"""
make_figure02_dataset_workflow.py: render Fig. 2, the dataset construction
workflow. Two annotation sources (Roboflow polygons, Fisser 3-class pkls)
flow through three deterministic cleaning operations (shadow merge, 40 m
component filter, IC pixel mask) into a binary train / validation / test
split. Pure diagram; no chip data required.

Routes through _fig_registry.write into <out_dir>/fig-archive/.

Usage:
  python scripts/make_figure02_dataset_workflow.py --out_dir paper-writing/figures
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _diagram_helpers import (
    PALETTE, arrow, box, caption_text, edge_point, setup_axes,
)
from _fig_registry import write as fig_write


# 1. Layout constants (axis 0..120 horizontal, 0..70 vertical).
# Wider canvas so arrows have visible length between nodes.
SOURCE_X = 8
TRUNK_X = [30, 55, 80]      # shadow merge, 40 m filter, IC mask
SPLIT_X = 108
SOURCE_W, SOURCE_H = 14, 10
TRUNK_W, TRUNK_H = 16, 10
SPLIT_W, SPLIT_H = 18, 6
SOURCE_RF_Y = 50
SOURCE_FI_Y = 22
TRUNK_Y = 36
SPLIT_TRAIN_Y = 50
SPLIT_VAL_Y = 40
SPLIT_TEST_Y = 30
CAPTION_Y = 12  # below the lowest source box


def main():
    parser = argparse.ArgumentParser(description="Fig. 2 dataset workflow")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(18, 8.5))
    fig.patch.set_facecolor("white")
    setup_axes(ax, xlim=(0, 120), ylim=(0, 70))

    # 2. Two source boxes (top: Roboflow, bottom: Fisser)
    src_rf = box(ax, SOURCE_X, SOURCE_RF_Y, SOURCE_W, SOURCE_H,
                  "Roboflow polygons",
                  sublabel="single-class iceberg,\nSAM3 + manual review",
                  fill=PALETTE["panel"], font_main=13, font_sub=10)
    src_fi = box(ax, SOURCE_X, SOURCE_FI_Y, SOURCE_W, SOURCE_H,
                  "Fisser (2024) arrays",
                  sublabel="3-class (ocean / iceberg / shadow),\nSZA < 65",
                  fill=PALETTE["panel2"], font_main=13, font_sub=10)

    # 3. Trunk operations (centred between source heights)
    op_shadow = box(ax, TRUNK_X[0], TRUNK_Y, TRUNK_W, TRUNK_H,
                     "shadow merge",
                     sublabel="class 2 -> 1\n(Fisser only)",
                     fill=PALETTE["panel"], font_main=13, font_sub=10)
    op_40m = box(ax, TRUNK_X[1], TRUNK_Y, TRUNK_W, TRUNK_H,
                  "40 m filter",
                  sublabel="drop CC < 16 px",
                  fill=PALETTE["panel"], font_main=13, font_sub=10)
    op_ic = box(ax, TRUNK_X[2], TRUNK_Y, TRUNK_W, TRUNK_H,
                 "IC pixel mask",
                 sublabel="train only,\nIC >= 15 %",
                 fill=PALETTE["panel"], font_main=13, font_sub=10)

    # 4. Three split outputs (right side)
    box(ax, SPLIT_X, SPLIT_TRAIN_Y, SPLIT_W, SPLIT_H,
         "train  (551 chips)", fill=PALETTE["cyan"], font_main=13)
    box(ax, SPLIT_X, SPLIT_VAL_Y, SPLIT_W, SPLIT_H,
         "validation  (137 chips)", fill=PALETTE["cyan"], font_main=13)
    box(ax, SPLIT_X, SPLIT_TEST_Y, SPLIT_W, SPLIT_H,
         "test  (228 chips)", fill=PALETTE["cyan"], font_main=13)

    # 5. Connections: sources -> shadow merge
    arrow(ax, edge_point(src_rf, "right"),
           (TRUNK_X[0] - TRUNK_W / 2, TRUNK_Y + 2))
    arrow(ax, edge_point(src_fi, "right"),
           (TRUNK_X[0] - TRUNK_W / 2, TRUNK_Y - 2))

    # 6. Trunk: shadow -> 40m -> IC
    arrow(ax, edge_point(op_shadow, "right"), edge_point(op_40m, "left"))
    arrow(ax, edge_point(op_40m, "right"), edge_point(op_ic, "left"))

    # 7. IC -> three splits
    ic_right = edge_point(op_ic, "right")
    for ysplit in (SPLIT_TRAIN_Y, SPLIT_VAL_Y, SPLIT_TEST_Y):
        arrow(ax, ic_right, (SPLIT_X - SPLIT_W / 2, ysplit))

    # 8. Bottom captions for each operation
    caption_text(ax, TRUNK_X[0], CAPTION_Y,
                  "binary targets so all\nsix methods share GT",
                  italic=True, font_size=11)
    caption_text(ax, TRUNK_X[1], CAPTION_Y,
                  "removes rasterisation\nartefacts (CC < 16 px)",
                  italic=True, font_size=11)
    caption_text(ax, TRUNK_X[2], CAPTION_Y,
                  "zeros bright non-annotated\npixels in train chips",
                  italic=True, font_size=11)
    caption_text(ax, SPLIT_X, CAPTION_Y,
                  "stratified 65 / 15 / 25;\ntest capped at 57 per bin",
                  italic=True, font_size=11)

    fig.suptitle("Figure 2. Dataset construction workflow",
                  fontsize=15, weight="bold", y=0.96)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # 9. Route through fig registry
    caption = (
        "Dataset construction workflow. Two annotation sources are merged "
        "through three deterministic cleaning operations to produce the "
        "binary v4_clean train / validation / test splits. "
        "Sources: Roboflow polygons (single-class iceberg, SAM3 smart-select "
        "+ manual review, SZA >= 65 deg) and Fisser (2024) arrays (3-class "
        "ocean / iceberg / shadow numpy arrays distributed as pickle "
        "files, SZA < 65 deg). "
        "Cleaning operations in order: (1) shadow merge - Fisser shadow "
        "pixels (class 2) are reclassified as iceberg (class 1) so every "
        "chip shares the same binary target; (2) 40 m component filter - "
        "connected components below 16 pixels (40 m root length at the "
        "10 m Sentinel-2 ground sample distance) are dropped from both "
        "sources to remove rasterisation artefacts and sub-resolution "
        "annotations; (3) annotation-aware ice-coverage (IC) pixel mask - "
        "bright non-annotated pixels in training chips whose ice-coverage "
        "fraction is >= 15 % are zeroed before training so the model does "
        "not learn that bright non-iceberg pixels (sea ice, melange) are "
        "negative examples. The IC mask applies only to training chips; "
        "validation and test chips are never masked. "
        "Output: a stratified 65 / 15 / 25 split yielding 551 train, 137 "
        "validation, and 228 test chips, with the test split capped at 57 "
        "chips per SZA bin."
    )
    archive = fig_write(
        fig=fig, slug="fig02_dataset_workflow",
        caption=caption, out_dir=args.out_dir, dpi=args.dpi,
    )
    svg_path = archive[:-4] + ".svg"
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written: {archive}")
    print(f"SVG companion: {svg_path}")


if __name__ == "__main__":
    main()
