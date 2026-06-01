"""
make_figure03_method_schematic.py: render Fig. 3, the six-method schematic.
A single test chip branches into two trunks: no-learning methods (TR, OT)
that operate on the image directly, and four UNet++ post-processing methods
(Phase B) that share the same softmax probability map. Pure diagram; no
chip data.

Color conventions:
- Red path (chip -> no learning -> TR -> segmentation) = Fisser (2024)
  baseline. Black paths everywhere else are added in this work.
- Phase B methods (UNet, UNet_TR, UNet_OT, UNet_CRF) carry navy text so the
  learning trunk is visually distinct.
- Teal segmentation boxes are the per-method outputs; the per-method
  motivation is rendered inside the box so the figure does not need a
  separate caption row.

Routes through _fig_registry.write into <out_dir>/fig-archive/. An editable
SVG companion is saved alongside the PNG.

Usage:
  python scripts/make_figure03_method_schematic.py --out_dir paper-writing/figures
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from _diagram_helpers import (
    PALETTE, arrow, box, edge_point, setup_axes,
)
from _fig_registry import write as fig_write


FISSER_RED = PALETTE["crimson"]   # marks the Fisser (2024) baseline path
PHASE_B_NAVY = PALETTE["navy"]    # marks UNet++ derived methods
SEG_TEAL = PALETTE["cyan"]        # segmentation-output color
PANEL_LW = 0.8                    # lighter box borders so they read as panels
HIGHLIGHT_LW = 1.6                # heavier border for highlighted boxes


def main():
    parser = argparse.ArgumentParser(description="Fig. 3 six-method schematic")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(19, 10))
    fig.patch.set_facecolor("white")
    setup_axes(ax, xlim=(0, 100), ylim=(0, 80))

    # 1. Test chip at top
    chip = box(ax, 50, 73, 22, 7,
                "Test chip", sublabel="B04 / B03 / B08, 256 x 256",
                fill=PALETTE["panel2"], lw=PANEL_LW,
                font_main=13, font_sub=10)

    # 2. Trunk headers. UNet++ probability map is the wider Phase B trunk.
    nolearn = box(ax, 18, 56, 20, 6, "no learning",
                   fill=PALETTE["grey"], lw=PANEL_LW,
                   text_color="white", font_main=13)
    pmap = box(ax, 65, 56, 42, 6, "UNet++ probability map",
                sublabel="one training run shared across four post-processors",
                fill=PHASE_B_NAVY, lw=PANEL_LW,
                text_color="white", font_main=13, font_sub=10)

    # 3. Method boxes. TR (Fisser-equivalent) carries a red border. The four
    # UNet methods carry navy text to mark them as Phase B.
    tr = box(ax, 9, 38, 14, 8, "TR",
              sublabel="fixed B08 >= 0.22\n(Fisser-equivalent)",
              fill=PALETTE["panel"], ec=FISSER_RED, lw=HIGHLIGHT_LW,
              font_main=14, font_sub=10)
    ot = box(ax, 27, 38, 14, 8, "OT",
              sublabel="per-chip Otsu\non B08",
              fill=PALETTE["panel"], lw=PANEL_LW,
              font_main=14, font_sub=10)

    # Spread the four UNet method boxes so they no longer touch under pmap.
    unet_centres = [48, 60, 72, 84]
    unet_labels = [
        ("UNet",      "argmax"),
        ("UNet_TR",   "P >= 0.22"),
        ("UNet_OT",   "per-chip Otsu\non P"),
        ("UNet_CRF",  "DenseCRF"),
    ]
    unet_boxes = []
    for cx, (lbl, sub) in zip(unet_centres, unet_labels):
        unet_boxes.append(
            box(ax, cx, 38, 10, 8, lbl, sublabel=sub,
                fill=PALETTE["panel"], lw=PANEL_LW,
                text_color=PHASE_B_NAVY,
                font_main=14, font_sub=10),
        )

    # 4. Segmentation outputs. The per-method motivation is now rendered
    # inside each teal box (legend defines this row as "segmentation"); the
    # box height is bumped from 5 to 8 so the two-line motivation fits.
    motivations = {
        9:  "Fisser-equivalent\nfixed threshold",
        27: "adaptive threshold,\nno learning",
        48: "direct binary\nfrom argmax",
        60: "fixed threshold\non softmax probs",
        72: "per-chip Otsu\non softmax probs",
        84: "boundary refinement\nwith bilateral cues",
    }
    seg_y = 16
    seg_h = 8
    seg_w_left = 12   # under TR/OT
    seg_w_right = 10  # under UNet methods (matched to method box width)

    box(ax, 9, seg_y, seg_w_left, seg_h, motivations[9],
        fill=SEG_TEAL, ec=FISSER_RED, lw=HIGHLIGHT_LW,
        text_color="white", font_main=10)
    box(ax, 27, seg_y, seg_w_left, seg_h, motivations[27],
        fill=SEG_TEAL, lw=PANEL_LW,
        text_color="white", font_main=10)
    for cx in unet_centres:
        box(ax, cx, seg_y, seg_w_right, seg_h, motivations[cx],
            fill=SEG_TEAL, lw=PANEL_LW,
            text_color="white", font_main=10)

    # 5. Arrows
    # 5a. Chip -> trunks. The chip -> no-learning arrow is red because the
    # Fisser path passes through it.
    arrow(ax, edge_point(chip, "bottom"), edge_point(nolearn, "top"),
          color=FISSER_RED, lw=HIGHLIGHT_LW)
    arrow(ax, edge_point(chip, "bottom"), edge_point(pmap, "top"))

    # 5b. Trunk -> methods. no-learning -> TR is red; OT and UNet branches
    # stay black.
    arrow(ax, edge_point(nolearn, "bottom"), edge_point(tr, "top"),
          color=FISSER_RED, lw=HIGHLIGHT_LW)
    arrow(ax, edge_point(nolearn, "bottom"), edge_point(ot, "top"))
    for m in unet_boxes:
        arrow(ax, edge_point(pmap, "bottom"), edge_point(m, "top"))

    # 5c. Method -> segmentation. TR's leaf arrow is red.
    method_seg_pairs = [(9, 9), (27, 27)] + [(c, c) for c in unet_centres]
    for method_x, seg_x in method_seg_pairs:
        is_fisser = (method_x == 9)
        arrow(ax, (method_x, 38 - 4),  # method box bottom
              (seg_x, seg_y + seg_h / 2),
              color=(FISSER_RED if is_fisser else None),
              lw=(HIGHLIGHT_LW if is_fisser else None))

    # 6. Group divider between non-learning and learning trunks
    ax.axvline(40, ymin=0.05, ymax=0.85, color="grey", lw=0.4, ls=":")

    # 7. Fisser 2024 attribution under TR's segmentation box
    ax.text(9, 8.5, "Fisser (2024)",
            ha="center", va="center",
            fontsize=12, weight="bold", color=FISSER_RED)

    # 8. Suptitle
    fig.suptitle("Figure 3. Six segmentation methods on the same test chip",
                 fontsize=15, weight="bold", y=0.97)

    # 9. Legend in the upper-right with arrow icons + color swatches.
    # Arrow rows for path coloring; swatch rows for Phase B and segmentation.
    legend_x = 71.5
    legend_top = 79.5
    legend_w = 27
    legend_h = 13
    # Background plate
    ax.add_patch(mpatches.FancyBboxPatch(
        (legend_x, legend_top - legend_h), legend_w, legend_h,
        boxstyle="round,pad=0.5",
        linewidth=0.8, edgecolor="#999", facecolor="white",
    ))

    def _arrow_icon(y, color):
        ax.annotate(
            "", xy=(legend_x + 5.5, y), xytext=(legend_x + 1.2, y),
            arrowprops=dict(arrowstyle="-|>", color=color,
                            linewidth=HIGHLIGHT_LW, mutation_scale=12),
        )

    def _swatch(y, fill, edge=None):
        ax.add_patch(mpatches.Rectangle(
            (legend_x + 1.5, y - 1.0), 4.0, 2.0,
            facecolor=fill,
            edgecolor=edge if edge else fill,
            linewidth=1.0,
        ))

    # Row 1: red arrow + Fisser baseline
    y1 = legend_top - 2.5
    _arrow_icon(y1, FISSER_RED)
    ax.text(legend_x + 7, y1, "Fisser (2024) baseline",
            ha="left", va="center", fontsize=10, color=FISSER_RED)
    # Row 2: black arrow + added in this work
    y2 = legend_top - 5.5
    _arrow_icon(y2, "black")
    ax.text(legend_x + 7, y2, "added in this work",
            ha="left", va="center", fontsize=10, color="black")
    # Row 3: navy swatch + Phase B (UNet++)
    y3 = legend_top - 8.5
    _swatch(y3, PHASE_B_NAVY)
    ax.text(legend_x + 7, y3, "Phase B (UNet++ derived)",
            ha="left", va="center", fontsize=10, color=PHASE_B_NAVY,
            weight="bold")
    # Row 4: teal swatch + segmentation
    y4 = legend_top - 11.5
    _swatch(y4, SEG_TEAL)
    ax.text(legend_x + 7, y4, "segmentation output",
            ha="left", va="center", fontsize=10, color=SEG_TEAL,
            weight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # 10. Save through fig registry + SVG companion
    caption = (
        "Six segmentation methods evaluated on the same test chip. The chip "
        "splits into a non-learning trunk (TR, OT) and a UNet++ probability "
        "map shared by four post-processors (UNet, UNet_TR, UNet_OT, "
        "UNet_CRF). Each leaf box is labelled with the per-method "
        "post-processing recipe and is filled teal as the per-method "
        "segmentation output. "
        "Color coding: the red path (chip -> no learning -> TR -> "
        "segmentation) is the Fisser (2024) baseline; black paths are "
        "added in this work. Phase B methods (the four UNet++ "
        "post-processors) carry navy text to mark them as the learning "
        "trunk."
    )
    archive = fig_write(
        fig=fig, slug="fig03_method_schematic",
        caption=caption, out_dir=args.out_dir, dpi=args.dpi,
    )
    svg_path = archive[:-4] + ".svg"
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written: {archive}")
    print(f"SVG companion: {svg_path}")


if __name__ == "__main__":
    main()
