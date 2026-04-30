"""
make_figure03_method_schematic.py: render Fig. 3, the six-method schematic.
A single test chip branches into two trunks: no-learning methods (TR, OT)
that operate on the image directly, and four UNet++ post-processing methods
that share the same softmax probability map. Pure diagram; no chip data.

Routes through _fig_registry.write into <out_dir>/fig-archive/.

Usage:
  python scripts/make_figure03_method_schematic.py --out_dir paper-writing/figures
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _diagram_helpers import (
    PALETTE, arrow, box, caption_text, edge_point, setup_axes,
)
from _fig_registry import write as fig_write


def main():
    parser = argparse.ArgumentParser(description="Fig. 3 six-method schematic")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(14, 8.5))
    fig.patch.set_facecolor("white")
    setup_axes(ax, xlim=(0, 100), ylim=(0, 80))

    # Layout: no-learning trunk centred at 18, UNet trunk centred at 65
    # UNet method centres at 50, 60, 72, 84. CRF max right edge = 84 + 5.5 = 89.5
    chip = box(ax, 50, 73, 22, 7,
                "Test chip", sublabel="B04 / B03 / B08, 256 x 256",
                fill=PALETTE["panel2"])

    # Left trunk: no-learning header + two methods
    nolearn = box(ax, 18, 56, 20, 6, "no learning",
                   fill=PALETTE["grey"], font_main=10)
    tr = box(ax, 9, 38, 14, 8, "TR (B0)",
              sublabel="fixed B08 >= 0.22\n(Fisser-equivalent)",
              fill=PALETTE["panel"])
    ot = box(ax, 27, 38, 14, 8, "OT (B1)",
              sublabel="per-chip Otsu\non B08",
              fill=PALETTE["panel"])
    box(ax, 9, 18, 12, 5, "segmentation",
         fill=PALETTE["cyan"], font_main=8)
    box(ax, 27, 18, 12, 5, "segmentation",
         fill=PALETTE["cyan"], font_main=8)

    # Right trunk: UNet++ probability map header + four methods
    pmap = box(ax, 65, 56, 38, 6, "UNet++ probability map",
                sublabel="one training run shared across four post-processors",
                fill=PALETTE["navy"])
    ax.texts[-2].set_color("white")  # main label
    ax.texts[-1].set_color("white")  # sublabel

    unet_centres = [50, 60, 72, 84]
    unet_labels = [
        ("UNet (B2)",       "argmax"),
        ("UNet_TR (B3)",    "P >= 0.22"),
        ("UNet_OT (B4)",    "per-chip Otsu\non P"),
        ("UNet_CRF (B5)",   "DenseCRF"),
    ]
    unet_boxes = []
    for cx, (lbl, sub) in zip(unet_centres, unet_labels):
        unet_boxes.append(
            box(ax, cx, 38, 10, 8, lbl, sublabel=sub,
                 fill=PALETTE["panel"], font_main=9, font_sub=7)
        )
    for cx in unet_centres:
        box(ax, cx, 18, 9, 5, "segmentation",
             fill=PALETTE["cyan"], font_main=7)

    # Chip -> trunks
    arrow(ax, edge_point(chip, "bottom"), edge_point(nolearn, "top"))
    arrow(ax, edge_point(chip, "bottom"), edge_point(pmap, "top"))

    # Trunk -> methods
    arrow(ax, edge_point(nolearn, "bottom"), edge_point(tr, "top"))
    arrow(ax, edge_point(nolearn, "bottom"), edge_point(ot, "top"))
    for m in unet_boxes:
        arrow(ax, edge_point(pmap, "bottom"), edge_point(m, "top"))

    # Method -> segmentation
    for cx in (9, 27) + tuple(unet_centres):
        arrow(ax, (cx, 33.5), (cx, 21))

    # Group divider
    ax.axvline(40, ymin=0.05, ymax=0.85, color="grey", lw=0.4, ls=":")

    fig.suptitle("Figure 3. Six segmentation methods on the same test chip",
                  fontsize=11, weight="bold", y=0.97)

    # Bottom legend / one-sentence motivation per leaf
    motivations = [
        (9,  "Fisser-equivalent\nfixed threshold"),
        (27, "adaptive threshold,\nno learning"),
        (50, "direct binary\nfrom argmax"),
        (60, "fixed threshold\non softmax probs"),
        (72, "per-chip Otsu\non softmax probs"),
        (84, "boundary refinement\nwith bilateral cues"),
    ]
    for cx, txt in motivations:
        caption_text(ax, cx, 12, txt, italic=True, font_size=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # 9. Route through fig registry
    caption = (
        "Six segmentation methods evaluated on the same test chip. Two "
        "methods operate directly on the input image (TR, OT). Four methods "
        "share the UNet++ probability map and differ only in post-processing "
        "(argmax, fixed threshold, per-chip Otsu, DenseCRF). One UNet++ "
        "training run produces all six method outputs."
    )
    archive = fig_write(
        fig=fig, slug="fig03_method_schematic",
        caption=caption, out_dir=args.out_dir, dpi=args.dpi,
    )
    plt.close(fig)
    print(f"Figure written: {archive}")


if __name__ == "__main__":
    main()
