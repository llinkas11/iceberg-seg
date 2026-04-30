"""
make_figure05_progression.py: render Fig. 5, the two-phase experimental
progression. Phase A is the dataset-cleaning and balancing chain (10 nodes).
Phase B is the six-method sweep on the frozen Phase A winner. Each node
carries a one-sentence reader-facing motivation; manifest names, scheme
letters, and YAML keys are deliberately absent. Pure diagram; no chip data.

Layout: text-only nodes (no boxes), single-axes diagram. Coordinates are
chosen so vertical row spacing exceeds two-line sentence height, eliminating
overlap. The fig registry saves with bbox_inches='tight' so the empty
axes margins are cropped automatically.

Routes through _fig_registry.write into <out_dir>/fig-archive/.

Usage:
  python scripts/make_figure05_progression.py --out_dir paper-writing/figures
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _diagram_helpers import PALETTE, arrow, caption_text, setup_axes
from _fig_registry import write as fig_write


# 1. Phase A nodes (10). Main chain A0->A4 across the top; balancing variants
#    A5/A6/A7 fork downward; A8/A9 combine balance steps.
A_NODES = [
    # (id, x, y, label, sentence)
    ("A0",  10, 83, "A0", "Fisser baseline\nlt65 only, drop GT0"),
    ("A1",  30, 83, "A1", "A0 + 29 null chips\n1:1 + undersample"),
    ("A2",  50, 83, "A2", "+ 40 m filter\n+ IC mask"),
    ("A3",  70, 83, "A3", "A2 + 29 null chips\n1:1 GT+/GT0"),
    ("A4",  90, 83, "A4", "+ augmentation\nhflip / vflip / rot90"),
    ("A5",  50, 64, "A5", "fixed 2:1\nbalancing"),
    ("A6",  70, 64, "A6", "adaptive 2:1\nbalancing"),
    ("A7",  90, 64, "A7", "size oversample\nRL bins (4x cap)"),
    ("A8",  60, 47, "A8", "fixed 2:1 +\nsize oversample"),
    ("A9",  80, 47, "A9", "adaptive 2:1 +\nsize oversample"),
]
A_EDGES = [
    # main chain across the top row
    ("A0", "A1"), ("A1", "A2"), ("A2", "A3"), ("A3", "A4"),
    # forks from A4 down into the balancing grid
    ("A4", "A5"), ("A4", "A6"), ("A4", "A7"),
    # combined-balance leaves
    ("A5", "A8"), ("A6", "A9"),
]

# 2. Phase B nodes (6, single row across the bottom). Wider x-spacing than the
#    Phase A rows so the longer two-line method labels do not collide.
B_NODES = [
    ("B0",  8, 18, "B0", "fixed B08 >= 0.22"),
    ("B1", 26, 18, "B1", "per-chip Otsu on B08"),
    ("B2", 44, 18, "B2", "UNet++ argmax"),
    ("B3", 62, 18, "B3", "UNet++ probs +\nfixed threshold"),
    ("B4", 80, 18, "B4", "UNet++ probs +\nper-chip Otsu"),
    ("B5", 98, 18, "B5", "UNet++ probs +\nDenseCRF"),
]

# 3. Virtual rectangle dimensions used only for arrow routing. No box is drawn.
NODE_HALF_W = 7
NODE_HALF_H = 4


def text_node(ax, x, y, label, sentence, *, label_color=None):
    """
    Render a text-only node centred at (x, y). Bold label on top, italic
    sentence underneath. Spacing tuned so two-line sentence fits within
    the virtual node bounds (NODE_HALF_H = 4).
    """
    color = label_color or PALETTE["ink"]
    ax.text(x, y + 1.4, label,
            ha="center", va="center", fontsize=11,
            weight="bold", color=color)
    ax.text(x, y - 1.4, sentence,
            ha="center", va="top", fontsize=7,
            style="italic", color=color)


def node_edge(x, y, side):
    """Anchor point on the virtual NODE_HALF_W x NODE_HALF_H rectangle."""
    if side == "right":  return (x + NODE_HALF_W, y)
    if side == "left":   return (x - NODE_HALF_W, y)
    if side == "top":    return (x, y + NODE_HALF_H)
    if side == "bottom": return (x, y - NODE_HALF_H)
    raise ValueError(side)


def main():
    parser = argparse.ArgumentParser(description="Fig. 5 progression visual")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(12.5, 7))
    fig.patch.set_facecolor("white")
    setup_axes(ax, xlim=(0, 100), ylim=(0, 100))

    # 4. Phase headers
    ax.text(50, 95, "Phase A. Dataset progression",
            ha="center", va="center", fontsize=12, weight="bold",
            color=PALETTE["ink"])
    ax.text(50, 30, "Phase B. Method sweep on the frozen Phase A winner",
            ha="center", va="center", fontsize=12, weight="bold",
            color=PALETTE["ink"])

    # 5. Phase A nodes
    a_pos = {nid: (x, y) for nid, x, y, _, _ in A_NODES}
    for nid, x, y, label, sentence in A_NODES:
        text_node(ax, x, y, label, sentence)

    # 6. Phase A arrows
    fork_targets = {("A4", "A5"), ("A4", "A6"), ("A4", "A7"),
                    ("A5", "A8"), ("A6", "A9")}
    for src, dst in A_EDGES:
        sx, sy = a_pos[src]
        dx, dy = a_pos[dst]
        if (src, dst) in fork_targets:
            arrow(ax, node_edge(sx, sy, "bottom"), node_edge(dx, dy, "top"))
        else:
            arrow(ax, node_edge(sx, sy, "right"), node_edge(dx, dy, "left"))

    # 7. Bridge from Phase A to Phase B: short vertical arrow centred on x=50,
    #    placed between the A8/A9 row and the Phase B header so the visual
    #    chain reads top -> bottom without overlapping any node.
    bridge_top = 41    # below A8/A9 (centred at y=47, half-h=4)
    bridge_bot = 34    # above Phase B header at y=30
    arrow(ax, (50, bridge_top), (50, bridge_bot),
          color=PALETTE["navy"], lw=1.5)
    ax.text(52, (bridge_top + bridge_bot) / 2, "frozen Phase A winner",
            ha="left", va="center", fontsize=9,
            color=PALETTE["navy"], style="italic")

    # 8. Phase B nodes (navy label colour to mark the phase)
    b_pos = {nid: (x, y) for nid, x, y, _, _ in B_NODES}
    for nid, x, y, label, sentence in B_NODES:
        text_node(ax, x, y, label, sentence, label_color=PALETTE["navy"])

    # 9. Phase B chain arrows
    for k in range(len(B_NODES) - 1):
        sx, sy = b_pos[B_NODES[k][0]]
        dx, dy = b_pos[B_NODES[k + 1][0]]
        arrow(ax, node_edge(sx, sy, "right"), node_edge(dx, dy, "left"))

    # 10. Caption strip
    caption_text(ax, 50, 6,
                 "Each step changes one controlled variable from its "
                 "predecessor. Phase A selects a dataset; Phase B applies "
                 "the six segmentation methods to the resulting model.",
                 italic=True, font_size=9)

    fig.suptitle("Figure 5. Two-phase experimental progression",
                 fontsize=12, weight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # 11. Route through fig registry (saves with bbox_inches='tight')
    caption = (
        "Two-phase experimental progression. Phase A walks the dataset axis, "
        "isolating one cleaning or balancing variable per step. The selected "
        "dataset is frozen and passed to Phase B, which sweeps the six "
        "segmentation methods of Fig. 3 in a single inference dispatch. "
        "Each step's motivation is reader-facing; implementation details "
        "(manifest identifiers, balancing-scheme labels, training "
        "hyperparameters) are deferred to the methods text."
    )
    archive = fig_write(
        fig=fig, slug="fig05_progression",
        caption=caption, out_dir=args.out_dir, dpi=args.dpi,
    )
    plt.close(fig)
    print(f"Figure written: {archive}")


if __name__ == "__main__":
    main()
