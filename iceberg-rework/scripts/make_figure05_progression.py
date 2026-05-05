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
    ("A0",  10, 83, "A0", "Fisser baseline\nlt65 only, drop nulls"),
    ("A1",  30, 83, "A1", "A0 + 29 null chips\n1:1 + undersample"),
    ("A2",  50, 83, "A2", "A1 + 40 m filter\n+ IC mask"),
    ("A3",  70, 83, "A3", "A2 + 29 null chips\n1:1 GT+ / null"),
    ("A4",  90, 83, "A4", "A3 + augmentation\nhflip / vflip / rot90"),
    ("A5",  50, 64, "A5", "A4 + fixed 2:1\nbalancing"),
    ("A6",  70, 64, "A6", "A4 + adaptive 2:1\nbalancing"),
    ("A7",  90, 64, "A7", "A4 + size oversample\nRL bins (4x cap)"),
    ("A8",  60, 47, "A8", "A5 + size oversample"),
    ("A9",  80, 47, "A9", "A6 + size oversample"),
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


def text_node(ax, x, y, label, sentence, *, label_color=None,
              selected=False):
    """
    Render a text-only node centred at (x, y). Description sentence is the
    visually dominant element; the A#/B# label is a small badge in the
    corner so the reader's eye lands on the description, not the ID.

    When `selected` is True the node is wrapped in a navy outlined box and
    a "selected" badge is added below the description, marking it as the
    Phase A configuration that feeds Phase B.
    """
    color = label_color or PALETTE["ink"]
    if selected:
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch(
            (x - NODE_HALF_W - 1, y - NODE_HALF_H - 1),
            2 * NODE_HALF_W + 2, 2 * NODE_HALF_H + 2,
            boxstyle="round,pad=0.4", linewidth=1.8,
            edgecolor=PALETTE["navy"], facecolor="#E8F0FA",
            zorder=0,
        ))
    # Description: large so it draws the eye (sentence beats ID).
    ax.text(x, y + 0.6, sentence,
            ha="center", va="center", fontsize=10,
            weight="bold", color=color)
    # A#/B# label: small, muted, positioned below as a footnote-style badge.
    ax.text(x, y - 2.8, label,
            ha="center", va="center", fontsize=8,
            color="#666",
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="white", edgecolor="#bbb",
                      linewidth=0.6))
    if selected:
        ax.text(x, y + NODE_HALF_H + 0.7, "SELECTED",
                ha="center", va="bottom", fontsize=8,
                weight="bold", color=PALETTE["navy"],
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=PALETTE["navy"], edgecolor="none"))
        # White-ink override on the SELECTED badge
        ax.texts[-1].set_color("white")


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

    fig, ax = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor("white")
    setup_axes(ax, xlim=(0, 100), ylim=(0, 100))

    # 4. Phase headers
    ax.text(50, 95, "Phase A. Dataset progression",
            ha="center", va="center", fontsize=15, weight="bold",
            color=PALETTE["ink"])
    ax.text(50, 30, "Phase B. Method sweep on the frozen Phase A winner",
            ha="center", va="center", fontsize=15, weight="bold",
            color=PALETTE["ink"])

    # 5. Phase A nodes. A2 is the preprocessing pipeline that trains the
    # baseline_v1 model used in Tables 1-3 and Fig. 6-10, so it is rendered
    # with a navy outline and a SELECTED badge.
    SELECTED_NODE = "A2"
    a_pos = {nid: (x, y) for nid, x, y, _, _ in A_NODES}
    for nid, x, y, label, sentence in A_NODES:
        text_node(ax, x, y, label, sentence,
                   selected=(nid == SELECTED_NODE))

    # 6. Phase A arrows. Long fork legs (A4 -> A5 / A6, A5 -> A8, A6 -> A9)
    # use elbow routing so each leg drops vertically out of the source then
    # turns horizontally into the target column, never crossing text.
    # Vertically aligned forks (A4 -> A7) stay as straight arrows.
    fork_targets = {("A4", "A5"), ("A4", "A6"), ("A4", "A7"),
                    ("A5", "A8"), ("A6", "A9")}
    elbow_left  = "angle,angleA=-90,angleB=0,rad=4"   # source above-and-right of target
    elbow_right = "angle,angleA=-90,angleB=180,rad=4" # source above-and-left of target
    for src, dst in A_EDGES:
        sx, sy = a_pos[src]
        dx, dy = a_pos[dst]
        if (src, dst) in fork_targets:
            if sx == dx:
                # Vertically aligned (A4 -> A7); straight arrow, no elbow
                arrow(ax, node_edge(sx, sy, "bottom"),
                          node_edge(dx, dy, "top"))
            else:
                conn = elbow_left if sx > dx else elbow_right
                arrow(ax, node_edge(sx, sy, "bottom"),
                          node_edge(dx, dy, "top"),
                          connectionstyle=conn)
        else:
            arrow(ax, node_edge(sx, sy, "right"), node_edge(dx, dy, "left"))

    # 7. Bridge from Phase A to Phase B. Routed in the column gap between
    # A8 and A9 (x=70) so it does not cut through any Phase A node. A2 is
    # already marked SELECTED with a navy outline + badge, and the caption
    # makes the source explicit, so the bridge is purely a visual link.
    bridge_x = 70
    bridge_top = 41    # below A8/A9 row (y=47, half-h=4)
    bridge_bot = 34    # above Phase B header at y=30
    arrow(ax, (bridge_x, bridge_top), (bridge_x, bridge_bot),
          color=PALETTE["navy"], lw=2.0)
    ax.text(bridge_x + 2, (bridge_top + bridge_bot) / 2,
            "A2 preprocessing\ntrains baseline_v1",
            ha="left", va="center", fontsize=11,
            color=PALETTE["navy"], style="italic", weight="bold")

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
                 italic=True, font_size=11)

    fig.suptitle("Figure 5. Two-phase experimental progression",
                 fontsize=11, weight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # 11. Route through fig registry (saves with bbox_inches='tight')
    caption = (
        "Two-phase experimental progression. Phase A walks the dataset axis, "
        "isolating one cleaning or balancing variable per step. A2 (40 m "
        "component filter + annotation-aware IC pixel mask) is the "
        "preprocessing pipeline marked SELECTED; A2 trains the baseline_v1 "
        "model whose six-method sweep produces the per-pair results in "
        "Tables 1-3 and Figs. 6-10. A3-A9 are balancing-variant explorations "
        "around the A2 baseline, not the Tables 1-3 source. Phase B sweeps "
        "the six segmentation methods of Fig. 3 in a single inference "
        "dispatch on the SELECTED model. Each step's motivation is "
        "reader-facing; implementation details (manifest identifiers, "
        "balancing-scheme labels, training hyperparameters) are deferred to "
        "the methods text."
    )
    archive = fig_write(
        fig=fig, slug="fig05_progression",
        caption=caption, out_dir=args.out_dir, dpi=args.dpi,
    )
    # Editable SVG companion alongside the PNG
    svg_path = archive[:-4] + ".svg"
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written: {archive}")
    print(f"SVG companion: {svg_path}")


if __name__ == "__main__":
    main()
