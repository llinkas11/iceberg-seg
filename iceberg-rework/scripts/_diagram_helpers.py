"""
_diagram_helpers.py: shared matplotlib primitives for the Methods-section
schematic figures (Figs. 2, 3, 4, 5). Box + arrow + label only; no chip data.

All functions take an Axes and draw onto it; the caller controls layout and
labels. Coordinates use whatever axis range the caller picks (typical: 0-100).
"""

import matplotlib.patches as mpatches
import numpy as np

# 1. Palette (consistent with Fig. 1)
PALETTE = {
    "gold":    "#e6b800",  # preliminary annotation, ground truth
    "cyan":    "#00b8b8",  # cleaned, matched, success
    "grey":    "#7f7f7f",  # intermediate, shadow
    "navy":    "#1a3a6e",  # UNet branch
    "crimson": "#c8102e",  # FN / FP / error markers
    "ink":     "#111111",  # outlines, arrows, labels
    "panel":   "#f5f5f7",  # box fill (light)
    "panel2":  "#e0e6f0",  # alternate box fill
}

DEFAULT_BOX_KW = dict(
    boxstyle="round,pad=0.4",
    linewidth=1.2,
    edgecolor=PALETTE["ink"],
    facecolor=PALETTE["panel"],
)
DEFAULT_ARROW_KW = dict(
    arrowstyle="-|>",
    linewidth=1.2,
    color=PALETTE["ink"],
    mutation_scale=14,
)


def box(ax, x, y, w, h, label, *,
        sublabel=None, fill=None, ec=None, lw=None,
        font_main=10, font_sub=8, italic_sub=True,
        text_color=None):
    """
    Draw a labelled rounded rectangle centred at (x, y) with size (w, h).
    Returns (cx, cy, w, h). Optional sublabel goes below the main label.
    `lw` overrides the default border linewidth so individual boxes can be
    drawn lighter or heavier than the pack default. `text_color` overrides
    the default ink color for the label and sublabel.
    """
    kw = dict(DEFAULT_BOX_KW)
    if fill is not None:
        kw["facecolor"] = fill
    if ec is not None:
        kw["edgecolor"] = ec
    if lw is not None:
        kw["linewidth"] = lw
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, **kw,
    )
    ax.add_patch(rect)

    color = text_color if text_color is not None else PALETTE["ink"]
    if sublabel:
        # Two stacked labels: main centred slightly above midline, sub below
        ax.text(x, y + h * 0.18, label,
                ha="center", va="center", fontsize=font_main,
                color=color, weight="bold")
        ax.text(x, y - h * 0.20, sublabel,
                ha="center", va="center", fontsize=font_sub,
                color=color,
                style="italic" if italic_sub else "normal",
                wrap=True)
    else:
        ax.text(x, y, label,
                ha="center", va="center", fontsize=font_main,
                color=color, weight="bold")
    return x, y, w, h


def arrow(ax, p_from, p_to, *, label=None, label_offset=(0, 1.2),
          color=None, lw=None, connectionstyle=None):
    """
    Draw an annotated arrow from p_from = (x1, y1) to p_to = (x2, y2).
    Optional label at midpoint, displaced by label_offset. `connectionstyle`
    is forwarded to arrowprops so callers can request elbow / curved routing
    (e.g. "angle,angleA=-90,angleB=180,rad=5") for paths that would
    otherwise cross text.
    """
    kw = dict(DEFAULT_ARROW_KW)
    if color is not None:
        kw["color"] = color
    if lw is not None:
        kw["linewidth"] = lw
    if connectionstyle is not None:
        kw["connectionstyle"] = connectionstyle
    ax.annotate("", xy=p_to, xytext=p_from, arrowprops=kw)
    if label:
        mx = 0.5 * (p_from[0] + p_to[0]) + label_offset[0]
        my = 0.5 * (p_from[1] + p_to[1]) + label_offset[1]
        ax.text(mx, my, label, ha="center", va="bottom",
                fontsize=8, style="italic", color=PALETTE["ink"])


def edge_point(box_xywh, side):
    """Return the (x, y) edge-midpoint of a box on the given side."""
    x, y, w, h = box_xywh
    if side == "right":  return (x + w / 2, y)
    if side == "left":   return (x - w / 2, y)
    if side == "top":    return (x, y + h / 2)
    if side == "bottom": return (x, y - h / 2)
    raise ValueError(side)


def caption_text(ax, x, y, text, *, font_size=8, italic=True, max_width=None):
    """Render a wrapped caption-style text block under a box."""
    style = "italic" if italic else "normal"
    if max_width is not None:
        # Manual word wrap based on character count heuristic.
        words = text.split()
        lines, cur = [], []
        for w in words:
            cur.append(w)
            if sum(len(x) for x in cur) + len(cur) - 1 > max_width:
                lines.append(" ".join(cur[:-1]))
                cur = [w]
        if cur:
            lines.append(" ".join(cur))
        text = "\n".join(lines)
    ax.text(x, y, text, ha="center", va="top", fontsize=font_size,
            style=style, color=PALETTE["ink"])


def setup_axes(ax, xlim=(0, 100), ylim=(0, 100)):
    """Strip ticks, set limits, equal aspect, no spines."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
