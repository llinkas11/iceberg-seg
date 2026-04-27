"""
_fig_registry.py: single-door for saving figures.

Every figure a script produces goes through `write(fig, slug, caption)`. The
registry does two things:

  1. Writes the figure to `fig-archive/<YYYYMMDD_HHMM>__<slug>.png`. The
     archive is append-only; old files are never overwritten, so regenerating
     a figure keeps the previous version on disk for audit.

  2. Updates `figures.md` so the slug points at the latest archive entry. If
     the slug already has a row, it is replaced (the old archive file stays).
     If it is new, a row is appended.

`write_table(headers, rows, title, slug, caption, out_dir)` is a sibling
shortcut for the common "render a small table as a PNG" pattern. It builds a
matplotlib table figure inline and forwards to `write` so table figures
land in the archive next to the rest.

Callers pick any `out_dir`. A project-wide convention is
`iceberg-rework/fig-archive/` + `iceberg-rework/figures.md`, but scripts can
pass a run-specific dir when they want per-run figures.
"""

import os
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_TS_FMT = "%Y%m%d_%H%M%S"
_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


def write(fig, slug, caption, out_dir, dpi=150):
    """
    Save `fig` to <out_dir>/fig-archive/<timestamp>__<slug>.png and update
    <out_dir>/figures.md so the slug row points at the new file.

    slug must be a lowercase snake_case identifier so the filename is
    filesystem-safe and the figures.md table stays scannable.

    Returns the archive path that was just written.
    """
    if not _SLUG_RE.match(slug):
        raise ValueError(
            f"slug {slug!r} must match {_SLUG_RE.pattern}; "
            "use lowercase snake_case so the filename and table row are stable"
        )

    archive_dir = os.path.join(out_dir, "fig-archive")
    os.makedirs(archive_dir, exist_ok=True)

    ts = datetime.now().strftime(_TS_FMT)
    fname = f"{ts}__{slug}.png"
    archive_path = os.path.join(archive_dir, fname)

    # bbox_inches='tight' matches what the original savefig calls used.
    fig.savefig(archive_path, dpi=dpi, bbox_inches="tight")

    _update_index(out_dir, slug, fname, caption)
    return archive_path


def _update_index(out_dir, slug, fname, caption):
    """
    Append-or-replace the slug's row in figures.md. The table is two columns:
    slug pointing at the archive filename (as a markdown image link), and
    caption prose.
    """
    index_path = os.path.join(out_dir, "figures.md")
    rel_link   = f"fig-archive/{fname}"
    row        = f"| `{slug}` | [{fname}]({rel_link}) | {caption.strip()} |"

    header = (
        "# Figures\n"
        "\n"
        "Live index of project figures. Every row points at the latest entry "
        "in `fig-archive/`; older entries for the same slug stay on disk. "
        "Regenerate via `_fig_registry.write(fig, slug, caption, out_dir)`.\n"
        "\n"
        "| slug | file | caption |\n"
        "|---|---|---|\n"
    )

    if not os.path.exists(index_path):
        with open(index_path, "w") as f:
            f.write(header)
            f.write(row + "\n")
        return

    lines = open(index_path).read().splitlines()
    # Match column 1 with the closing backtick + trailing pipe so a slug like
    # `iou_delta` cannot prefix-match `iou_delta_heatmap` and overwrite the
    # wrong row.
    token = f"| `{slug}` |"
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith(token):
            lines[i] = row
            replaced = True
            break

    if not replaced:
        # Append below the last non-empty line so the table stays contiguous.
        lines.append(row)

    with open(index_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_table(headers, rows, title, slug, caption, out_dir, *, col_widths=None):
    """
    Render a small (headers, rows) table as a PNG figure and route it through
    `write`. Used for tables we want to publish alongside plots, not for
    program-readable CSVs (those go to disk directly).

    The header row is dark with white bold text; data rows alternate a light
    grey shade. Cell text is centred. col_widths is reserved for future use.
    """
    n_cols = len(headers)
    n_rows = len(rows)
    fig_w = max(8, n_cols * 1.8)
    fig_h = max(2, 0.4 * (n_rows + 2))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=headers, loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (r, _), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f0f0")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    write(fig, slug=slug, caption=caption, out_dir=out_dir)
    plt.close(fig)
