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

Callers pick any `out_dir`. A project-wide convention is
`iceberg-rework/fig-archive/` + `iceberg-rework/figures.md`, but scripts can
pass a run-specific dir when they want per-run figures.
"""

import os
import re
from datetime import datetime


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
    # Find the slug's existing row by a backtick-wrapped match in column 1.
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
