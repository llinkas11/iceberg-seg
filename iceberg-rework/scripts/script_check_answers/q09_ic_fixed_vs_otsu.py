"""
q09_ic_fixed_vs_otsu.py: empirical answer to script-check question 9.

Question (from script-check-README.md, otsu_threshold_tifs.py):
  "IC filter on the Otsu result. Same 15 % rule as TR but evaluated against
  the Otsu threshold for that chip. Order of operations: compute Otsu, then
  test whether 15 % of pixels exceed it. Is this the right order, or should
  the IC test be against a fixed reference threshold so that 'sea ice
  contamination' means the same thing across chips?"

What this script does:
  1. Glob every chip TIF and compute (B08 >= 0.22).mean() (the fixed-IC
     fraction the TR pipeline uses) and (B08 >= per-chip Otsu).mean() (the
     production OT IC fraction).
  2. For each candidate IC cutoff in {0.15, 0.20}, compare which chips skip
     under the production OT rule (Otsu-IC > cutoff) vs the proposed fixed-
     reference rule (TR-IC > cutoff).
  3. Tabulate flip-in (skip-under-fixed but kept-under-Otsu) and flip-out
     (kept-under-fixed but skip-under-Otsu) per (region, SZA bin) at the
     production cutoff (0.15) so the disagreement set has a per-bin
     decomposition.
  4. Emit a per-chip CSV plus two PNGs: a scatter of fixed-IC vs Otsu-IC
     fractions colored by the production decision, and a per-(region, SZA
     bin) flip-rate bar chart.

Inputs:
  --chips_root   Directory containing <region>/<sza_bin>/tifs/*.tif.
  --b08_idx      Band index of B08 in the chip TIF (default 2).
  --threshold    Reference reflectance threshold for the fixed-IC test
                 (default 0.22, matching threshold_tifs.py).
  --ic_threshold Production IC cutoff (default 0.15).
  --out_root     Parent directory under which a slug folder is created.

Outputs (under <out_root>/q09_ic_fixed_vs_otsu/):
  <ts>__q09_ic_fixed_vs_otsu.csv               one row per chip
  <ts>__q09_ic_fixed_vs_otsu__overview.png     scatter of fixed vs Otsu IC frac
  <ts>__q09_ic_fixed_vs_otsu__by_sza_region.png  per-bin flip-rate bars
"""

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from skimage.filters import threshold_otsu

from _common import (
    SZA_BINS, REGIONS,
    list_chip_tifs, make_slug_dir, resolve_chips_root, resolve_out_root, stamp,
)


SLUG = "q09_ic_fixed_vs_otsu"
EXTRA_CUTOFFS = [0.15, 0.20]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chips_root", default=str(resolve_chips_root()),
                   help="Root containing <region>/<sza_bin>/tifs/*.tif")
    p.add_argument("--b08_idx", type=int, default=2,
                   help="0-indexed band of B08 in the chip TIF")
    p.add_argument("--threshold", type=float, default=0.22,
                   help="Fixed reference reflectance threshold for IC test")
    p.add_argument("--ic_threshold", type=float, default=0.15,
                   help="Production IC cutoff (default 0.15)")
    p.add_argument("--out_root", default=str(resolve_out_root()),
                   help="Parent directory; a slug subfolder is created under it")
    return p.parse_args()


def compute(tif_path, b08_idx, fixed_thresh):
    """Return (otsu, fixed_ic_frac, otsu_ic_frac) or None if invalid chip."""
    with rasterio.open(tif_path) as src:
        if src.count <= b08_idx:
            return None
        b08 = src.read(b08_idx + 1).astype(np.float32)
    flat = b08.ravel()
    if np.unique(flat).size < 2:
        return None
    otsu = float(threshold_otsu(flat))
    fixed_ic = float((flat >= fixed_thresh).mean())
    otsu_ic = float((flat >= otsu).mean())
    return otsu, fixed_ic, otsu_ic


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    out_dir = make_slug_dir(SLUG, args.out_root)
    chip_rows = list_chip_tifs(args.chips_root)
    if not chip_rows:
        raise SystemExit(f"No chips found under {args.chips_root}")
    print(f"Found {len(chip_rows)} chips under {args.chips_root}")

    # 1. Per-chip fixed-IC and Otsu-IC fractions
    rows = []
    skipped = 0
    for tif, region, sza_bin in chip_rows:
        res = compute(tif, args.b08_idx, args.threshold)
        if res is None:
            skipped += 1
            continue
        otsu, fixed_ic, otsu_ic = res
        rows.append({
            "chip_stem":   tif.stem,
            "region":      region,
            "sza_bin":     sza_bin,
            "otsu":        otsu,
            "fixed_ic_frac": fixed_ic,
            "otsu_ic_frac":  otsu_ic,
        })
    if skipped:
        print(f"Skipped {skipped} chips (too few bands or constant)")
    print(f"Computed pair on {len(rows)} chips")

    if not rows:
        raise SystemExit("No chips evaluated; nothing to plot.")

    # 2. Disagreement decomposition at the production 0.15 cutoff
    print("\nFlip counts per cutoff (Otsu-IC > cutoff vs Fixed-IC > cutoff):")
    flip_summary = {}
    for cutoff in EXTRA_CUTOFFS:
        flip_in = 0   # fixed says SKIP, otsu says KEEP
        flip_out = 0  # fixed says KEEP, otsu says SKIP
        agree = 0
        for r in rows:
            fixed_skip = r["fixed_ic_frac"] > cutoff
            otsu_skip  = r["otsu_ic_frac"]  > cutoff
            if fixed_skip and not otsu_skip:
                flip_in += 1
            elif otsu_skip and not fixed_skip:
                flip_out += 1
            else:
                agree += 1
        flip_summary[cutoff] = (flip_in, flip_out, agree)
        n = len(rows)
        print(f"  cutoff={cutoff:.2f}  agree={agree} ({agree/n:.2%})  "
              f"fixed-only-skip={flip_in} ({flip_in/n:.2%})  "
              f"otsu-only-skip={flip_out} ({flip_out/n:.2%})")

    # 3. Add per-cutoff flip flags to the rows for the CSV
    cutoff = args.ic_threshold
    for r in rows:
        fixed_skip = r["fixed_ic_frac"] > cutoff
        otsu_skip  = r["otsu_ic_frac"]  > cutoff
        r["fixed_skip"] = fixed_skip
        r["otsu_skip"] = otsu_skip
        if fixed_skip and not otsu_skip:
            r["flip"] = "fixed_only_skip"
        elif otsu_skip and not fixed_skip:
            r["flip"] = "otsu_only_skip"
        elif otsu_skip and fixed_skip:
            r["flip"] = "both_skip"
        else:
            r["flip"] = "both_keep"

    # 4. Write per-chip CSV
    fieldnames = ["chip_stem", "region", "sza_bin", "otsu",
                  "fixed_ic_frac", "otsu_ic_frac",
                  "fixed_skip", "otsu_skip", "flip"]
    csv_path = out_dir / f"{stamp(SLUG)}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in fieldnames} for r in rows])
    print(f"CSV: {csv_path}")

    # 5. Overview scatter: fixed vs Otsu IC frac, colored by the production decision
    fixed_ic = np.array([r["fixed_ic_frac"] for r in rows], dtype=np.float64)
    otsu_ic  = np.array([r["otsu_ic_frac"]  for r in rows], dtype=np.float64)
    flips    = np.array([r["flip"] for r in rows])
    color_map = {
        "both_keep":       "#37474F",
        "both_skip":       "#90A4AE",
        "fixed_only_skip": "#1976D2",
        "otsu_only_skip":  "#D32F2F",
    }

    fig, ax = plt.subplots(figsize=(8, 7))
    for label, color in color_map.items():
        mask = flips == label
        ax.scatter(fixed_ic[mask], otsu_ic[mask],
                   s=8, alpha=0.5, color=color, edgecolor="none",
                   label=f"{label} (n={int(mask.sum())})")
    ax.axvline(args.ic_threshold, color="#1976D2", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(args.ic_threshold, color="#D32F2F", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel(f"Fixed-IC fraction (B08 >= {args.threshold}).mean()")
    ax.set_ylabel("Otsu-IC fraction (B08 >= per-chip Otsu).mean()")
    ax.set_title(f"Q9: fixed-IC vs Otsu-IC at cutoff {args.ic_threshold} (n={len(rows)})")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    overview_path = out_dir / f"{stamp(SLUG, 'overview')}.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {overview_path}")

    # 6. Per-(region, SZA bin) flip rates at the production cutoff
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.35
    group_labels = []
    fixed_only_pct = []
    otsu_only_pct  = []
    for region in REGIONS:
        for sza_bin in SZA_BINS:
            sub = [r for r in rows if r["region"] == region and r["sza_bin"] == sza_bin]
            group_labels.append(f"{region}\n{sza_bin}\n(n={len(sub)})")
            if not sub:
                fixed_only_pct.append(0.0); otsu_only_pct.append(0.0); continue
            n = len(sub)
            fixed_only_pct.append(sum(1 for r in sub if r["flip"] == "fixed_only_skip") / n * 100)
            otsu_only_pct.append( sum(1 for r in sub if r["flip"] == "otsu_only_skip")  / n * 100)
    x = np.arange(len(group_labels))
    ax.bar(x - width / 2, fixed_only_pct, width=width, color="#1976D2",
           label="fixed-only skip")
    ax.bar(x + width / 2, otsu_only_pct, width=width, color="#D32F2F",
           label="otsu-only skip")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("Percent of chips that flip (%)")
    ax.set_title(f"Q9: disagreement between fixed-IC and Otsu-IC at cutoff {args.ic_threshold}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    by_path = out_dir / f"{stamp(SLUG, 'by_sza_region')}.png"
    fig.savefig(by_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {by_path}")


if __name__ == "__main__":
    main()
