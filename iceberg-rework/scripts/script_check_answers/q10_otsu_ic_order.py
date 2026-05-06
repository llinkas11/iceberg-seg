"""
q10_otsu_ic_order.py: empirical answer to script-check question 10.

Question (paraphrased from script-check-README.md, otsu_threshold_tifs.py):
  "Order of operations: compute Otsu, then test whether 15 % of pixels exceed
  it. Is this the right order, or should the IC test be against a fixed
  reference threshold so that 'sea ice contamination' means the same thing
  across chips?" (specifically: does the order IC-then-Otsu vs Otsu-then-IC
  matter for which chips end up skipped?)

What this script does:
  1. Glob every chip TIF and compute (a) the production OT order
     (compute Otsu on the full chip; skip if (B08 >= otsu).mean() > 0.15;
     also skip if otsu < 0.10 floor) and (b) the alternate order (apply the
     0.22 IC test first, skipping the chip if (B08 >= 0.22).mean() > 0.15;
     compute Otsu only on the non-skipped chips).
  2. Tabulate skip/keep decisions per chip under each order and report the
     symmetric-difference set per (region, SZA bin).
  3. Emit a per-chip CSV plus two PNGs: a confusion-style stacked bar showing
     the four outcome categories (both keep / both skip / order-A only skip
     / order-B only skip) and a per-(region, SZA bin) breakdown.

Inputs:
  --chips_root, --b08_idx, --threshold, --ic_threshold, --otsu_floor, --out_root.

Outputs (under <out_root>/q10_otsu_ic_order/):
  <ts>__q10_otsu_ic_order.csv               one row per chip
  <ts>__q10_otsu_ic_order__overview.png     overall outcome stack
  <ts>__q10_otsu_ic_order__by_sza_region.png  per-bin breakdown
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


SLUG = "q10_otsu_ic_order"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chips_root", default=str(resolve_chips_root()),
                   help="Root containing <region>/<sza_bin>/tifs/*.tif")
    p.add_argument("--b08_idx", type=int, default=2,
                   help="0-indexed band of B08 in the chip TIF")
    p.add_argument("--threshold", type=float, default=0.22,
                   help="Reflectance threshold for the fixed IC test (default 0.22)")
    p.add_argument("--ic_threshold", type=float, default=0.15,
                   help="IC cutoff (default 0.15)")
    p.add_argument("--otsu_floor", type=float, default=0.10,
                   help="Production Otsu floor (default 0.10)")
    p.add_argument("--out_root", default=str(resolve_out_root()),
                   help="Parent directory; a slug subfolder is created under it")
    return p.parse_args()


def evaluate_chip(tif_path, b08_idx, threshold, ic_thresh, floor):
    """
    Return (otsu_then_ic_skip, ic_then_otsu_skip, otsu, fixed_ic_frac,
    otsu_ic_frac) or None if invalid chip.

    Order A (production OT): compute Otsu -> apply floor -> apply Otsu-IC.
    Order B (alt):           apply fixed IC at 0.22 -> compute Otsu -> apply floor.
    """
    with rasterio.open(tif_path) as src:
        if src.count <= b08_idx:
            return None
        b08 = src.read(b08_idx + 1).astype(np.float32)
    flat = b08.ravel()
    if np.unique(flat).size < 2:
        return None

    otsu = float(threshold_otsu(flat))
    fixed_ic = float((flat >= threshold).mean())
    otsu_ic  = float((flat >= otsu).mean())

    # Order A (production OT): Otsu first, then floor, then Otsu-IC.
    a_skip_floor = otsu < floor
    a_skip_ic    = otsu_ic > ic_thresh
    a_skip = bool(a_skip_floor or a_skip_ic)

    # Order B: fixed IC first; only chips that pass it go on to Otsu.
    if fixed_ic > ic_thresh:
        b_skip = True
        b_skip_floor = False  # never reached
        b_skip_ic    = True
    else:
        b_skip_floor = otsu < floor
        b_skip_ic    = False
        b_skip = bool(b_skip_floor)

    return {
        "otsu":         otsu,
        "fixed_ic":     fixed_ic,
        "otsu_ic":      otsu_ic,
        "a_skip":       a_skip,
        "a_skip_floor": a_skip_floor,
        "a_skip_ic":    a_skip_ic,
        "b_skip":       b_skip,
        "b_skip_floor": b_skip_floor,
        "b_skip_ic":    b_skip_ic,
    }


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    out_dir = make_slug_dir(SLUG, args.out_root)
    chip_rows = list_chip_tifs(args.chips_root)
    if not chip_rows:
        raise SystemExit(f"No chips found under {args.chips_root}")
    print(f"Found {len(chip_rows)} chips under {args.chips_root}")

    rows = []
    skipped = 0
    for tif, region, sza_bin in chip_rows:
        res = evaluate_chip(tif, args.b08_idx, args.threshold,
                            args.ic_threshold, args.otsu_floor)
        if res is None:
            skipped += 1
            continue
        if res["a_skip"] and res["b_skip"]:
            outcome = "both_skip"
        elif res["a_skip"]:
            outcome = "a_only_skip"   # OT-then-IC skipped, fixed-IC kept
        elif res["b_skip"]:
            outcome = "b_only_skip"   # fixed-IC skipped, OT-then-IC kept
        else:
            outcome = "both_keep"
        row = {
            "chip_stem":  tif.stem,
            "region":     region,
            "sza_bin":    sza_bin,
            "otsu":       res["otsu"],
            "fixed_ic":   res["fixed_ic"],
            "otsu_ic":    res["otsu_ic"],
            "a_skip":     res["a_skip"],
            "b_skip":     res["b_skip"],
            "outcome":    outcome,
        }
        rows.append(row)

    if skipped:
        print(f"Skipped {skipped} chips (too few bands or constant)")
    print(f"Evaluated {len(rows)} chips")
    if not rows:
        raise SystemExit("No chips evaluated; nothing to plot.")

    # 1. Write per-chip CSV
    csv_path = out_dir / f"{stamp(SLUG)}.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {csv_path}")

    # 2. Headline outcome counts
    outcomes = ["both_keep", "both_skip", "a_only_skip", "b_only_skip"]
    counts = {o: sum(1 for r in rows if r["outcome"] == o) for o in outcomes}
    n = len(rows)
    print("\nOutcome counts:")
    for o in outcomes:
        print(f"  {o:<14}: {counts[o]:>6}  ({counts[o] / n:.2%})")
    flip_rate = (counts["a_only_skip"] + counts["b_only_skip"]) / n
    print(f"  total flip-rate (symmetric difference): {flip_rate:.2%}")

    # 3. Overview stacked bar (one bar, four segments)
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {"both_keep": "#37474F", "both_skip": "#90A4AE",
               "a_only_skip": "#D32F2F", "b_only_skip": "#1976D2"}
    bottom = 0
    for o in outcomes:
        height = counts[o] / n * 100
        ax.bar(["all chips"], [height], bottom=[bottom],
               color=palette[o], label=f"{o} ({counts[o]})")
        bottom += height
    ax.set_ylabel("Percent of chips (%)")
    ax.set_title(f"Q10: outcome under Otsu-then-IC vs IC-then-Otsu (n={n})")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    overview_path = out_dir / f"{stamp(SLUG, 'overview')}.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {overview_path}")

    # 4. Per-(region, SZA bin) breakdown
    fig, ax = plt.subplots(figsize=(11, 5))
    group_labels = []
    pcts = {o: [] for o in outcomes}
    for region in REGIONS:
        for sza_bin in SZA_BINS:
            sub = [r for r in rows if r["region"] == region and r["sza_bin"] == sza_bin]
            group_labels.append(f"{region}\n{sza_bin}\n(n={len(sub)})")
            if not sub:
                for o in outcomes:
                    pcts[o].append(0.0)
                continue
            for o in outcomes:
                pcts[o].append(sum(1 for r in sub if r["outcome"] == o) / len(sub) * 100)

    x = np.arange(len(group_labels))
    bottom = np.zeros(len(group_labels))
    for o in outcomes:
        ax.bar(x, pcts[o], bottom=bottom, color=palette[o], label=o)
        bottom = bottom + np.array(pcts[o])
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("Percent of chips (%)")
    ax.set_title("Q10: order-of-operations outcome per (region, SZA bin)")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    by_path = out_dir / f"{stamp(SLUG, 'by_sza_region')}.png"
    fig.savefig(by_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {by_path}")


if __name__ == "__main__":
    main()
