"""
q07_otsu_floor_distribution.py: empirical answer to script-check question 7.

Question (from script-check-README.md, otsu_threshold_tifs.py):
  "min_otsu_thresh = 0.10 ... is 0.10 the right floor in offset-uncorrected
  reflectance, or should it be 0.20 (i.e. 0.10 after the +0.10 offset)?"

What this script does:
  1. Glob every chip TIF and compute threshold_otsu(B08) per chip with no
     floor applied, mirroring the OT pre-floor step in otsu_threshold_tifs.py.
  2. Distribute the per-chip Otsu thresholds and tabulate the skip-rate at
     candidate floors {0.10, 0.15, 0.20}.
  3. Emit a per-chip CSV and two PNGs: an aggregate histogram with floor
     lines and a per-(region, SZA bin) violin plot.

Inputs:
  --chips_root, --b08_idx, --out_root.

Outputs (under <out_root>/q07_otsu_floor_distribution/):
  <ts>__q07_otsu_floor_distribution.csv               one row per chip
  <ts>__q07_otsu_floor_distribution__overview.png     histogram with floors
  <ts>__q07_otsu_floor_distribution__by_sza_region.png  violin per (region, sza)

Usage (Mac smoke test):
  python iceberg-rework/scripts/script_check_answers/q07_otsu_floor_distribution.py

Deploy to moosehead:
  rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/script_check_answers/ \
    llinkas@moosehead.bowdoin.edu:~/iceberg-rework/scripts/script_check_answers/
"""

import argparse
import csv
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


SLUG = "q07_otsu_floor_distribution"
FLOORS = [0.10, 0.15, 0.20]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chips_root", default=str(resolve_chips_root()),
                   help="Root containing <region>/<sza_bin>/tifs/*.tif")
    p.add_argument("--b08_idx", type=int, default=2,
                   help="0-indexed band of B08 in the chip TIF")
    p.add_argument("--out_root", default=str(resolve_out_root()),
                   help="Parent directory; a slug subfolder is created under it")
    return p.parse_args()


def compute_otsu(tif_path, b08_idx):
    """
    Run skimage.filters.threshold_otsu on the B08 band of a chip. Returns
    (otsu_thresh, n_unique). n_unique=1 chips return otsu_thresh=NaN since
    Otsu is undefined on a constant image.
    """
    with rasterio.open(tif_path) as src:
        if src.count <= b08_idx:
            return None, None
        b08 = src.read(b08_idx + 1).astype(np.float32)
    flat = b08.ravel()
    n_unique = int(np.unique(flat).size)
    if n_unique < 2:
        return float("nan"), n_unique
    return float(threshold_otsu(flat)), n_unique


def main():
    args = parse_args()

    # 1. Resolve output dir and discover chips
    out_dir = make_slug_dir(SLUG, args.out_root)
    chip_rows = list_chip_tifs(args.chips_root)
    if not chip_rows:
        raise SystemExit(f"No chips found under {args.chips_root}")
    print(f"Found {len(chip_rows)} chips under {args.chips_root}")

    # 2. Compute Otsu per chip
    rows = []
    too_few_bands = 0
    for tif, region, sza_bin in chip_rows:
        thresh, n_unique = compute_otsu(tif, args.b08_idx)
        if thresh is None:
            too_few_bands += 1
            continue
        rows.append({
            "chip_stem":    tif.stem,
            "region":       region,
            "sza_bin":      sza_bin,
            "otsu_thresh":  thresh,
            "n_unique_b08": n_unique,
        })
    if too_few_bands:
        print(f"Skipped {too_few_bands} chips with too few bands")
    print(f"Computed Otsu for {len(rows)} chips")

    # 3. Write per-chip CSV
    csv_path = out_dir / f"{stamp(SLUG)}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chip_stem", "region", "sza_bin", "otsu_thresh", "n_unique_b08"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {csv_path}")

    # 4. Aggregate skip-rate at each floor and print summary
    threshes = np.array([r["otsu_thresh"] for r in rows], dtype=np.float64)
    valid = ~np.isnan(threshes)
    print(f"\nSkip-rate at each Otsu floor:")
    for f_floor in FLOORS:
        skipped = int(((threshes < f_floor) & valid).sum()) + int((~valid).sum())
        rate = skipped / len(threshes)
        print(f"  otsu < {f_floor:.2f}: {rate:.3%}  ({skipped}/{len(threshes)})")

    # 5. Overview histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, max(0.6, np.nanmax(threshes) * 1.05), 60)
    ax.hist(threshes[valid], bins=bins, color="#37474F", alpha=0.85)
    # Stagger floor labels so close-together floors do not overprint.
    ymax = ax.get_ylim()[1]
    label_y_fracs = [0.95, 0.85, 0.75]
    for f_floor, y_frac in zip(FLOORS, label_y_fracs):
        ax.axvline(f_floor, color="#D32F2F" if f_floor == 0.10 else "#90A4AE",
                   lw=1.0, ls="--", alpha=0.9)
        ax.text(f_floor, ymax * y_frac, f" {f_floor:.2f}",
                color="#D32F2F" if f_floor == 0.10 else "#37474F", fontsize=8, va="top")
    # Mark the offset-corrected equivalent of each floor (+0.10).
    for f_floor in FLOORS:
        ax.axvline(f_floor + 0.10, color="#1976D2", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("per-chip Otsu threshold on B08 (offset-uncorrected reflectance)")
    ax.set_ylabel("Chip count")
    ax.set_title(f"Q7: per-chip Otsu threshold distribution (n={int(valid.sum())} valid)")
    ax.grid(True, alpha=0.3, linestyle="--")
    overview_path = out_dir / f"{stamp(SLUG, 'overview')}.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {overview_path}")

    # 6. Per-(region, sza_bin) violin plot
    groups = []
    labels = []
    for r in REGIONS:
        for s in SZA_BINS:
            vals = np.array([row["otsu_thresh"] for row in rows
                             if row["region"] == r and row["sza_bin"] == s
                             and not np.isnan(row["otsu_thresh"])])
            if len(vals) == 0:
                continue
            groups.append(vals)
            labels.append(f"{r}\n{s}\n(n={len(vals)})")

    fig, ax = plt.subplots(figsize=(11, 5))
    if groups:
        parts = ax.violinplot(groups, showmedians=True, widths=0.85)
        for body in parts["bodies"]:
            body.set_facecolor("#90A4AE")
            body.set_alpha(0.7)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=9)
    for f_floor in FLOORS:
        ax.axhline(f_floor, color="#D32F2F" if f_floor == 0.10 else "#90A4AE",
                   lw=1.0, ls="--", alpha=0.7)
    ax.set_ylabel("per-chip Otsu threshold on B08")
    ax.set_title("Q7: per-chip Otsu threshold by (region, SZA bin); dashed lines mark candidate floors")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    by_path = out_dir / f"{stamp(SLUG, 'by_sza_region')}.png"
    fig.savefig(by_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {by_path}")


if __name__ == "__main__":
    main()
