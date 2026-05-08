"""
q12_tophat_se_radius.py: empirical answer to script-check question 12.

Question (from script-check-README.md, tophat_recover.py):
  "Structuring element. disk(10) at 10 m pixels, i.e. SE radius 100 m. Drawn
  from Fisser's reported small-iceberg cap. Is the disk shape and that
  radius defensible for the 40 m to 100 m size range we want to recover, or
  should we sweep over multiple radii and combine?"

What this script does:
  1. Iterate the v4_clean test_chips set (228 chips, 4 SZA bins).
  2. For each chip read the B08 band, run white-tophat at disk(r) for
     r in {5, 10, 15, 20}, threshold the response at 0.05 (the production
     th_thresh), drop components below 16 px (40 m root length), subtract
     the production base mask (UNet pred per chip), and count recovered
     polygons + their area distribution.
  3. Tabulate per-chip recovered polygon counts and area histograms per
     radius.
  4. Emit a per-(chip, radius) CSV plus two PNGs: an area-distribution
     histogram per radius and a per-(SZA bin, radius) recovered-count bar
     chart.

Inputs:
  --chips_root   v4_clean test_chips root (default per HPC).
  --base_root    UNet test predictions root (default per HPC).
  --th_thresh    Top-hat response threshold (default 0.05).
  --min_area_px  Min component size in pixels (default 16).
  --radii        Comma list of disk radii to test (default 5,10,15,20).
  --out_root     Parent directory; a slug subfolder is created under it.

Outputs (under <out_root>/q12_tophat_se_radius/):
  <ts>__q12_tophat_se_radius.csv               one row per (chip, radius)
  <ts>__q12_tophat_se_radius__overview.png     area histograms per radius
  <ts>__q12_tophat_se_radius__by_sza_region.png  recovered counts per (sza, r)
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
from skimage.measure import label
from skimage.morphology import disk, white_tophat

from _common import (
    SZA_BINS,
    make_slug_dir, resolve_out_root, stamp,
)


SLUG = "q12_tophat_se_radius"
PIXEL_AREA_M2 = 100.0


def _hpc_v4_test_chips_root():
    candidates = [
        Path("/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/data/v4_clean/test_chips"),
        Path("/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/data/v4_clean/test_chips"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _hpc_base_root():
    candidates = [
        Path("/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/results/model_comparison_20260423_stage1_vs_baseline/area_comparison/baseline_v3_balanced_aug/test"),
        Path("/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/results/model_comparison_20260423_stage1_vs_baseline/area_comparison/baseline_v3_balanced_aug/test"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chips_root", default=str(_hpc_v4_test_chips_root()),
                   help="v4_clean test_chips root (sza_lt65/, sza_65_70/, ...)")
    p.add_argument("--base_root", default=str(_hpc_base_root()),
                   help="UNet test predictions root containing per-sza/UNet/geotiffs/")
    p.add_argument("--th_thresh", type=float, default=0.05,
                   help="Top-hat response threshold in reflectance units")
    p.add_argument("--min_area_px", type=int, default=16,
                   help="Min component size in pixels (40 m root length)")
    p.add_argument("--radii", default="5,10,15,20",
                   help="Comma list of disk radii to test")
    p.add_argument("--out_root", default=str(resolve_out_root()),
                   help="Parent directory; a slug subfolder is created under it")
    return p.parse_args()


def list_v4_test_chips(chips_root):
    """Return list of (chip_path, sza_bin, stem) under chips_root."""
    out = []
    for sza in SZA_BINS:
        d = Path(chips_root) / sza
        if not d.is_dir():
            continue
        for tif in sorted(d.glob("*.tif")):
            out.append((tif, sza, tif.stem))
    return out


def read_b08(tif_path):
    with rasterio.open(tif_path) as src:
        if src.count < 3:
            return None
        return src.read(3).astype(np.float32)


def read_base_mask(base_root, sza_bin, stem, ref_shape):
    """Try to load <base_root>/<sza_bin>/UNet/geotiffs/<stem>_pred.tif."""
    p = Path(base_root) / sza_bin / "UNet" / "geotiffs" / f"{stem}_pred.tif"
    if p.exists():
        with rasterio.open(p) as src:
            arr = src.read(1)
            return (arr > 0).astype(np.uint8)
    # Try the variant with the double-underscore stem (some checkpoints write that).
    alt = Path(base_root) / sza_bin / "UNet" / "geotiffs" / f"{stem}__pred.tif"
    if alt.exists():
        with rasterio.open(alt) as src:
            arr = src.read(1)
            return (arr > 0).astype(np.uint8)
    return None


def recover(b08, base_mask, radius, th_thresh, min_area_px):
    """Run white-tophat recovery, subtract base mask, return list of areas in m^2."""
    response = white_tophat(b08, disk(radius))
    candidate = (response >= th_thresh).astype(np.uint8)
    if base_mask is not None:
        candidate = candidate & (base_mask == 0).astype(np.uint8)
    labels = label(candidate, connectivity=2)
    if labels.max() == 0:
        return []
    areas = []
    for li in range(1, labels.max() + 1):
        n_px = int((labels == li).sum())
        if n_px >= min_area_px:
            areas.append(n_px * PIXEL_AREA_M2)
    return areas


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    out_dir = make_slug_dir(SLUG, args.out_root)
    radii = [int(x) for x in args.radii.split(",")]
    print(f"Radii: {radii}")
    print(f"chips_root: {args.chips_root}")
    print(f"base_root:  {args.base_root}")

    chips = list_v4_test_chips(args.chips_root)
    print(f"Found {len(chips)} v4_clean test chips")

    rows = []
    n_skipped = 0
    n_no_base = 0
    per_radius_areas = {r: [] for r in radii}
    for tif, sza, stem in chips:
        b08 = read_b08(tif)
        if b08 is None:
            n_skipped += 1
            continue
        base_mask = read_base_mask(args.base_root, sza, stem, b08.shape)
        if base_mask is None:
            n_no_base += 1
        for r in radii:
            areas = recover(b08, base_mask, r, args.th_thresh, args.min_area_px)
            rows.append({
                "chip_stem":  stem,
                "sza_bin":    sza,
                "radius_px":  r,
                "n_recovered": len(areas),
                "total_recovered_m2": float(sum(areas)),
                "max_area_m2": float(max(areas)) if areas else 0.0,
                "median_area_m2": float(np.median(areas)) if areas else 0.0,
                "has_base_mask": base_mask is not None,
            })
            per_radius_areas[r].extend(areas)

    if n_skipped:
        print(f"Skipped {n_skipped} chips with too few bands")
    if n_no_base:
        print(f"{n_no_base} chips had no base mask (top-hat against all-zero base)")
    print(f"Wrote {len(rows)} (chip, radius) rows")

    if not rows:
        raise SystemExit("No rows; nothing to plot.")

    csv_path = out_dir / f"{stamp(SLUG)}.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {csv_path}")

    # 1. Headline counts per radius
    print("\nRecovered polygon counts per radius (over v4_clean test set):")
    for r in radii:
        sub = [row for row in rows if row["radius_px"] == r]
        n_polys = sum(row["n_recovered"] for row in sub)
        total_area = sum(row["total_recovered_m2"] for row in sub)
        print(f"  disk({r}): n_polys={n_polys}  total_recovered={total_area/1e6:.4f} km^2 "
              f"({n_polys/len(sub):.2f} polys/chip)")

    # 2. Overview: area histograms per radius
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.cm.viridis
    for i, r in enumerate(radii):
        areas = np.array(per_radius_areas[r], dtype=np.float64)
        if areas.size == 0:
            continue
        bins = np.geomspace(max(areas.min(), 100.0), max(areas.max(), 1e4), 40)
        ax.hist(areas, bins=bins, alpha=0.5,
                color=cmap(i / max(1, len(radii) - 1)),
                label=f"disk({r})  n={areas.size}", histtype="step", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Recovered polygon area (m^2)")
    ax.set_ylabel("Polygon count")
    ax.set_title(f"Q12: top-hat recovered area distribution per disk radius "
                 f"(th_thresh={args.th_thresh}, min_area_px={args.min_area_px})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    overview_path = out_dir / f"{stamp(SLUG, 'overview')}.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {overview_path}")

    # 3. Per-(SZA bin, radius) recovered count bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(radii)
    for i, r in enumerate(radii):
        vals = []
        for sza in SZA_BINS:
            sub = [row for row in rows
                   if row["radius_px"] == r and row["sza_bin"] == sza]
            vals.append(sum(row["n_recovered"] for row in sub))
        x = np.arange(len(SZA_BINS))
        ax.bar(x + (i - (len(radii) - 1) / 2) * width, vals, width=width,
               color=cmap(i / max(1, len(radii) - 1)),
               label=f"disk({r})")
    ax.set_xticks(np.arange(len(SZA_BINS)))
    ax.set_xticklabels(SZA_BINS, fontsize=9)
    ax.set_ylabel("Recovered polygon count (sum across test chips)")
    ax.set_title("Q12: top-hat recovered polygons per (SZA bin, radius)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    by_path = out_dir / f"{stamp(SLUG, 'by_sza_region')}.png"
    fig.savefig(by_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {by_path}")


if __name__ == "__main__":
    main()
