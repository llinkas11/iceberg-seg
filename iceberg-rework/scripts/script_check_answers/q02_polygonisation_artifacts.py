"""
q02_polygonisation_artifacts.py: empirical answer to script-check question 2.

Question (from script-check-README.md, threshold_tifs.py):
  "Connected-component polygonisation. rasterio.features.shapes on the binary
  mask, no morphological cleanup before vectorisation. Any concerns about
  salt-and-pepper artifacts inflating the polygon count? The 100 m^2 min-area
  cutoff (~10 x 10 m) does most of the cleanup downstream."

What this script does:
  1. Glob every chip TIF and build the production B08 >= 0.22 binary mask
     (same as threshold_tifs.py, with no morphological cleanup).
  2. Polygonise the mask with rasterio.features.shapes at 4-connectivity (the
     default), record the polygon count and area-distribution PRE the 100 m^2
     filter, including counts at <= 1 px and <= 2 px.
  3. Repeat the polygonisation after a single iteration of binary opening
     (skimage.morphology.binary_opening) on the mask, comparing polygon count
     and total post-100 m^2 area against the production path.
  4. Skip chips that fail the IC chip-rejection rule (>15% > 0.22) so the
     numbers reflect the chips that actually feed downstream area retrieval.
  5. Emit a per-chip CSV plus two PNGs: the polygon-area histogram pre-filter
     and a per-(region, SZA bin) bar chart of the opening delta.

Inputs:
  --chips_root   Directory containing <region>/<sza_bin>/tifs/*.tif.
  --b08_idx      Band index of B08 in the chip TIF (default 2).
  --threshold    Reflectance threshold for the binary mask (default 0.22).
  --min_area_m2  Production min-area cutoff used for the post-filter compare
                 (default 100, mirroring threshold_tifs.py MIN_AREA_M2).
  --ic_threshold Skip-chip cutoff on (B08 >= threshold).mean() (default 0.15).
  --out_root     Parent directory under which a slug folder is created.

Outputs (under <out_root>/q02_polygonisation_artifacts/):
  <ts>__q02_polygonisation_artifacts.csv               one row per evaluated chip
  <ts>__q02_polygonisation_artifacts__overview.png     polygon-area histogram
  <ts>__q02_polygonisation_artifacts__by_sza_region.png  opening delta per bin
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
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape
from skimage.morphology import binary_opening, disk

from _common import (
    SZA_BINS, REGIONS,
    list_chip_tifs, make_slug_dir, resolve_chips_root, resolve_out_root, stamp,
)


SLUG = "q02_polygonisation_artifacts"
PIXEL_AREA_M2 = 100.0  # 10 m x 10 m S2 pixel area


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chips_root", default=str(resolve_chips_root()),
                   help="Root containing <region>/<sza_bin>/tifs/*.tif")
    p.add_argument("--b08_idx", type=int, default=2,
                   help="0-indexed band of B08 in the chip TIF")
    p.add_argument("--threshold", type=float, default=0.22,
                   help="Reflectance threshold for the binary mask (default 0.22)")
    p.add_argument("--min_area_m2", type=float, default=100.0,
                   help="Production min-area cutoff for the post-filter compare")
    p.add_argument("--ic_threshold", type=float, default=0.15,
                   help="IC chip-rejection cutoff; chips above this are skipped")
    p.add_argument("--out_root", default=str(resolve_out_root()),
                   help="Parent directory; a slug subfolder is created under it")
    return p.parse_args()


def polygonise(mask, transform):
    """
    Run rasterio.features.shapes on a binary mask and return a list of (geom,
    area_m2) tuples for value=1 polygons. No min-area cutoff applied here.
    """
    out = []
    for geom_dict, val in rio_shapes(mask.astype(np.uint8), transform=transform):
        if val != 1:
            continue
        geom = shape(geom_dict)
        if geom.is_empty:
            continue
        out.append((geom, float(geom.area)))
    return out


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    # 1. Resolve output dir and discover chips
    out_dir = make_slug_dir(SLUG, args.out_root)
    chip_rows = list_chip_tifs(args.chips_root)
    if not chip_rows:
        raise SystemExit(f"No chips found under {args.chips_root}")
    print(f"Found {len(chip_rows)} chips under {args.chips_root}")

    # 2. Iterate chips: production polygonisation, opening polygonisation
    rows = []
    all_areas_prod_pre = []  # for the area histogram (pre-100m2 filter)
    too_few_bands = 0
    ic_skipped = 0
    for tif, region, sza_bin in chip_rows:
        with rasterio.open(tif) as src:
            if src.count <= args.b08_idx:
                too_few_bands += 1
                continue
            b08 = src.read(args.b08_idx + 1).astype(np.float32)
            transform = src.transform

        mask_prod = (b08 >= args.threshold)
        ic_frac = float(mask_prod.mean())
        if ic_frac > args.ic_threshold:
            ic_skipped += 1
            continue

        # 2a. Production path: shapes on raw mask
        polys_prod = polygonise(mask_prod, transform)
        areas_prod = np.array([a for _, a in polys_prod], dtype=np.float64)
        n_prod = len(areas_prod)
        n_1px = int((areas_prod <= PIXEL_AREA_M2 + 1e-3).sum())
        n_2px = int((areas_prod <= 2 * PIXEL_AREA_M2 + 1e-3).sum())
        n_prod_kept = int((areas_prod >= args.min_area_m2).sum())
        area_prod_kept = float(areas_prod[areas_prod >= args.min_area_m2].sum())

        # 2b. Opening path: 1-iter binary opening with disk(1) before shapes
        mask_open = binary_opening(mask_prod, disk(1))
        polys_open = polygonise(mask_open, transform)
        areas_open = np.array([a for _, a in polys_open], dtype=np.float64)
        n_open = len(areas_open)
        n_open_kept = int((areas_open >= args.min_area_m2).sum())
        area_open_kept = float(areas_open[areas_open >= args.min_area_m2].sum())

        rows.append({
            "chip_stem":         tif.stem,
            "region":            region,
            "sza_bin":           sza_bin,
            "ic_frac":           ic_frac,
            "n_polys_prod":      n_prod,
            "n_polys_le_1px":    n_1px,
            "n_polys_le_2px":    n_2px,
            "n_polys_prod_kept": n_prod_kept,
            "area_m2_prod_kept": area_prod_kept,
            "n_polys_open":      n_open,
            "n_polys_open_kept": n_open_kept,
            "area_m2_open_kept": area_open_kept,
        })
        all_areas_prod_pre.extend(areas_prod.tolist())

    if too_few_bands:
        print(f"Skipped {too_few_bands} chips with too few bands")
    if ic_skipped:
        print(f"Skipped {ic_skipped} chips for IC > {args.ic_threshold}")
    print(f"Evaluated {len(rows)} chips")

    if not rows:
        raise SystemExit("No chips passed the IC filter; nothing to plot.")

    # 3. Write per-chip CSV
    fieldnames = list(rows[0].keys())
    csv_path = out_dir / f"{stamp(SLUG)}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {csv_path}")

    # 4. Headline aggregates
    n_prod_total = sum(r["n_polys_prod"] for r in rows)
    n_1px_total  = sum(r["n_polys_le_1px"] for r in rows)
    n_2px_total  = sum(r["n_polys_le_2px"] for r in rows)
    n_prod_kept_total = sum(r["n_polys_prod_kept"] for r in rows)
    n_open_kept_total = sum(r["n_polys_open_kept"] for r in rows)
    area_prod_kept_total = sum(r["area_m2_prod_kept"] for r in rows)
    area_open_kept_total = sum(r["area_m2_open_kept"] for r in rows)

    pct_1px = n_1px_total / max(n_prod_total, 1)
    pct_2px = n_2px_total / max(n_prod_total, 1)
    poly_delta_pct = (n_open_kept_total - n_prod_kept_total) / max(n_prod_kept_total, 1)
    area_delta_pct = (area_open_kept_total - area_prod_kept_total) / max(area_prod_kept_total, 1.0)

    print("\nPolygon counts pre-100 m^2 filter:")
    print(f"  total polygons (production):  {n_prod_total}")
    print(f"  <= 1-pixel polygons: {n_1px_total} ({pct_1px:.2%})")
    print(f"  <= 2-pixel polygons: {n_2px_total} ({pct_2px:.2%})")
    print("\nKept polygons (>= 100 m^2):")
    print(f"  production:        {n_prod_kept_total}  area={area_prod_kept_total/1e6:.2f} km^2")
    print(f"  after 1x opening:  {n_open_kept_total}  area={area_open_kept_total/1e6:.2f} km^2")
    print(f"  poly-count delta (opening - production): {poly_delta_pct:+.2%}")
    print(f"  area delta        (opening - production): {area_delta_pct:+.2%}")

    # 5. Overview: polygon-area histogram pre-filter
    areas_arr = np.asarray(all_areas_prod_pre, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bins = np.geomspace(max(areas_arr.min(), PIXEL_AREA_M2 / 2), max(areas_arr.max(), 1e4), 60)
    ax.hist(areas_arr, bins=bins, color="#37474F", alpha=0.85)
    ax.axvline(PIXEL_AREA_M2, color="#1976D2", lw=1.0, ls=":", alpha=0.8)
    ax.text(PIXEL_AREA_M2, ax.get_ylim()[1] * 0.95, " 1 px (100 m^2)",
            color="#1976D2", fontsize=8, va="top")
    ax.axvline(2 * PIXEL_AREA_M2, color="#1976D2", lw=1.0, ls=":", alpha=0.6)
    ax.text(2 * PIXEL_AREA_M2, ax.get_ylim()[1] * 0.85, " 2 px (200 m^2)",
            color="#1976D2", fontsize=8, va="top")
    ax.axvline(args.min_area_m2, color="#D32F2F", lw=1.2, ls="--", alpha=0.9)
    ax.text(args.min_area_m2, ax.get_ylim()[1] * 0.75,
            f"  prod cutoff = {args.min_area_m2:.0f} m^2",
            color="#D32F2F", fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Polygon area (m^2), pre-cutoff")
    ax.set_ylabel("Polygon count")
    ax.set_title(f"Q2: TR polygon-area distribution pre 100 m^2 filter "
                 f"(n_chips={len(rows)}, n_polys={len(areas_arr)})")
    ax.grid(True, alpha=0.3, linestyle="--")
    overview_path = out_dir / f"{stamp(SLUG, 'overview')}.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {overview_path}")

    # 6. Per-(region, SZA bin): opening delta on kept-polygon count
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.35
    group_labels = []
    prod_vals = []
    open_vals = []
    for r in REGIONS:
        for s in SZA_BINS:
            sub = [row for row in rows if row["region"] == r and row["sza_bin"] == s]
            if not sub:
                group_labels.append(f"{r}\n{s}\n(n=0)")
                prod_vals.append(0.0)
                open_vals.append(0.0)
                continue
            group_labels.append(f"{r}\n{s}\n(n={len(sub)})")
            prod_vals.append(sum(row["n_polys_prod_kept"] for row in sub))
            open_vals.append(sum(row["n_polys_open_kept"] for row in sub))
    x = np.arange(len(group_labels))
    ax.bar(x - width / 2, prod_vals, width=width, color="#37474F", label="production")
    ax.bar(x + width / 2, open_vals, width=width, color="#1976D2",
           label="after 1x binary_opening")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("Kept polygon count (>= 100 m^2)")
    ax.set_title("Q2: production vs binary-opening polygon counts per (region, SZA bin)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    by_path = out_dir / f"{stamp(SLUG, 'by_sza_region')}.png"
    fig.savefig(by_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {by_path}")


if __name__ == "__main__":
    main()
