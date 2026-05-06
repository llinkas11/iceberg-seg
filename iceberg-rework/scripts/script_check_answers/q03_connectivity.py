"""
q03_connectivity.py: empirical answer to script-check question 3.

Question (from script-check-README.md, threshold_tifs.py):
  "No separation of touching icebergs. Two icebergs whose pixels are
  8-connected through one bright pixel become one polygon. Is that acceptable
  here, or should we use 4-connectivity, watershed, or distance-transform
  splitting?"

What this script does:
  1. Glob every chip TIF and build the production B08 >= 0.22 binary mask
     (same as threshold_tifs.py), skipping chips that fail the IC rule.
  2. Polygonise the mask twice via rasterio.features.shapes:
       - 4-connectivity (the production default in threshold_tifs.py)
       - 8-connectivity
     and apply the 100 m^2 min-area cutoff to both.
  3. Tabulate per-chip polygon-count delta and area delta (4 - 8) so a
     reviewer can read the magnitude of "icebergs merged via diagonal
     contact" directly.
  4. Emit a per-chip CSV plus two PNGs: an aggregate scatter of polygon
     counts at the two connectivities and a per-(region, SZA bin) bar chart
     of the count and area deltas.

Inputs:
  --chips_root   Directory containing <region>/<sza_bin>/tifs/*.tif.
  --b08_idx      Band index of B08 in the chip TIF (default 2).
  --threshold    Reflectance threshold for the binary mask (default 0.22).
  --min_area_m2  Min-area cutoff applied to both polygon sets (default 100).
  --ic_threshold Skip-chip cutoff on (B08 >= threshold).mean() (default 0.15).
  --out_root     Parent directory under which a slug folder is created.

Outputs (under <out_root>/q03_connectivity/):
  <ts>__q03_connectivity.csv               one row per evaluated chip
  <ts>__q03_connectivity__overview.png     scatter of n4 vs n8 per chip
  <ts>__q03_connectivity__by_sza_region.png  per-bin count/area delta bars
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

from _common import (
    SZA_BINS, REGIONS,
    list_chip_tifs, make_slug_dir, resolve_chips_root, resolve_out_root, stamp,
)


SLUG = "q03_connectivity"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chips_root", default=str(resolve_chips_root()),
                   help="Root containing <region>/<sza_bin>/tifs/*.tif")
    p.add_argument("--b08_idx", type=int, default=2,
                   help="0-indexed band of B08 in the chip TIF")
    p.add_argument("--threshold", type=float, default=0.22,
                   help="Reflectance threshold for the binary mask (default 0.22)")
    p.add_argument("--min_area_m2", type=float, default=100.0,
                   help="Min-area cutoff applied to both connectivity sets")
    p.add_argument("--ic_threshold", type=float, default=0.15,
                   help="IC chip-rejection cutoff; chips above this are skipped")
    p.add_argument("--out_root", default=str(resolve_out_root()),
                   help="Parent directory; a slug subfolder is created under it")
    return p.parse_args()


def polygonise_at(mask, transform, connectivity, min_area_m2):
    """Polygonise mask at the given connectivity and apply the min-area cutoff."""
    n = 0
    total_area = 0.0
    for geom_dict, val in rio_shapes(mask.astype(np.uint8),
                                     transform=transform,
                                     connectivity=connectivity):
        if val != 1:
            continue
        geom = shape(geom_dict)
        if geom.is_empty or geom.area < min_area_m2:
            continue
        n += 1
        total_area += float(geom.area)
    return n, total_area


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    # 1. Resolve output dir and discover chips
    out_dir = make_slug_dir(SLUG, args.out_root)
    chip_rows = list_chip_tifs(args.chips_root)
    if not chip_rows:
        raise SystemExit(f"No chips found under {args.chips_root}")
    print(f"Found {len(chip_rows)} chips under {args.chips_root}")

    # 2. Iterate chips: polygonise at 4- and 8-connectivity
    rows = []
    too_few_bands = 0
    ic_skipped = 0
    for tif, region, sza_bin in chip_rows:
        with rasterio.open(tif) as src:
            if src.count <= args.b08_idx:
                too_few_bands += 1
                continue
            b08 = src.read(args.b08_idx + 1).astype(np.float32)
            transform = src.transform

        mask = (b08 >= args.threshold)
        ic_frac = float(mask.mean())
        if ic_frac > args.ic_threshold:
            ic_skipped += 1
            continue

        n4, area4 = polygonise_at(mask, transform, 4, args.min_area_m2)
        n8, area8 = polygonise_at(mask, transform, 8, args.min_area_m2)

        rows.append({
            "chip_stem":    tif.stem,
            "region":       region,
            "sza_bin":      sza_bin,
            "ic_frac":      ic_frac,
            "n_polys_c4":   n4,
            "n_polys_c8":   n8,
            "area_m2_c4":   area4,
            "area_m2_c8":   area8,
            "delta_n":      n4 - n8,         # positive => 4-conn splits a join
            "delta_area_m2": area4 - area8,  # near zero unless tiny boundary pixels added
        })

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
    n4_total   = sum(r["n_polys_c4"] for r in rows)
    n8_total   = sum(r["n_polys_c8"] for r in rows)
    a4_total   = sum(r["area_m2_c4"] for r in rows)
    a8_total   = sum(r["area_m2_c8"] for r in rows)
    n_diff_chips = sum(1 for r in rows if r["delta_n"] != 0)
    pct_diff_chips = n_diff_chips / max(len(rows), 1)
    rel_count_delta = (n4_total - n8_total) / max(n8_total, 1)
    rel_area_delta = (a4_total - a8_total) / max(a8_total, 1.0)

    print("\nPolygon counts (post 100 m^2 cutoff):")
    print(f"  4-connectivity: {n4_total}  area={a4_total/1e6:.2f} km^2")
    print(f"  8-connectivity: {n8_total}  area={a8_total/1e6:.2f} km^2")
    print(f"  count delta (4 - 8) = {n4_total - n8_total} ({rel_count_delta:+.3%})")
    print(f"  area  delta (4 - 8) = {(a4_total - a8_total)/1e6:+.4f} km^2 ({rel_area_delta:+.3%})")
    print(f"  chips with non-zero delta_n: {n_diff_chips}/{len(rows)} ({pct_diff_chips:.2%})")

    # 5. Overview scatter: n4 vs n8 per chip
    n4 = np.array([r["n_polys_c4"] for r in rows], dtype=np.int64)
    n8 = np.array([r["n_polys_c8"] for r in rows], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(n8, n4, s=12, alpha=0.45, color="#37474F", edgecolor="none")
    lim = max(n4.max(), n8.max(), 1)
    ax.plot([0, lim], [0, lim], color="#1976D2", lw=1.0, ls="--", alpha=0.8,
            label="y = x (no split)")
    ax.set_xlabel("n_polys at 8-connectivity (per chip)")
    ax.set_ylabel("n_polys at 4-connectivity (per chip)")
    ax.set_title(f"Q3: 4-conn vs 8-conn polygon counts per chip "
                 f"(n_chips={len(rows)}, count delta = {rel_count_delta:+.3%})")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")
    overview_path = out_dir / f"{stamp(SLUG, 'overview')}.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {overview_path}")

    # 6. Per-(region, SZA bin) count and area deltas
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    group_labels = []
    count_deltas = []
    area_deltas = []
    for r in REGIONS:
        for s in SZA_BINS:
            sub = [row for row in rows if row["region"] == r and row["sza_bin"] == s]
            if not sub:
                group_labels.append(f"{r}\n{s}\n(n=0)")
                count_deltas.append(0.0)
                area_deltas.append(0.0)
                continue
            group_labels.append(f"{r}\n{s}\n(n={len(sub)})")
            n4 = sum(row["n_polys_c4"] for row in sub)
            n8 = sum(row["n_polys_c8"] for row in sub)
            a4 = sum(row["area_m2_c4"] for row in sub)
            a8 = sum(row["area_m2_c8"] for row in sub)
            count_deltas.append((n4 - n8) / max(n8, 1))
            area_deltas.append((a4 - a8) / max(a8, 1.0))

    x = np.arange(len(group_labels))
    axes[0].bar(x, np.array(count_deltas) * 100, color="#37474F")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(group_labels, fontsize=8)
    axes[0].set_ylabel("(n4 - n8) / n8 (%)")
    axes[0].set_title("Polygon-count delta (4-conn relative to 8-conn)")
    axes[0].axhline(0.0, color="#90A4AE", lw=0.8, ls="--", alpha=0.6)
    axes[0].grid(True, axis="y", alpha=0.3, linestyle="--")
    axes[1].bar(x, np.array(area_deltas) * 100, color="#1976D2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(group_labels, fontsize=8)
    axes[1].set_ylabel("(area4 - area8) / area8 (%)")
    axes[1].set_title("Total-area delta (4-conn relative to 8-conn)")
    axes[1].axhline(0.0, color="#90A4AE", lw=0.8, ls="--", alpha=0.6)
    axes[1].grid(True, axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(f"Q3: per-(region, SZA bin) connectivity deltas (n_chips={len(rows)})")
    by_path = out_dir / f"{stamp(SLUG, 'by_sza_region')}.png"
    fig.savefig(by_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {by_path}")


if __name__ == "__main__":
    main()
