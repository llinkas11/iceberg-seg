"""
q14_tophat_crs_audit.py: empirical answer to script-check question 14.

Question (from script-check-README.md, tophat_recover.py):
  "Subtraction of base mask. Recovered candidates are
  `(response >= th_thresh) AND NOT base_mask`. Base mask is built from
  `<stem>_pred.tif` if present, else by rasterising the base method's
  polygons. Any concerns about the rasterised-polygon path on per-chip
  CRS mismatches?"

What this script does:
  1. Enumerate every chip TIF under chips_root and read its CRS via rasterio.
  2. Enumerate every base-method GeoPackage under
     <results_root>/area_comparison/baseline_v3_balanced_aug/test/<sza_bin>/<method>/gpkgs/
     and read each GeoDataFrame's CRS.
  3. Cross-reference by chip stem: for every chip with a corresponding
     gpkg in any base method, compare the chip CRS to the gpkg CRS.
  4. Tabulate (chip CRS, gpkg CRS, agree?) per chip + method, and emit:
     - per-chip CSV listing every (chip_stem, region, sza_bin, method,
       chip_crs, gpkg_crs, mismatch).
     - overview bar chart of CRS-mismatch counts per (region, sza_bin).

Inputs:
  --chips_root    Root containing <region>/<sza_bin>/tifs/*.tif.
  --results_root  Root containing area_comparison/.../test/ tree.
  --out_root      Parent directory; a slug subfolder is created under it.

Outputs (under <out_root>/q14_tophat_crs_audit/):
  <ts>__q14_tophat_crs_audit.csv               one row per (chip, method)
  <ts>__q14_tophat_crs_audit__overview.png     mismatch count per (region, sza)

Usage on moosehead:
  python3 q14_tophat_crs_audit.py

Deploy:
  rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/script_check_answers/ \
    bowdoin:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/script_check_answers/
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import geopandas as gpd

from _common import (
    SZA_BINS, REGIONS,
    list_chip_tifs, make_slug_dir, resolve_chips_root, resolve_out_root, stamp,
)


SLUG = "q14_tophat_crs_audit"
BASE_METHODS = ["TR", "OT", "UNet", "UNet_TR", "UNet_OT", "UNet_CRF"]


def _resolve_results_root():
    """HPC + Mac fallback for the model-comparison test results tree."""
    hpc = Path("/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/results/model_comparison_20260423_stage1_vs_baseline/area_comparison/baseline_v3_balanced_aug/test")
    if hpc.exists():
        return hpc
    return Path("/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/results/model_comparison_20260423_stage1_vs_baseline/area_comparison/baseline_v3_balanced_aug/test")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chips_root", default=str(resolve_chips_root()),
                   help="Root containing <region>/<sza_bin>/tifs/*.tif")
    p.add_argument("--results_root", default=str(_resolve_results_root()),
                   help="Root containing test/<sza_bin>/<method>/gpkgs/")
    p.add_argument("--out_root", default=str(resolve_out_root()),
                   help="Parent directory; a slug subfolder is created under it")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Resolve output dir
    out_dir = make_slug_dir(SLUG, args.out_root)
    print(f"Chips root:   {args.chips_root}")
    print(f"Results root: {args.results_root}")

    # 2. Index chip CRS by stem
    chip_rows = list_chip_tifs(args.chips_root)
    print(f"Reading CRS from {len(chip_rows)} chip TIFs ...")
    chip_crs_by_stem = {}
    chip_meta_by_stem = {}
    for tif, region, sza_bin in chip_rows:
        try:
            with rasterio.open(tif) as src:
                crs_str = str(src.crs) if src.crs is not None else "NONE"
        except Exception as exc:
            crs_str = f"ERROR: {exc}"
        chip_crs_by_stem[tif.stem] = crs_str
        chip_meta_by_stem[tif.stem] = (region, sza_bin)
    print(f"Chip CRS values seen: {sorted(set(chip_crs_by_stem.values()))[:10]}")

    # 3. Walk base-method gpkgs and pair with chip CRS
    rows = []
    results_root = Path(args.results_root)
    for sza_bin in SZA_BINS:
        for method in BASE_METHODS:
            gpkg_dir = results_root / sza_bin / method / "gpkgs"
            if not gpkg_dir.is_dir():
                continue
            for gpkg in sorted(gpkg_dir.glob("*_icebergs.gpkg")):
                stem = gpkg.name[:-len("_icebergs.gpkg")]
                if stem not in chip_crs_by_stem:
                    # gpkg references a chip not in our chip pool.
                    continue
                try:
                    gdf = gpd.read_file(gpkg)
                    gpkg_crs = str(gdf.crs) if gdf.crs is not None else "NONE"
                except Exception as exc:
                    gpkg_crs = f"ERROR: {exc}"
                chip_crs = chip_crs_by_stem[stem]
                region, _ = chip_meta_by_stem[stem]
                mismatch = int(chip_crs != gpkg_crs)
                rows.append({
                    "chip_stem":  stem,
                    "region":     region,
                    "sza_bin":    sza_bin,
                    "method":     method,
                    "chip_crs":   chip_crs,
                    "gpkg_crs":   gpkg_crs,
                    "mismatch":   mismatch,
                })
    print(f"Recorded CRS pairs for {len(rows)} (chip, method) combinations")

    # 4. Write per-chip CSV
    csv_path = out_dir / f"{stamp(SLUG)}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "chip_stem", "region", "sza_bin", "method",
            "chip_crs", "gpkg_crs", "mismatch",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {csv_path}")

    if not rows:
        print("No (chip, method) gpkg pairs found; nothing to plot.")
        return

    # 5. Aggregate mismatch counts
    mismatch_total = sum(r["mismatch"] for r in rows)
    print(f"\nTotal mismatches: {mismatch_total}/{len(rows)} ({mismatch_total/len(rows):.3%})")
    crs_pairs = sorted({(r["chip_crs"], r["gpkg_crs"]) for r in rows})
    print("Distinct (chip_crs, gpkg_crs) pairs:")
    for cc, gc in crs_pairs:
        n = sum(1 for r in rows if r["chip_crs"] == cc and r["gpkg_crs"] == gc)
        flag = "MISMATCH" if cc != gc else "ok"
        print(f"  [{flag}] chip={cc}  gpkg={gc}  n={n}")

    # 6. Per-(region, SZA bin) bar chart of mismatch fraction
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = [f"{r}\n{s}" for r in REGIONS for s in SZA_BINS]
    x = np.arange(len(x_labels))
    bar_vals = []
    for r in REGIONS:
        for s in SZA_BINS:
            bin_rows = [row for row in rows
                        if row["region"] == r and row["sza_bin"] == s]
            n = len(bin_rows)
            mm = sum(row["mismatch"] for row in bin_rows)
            bar_vals.append(mm / n if n else 0.0)
    ax.bar(x, bar_vals, color="#D32F2F", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("CRS mismatch fraction")
    ax.set_title(f"Q14: tophat base-mask CRS mismatch per (region, SZA bin) "
                  f"(total: {mismatch_total}/{len(rows)} pairs)")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    overview_path = out_dir / f"{stamp(SLUG, 'overview')}.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {overview_path}")


if __name__ == "__main__":
    main()
