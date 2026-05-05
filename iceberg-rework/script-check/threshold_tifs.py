"""
threshold_tifs.py: Apply Fisser B08 >= 0.12 NIR threshold to S2 chip .tifs.

Mirrors the output format of predict_tifs.py so compare_areas.py can load both.

Usage:
  python threshold_tifs.py \\
      --chips_dir chips/KQ/sza_65_70/tifs \\
      --out_dir   georef_predictions/KQ/sza_65_70

Output:
  out_dir/all_icebergs.gpkg  iceberg polygons with area_m2
  out_dir/method_config.json parameters used by this run
  out_dir/skipped_chips.csv  chips excluded, with a reason

Note:
  --b08_idx is the 0-indexed band position of B08 in the chip stack.
  Default is 2 (i.e. bands were stacked as B04, B03, B08 by chip_sentinel2.py).
  If you used a different band order, adjust accordingly.
"""

import os
import argparse
import warnings
from glob import glob

import numpy as np
import rasterio as rio
from rasterio.features import shapes as rio_shapes
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd

from _method_common import (
    write_method_config, write_skipped_chips,
    SKIP_TOO_FEW_BANDS, SKIP_IC_BLOCK_FILTER,
)

warnings.filterwarnings("ignore")

THRESHOLD   = 0.22   # Fisser 2024 B08 NIR reflectance threshold (0.12) + 0.10 DN offset correction
                     # All scenes have processing baseline ≥4.0 (N0500/N0510), which adds +1000 DN
                     # chip_sentinel2.py does not subtract this offset, so reflectances are +0.1 high
                     # 0.22 here = 0.12 in Fisser's corrected reflectance space
MIN_AREA_M2 = 100    # minimum polygon area in m2 (~10x10 m)
IC_THRESHOLD = 0.15  # Fisser 2025 IC block filter: skip chip if >15% of pixels exceed NIR threshold
                     # Flags chips dominated by sea ice rather than open water with icebergs


def apply_threshold(chips_dir, out_dir, b08_idx=2, threshold=THRESHOLD, min_area_m2=MIN_AREA_M2, ic_threshold=IC_THRESHOLD):
    os.makedirs(out_dir, exist_ok=True)

    tif_files = sorted(glob(os.path.join(chips_dir, "*.tif")))
    if not tif_files:
        print(f"No .tif files found in {chips_dir}")
        return

    print(f"Found {len(tif_files)} chips  threshold={threshold}  b08_idx={b08_idx}  ic_threshold={ic_threshold}")

    all_gdfs   = []
    skipped    = []   # one row per chip we refused to score

    for i, tif_path in enumerate(tif_files):
        stem = os.path.splitext(os.path.basename(tif_path))[0]

        with rio.open(tif_path) as src:
            chip = src.read().astype(np.float32)  # (C, H, W)
            meta = src.meta.copy()

        if chip.shape[0] <= b08_idx:
            print(f"  [{i+1}/{len(tif_files)}] SKIP {stem}: only {chip.shape[0]} bands")
            skipped.append({"chip_stem": stem, "reason": SKIP_TOO_FEW_BANDS,
                            "n_bands": chip.shape[0]})
            continue

        b08 = chip[b08_idx]

        # IC block filter (Fisser 2025): skip sea-ice-dominated chips
        ic_frac = float((b08 >= threshold).mean())
        if ic_frac > ic_threshold:
            print(f"  [{i+1:>4}/{len(tif_files)}] IC   {stem[:60]}  ic_frac={ic_frac:.2f}")
            skipped.append({"chip_stem": stem, "reason": SKIP_IC_BLOCK_FILTER,
                            "ic_frac": f"{ic_frac:.4f}"})
            continue

        iceberg_mask = (b08 >= threshold).astype(np.uint8)

        records = []
        for geom_dict, val in rio_shapes(iceberg_mask, transform=meta["transform"]):
            if val == 0:
                continue
            geom = shape(geom_dict)
            if geom.is_empty or geom.area < min_area_m2:
                continue
            records.append({
                "geometry"   : geom,
                "class_id"   : 1,
                "class_name" : "iceberg",
                "area_m2"    : round(geom.area, 2),
                "source_file": os.path.basename(tif_path),
            })

        print(f"  [{i+1:>4}/{len(tif_files)}] {stem[:60]}  icebergs={len(records)}")

        if records:
            gdf = gpd.GeoDataFrame(records, crs=meta["crs"])
            all_gdfs.append(gdf)

    # Write provenance files before the empty-run short-circuit below, so an
    # empty result still produces method_config.json + skipped_chips.csv the
    # evaluator can join on.
    cfg_path = write_method_config(
        out_dir, "TR",
        params={
            "chips_dir":    os.path.abspath(chips_dir),
            "threshold":    threshold,
            "min_area_m2":  min_area_m2,
            "b08_idx":      b08_idx,
            "ic_threshold": ic_threshold,
        },
    )
    skip_path = write_skipped_chips(out_dir, skipped)

    n_ic      = sum(1 for r in skipped if r["reason"] == SKIP_IC_BLOCK_FILTER)
    n_skipped = sum(1 for r in skipped if r["reason"] == SKIP_TOO_FEW_BANDS)

    if not all_gdfs:
        print("\nNo icebergs detected across all chips.")
        if n_ic:
            print(f"IC-filtered: {n_ic} chips (sea ice contamination)")
        if n_skipped:
            print(f"Skipped:     {n_skipped} chips (too few bands)")
        print(f"Method config : {cfg_path}")
        print(f"Skipped chips : {skip_path}")
        return

    target_crs = all_gdfs[0].crs
    reprojected = [gdf.to_crs(target_crs) if gdf.crs != target_crs else gdf for gdf in all_gdfs]
    merged = gpd.GeoDataFrame(pd.concat(reprojected, ignore_index=True), crs=target_crs)
    merged["iceberg_id"] = range(1, len(merged) + 1)

    icebergs = merged[merged["class_name"] == "iceberg"]
    print(f"\n{'-'*50}")
    print(f"Total iceberg polygons : {len(icebergs)}")
    if len(icebergs) > 0:
        print(f"  min  = {icebergs['area_m2'].min():.1f} m2")
        print(f"  mean = {icebergs['area_m2'].mean():.1f} m2")
        print(f"  max  = {icebergs['area_m2'].max():.1f} m2")
    if n_ic:
        print(f"IC-filtered: {n_ic} chips (sea ice contamination)")
    if n_skipped:
        print(f"Skipped:     {n_skipped} chips (too few bands)")
    print(f"{'-'*50}")

    out_path = os.path.join(out_dir, "all_icebergs.gpkg")
    merged.to_file(out_path, driver="GPKG")
    print(f"\nSaved         : {out_path}")
    print(f"Method config : {cfg_path}")
    print(f"Skipped chips : {skip_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply Fisser B08 ≥ 0.12 NIR threshold to S2 chip .tifs"
    )
    parser.add_argument("--chips_dir", required=True,
                        help="Directory of .tif chip files (same dir used by predict_tifs.py --imgs_dir)")
    parser.add_argument("--out_dir",   required=True,
                        help="Output directory, use the same path as predict_tifs.py --out_dir")
    parser.add_argument("--b08_idx",   type=int,   default=2,
                        help="0-indexed band position of B08 in chip stack (default: 2 for B04/B03/B08 order)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"NIR reflectance threshold (default: {THRESHOLD})")
    parser.add_argument("--min_area",     type=float, default=MIN_AREA_M2,
                        help=f"Min iceberg area in m2 (default: {MIN_AREA_M2})")
    parser.add_argument("--ic_threshold", type=float, default=IC_THRESHOLD,
                        help=f"IC block filter: skip chip if fraction of bright pixels exceeds this (default: {IC_THRESHOLD})")
    args = parser.parse_args()

    apply_threshold(args.chips_dir, args.out_dir, args.b08_idx, args.threshold, args.min_area, args.ic_threshold)


if __name__ == "__main__":
    main()
