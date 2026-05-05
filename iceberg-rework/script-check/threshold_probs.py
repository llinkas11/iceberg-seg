"""
threshold_probs.py: UNet + TR method.

Applies a fixed NIR threshold to the UNet++ iceberg probability band.
Instead of argmax, labels a pixel as iceberg if P(iceberg) >= threshold.

Input:  softmax prob .tifs from predict_tifs.py
        (2-band float32 GeoTIFF: band 1=ocean, band 2=iceberg)
Output: all_icebergs.gpkg  (same format as threshold_tifs.py)

Usage:
  python threshold_probs.py \\
      --probs_dir area_comparison/KQ/sza_70_75/UNet/probs \\
      --out_dir   area_comparison/KQ/sza_70_75/UNet_TR

rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/threshold_probs.py smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
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
    write_method_config, write_skipped_chips, SKIP_TOO_FEW_PROB_BANDS,
)

warnings.filterwarnings("ignore", category=UserWarning)

ICEBERG_BAND = 1   # 0-indexed band in the prob .tif (ocean=0, iceberg=1)
THRESHOLD    = 0.22  # matches the Fisser threshold used in threshold_tifs.py
MIN_AREA_M2  = 100.0


def main():
    parser = argparse.ArgumentParser(
        description="UNet+TR: threshold UNet++ iceberg probability band"
    )
    parser.add_argument("--probs_dir",   required=True,
        help="Directory of *_probs.tif files from predict_tifs.py")
    parser.add_argument("--out_dir",     required=True,
        help="Output directory")
    parser.add_argument("--threshold",   type=float, default=THRESHOLD,
        help=f"P(iceberg) threshold (default {THRESHOLD})")
    parser.add_argument("--min_area_m2", type=float, default=MIN_AREA_M2)
    args = parser.parse_args()

    gpkg_dir = os.path.join(args.out_dir, "gpkgs")
    os.makedirs(gpkg_dir, exist_ok=True)

    prob_files = sorted(glob(os.path.join(args.probs_dir, "*_probs.tif")))
    print(f"Found {len(prob_files)} prob .tifs  |  threshold={args.threshold}")

    all_gdfs = []
    skipped  = []

    for i, prob_path in enumerate(prob_files):
        stem = os.path.basename(prob_path).replace("_probs.tif", "")

        with rio.open(prob_path) as src:
            probs = src.read().astype(np.float32)   # (C, H, W)
            meta  = src.meta.copy()

        if probs.shape[0] <= ICEBERG_BAND:
            skipped.append({"chip_stem": stem, "reason": SKIP_TOO_FEW_PROB_BANDS,
                            "n_bands": probs.shape[0]})
            continue

        iceberg_prob = probs[ICEBERG_BAND]
        mask = (iceberg_prob >= args.threshold).astype(np.uint8)

        records = []
        for geom_dict, val in rio_shapes(mask, transform=meta["transform"]):
            if val == 0:
                continue
            geom = shape(geom_dict)
            if geom.is_empty or geom.area < args.min_area_m2:
                continue
            records.append({"geometry": geom, "area_m2": geom.area,
                            "source_file": stem})

        n = len(records)
        print(f"  [{i+1:>4}/{len(prob_files)}] {stem[:60]}  icebergs={n}")

        if records:
            gdf = gpd.GeoDataFrame(records, crs=meta["crs"])
            gdf.to_file(os.path.join(gpkg_dir, f"{stem}_icebergs.gpkg"), driver="GPKG")
            all_gdfs.append(gdf)

    cfg_path = write_method_config(
        args.out_dir, "UNet_TR",
        params={
            "probs_dir":    os.path.abspath(args.probs_dir),
            "iceberg_band": ICEBERG_BAND,
            "threshold":    args.threshold,
            "min_area_m2":  args.min_area_m2,
        },
    )
    skip_path = write_skipped_chips(args.out_dir, skipped)

    if all_gdfs:
        target_crs = all_gdfs[0].crs
        reprojected = [gdf.to_crs(target_crs) if gdf.crs != target_crs else gdf for gdf in all_gdfs]
        merged = gpd.GeoDataFrame(pd.concat(reprojected, ignore_index=True),
                                  crs=target_crs)
        merged["area_m2"] = merged["area_m2"].round(2)
        out = os.path.join(args.out_dir, "all_icebergs.gpkg")
        merged.to_file(out, driver="GPKG")
        print(f"\nTotal icebergs : {len(merged)}")
        print(f"Saved          : {out}")
    else:
        print("\nNo icebergs detected.")
    print(f"Method config  : {cfg_path}")
    print(f"Skipped chips  : {skip_path}")


if __name__ == "__main__":
    main()