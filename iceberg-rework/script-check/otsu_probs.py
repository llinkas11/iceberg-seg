"""
otsu_probs.py: UNet + OT method.

Applies per-chip Otsu thresholding to the UNet++ iceberg probability band.
Finds the threshold that best separates the P(iceberg) histogram into
ocean-like vs iceberg-like regions within each chip.

Input:  softmax prob .tifs from predict_tifs.py
        (2-band float32 GeoTIFF: band 1=ocean, band 2=iceberg)
Output: all_icebergs.gpkg  (same format as otsu_threshold_tifs.py)

Usage:
  python otsu_probs.py \\
      --probs_dir area_comparison/KQ/sza_70_75/UNet/probs \\
      --out_dir   area_comparison/KQ/sza_70_75/UNet_OT

rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/otsu_probs.py smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import os
import argparse
import warnings
from glob import glob

import numpy as np
import rasterio as rio
from rasterio.features import shapes as rio_shapes
from skimage.filters import threshold_otsu
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd

from _method_common import (
    write_method_config, write_skipped_chips,
    SKIP_TOO_FEW_PROB_BANDS, SKIP_FLAT_PROB, SKIP_IC_BLOCK_FILTER,
)

warnings.filterwarnings("ignore", category=UserWarning)

ICEBERG_BAND  = 1      # 0-indexed: ocean=0, iceberg=1
MIN_AREA_M2   = 100.0
IC_THRESHOLD  = 0.15   # skip chip if >15% pixels exceed Otsu (sea-ice filter)


def main():
    parser = argparse.ArgumentParser(
        description="UNet+OT: Otsu threshold on UNet++ iceberg probability band"
    )
    parser.add_argument("--probs_dir",    required=True,
        help="Directory of *_probs.tif files from predict_tifs.py")
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--min_area_m2",  type=float, default=MIN_AREA_M2)
    parser.add_argument("--ic_threshold", type=float, default=IC_THRESHOLD,
        help="Skip chip if >this fraction exceeds Otsu (sea-ice filter)")
    args = parser.parse_args()

    gpkg_dir = os.path.join(args.out_dir, "gpkgs")
    os.makedirs(gpkg_dir, exist_ok=True)

    prob_files = sorted(glob(os.path.join(args.probs_dir, "*_probs.tif")))
    print(f"Found {len(prob_files)} prob .tifs")

    all_gdfs = []
    skipped  = []

    for i, prob_path in enumerate(prob_files):
        stem = os.path.basename(prob_path).replace("_probs.tif", "")

        with rio.open(prob_path) as src:
            probs = src.read().astype(np.float32)
            meta  = src.meta.copy()

        if probs.shape[0] <= ICEBERG_BAND:
            skipped.append({"chip_stem": stem, "reason": SKIP_TOO_FEW_PROB_BANDS,
                            "n_bands": probs.shape[0]})
            continue

        iceberg_prob = np.nan_to_num(probs[ICEBERG_BAND], nan=0.0)
        flat = iceberg_prob.flatten()

        # Otsu needs a bimodal distribution, so skip chips with flat prob
        if flat.max() - flat.min() < 0.01:
            print(f"  [{i+1:>4}/{len(prob_files)}] SKIP (flat prob)  {stem[:60]}")
            skipped.append({"chip_stem": stem, "reason": SKIP_FLAT_PROB,
                            "prob_range": f"{float(flat.max() - flat.min()):.4f}"})
            continue

        thresh  = float(threshold_otsu(flat))
        ic_frac = float((iceberg_prob >= thresh).mean())
        if ic_frac > args.ic_threshold:
            print(f"  [{i+1:>4}/{len(prob_files)}] SKIP (IC-filtered ic_frac={ic_frac:.2f})  {stem[:50]}")
            skipped.append({"chip_stem": stem, "reason": SKIP_IC_BLOCK_FILTER,
                            "otsu_thresh": f"{thresh:.4f}",
                            "ic_frac":     f"{ic_frac:.4f}"})
            continue

        mask = (iceberg_prob >= thresh).astype(np.uint8)

        records = []
        for geom_dict, val in rio_shapes(mask, transform=meta["transform"]):
            if val == 0:
                continue
            geom = shape(geom_dict)
            if geom.is_empty or geom.area < args.min_area_m2:
                continue
            records.append({"geometry": geom, "area_m2": geom.area,
                            "source_file": stem, "otsu_thresh": round(thresh, 4)})

        n = len(records)
        print(f"  [{i+1:>4}/{len(prob_files)}] {stem[:55]}  thr={thresh:.3f}  icebergs={n}")

        if records:
            gdf = gpd.GeoDataFrame(records, crs=meta["crs"])
            gdf.to_file(os.path.join(gpkg_dir, f"{stem}_icebergs.gpkg"), driver="GPKG")
            all_gdfs.append(gdf)

    cfg_path = write_method_config(
        args.out_dir, "UNet_OT",
        params={
            "probs_dir":    os.path.abspath(args.probs_dir),
            "iceberg_band": ICEBERG_BAND,
            "min_area_m2":  args.min_area_m2,
            "ic_threshold": args.ic_threshold,
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