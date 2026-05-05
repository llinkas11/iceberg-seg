"""
densecrf_tifs.py: UNet + CRF method.

Applies DenseCRF post-processing to UNet++ softmax probabilities.
Uses the chip image for the bilateral pairwise term (boundary-preserving smoothing).
Core CRF logic reused from partner's crf_utils.py, do not duplicate.

Inputs:
  - *_probs.tif  : 2-band float32 softmax probs from predict_tifs.py (ocean, iceberg)
  - *tif chips   : original chips (for bilateral term)
Output:
  all_icebergs.gpkg

Requires crf_utils.py in the same directory (copied from partner's sandbox).

Usage:
  python densecrf_tifs.py \\
      --probs_dir area_comparison/KQ/sza_70_75/UNet/probs \\
      --chips_dir /mnt/research/.../chips/KQ/sza_70_75/tifs \\
      --out_dir   area_comparison/KQ/sza_70_75/UNet_CRF

rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/densecrf_tifs.py smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import os
import sys
import argparse
import warnings
from glob import glob

import numpy as np
import rasterio as rio
from rasterio.features import shapes as rio_shapes
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd

# crf_utils.py must be in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crf_utils import apply_densecrf
from _method_common import (
    write_method_config, write_skipped_chips, SKIP_CHIP_TIF_MISSING,
)

warnings.filterwarnings("ignore", category=UserWarning)

MIN_AREA_M2 = 100.0

# Default CRF params from partner's sandbox run_001 (best of 2 tested)
DEFAULT_PARAMS = {
    "sxy_gaussian":    3,
    "compat_gaussian": 3,
    "sxy_bilateral":   40,
    "srgb_bilateral":  3,
    "compat_bilateral":4,
    "iterations":      5,
}


def find_chip(chips_dir, stem):
    """Find chip .tif matching a prob stem (strip _probs suffix if present)."""
    chip_stem = stem.replace("_probs", "")
    path = os.path.join(chips_dir, f"{chip_stem}.tif")
    if os.path.exists(path):
        return path
    # fallback: glob
    matches = glob(os.path.join(chips_dir, f"{chip_stem}*.tif"))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(
        description="UNet+CRF: DenseCRF post-processing on UNet++ softmax probs"
    )
    parser.add_argument("--probs_dir",    required=True,
        help="Directory of *_probs.tif from predict_tifs.py")
    parser.add_argument("--chips_dir",    required=True,
        help="Directory of original chip .tifs (for CRF bilateral term)")
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--min_area_m2",  type=float, default=MIN_AREA_M2)
    # CRF params (override defaults if needed)
    parser.add_argument("--sxy_gaussian",    type=float, default=DEFAULT_PARAMS["sxy_gaussian"])
    parser.add_argument("--compat_gaussian", type=float, default=DEFAULT_PARAMS["compat_gaussian"])
    parser.add_argument("--sxy_bilateral",   type=float, default=DEFAULT_PARAMS["sxy_bilateral"])
    parser.add_argument("--srgb_bilateral",  type=float, default=DEFAULT_PARAMS["srgb_bilateral"])
    parser.add_argument("--compat_bilateral",type=float, default=DEFAULT_PARAMS["compat_bilateral"])
    parser.add_argument("--iterations",      type=int,   default=DEFAULT_PARAMS["iterations"])
    args = parser.parse_args()

    params = {
        "sxy_gaussian":     args.sxy_gaussian,
        "compat_gaussian":  args.compat_gaussian,
        "sxy_bilateral":    args.sxy_bilateral,
        "srgb_bilateral":   args.srgb_bilateral,
        "compat_bilateral": args.compat_bilateral,
        "iterations":       args.iterations,
    }

    gpkg_dir = os.path.join(args.out_dir, "gpkgs")
    os.makedirs(gpkg_dir, exist_ok=True)

    prob_files = sorted(glob(os.path.join(args.probs_dir, "*_probs.tif")))
    print(f"Found {len(prob_files)} prob .tifs")
    print(f"CRF params: {params}\n")

    all_gdfs = []
    skipped  = []

    for i, prob_path in enumerate(prob_files):
        stem = os.path.basename(prob_path).replace("_probs.tif", "")

        chip_path = find_chip(args.chips_dir, stem)
        if chip_path is None:
            print(f"  [{i+1:>4}/{len(prob_files)}] NO CHIP  {stem[:60]}")
            skipped.append({"chip_stem": stem, "reason": SKIP_CHIP_TIF_MISSING})
            continue

        with rio.open(prob_path) as src:
            probs = src.read().astype(np.float32)   # (3, H, W)
            meta  = src.meta.copy()

        with rio.open(chip_path) as src:
            chip = src.read().astype(np.float32)    # (3, H, W)

        refined = apply_densecrf(probs, chip, params)  # (H, W) uint8

        mask = (refined == 1).astype(np.uint8)   # iceberg class

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
        args.out_dir, "UNet_CRF",
        params={
            "probs_dir":   os.path.abspath(args.probs_dir),
            "chips_dir":   os.path.abspath(args.chips_dir),
            "min_area_m2": args.min_area_m2,
            "crf":         params,
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
        print(f"\nTotal icebergs  : {len(merged)}")
        print(f"Saved           : {out}")
    else:
        print("\nNo icebergs detected.")

    n_no_chip = sum(1 for r in skipped if r["reason"] == SKIP_CHIP_TIF_MISSING)
    if n_no_chip:
        print(f"Skipped (no chip): {n_no_chip}")
    print(f"Method config    : {cfg_path}")
    print(f"Skipped chips    : {skip_path}")


if __name__ == "__main__":
    main()