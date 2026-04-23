"""
otsu_threshold_tifs.py — Per-chip Otsu threshold on B08 for iceberg detection.

Computes Otsu threshold independently for each chip, applies floor/ceiling
guards, skips chips likely dominated by sea ice, and vectorizes iceberg pixels
to GeoPackage format matching predict_tifs.py and threshold_tifs.py outputs.

Usage:
  python otsu_threshold_tifs.py \\
      --chips_dir /mnt/research/.../chips/KQ/sza_lt65/tifs \\
      --out_dir   /mnt/research/.../area_comparison/KQ/sza_lt65/otsu

Output:
  out_dir/all_icebergs_otsu.gpkg   — merged iceberg polygons, all chips

Filtering logic:
  1. Otsu threshold computed on non-zero B08 pixels per chip.
  2. If threshold < otsu_floor (default 0.10): chip is too radiometrically flat,
     likely open ocean or cloud — skip.
  3. If threshold > otsu_ceil (default 0.50): sparse histogram produces unstable
     high thresholds — clip to otsu_ceil.
  4. If fraction of pixels above threshold > sea_ice_frac (default 0.15): chip
     is likely dominated by sea ice, not icebergs — skip.
  5. Polygons smaller than min_area_m2 (default 100 m2 = 10x10 m) are dropped.

Requires: rasterio, geopandas, shapely, scikit-image, numpy, pandas
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
from skimage.filters import threshold_otsu

warnings.filterwarnings("ignore", category=UserWarning)

# Defaults
OTSU_FLOOR    = 0.10   # chips with Otsu threshold below this are skipped (too flat)
OTSU_CEIL     = 0.50   # Otsu threshold is clipped to this maximum
SEA_ICE_FRAC  = 0.15   # skip chips where >15% pixels are above threshold
MIN_AREA_M2   = 100    # minimum polygon area to keep (m2)


def run_otsu(chips_dir, out_dir, b08_idx=2,
             otsu_floor=OTSU_FLOOR, otsu_ceil=OTSU_CEIL,
             sea_ice_frac=SEA_ICE_FRAC, min_area_m2=MIN_AREA_M2):
    """
    Apply per-chip Otsu threshold to B08 band and vectorize iceberg pixels.

    Parameters
    ----------
    chips_dir   : str  — directory containing .tif chip files
    out_dir     : str  — output directory for all_icebergs_otsu.gpkg
    b08_idx     : int  — 0-indexed band position of B08 (default 2 for B04/B03/B08)
    otsu_floor  : float — minimum acceptable Otsu threshold (default 0.10)
    otsu_ceil   : float — maximum Otsu threshold; clipped if exceeded (default 0.50)
    sea_ice_frac: float — skip chip if bright fraction exceeds this (default 0.15)
    min_area_m2 : float — minimum polygon area in m2 (default 100)
    """
    os.makedirs(out_dir, exist_ok=True)

    tif_files = sorted(glob(os.path.join(chips_dir, "*.tif")))
    if not tif_files:
        print(f"No .tif files found in {chips_dir}")
        return

    print(f"Found {len(tif_files)} chips  "
          f"otsu_floor={otsu_floor}  otsu_ceil={otsu_ceil}  "
          f"sea_ice_frac={sea_ice_frac}")

    all_gdfs  = []
    n_skipped = 0
    n_clipped = 0

    for tif_path in tif_files:
        chip_name = os.path.splitext(os.path.basename(tif_path))[0]

        with rio.open(tif_path) as src:
            chip      = src.read().astype(np.float32)   # (bands, H, W)
            meta      = src.meta.copy()
            transform = src.transform
            crs       = src.crs

        # 1. Extract B08
        if chip.shape[0] <= b08_idx:
            print(f"  SKIP {chip_name}: only {chip.shape[0]} bands, need b08_idx={b08_idx}")
            n_skipped += 1
            continue

        b08 = chip[b08_idx]

        # 2. Compute Otsu threshold on non-zero pixels
        nonzero = b08[b08 > 0]
        if len(nonzero) < 100:
            n_skipped += 1
            continue

        try:
            thresh = float(threshold_otsu(nonzero))
        except Exception:
            n_skipped += 1
            continue

        # 3. Floor guard: skip flat chips
        if thresh < otsu_floor:
            n_skipped += 1
            continue

        # 4. Ceiling guard: clip unstable high thresholds
        if thresh > otsu_ceil:
            thresh = otsu_ceil
            n_clipped += 1

        # 5. Sea-ice guard: skip chips where too many pixels are bright
        bright_frac = float(np.mean(b08 > thresh))
        if bright_frac > sea_ice_frac:
            n_skipped += 1
            continue

        # 6. Apply threshold
        iceberg_mask = (b08 >= thresh).astype(np.uint8)

        if iceberg_mask.sum() == 0:
            continue

        # 7. Vectorize
        pixel_area_m2 = abs(transform.a * transform.e)   # m2 per pixel (10x10 = 100)

        rows = []
        for geom_dict, val in rio_shapes(iceberg_mask, transform=transform):
            if val != 1:
                continue
            geom = shape(geom_dict)
            if geom.is_empty:
                continue
            area_m2 = geom.area
            if area_m2 < min_area_m2:
                continue
            rows.append({
                "geometry":   geom,
                "area_m2":    round(area_m2, 2),
                "class_name": "iceberg",
                "chip_name":  chip_name,
                "otsu_thresh": round(thresh, 4),
            })

        if not rows:
            continue

        gdf = gpd.GeoDataFrame(rows, crs=crs)
        all_gdfs.append(gdf)

    # 8. Merge and save
    print(f"  Skipped {n_skipped} chips (flat/sea-ice/error), "
          f"clipped {n_clipped} thresholds to {otsu_ceil}")

    if not all_gdfs:
        print("  No iceberg polygons found — all_icebergs_otsu.gpkg not written.")
        return

    merged = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True),
                               crs=all_gdfs[0].crs)
    out_path = os.path.join(out_dir, "all_icebergs_otsu.gpkg")
    merged.to_file(out_path, driver="GPKG")
    print(f"  Saved {len(merged)} polygons → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-chip Otsu B08 threshold for iceberg detection"
    )
    parser.add_argument("--chips_dir",     required=True,
                        help="Directory containing .tif chip files")
    parser.add_argument("--out_dir",       required=True,
                        help="Output directory for all_icebergs_otsu.gpkg")
    parser.add_argument("--b08_idx",       type=int,   default=2,
                        help="0-indexed band position of B08 (default: 2)")
    parser.add_argument("--otsu_floor",    type=float, default=OTSU_FLOOR,
                        help=f"Minimum Otsu threshold; skip chip if below (default: {OTSU_FLOOR})")
    parser.add_argument("--otsu_ceil",     type=float, default=OTSU_CEIL,
                        help=f"Maximum Otsu threshold; clip if exceeded (default: {OTSU_CEIL})")
    parser.add_argument("--sea_ice_frac",  type=float, default=SEA_ICE_FRAC,
                        help=f"Skip chip if bright fraction > this (default: {SEA_ICE_FRAC})")
    parser.add_argument("--min_area_m2",   type=float, default=MIN_AREA_M2,
                        help=f"Minimum polygon area in m2 (default: {MIN_AREA_M2})")
    args = parser.parse_args()

    run_otsu(
        chips_dir    = args.chips_dir,
        out_dir      = args.out_dir,
        b08_idx      = args.b08_idx,
        otsu_floor   = args.otsu_floor,
        otsu_ceil    = args.otsu_ceil,
        sea_ice_frac = args.sea_ice_frac,
        min_area_m2  = args.min_area_m2,
    )


if __name__ == "__main__":
    main()
