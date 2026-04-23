"""
threshold_tifs.py — Apply Fisser B08 ≥ 0.12 NIR threshold to S2 chip .tifs.

Mirrors the output format of predict_tifs.py so compare_areas.py can load both.

Usage:
  python threshold_tifs.py \\
      --chips_dir chips/KQ/sza_65_70/tifs \\
      --out_dir   georef_predictions/KQ/sza_65_70

Output:
  out_dir/all_icebergs_threshold.gpkg  — iceberg polygons with area_m2

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

warnings.filterwarnings("ignore")

THRESHOLD   = 0.12   # Fisser 2024 B08 NIR reflectance threshold
MIN_AREA_M2 = 100    # minimum polygon area in m² (~10×10 m)


def apply_threshold(chips_dir, out_dir, b08_idx=2, threshold=THRESHOLD, min_area_m2=MIN_AREA_M2):
    os.makedirs(out_dir, exist_ok=True)

    tif_files = sorted(glob(os.path.join(chips_dir, "*.tif")))
    if not tif_files:
        print(f"No .tif files found in {chips_dir}")
        return

    print(f"Found {len(tif_files)} chips  threshold={threshold}  b08_idx={b08_idx}")

    all_gdfs = []

    for i, tif_path in enumerate(tif_files):
        stem = os.path.splitext(os.path.basename(tif_path))[0]

        with rio.open(tif_path) as src:
            chip = src.read().astype(np.float32)  # (C, H, W)
            meta = src.meta.copy()

        if chip.shape[0] <= b08_idx:
            print(f"  [{i+1}/{len(tif_files)}] SKIP {stem} — only {chip.shape[0]} bands")
            continue

        b08          = chip[b08_idx]
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

    if not all_gdfs:
        print("\nNo icebergs detected across all chips.")
        return

    merged = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=all_gdfs[0].crs)
    merged["iceberg_id"] = range(1, len(merged) + 1)

    icebergs = merged[merged["class_name"] == "iceberg"]
    print(f"\n{'─'*50}")
    print(f"Total iceberg polygons : {len(icebergs)}")
    if len(icebergs) > 0:
        print(f"  min  = {icebergs['area_m2'].min():.1f} m²")
        print(f"  mean = {icebergs['area_m2'].mean():.1f} m²")
        print(f"  max  = {icebergs['area_m2'].max():.1f} m²")
    print(f"{'─'*50}")

    out_path = os.path.join(out_dir, "all_icebergs_threshold.gpkg")
    merged.to_file(out_path, driver="GPKG")
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply Fisser B08 ≥ 0.12 NIR threshold to S2 chip .tifs"
    )
    parser.add_argument("--chips_dir", required=True,
                        help="Directory of .tif chip files (same dir used by predict_tifs.py --imgs_dir)")
    parser.add_argument("--out_dir",   required=True,
                        help="Output directory — use the same path as predict_tifs.py --out_dir")
    parser.add_argument("--b08_idx",   type=int,   default=2,
                        help="0-indexed band position of B08 in chip stack (default: 2 for B04/B03/B08 order)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"NIR reflectance threshold (default: {THRESHOLD})")
    parser.add_argument("--min_area",  type=float, default=MIN_AREA_M2,
                        help=f"Min iceberg area in m² (default: {MIN_AREA_M2})")
    args = parser.parse_args()

    apply_threshold(args.chips_dir, args.out_dir, args.b08_idx, args.threshold, args.min_area)


if __name__ == "__main__":
    main()
