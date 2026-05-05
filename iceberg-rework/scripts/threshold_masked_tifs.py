"""
threshold_masked_tifs.py — Apply Fisser B08 ≥ 0.12 NIR threshold restricted to
open-water pixels identified by an NDWI water mask.

NDWI = (B03 - B08) / (B03 + B08 + ε)
Pixels where NDWI > ndwi_threshold are classified as open water.
B08 ≥ 0.12 is then applied ONLY within open-water pixels.

This prevents bright sea ice, clouds, and snow from being counted as icebergs,
giving a fairer comparison to UNet++ which learned to distinguish these classes.

Chip band order (set by chip_sentinel2.py): B04=0, B03=1, B08=2

Usage:
  python threshold_masked_tifs.py \\
      --chips_dir chips/KQ/sza_gt75/tifs \\
      --out_dir   area_comparison/KQ/sza_gt75/threshold_masked

Output:
  out_dir/all_icebergs_threshold_masked.gpkg
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

NIR_THRESHOLD  = 0.22   # Fisser 2024 B08 threshold (0.12) + 0.10 DN offset correction
                        # All scenes baseline ≥4.0: chip_sentinel2.py does not subtract +1000 DN offset
NDWI_THRESHOLD = 0.0    # NDWI > 0 → open water (negative = ice/land/cloud)
MIN_AREA_M2    = 100    # ~10×10 m minimum polygon
IC_THRESHOLD   = 0.15   # Fisser 2025 IC block filter (eq. 2): skip chip if >15% of pixels exceed NIR threshold
                        # Flags chips dominated by sea ice rather than open water with icebergs
                        # Original Fisser 2025 rule: 10 km block; we apply at 2.56 km chip level (pre-tiled chips)


def _apply_thresholds(
    b03,
    b08,
    transform,
    source_name,
    nir_threshold=NIR_THRESHOLD,
    ndwi_threshold=NDWI_THRESHOLD,
    min_area_m2=MIN_AREA_M2,
    ic_threshold=IC_THRESHOLD,
):
    """Apply IC + NDWI + B08 thresholds to band arrays.

    Inputs:
        b03, b08    : 2-D float reflectance arrays.
        transform   : rasterio affine for polygonization.
        source_name : basename string stored on each polygon record.
    Returns dict: ic_skipped, records, n_polygons, total_area_m2,
                  water_px, iceberg_px, ic_frac.
    """
    out = {
        "ic_skipped"   : False,
        "records"      : [],
        "n_polygons"   : 0,
        "total_area_m2": 0.0,
        "water_px"     : 0,
        "iceberg_px"   : 0,
        "ic_frac"      : float("nan"),
    }

    # 1. IC block filter (Fisser 2025): skip sea-ice-dominated chips
    ic_frac = float((b08 >= nir_threshold).mean())
    out["ic_frac"] = ic_frac
    if ic_frac > ic_threshold:
        out["ic_skipped"] = True
        return out

    # 2. NDWI water mask: open water has positive NDWI
    ndwi       = (b03 - b08) / (b03 + b08 + 1e-6)
    water_mask = (ndwi > ndwi_threshold).astype(np.uint8)
    out["water_px"] = int(water_mask.sum())

    # 3. Iceberg mask: bright NIR within open-water pixels
    iceberg_mask = ((b08 >= nir_threshold) & (water_mask == 1)).astype(np.uint8)
    out["iceberg_px"] = int(iceberg_mask.sum())

    # 4. Polygonize and filter by min-area
    records = []
    for geom_dict, val in rio_shapes(iceberg_mask, transform=transform):
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
            "source_file": source_name,
        })

    out["records"]       = records
    out["n_polygons"]    = len(records)
    out["total_area_m2"] = float(sum(r["area_m2"] for r in records))
    return out


def detect_icebergs_in_chip(
    tif_path,
    b03_idx=1,
    b08_idx=2,
    nir_threshold=NIR_THRESHOLD,
    ndwi_threshold=NDWI_THRESHOLD,
    min_area_m2=MIN_AREA_M2,
    ic_threshold=IC_THRESHOLD,
):
    """Run NDWI-masked Fisser threshold on one chip read from disk.

    Inputs:
        tif_path         : path to a Sentinel-2 chip GeoTIFF.
        b03_idx, b08_idx : 0-indexed band positions.
    Returns dict: band_skipped, n_bands, crs, plus all keys from
                  _apply_thresholds (ic_skipped, records, n_polygons,
                  total_area_m2, water_px, iceberg_px, ic_frac).
    """
    out = {
        "band_skipped" : False,
        "n_bands"      : 0,
        "crs"          : None,
        "ic_skipped"   : False,
        "records"      : [],
        "n_polygons"   : 0,
        "total_area_m2": 0.0,
        "water_px"     : 0,
        "iceberg_px"   : 0,
        "ic_frac"      : float("nan"),
    }

    # 1. Read chip
    with rio.open(tif_path) as src:
        chip = src.read().astype(np.float32)
        meta = src.meta.copy()
    out["crs"]     = meta["crs"]
    out["n_bands"] = chip.shape[0]

    # 2. Validate band count
    if out["n_bands"] <= max(b03_idx, b08_idx):
        out["band_skipped"] = True
        return out

    # 3. Apply thresholds on band arrays
    inner = _apply_thresholds(
        chip[b03_idx],
        chip[b08_idx],
        meta["transform"],
        os.path.basename(tif_path),
        nir_threshold  = nir_threshold,
        ndwi_threshold = ndwi_threshold,
        min_area_m2    = min_area_m2,
        ic_threshold   = ic_threshold,
    )
    out.update(inner)
    return out


def apply_masked_threshold(
    chips_dir,
    out_dir,
    b03_idx=1,
    b08_idx=2,
    nir_threshold=NIR_THRESHOLD,
    ndwi_threshold=NDWI_THRESHOLD,
    min_area_m2=MIN_AREA_M2,
    ic_threshold=IC_THRESHOLD,
):
    os.makedirs(out_dir, exist_ok=True)

    tif_files = sorted(glob(os.path.join(chips_dir, "*.tif")))
    if not tif_files:
        print(f"No .tif files found in {chips_dir}")
        return

    print(f"Found {len(tif_files)} chips")
    print(f"  NIR threshold : B08 >= {nir_threshold}")
    print(f"  NDWI mask     : NDWI > {ndwi_threshold} (open water only)")
    print(f"  IC filter     : skip chip if bright-pixel fraction > {ic_threshold}")
    print(f"  Min area      : {min_area_m2} m²\n")

    all_gdfs  = []
    n_skipped = 0
    n_ic      = 0

    for i, tif_path in enumerate(tif_files):
        stem = os.path.splitext(os.path.basename(tif_path))[0]

        # 1. Detect icebergs on this chip
        res = detect_icebergs_in_chip(
            tif_path,
            b03_idx        = b03_idx,
            b08_idx        = b08_idx,
            nir_threshold  = nir_threshold,
            ndwi_threshold = ndwi_threshold,
            min_area_m2    = min_area_m2,
            ic_threshold   = ic_threshold,
        )

        # 2. Log skipped chips
        if res["band_skipped"]:
            print(f"  [{i+1}/{len(tif_files)}] SKIP {stem}: only {res['n_bands']} band(s)")
            n_skipped += 1
            continue

        if res["ic_skipped"]:
            print(
                f"  [{i+1:>4}/{len(tif_files)}] IC   {stem[:55]}  ic_frac={res['ic_frac']:.2f}"
            )
            n_ic += 1
            continue

        # 3. Log per-chip detection counts
        print(
            f"  [{i+1:>4}/{len(tif_files)}] {stem[:55]}  "
            f"water_px={res['water_px']:>6}  icebergs={res['n_polygons']}"
        )

        # 4. Accumulate polygon records into a GeoDataFrame
        if res["records"]:
            gdf = gpd.GeoDataFrame(res["records"], crs=res["crs"])
            all_gdfs.append(gdf)

    if not all_gdfs:
        print("\nNo icebergs detected across all chips.")
        if n_ic:
            print(f"IC-filtered: {n_ic} chips (sea ice contamination)")
        if n_skipped:
            print(f"Skipped:     {n_skipped} chips (too few bands)")
        return

    target_crs = all_gdfs[0].crs
    reprojected = [gdf.to_crs(target_crs) if gdf.crs != target_crs else gdf for gdf in all_gdfs]
    merged = gpd.GeoDataFrame(pd.concat(reprojected, ignore_index=True), crs=target_crs)
    merged["iceberg_id"] = range(1, len(merged) + 1)

    icebergs = merged[merged["class_name"] == "iceberg"]
    print(f"\n{'─'*50}")
    print(f"Total iceberg polygons : {len(icebergs)}")
    if len(icebergs) > 0:
        print(f"  min    = {icebergs['area_m2'].min():.1f} m²")
        print(f"  median = {icebergs['area_m2'].median():.1f} m²")
        print(f"  mean   = {icebergs['area_m2'].mean():.1f} m²")
        print(f"  max    = {icebergs['area_m2'].max():.1f} m²")
        print(f"  total  = {icebergs['area_m2'].sum()/1e6:.4f} km²")
    if n_ic:
        print(f"IC-filtered: {n_ic} chips (sea ice contamination)")
    if n_skipped:
        print(f"Skipped:     {n_skipped} chips (too few bands)")
    print(f"{'─'*50}")

    out_path = os.path.join(out_dir, "all_icebergs_threshold_masked.gpkg")
    merged.to_file(out_path, driver="GPKG")
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply NDWI-masked Fisser B08 threshold to S2 chip .tifs"
    )
    parser.add_argument("--chips_dir",      required=True,
                        help="Directory of .tif chip files")
    parser.add_argument("--out_dir",        required=True,
                        help="Output directory for all_icebergs_threshold_masked.gpkg")
    parser.add_argument("--b03_idx",        type=int,   default=1,
                        help="0-indexed band position of B03 in chip stack (default: 1)")
    parser.add_argument("--b08_idx",        type=int,   default=2,
                        help="0-indexed band position of B08 in chip stack (default: 2)")
    parser.add_argument("--threshold",      type=float, default=NIR_THRESHOLD,
                        help=f"NIR reflectance threshold (default: {NIR_THRESHOLD})")
    parser.add_argument("--ndwi_threshold", type=float, default=NDWI_THRESHOLD,
                        help=f"NDWI cutoff for open-water mask (default: {NDWI_THRESHOLD})")
    parser.add_argument("--min_area",       type=float, default=MIN_AREA_M2,
                        help=f"Min polygon area in m² (default: {MIN_AREA_M2})")
    parser.add_argument("--ic_threshold",   type=float, default=IC_THRESHOLD,
                        help=f"IC block filter: skip chip if bright-pixel fraction exceeds this (default: {IC_THRESHOLD})")
    args = parser.parse_args()

    apply_masked_threshold(
        chips_dir      = args.chips_dir,
        out_dir        = args.out_dir,
        b03_idx        = args.b03_idx,
        b08_idx        = args.b08_idx,
        nir_threshold  = args.threshold,
        ndwi_threshold = args.ndwi_threshold,
        min_area_m2    = args.min_area,
        ic_threshold   = args.ic_threshold,
    )


if __name__ == "__main__":
    main()