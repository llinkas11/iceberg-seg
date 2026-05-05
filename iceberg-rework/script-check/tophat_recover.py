"""
tophat_recover.py: small-iceberg recovery via white top-hat on B08.

For each input chip in --chips_dir, reads the B08 band, applies a white
top-hat morphological filter (disk structuring element of radius
--se_radius pixels), thresholds the response at --th_thresh, drops
connected components below --min_area_px, subtracts pixels already
covered by the base method's per-chip prediction, and writes a per-chip
gpkg containing the union of base polygons + recovered polygons.

Outputs (per --out_dir):
  gpkgs/<chip_stem>_icebergs.gpkg   per-chip merged polygons (base + TH)
  all_icebergs.gpkg                 concat across chips
  recovery_stats.csv                per-chip counts of base / TH / total
  method_config.json                provenance: SE radius, threshold, base method id

Usage:
  python scripts/tophat_recover.py \
      --chips_dir   data/v4_clean/test_chips/sza_lt65 \
      --base_dir    runs/exp_baseline_v1/<ts>/inference/sza_lt65/UNet \
      --out_dir     runs/exp_baseline_v1/<ts>/inference/sza_lt65/UNet_TH

Notes:
- Designed as a post-processor; does not need a UNet checkpoint.
- Input chip and prediction tifs must share pixel grid (256x256, 10 m).
- Idempotent at the same parameters: rerunning overwrites outputs.
"""

import argparse
import csv
import json
import os
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.features import rasterize as rio_rasterize, shapes as rio_shapes
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape as shapely_shape
from skimage.morphology import disk, white_tophat
from skimage.measure import label

from _method_common import get_git_sha, sha256_of_file

# 1. Defaults
DEFAULT_SE_RADIUS = 10       # 100 m at 10 m pixels (Fisser cap on small icebergs)
DEFAULT_TH_THRESH = 0.05     # response threshold in reflectance units
DEFAULT_MIN_AREA_PX = 16     # 40 m root length, matches the global cutoff
PIXEL_AREA_M2 = 100.0


def read_b08(tif_path):
    """Return the B08 band (band 3) as a float32 array."""
    with rasterio.open(tif_path) as src:
        if src.count < 3:
            raise ValueError(f"{tif_path}: expected >= 3 bands, got {src.count}")
        b08 = src.read(3).astype(np.float32)
        transform = src.transform
        crs = src.crs
    return b08, transform, crs


def read_base_polygons(base_dir, stem, all_polys_cache=None):
    """
    Load base method polygons for one chip. Three fallbacks in order:
      1. per-chip gpkgs/<stem>_icebergs.gpkg
      2. all_icebergs.gpkg sliced by source_file == <stem>.tif
      3. None
    """
    p = os.path.join(base_dir, "gpkgs", f"{stem}_icebergs.gpkg")
    if os.path.exists(p):
        return gpd.read_file(p)

    if all_polys_cache is None:
        all_p = os.path.join(base_dir, "all_icebergs.gpkg")
        if not os.path.exists(all_p):
            return None
        all_polys_cache = gpd.read_file(all_p)

    src_name = f"{stem}.tif"
    sub = all_polys_cache[all_polys_cache.get("source_file") == src_name]
    return sub if len(sub) > 0 else None


def read_base_mask(base_dir, stem, ref_shape, ref_transform, all_polys_cache=None):
    """
    Build a binary mask of the base method's polygon footprint on the chip's
    pixel grid. Prefers <stem>_pred.tif under geotiffs/, falls back to
    rasterising the polygons read by read_base_polygons.
    Returns a uint8 array shaped like ref_shape, or None when no base
    information exists for this chip.
    """
    # 1. Direct geotiff path (UNet only writes this; others do not)
    pred_tif = os.path.join(base_dir, "geotiffs", f"{stem}_pred.tif")
    if os.path.exists(pred_tif):
        with rasterio.open(pred_tif) as src:
            return (src.read(1) > 0).astype(np.uint8)

    # 2. Rasterise polygons (per-chip gpkg or all_icebergs slice)
    polys = read_base_polygons(base_dir, stem, all_polys_cache=all_polys_cache)
    if polys is None or len(polys) == 0:
        return None
    geoms = [(g, 1) for g in polys.geometry if g is not None and not g.is_empty]
    if not geoms:
        return None
    return rio_rasterize(
        geoms,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0,
        dtype="uint8",
    )


def recover_tophat(b08, base_mask, se_radius, th_thresh, min_area_px):
    """
    Run the white top-hat recovery step.
    Returns a binary mask of NEW iceberg pixels (not already in base_mask).
    """
    # 1. White top-hat highlights bright spots smaller than the SE
    se = disk(se_radius)
    response = white_tophat(b08, se)

    # 2. Threshold + drop pixels already covered by the base method
    candidate = (response >= th_thresh).astype(np.uint8)
    if base_mask is not None:
        candidate &= (base_mask == 0).astype(np.uint8)

    # 3. Filter connected components below the size cutoff
    labels, n = label(candidate, connectivity=2, return_num=True)
    if n == 0:
        return np.zeros_like(candidate)
    sizes = np.bincount(labels.ravel())
    keep = np.zeros_like(sizes, dtype=bool)
    keep[1:] = sizes[1:] >= min_area_px
    return keep[labels].astype(np.uint8)


def mask_to_polygons(mask, transform, source_file):
    """Vectorise a binary mask into shapely polygons; returns list of records."""
    if mask.sum() == 0:
        return []
    rows = []
    iceberg_id = 1
    for geom, val in rio_shapes(mask, mask=mask.astype(bool), transform=transform):
        if val == 0:
            continue
        poly = shapely_shape(geom)
        rows.append({
            "class_id":    1,
            "class_name":  "iceberg",
            "area_m2":     poly.area,
            "source_file": source_file,
            "iceberg_id":  iceberg_id,
            "geometry":    poly,
        })
        iceberg_id += 1
    return rows


def main():
    parser = argparse.ArgumentParser(description="White top-hat small-iceberg recovery on a base method's outputs")
    parser.add_argument("--chips_dir", required=True, help="dir of input *.tif (B04/B03/B08)")
    parser.add_argument("--base_dir",  required=True, help="dir of base method outputs (must have geotiffs/ + gpkgs/)")
    parser.add_argument("--out_dir",   required=True, help="dir to write the recovered method outputs")
    parser.add_argument("--base_method_id", default="UNet", help="label for the base method (provenance only)")
    parser.add_argument("--se_radius", type=int, default=DEFAULT_SE_RADIUS,
                        help="disk SE radius in pixels (default 10 = 100 m)")
    parser.add_argument("--th_thresh", type=float, default=DEFAULT_TH_THRESH,
                        help="top-hat response threshold in reflectance units")
    parser.add_argument("--min_area_px", type=int, default=DEFAULT_MIN_AREA_PX,
                        help="drop recovered components below this pixel area")
    args = parser.parse_args()

    # 1. Stage output directories
    os.makedirs(os.path.join(args.out_dir, "gpkgs"), exist_ok=True)

    # 2. Iterate chips
    chip_paths = sorted(
        os.path.join(args.chips_dir, f)
        for f in os.listdir(args.chips_dir) if f.endswith(".tif")
    )
    if not chip_paths:
        raise SystemExit(f"no tifs found under {args.chips_dir}")

    stats_rows = []
    all_polys = []

    # Cache the cross-chip all_icebergs.gpkg once: TR/OT only emit this file,
    # so re-reading it per chip would dominate runtime.
    all_polys_cache = None
    all_icebergs_path = os.path.join(args.base_dir, "all_icebergs.gpkg")
    if os.path.exists(all_icebergs_path):
        all_polys_cache = gpd.read_file(all_icebergs_path)

    for chip_path in chip_paths:
        stem = os.path.splitext(os.path.basename(chip_path))[0]
        b08, transform, _crs = read_b08(chip_path)
        base_mask = read_base_mask(
            args.base_dir, stem, b08.shape, transform, all_polys_cache,
        )

        # 3. Run recovery on this chip
        recovered = recover_tophat(
            b08=b08,
            base_mask=base_mask,
            se_radius=args.se_radius,
            th_thresh=args.th_thresh,
            min_area_px=args.min_area_px,
        )

        recovered_polys = mask_to_polygons(recovered, transform,
                                            source_file=os.path.basename(chip_path))

        # 4. Merge with base polygons
        base_gdf = read_base_polygons(args.base_dir, stem, all_polys_cache)
        n_base = len(base_gdf) if base_gdf is not None else 0
        n_recov = len(recovered_polys)

        if base_gdf is not None and n_base > 0:
            recov_gdf = gpd.GeoDataFrame(recovered_polys, crs=base_gdf.crs) if recovered_polys else None
            merged = pd.concat([base_gdf, recov_gdf], ignore_index=True) if recov_gdf is not None else base_gdf
        elif recovered_polys:
            merged = gpd.GeoDataFrame(recovered_polys)
        else:
            merged = gpd.GeoDataFrame(columns=["class_id", "class_name", "area_m2",
                                                "source_file", "iceberg_id", "geometry"])

        # Re-id sequentially after the merge
        if len(merged) > 0:
            merged["iceberg_id"] = list(range(1, len(merged) + 1))

        out_gpkg = os.path.join(args.out_dir, "gpkgs", f"{stem}_icebergs.gpkg")
        if len(merged) > 0:
            merged.to_file(out_gpkg, driver="GPKG")
        all_polys.append(merged)

        stats_rows.append({
            "chip_stem":        stem,
            "n_base_polygons":  n_base,
            "n_recovered":      n_recov,
            "n_total":          n_base + n_recov,
        })

    # 5. Concatenated gpkg + stats CSV
    # Fisser synthetic chips have CRS = None, real Roboflow chips span UTM 24N
    # and 25N, so a single CRS for the cross-chip concat does not exist. The
    # eval script reads per-chip gpkgs, so the combined file is informational
    # only; warn but continue if the concat fails.
    try:
        combined = pd.concat(all_polys, ignore_index=True) if all_polys else pd.DataFrame()
        if len(combined) > 0:
            combined.to_file(os.path.join(args.out_dir, "all_icebergs.gpkg"), driver="GPKG")
    except Exception as exc:
        print(f"WARN: cross-chip all_icebergs.gpkg skipped ({exc.__class__.__name__}: {exc})")

    stats_path = os.path.join(args.out_dir, "recovery_stats.csv")
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["chip_stem", "n_base_polygons", "n_recovered", "n_total"]
        )
        writer.writeheader()
        writer.writerows(stats_rows)

    # 6. Method config (provenance)
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = {
        "method":           f"{args.base_method_id}_TH",
        "base_method":      args.base_method_id,
        "base_dir":         os.path.abspath(args.base_dir),
        "chips_dir":        os.path.abspath(args.chips_dir),
        "se_radius":        args.se_radius,
        "th_thresh":        args.th_thresh,
        "min_area_px":      args.min_area_px,
        "n_chips_processed": len(chip_paths),
        "n_polygons_base":  int(sum(r["n_base_polygons"] for r in stats_rows)),
        "n_polygons_recov": int(sum(r["n_recovered"] for r in stats_rows)),
        "n_polygons_total": int(sum(r["n_total"] for r in stats_rows)),
        "git_sha":          get_git_sha(repo_dir),
        "created_utc":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(args.out_dir, "method_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"top-hat recovery written to {args.out_dir}/")
    print(f"  base polygons:      {cfg['n_polygons_base']}")
    print(f"  recovered polygons: {cfg['n_polygons_recov']}")
    print(f"  total:              {cfg['n_polygons_total']}")


if __name__ == "__main__":
    main()
