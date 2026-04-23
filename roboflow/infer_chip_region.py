"""
Infer the AOI region prefix for a georeferenced chip using the Greenland AOI GeoPackage.

Example:
  python3 roboflow/infer_chip_region.py \
      --chip S2-iceberg-areas/S2UnetPlusPlus/imgs/example.tif \
      --aoi S2-iceberg-areas/aois_greenland_area_distributions.gpkg
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer AOI region prefix from chip location.")
    parser.add_argument("--chip", required=True, help="Path to georeferenced chip .tif")
    parser.add_argument("--aoi", required=True, help="Path to AOI GeoPackage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chip_path = Path(args.chip).expanduser().resolve()
    aoi_path = Path(args.aoi).expanduser().resolve()

    with rasterio.open(chip_path) as src:
        chip_bounds_5938 = transform_bounds(src.crs, "EPSG:5938", *src.bounds)

    cx = (chip_bounds_5938[0] + chip_bounds_5938[2]) / 2
    cy = (chip_bounds_5938[1] + chip_bounds_5938[3]) / 2

    conn = sqlite3.connect(aoi_path)
    rows = conn.execute(
        """
        SELECT a.REGION, r.minx, r.maxx, r.miny, r.maxy
        FROM rtree_aois_greenland_area_distributions_geom AS r
        JOIN aois_greenland_area_distributions AS a
          ON a.fid = r.id
        """
    ).fetchall()
    conn.close()

    matches = []
    for region, minx, maxx, miny, maxy in rows:
        if minx <= cx <= maxx and miny <= cy <= maxy:
            matches.append(region)

    print(f"chip_center_epsg5938=({cx:.3f}, {cy:.3f})")
    print("matching_regions=" + ",".join(matches))
    if matches:
        prefixes = []
        for region in matches:
            prefix = str(region).split("-")[0]
            if prefix not in prefixes:
                prefixes.append(prefix)
        print("region_prefixes=" + ",".join(prefixes))
    else:
        print("region_prefixes=")


if __name__ == "__main__":
    main()
