"""
extract_chip_sza.py

Emit per-chip solar zenith angle (degrees) for every chip used in
area_comparison. Primary source is the Sentinel-2 L1C tile metadata
(GRANULE/*/MTD_TL.xml, Sun_Angles_Grid), bilinearly sampled at the chip
centroid. Fallback is pysolar from the scene acquisition time and chip
centroid lat/lon. Both are written so downstream consumers can QA.

Output:
  {out_csv}     default /mnt/research/.../area_comparison/chip_sza.csv

Columns:
  region, sza_bin, chip_stem, scene_id, acq_datetime_utc,
  lat, lon, sza_deg_mtd, sza_deg_pysolar, sza_deg, source

sza_deg_mtd       : bilinear interp of the 22x22 Sun_Angles_Grid at chip centroid.
sza_deg_pysolar   : 90 - pysolar.get_altitude(lat, lon, acq_datetime_utc).
sza_deg / source  : MTD value when available, else pysolar, else NaN.

Usage:
  python extract_chip_sza.py \\
      --chips_root /mnt/research/.../S2-iceberg-areas/chips \\
      --safe_root  /mnt/research/v.gomezgilyaspik/students/smishra/sentinel2_downloads \\
      --out_csv    /mnt/research/.../S2-iceberg-areas/area_comparison/chip_sza.csv

Rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/extract_chip_sza.py \\
      smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import os
import re
import glob
import argparse
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

DEFAULT_CHIPS_ROOT = "/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/chips"
DEFAULT_SAFE_ROOT  = "/mnt/research/v.gomezgilyaspik/students/smishra/sentinel2_downloads"
DEFAULT_OUT_CSV    = "/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/area_comparison/chip_sza.csv"

SCENE_RE = re.compile(
    r"^(?P<scene>S2[AB]_MSIL1C_(?P<datestr>\d{8}T\d{6})_N\d{4}_R\d{3}_T\w{5}_\d{8}T\d{6})"
    r"_r(?P<row>\d+)_c(?P<col>\d+)$"
)


# 1. Chip-name parsing -------------------------------------------------------

def parse_chip_stem(chip_stem):
    """
    chip_stem example:
      S2A_MSIL1C_20210906T135031_N0500_R110_T25WDQ_20230117T021638_r0000_c10240

    Returns (scene_id, acq_datetime_utc, tile_row_start_px, tile_col_start_px)
    where the pixel offsets are the top-left of the 256x256 chip inside the
    full 10980x10980 tile. Returns None if unparseable.
    """
    m = SCENE_RE.match(chip_stem)
    if not m:
        return None
    scene_id = m.group("scene")
    try:
        dt = datetime.strptime(m.group("datestr"), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return scene_id, dt, int(m.group("row")), int(m.group("col"))


# 2. MTD_TL.xml parsing ------------------------------------------------------

def find_mtd_tl(safe_root, scene_id):
    """Locate GRANULE/*/MTD_TL.xml for a scene_id under safe_root."""
    # Tolerate both unzipped .SAFE dirs and sibling .SAFE inside region folders.
    patterns = [
        os.path.join(safe_root, "**", f"{scene_id}.SAFE", "GRANULE", "*", "MTD_TL.xml"),
        os.path.join(safe_root, f"{scene_id}.SAFE", "GRANULE", "*", "MTD_TL.xml"),
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            return matches[0]
    return None


def parse_sun_zenith_grid(mtd_path):
    """
    Read the 22x22 Sun_Angles_Grid/Zenith Values_List from MTD_TL.xml.
    Returns (zenith_grid_2d_deg, mean_zenith_deg, col_step_m, row_step_m)
    or (None, None, None, None) on parse failure.

    The S2 L1C 10m tile is 10980x10980 pixels. The Sun_Angles_Grid is
    sampled every 5000 m => 23x23 entries covering 0..10980 m extent, but
    S2 delivers 22x22 rows/cols of 5km step (indices 0..21 => 0..105 km).
    COL_STEP / ROW_STEP in the XML give the exact step size in meters.
    """
    try:
        tree = ET.parse(mtd_path)
    except ET.ParseError:
        return None, None, None, None
    root = tree.getroot()

    # Strip XML namespaces for simpler XPath-like traversal.
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    mean_zen = None
    mean_el = root.find(".//Tile_Angles/Mean_Sun_Angle/ZENITH_ANGLE")
    if mean_el is not None and mean_el.text:
        try:
            mean_zen = float(mean_el.text)
        except ValueError:
            mean_zen = None

    sun_zen = root.find(".//Tile_Angles/Sun_Angles_Grid/Zenith")
    if sun_zen is None:
        return None, mean_zen, None, None

    col_step_el = sun_zen.find("COL_STEP")
    row_step_el = sun_zen.find("ROW_STEP")
    col_step_m  = float(col_step_el.text) if col_step_el is not None else 5000.0
    row_step_m  = float(row_step_el.text) if row_step_el is not None else 5000.0

    values_el = sun_zen.find("Values_List")
    if values_el is None:
        return None, mean_zen, col_step_m, row_step_m
    rows = []
    for v in values_el.findall("VALUES"):
        if v.text is None:
            continue
        rows.append([float(x) if x not in ("", "NaN") else np.nan
                     for x in v.text.strip().split()])
    if not rows:
        return None, mean_zen, col_step_m, row_step_m
    grid = np.array(rows, dtype=float)
    return grid, mean_zen, col_step_m, row_step_m


# 3. Bilinear interp of sun-angle grid ---------------------------------------

def interp_sza_from_grid(grid, col_step_m, row_step_m,
                         chip_x_m, chip_y_m, tile_origin_x, tile_origin_y):
    """
    Bilinear interpolation of the sun-angle grid at chip centroid.

    grid        : (Ny, Nx) array of zenith (deg) aligned to tile origin.
    col_step_m  : horizontal grid spacing (m)
    row_step_m  : vertical grid spacing (m)
    chip_x_m    : chip centroid easting (same CRS as tile)
    chip_y_m    : chip centroid northing
    tile_origin_x, tile_origin_y : top-left corner of tile in the UTM CRS

    Returns SZA in degrees, or NaN if the chip falls outside the grid.
    """
    if grid is None:
        return np.nan
    # Local coords inside the tile, in meters (x increases east, y decreases south).
    dx = chip_x_m - tile_origin_x                   # easting offset from origin
    dy = tile_origin_y - chip_y_m                   # southing offset (row grows down)
    fx = dx / col_step_m
    fy = dy / row_step_m
    ny, nx = grid.shape
    if fx < 0 or fy < 0 or fx > (nx - 1) or fy > (ny - 1):
        return np.nan
    ix = int(np.floor(fx)); iy = int(np.floor(fy))
    tx = fx - ix;           ty = fy - iy
    ix2 = min(ix + 1, nx - 1); iy2 = min(iy + 1, ny - 1)
    v00 = grid[iy,  ix ]; v10 = grid[iy,  ix2]
    v01 = grid[iy2, ix ]; v11 = grid[iy2, ix2]
    top    = v00 * (1 - tx) + v10 * tx
    bottom = v01 * (1 - tx) + v11 * tx
    return float(top * (1 - ty) + bottom * ty)


# 4. pysolar fallback --------------------------------------------------------

def pysolar_sza(lat, lon, dt_utc):
    """90 - altitude(deg). Returns NaN on import or compute failure."""
    try:
        from pysolar.solar import get_altitude
    except Exception:
        return np.nan
    try:
        alt = get_altitude(lat, lon, dt_utc)
        return float(90.0 - alt)
    except Exception:
        return np.nan


# 5. Per-chip processing -----------------------------------------------------

def chip_centroid_latlon(tif_path):
    """Return (lat, lon, easting, northing, crs, tile_origin_x, tile_origin_y)."""
    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        cx, cy = src.transform * (w / 2.0, h / 2.0)
        crs    = src.crs
        # Tile origin depends on the chip's own affine; the top-left of THIS
        # chip is transform * (0, 0). For bilinear we pass that as the local
        # origin; grid is tile-wide, but since we only need chip centroid
        # inside the tile grid, and the chip CRS == tile CRS, offsets are
        # self-consistent when the MTD grid is anchored at the tile corner.
        # We don't know the full tile corner from a chip alone, so we store
        # the chip top-left here; grid interp adapts.
        origin_x, origin_y = src.transform * (0.0, 0.0)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(cx, cy)
    return lat, lon, cx, cy, crs, origin_x, origin_y


def tile_origin_from_mtd(mtd_path, crs):
    """
    Extract the tile top-left easting/northing from Tile_Geocoding in MTD_TL.
    Needed because the Sun_Angles_Grid is indexed from the tile corner, not
    from an individual chip corner.
    """
    try:
        tree = ET.parse(mtd_path); root = tree.getroot()
    except ET.ParseError:
        return None, None
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    for geoposition in root.findall(".//Tile_Geocoding/Geoposition"):
        if geoposition.get("resolution") == "10":
            ulx = geoposition.find("ULX"); uly = geoposition.find("ULY")
            if ulx is not None and uly is not None:
                return float(ulx.text), float(uly.text)
    return None, None


# 6. Driver ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-chip SZA from MTD_TL + pysolar.")
    parser.add_argument("--chips_root", default=DEFAULT_CHIPS_ROOT,
                        help="Root containing {REGION}/{sza_bin}/tifs/*.tif")
    parser.add_argument("--safe_root",  default=DEFAULT_SAFE_ROOT,
                        help="Root containing downloaded .SAFE folders.")
    parser.add_argument("--out_csv",    default=DEFAULT_OUT_CSV,
                        help="Output CSV path.")
    parser.add_argument("--region",     default=None, help="kq | sk")
    parser.add_argument("--sza_bin",    default=None)
    args = parser.parse_args()

    records = []
    missed_mtd = set()

    # 6a. Walk chips/{REGION}/{sza_bin}/tifs/*.tif
    chips_root = Path(args.chips_root)
    for region_dir in sorted(chips_root.iterdir()):
        if not region_dir.is_dir():
            continue
        region = region_dir.name.lower()
        if args.region and region != args.region.lower():
            continue
        for sza_dir in sorted(region_dir.iterdir()):
            if not sza_dir.is_dir():
                continue
            sza_bin = sza_dir.name
            if args.sza_bin and sza_bin != args.sza_bin:
                continue
            tif_dir = sza_dir / "tifs"
            if not tif_dir.is_dir():
                continue
            for tif_path in sorted(tif_dir.glob("*.tif")):
                chip_stem = tif_path.stem
                parsed = parse_chip_stem(chip_stem)
                if parsed is None:
                    continue
                scene_id, dt_utc, _r, _c = parsed
                lat, lon, cx, cy, crs, _ox, _oy = chip_centroid_latlon(tif_path)

                # 6b. MTD-based SZA via bilinear interp on the 22x22 grid.
                sza_mtd = np.nan
                mtd_path = find_mtd_tl(args.safe_root, scene_id)
                if mtd_path:
                    grid, _mean_zen, col_step, row_step = parse_sun_zenith_grid(mtd_path)
                    tox, toy = tile_origin_from_mtd(mtd_path, crs)
                    if grid is not None and tox is not None:
                        sza_mtd = interp_sza_from_grid(
                            grid, col_step, row_step,
                            cx, cy, tox, toy,
                        )
                else:
                    missed_mtd.add(scene_id)

                # 6c. pysolar fallback / cross-check.
                sza_py = pysolar_sza(lat, lon, dt_utc)

                # 6d. Choose canonical SZA.
                if np.isfinite(sza_mtd):
                    sza, source = sza_mtd, "mtd"
                elif np.isfinite(sza_py):
                    sza, source = sza_py, "pysolar"
                else:
                    sza, source = np.nan, "none"

                records.append({
                    "region":            region,
                    "sza_bin":           sza_bin,
                    "chip_stem":         chip_stem,
                    "scene_id":          scene_id,
                    "acq_datetime_utc":  dt_utc.isoformat(),
                    "lat":               round(lat, 6),
                    "lon":               round(lon, 6),
                    "sza_deg_mtd":       round(sza_mtd, 4) if np.isfinite(sza_mtd) else np.nan,
                    "sza_deg_pysolar":   round(sza_py, 4)  if np.isfinite(sza_py)  else np.nan,
                    "sza_deg":           round(sza, 4)     if np.isfinite(sza)     else np.nan,
                    "source":            source,
                })

    # 6e. Write CSV and print summary.
    df = pd.DataFrame(records)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    if len(df):
        diffs = (df["sza_deg_mtd"] - df["sza_deg_pysolar"]).abs()
        diffs = diffs[np.isfinite(diffs)]
        print(f"  MTD source rows     : {(df['source']=='mtd').sum()}")
        print(f"  pysolar fallback    : {(df['source']=='pysolar').sum()}")
        print(f"  no SZA              : {(df['source']=='none').sum()}")
        if len(diffs):
            print(f"  |MTD - pysolar| max : {diffs.max():.3f} deg")
            print(f"  |MTD - pysolar| p95 : {diffs.quantile(0.95):.3f} deg")
            print(f"  fraction within 0.5 : {(diffs <= 0.5).mean():.3f}")
    if missed_mtd:
        print(f"\n{len(missed_mtd)} scene(s) with no MTD_TL.xml under safe_root:")
        for s in sorted(missed_mtd)[:5]:
            print(f"    {s}")


if __name__ == "__main__":
    main()
