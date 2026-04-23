"""
scripts/import_chips.py
=======================
Import chip PNG images and UNet++ predictions into the iceberg-labeler database.

What it does
------------
1. Scans CHIPS_SOURCE_ROOT/<region>/<sza_bin>/pngs/ for PNG preview images.
2. For each PNG, locates the corresponding GeoTIFF (same stem, .tif extension).
3. Reads the chip's affine transform from the GeoTIFF using rasterio.
4. Locates the per-chip GeoPackage in GPKG_SOURCE_ROOT/<REGION>/<sza_bin>/unet/gpkgs/.
5. Converts each predicted polygon from UTM → pixel coordinates.
6. Inserts the chip and its predictions into the SQLite database.

Chips already in the database (by filename) are skipped — safe to re-run.

Usage
-----
# From the iceberg-labeler/ directory:
python scripts/import_chips.py

# Override source roots via CLI:
python scripts/import_chips.py \\
    --chips-root /path/to/S2-iceberg-areas/chips \\
    --gpkg-root  /path/to/S2-iceberg-areas/area_comparison

# Dry run (show what would be imported without writing to DB):
python scripts/import_chips.py --dry-run

# Filter to specific region and/or SZA bin:
python scripts/import_chips.py --region kq --sza-bin sza_65_70
"""

import sys
import os
import json
import argparse
import shutil
from pathlib import Path

# Allow importing from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import mapping

import config
from database import SessionLocal, init_db
from models import Chip, Prediction


# ── Coordinate conversion ─────────────────────────────────────────────────────

def utm_polygon_to_pixel(geom, inv_transform, chip_h=256):
    """
    Convert a Shapely polygon from UTM → pixel coordinates.

    Parameters
    ----------
    geom          : Shapely Polygon in UTM CRS
    inv_transform : rasterio inverse affine transform (~src.transform)
    chip_h        : chip height in pixels (used for row clamping)

    Returns
    -------
    List of [col, row] pairs (floats), or None if the polygon is degenerate.
    """
    coords = []
    try:
        exterior = list(geom.exterior.coords)
    except AttributeError:
        return None  # multi-polygon or empty — skip

    for x_utm, y_utm in exterior:
        col, row = inv_transform * (x_utm, y_utm)
        # Clamp to chip bounds
        col = max(0.0, min(255.0, col))
        row = max(0.0, min(255.0, row))
        coords.append([round(col, 2), round(row, 2)])

    # Need at least 3 distinct points
    unique = set(tuple(c) for c in coords)
    if len(unique) < 3:
        return None

    return coords


# ── GeoPackage lookup ─────────────────────────────────────────────────────────

def find_gpkg(gpkg_root: Path, region: str, sza_bin: str, chip_stem: str) -> Path | None:
    """
    Find the per-chip GeoPackage file.

    Expected path:
        GPKG_ROOT/<REGION>/<sza_bin>/unet/gpkgs/<chip_stem>_icebergs.gpkg

    The region directory may be uppercase (KQ) while the region parameter is
    lowercase (kq) — both are tried.
    """
    for r in [region.upper(), region.lower(), region]:
        candidate = gpkg_root / r / sza_bin / "unet" / "gpkgs" / f"{chip_stem}_icebergs.gpkg"
        if candidate.exists():
            return candidate
    return None


# ── Main import logic ─────────────────────────────────────────────────────────

def import_chips(
    chips_root: Path,
    gpkg_root:  Path,
    chips_dir:  Path,
    dry_run:    bool = False,
    region_filter: str | None = None,
    sza_bin_filter: str | None = None,
    verbose:    bool = True,
):
    db = SessionLocal()

    # Pre-load existing filenames to skip duplicates
    existing = {row[0] for row in db.query(Chip.filename).all()}

    stats = {"found": 0, "skipped": 0, "imported": 0, "no_tif": 0,
             "no_gpkg": 0, "no_preds": 0, "errors": 0}

    # Walk region / sza_bin subdirectories
    for region_dir in sorted(chips_root.iterdir()):
        if not region_dir.is_dir():
            continue
        region = region_dir.name.lower()
        if region_filter and region != region_filter.lower():
            continue

        for sza_dir in sorted(region_dir.iterdir()):
            if not sza_dir.is_dir():
                continue
            sza_bin = sza_dir.name
            if sza_bin_filter and sza_bin != sza_bin_filter:
                continue

            png_dir = sza_dir / "pngs"
            tif_dir = sza_dir / "tifs"

            if not png_dir.exists():
                if verbose:
                    print(f"  [skip] no pngs/ dir: {sza_dir}")
                continue

            png_files = sorted(png_dir.glob("*.png"))
            if verbose:
                print(f"\n{region}/{sza_bin} — {len(png_files)} PNGs")

            for png_path in png_files:
                stats["found"] += 1
                stem     = png_path.stem
                filename = png_path.name

                # Skip already-imported chips
                if filename in existing:
                    stats["skipped"] += 1
                    continue

                # ── Find TIF ────────────────────────────────────────────────
                # PNG names may have a band suffix like "_B08" that TIFs lack
                tif_stem = stem
                tif_path = tif_dir / (tif_stem + ".tif")
                if not tif_path.exists():
                    # Try stripping the last _BAND suffix (e.g. "_B08")
                    import re
                    stripped = re.sub(r'_B\d+$', '', stem)
                    if stripped != stem:
                        tif_path = tif_dir / (stripped + ".tif")
                        tif_stem = stripped
                if not tif_path.exists():
                    stats["no_tif"] += 1
                    if verbose:
                        print(f"  [no tif] {stem}")
                    continue

                # ── Read affine transform from TIF ───────────────────────────
                try:
                    with rasterio.open(tif_path) as src:
                        transform = src.transform
                        chip_h    = src.height
                        chip_w    = src.width
                    inv_transform = ~transform
                except Exception as e:
                    stats["errors"] += 1
                    print(f"  [ERROR reading TIF] {stem}: {e}")
                    continue

                # ── Find GeoPackage ─────────────────────────────────────────
                gpkg_path = find_gpkg(gpkg_root, region, sza_bin, tif_stem)
                if gpkg_path is None:
                    stats["no_gpkg"] += 1
                    if verbose:
                        print(f"  [no gpkg] {stem}")
                    # Import chip with zero predictions (still useful)
                    # continue  # uncomment to skip chips without predictions

                # ── Use source PNG path directly (no copy needed) ────────────
                dest_png = png_path   # serve images from original location

                # ── Parse predictions ───────────────────────────────────────
                pred_records = []
                if gpkg_path is not None:
                    try:
                        gdf = gpd.read_file(gpkg_path)
                        for _, row in gdf.iterrows():
                            geom       = row.geometry
                            class_name = str(row.get("class_name", "iceberg"))
                            area_m2    = float(row.get("area_m2", 0.0))

                            pixel_coords = utm_polygon_to_pixel(
                                geom, inv_transform, chip_h=chip_h
                            )
                            if pixel_coords is None:
                                continue
                            pred_records.append({
                                "class_name":   class_name,
                                "pixel_coords": json.dumps(pixel_coords),
                                "area_m2":      round(area_m2, 2),
                            })
                    except Exception as e:
                        stats["errors"] += 1
                        print(f"  [ERROR reading gpkg] {stem}: {e}")
                        continue

                if verbose:
                    print(f"  [import] {stem:70s}  {len(pred_records):4d} preds")

                if not dry_run:
                    chip = Chip(
                        filename         = filename,
                        image_path       = str(dest_png),
                        tif_path         = str(tif_path),
                        region           = region,
                        sza_bin          = sza_bin,
                        width            = chip_w,
                        height           = chip_h,
                        prediction_count = len(pred_records),
                    )
                    db.add(chip)
                    db.flush()  # get chip.id

                    for pr in pred_records:
                        db.add(Prediction(
                            chip_id      = chip.id,
                            class_name   = pr["class_name"],
                            pixel_coords = pr["pixel_coords"],
                            area_m2      = pr["area_m2"],
                        ))

                    db.commit()
                    existing.add(filename)

                stats["imported"] += 1

    db.close()

    print("\n" + "─" * 50)
    print(f"PNG files found     : {stats['found']}")
    print(f"Already in DB       : {stats['skipped']}")
    print(f"Imported            : {stats['imported']}")
    print(f"Missing TIF         : {stats['no_tif']}")
    print(f"Missing GeoPackage  : {stats['no_gpkg']}")
    print(f"Errors              : {stats['errors']}")
    if dry_run:
        print("\n[DRY RUN — no changes written to database]")
    print("─" * 50)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Import iceberg chips + UNet++ predictions into the labeler DB"
    )
    parser.add_argument(
        "--chips-root",
        default=config.CHIPS_SOURCE_ROOT,
        help=f"Root of chip directories (default: {config.CHIPS_SOURCE_ROOT})"
    )
    parser.add_argument(
        "--gpkg-root",
        default=config.GPKG_SOURCE_ROOT,
        help=f"Root of GeoPackage directories (default: {config.GPKG_SOURCE_ROOT})"
    )
    parser.add_argument(
        "--chips-dir",
        default=config.CHIPS_DIR,
        help=f"Destination directory for PNG copies (default: {config.CHIPS_DIR})"
    )
    parser.add_argument("--region",  default=None, help="Filter to one region (e.g. kq)")
    parser.add_argument("--sza-bin", default=None, help="Filter to one SZA bin (e.g. sza_65_70)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be imported without writing anything")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-chip output")
    args = parser.parse_args()

    chips_root = Path(args.chips_root)
    gpkg_root  = Path(args.gpkg_root)
    chips_dir  = Path(args.chips_dir)

    for p, name in [(chips_root, "chips-root"), (gpkg_root, "gpkg-root")]:
        if not p.exists():
            print(f"ERROR: {name} does not exist: {p}")
            sys.exit(1)

    if not args.dry_run:
        chips_dir.mkdir(parents=True, exist_ok=True)
        init_db()

    print(f"Chips root : {chips_root}")
    print(f"GPKG root  : {gpkg_root}")
    print(f"Chips dir  : {chips_dir}")
    if args.dry_run:
        print("[DRY RUN]")

    import_chips(
        chips_root     = chips_root,
        gpkg_root      = gpkg_root,
        chips_dir      = chips_dir,
        dry_run        = args.dry_run,
        region_filter  = args.region,
        sza_bin_filter = args.sza_bin,
        verbose        = not args.quiet,
    )


if __name__ == "__main__":
    main()
