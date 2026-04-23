"""
export_reference_gpkg.py

Export labeler-validated iceberg polygons from the iceberg-labeler SQLite
database into per-chip UTM GeoPackages. These serve as the "visually
delineated" reference set A_ref in Fisser-style relative error analysis.

Inverts the UTM->pixel transform used in scripts/import_chips.py
(utm_polygon_to_pixel, L56-88) so pixel-space labeler polygons become
georeferenced UTM polygons with area in m^2.

Inputs:
  - iceberg-labeler SQLite (DATABASE_URL from config.py by default)
  - Each chip's source GeoTIFF (Chip.tif_path) provides affine + CRS

Outputs (one GeoPackage per chip):
  {out_root}/{REGION}/{sza_bin}/reference/gpkgs/{chip_stem}_reference.gpkg

  Columns: geometry, class_name, area_m2, root_length_m, source_file,
           iceberg_id, action, tags, labeler_id

Filters (Fisser-comparable):
  1. AnnotationResult.chip_verdict in {accepted, edited}
  2. PolygonDecision.action        in {accepted, added, modified}
  3. Chip dropped if chip-level tags contain any of:
     cloud, ambiguous, land-edge, melange
     (sea-ice and dark-water are retained; those are physical conditions
      we want to quantify, not labels of bad ground truth.)
  4. Polygon dropped if sqrt(area_m2) < min_root_length_m (default 40 m,
     matches S2 GSD resolvability and paper-writing/plan.md).

Usage:
  python scripts/export_reference_gpkg.py \\
      --out_root /mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/area_comparison

Rsync:
  rsync -av /Users/smishra/iceberg-labeler/scripts/export_reference_gpkg.py \\
      smishra@moosehead.bowdoin.edu:~/iceberg-labeler/scripts/
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path

import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

# Make parent dir importable so we can reuse the labeler's ORM + DB session.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from database import SessionLocal           # noqa: E402
from models import (                        # noqa: E402
    Chip, Assignment, Labeler, AnnotationResult, PolygonDecision,
)

DEFAULT_OUT_ROOT = "/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/area_comparison"

# Chip is DROPPED if any of these tags is present (non-iceberg / invalid GT).
# sea-ice and dark-water are KEPT (they are study conditions, not GT flaws).
DROP_TAGS = {"cloud", "ambiguous", "land-edge", "melange"}

KEEP_VERDICTS = {"accepted", "edited"}
KEEP_ACTIONS  = {"accepted", "added", "modified"}


# 1. Coordinate conversion: pixel -> UTM -------------------------------------

def pixel_polygon_to_utm(pixel_coords, transform):
    """
    Convert a polygon from pixel coords to UTM via rasterio affine.

    pixel_coords : list of [col, row] pairs (JSON from PolygonDecision)
    transform    : rasterio.Affine (src.transform of the chip GeoTIFF)

    Returns a Shapely Polygon in UTM, or None if the polygon is degenerate.
    """
    if not pixel_coords or len(pixel_coords) < 3:
        return None
    utm_pts = []
    for col, row in pixel_coords:
        x, y = transform * (float(col), float(row))
        utm_pts.append((x, y))
    if len({(round(x, 4), round(y, 4)) for x, y in utm_pts}) < 3:
        return None
    poly = Polygon(utm_pts)
    if not poly.is_valid:
        poly = poly.buffer(0)                 # clean self-intersections
        if poly.is_empty or poly.area == 0:
            return None
    return poly


# 2. Tag parsing -------------------------------------------------------------

def should_drop_chip(tags_str):
    """Return True if the chip-level tag string contains any DROP_TAGS."""
    if not tags_str:
        return False
    tags = {t.strip() for t in tags_str.split(",") if t.strip()}
    return bool(tags & DROP_TAGS)


# 3. Per-chip export ---------------------------------------------------------

def export_chip(db, chip, out_root, min_root_length_m, stats):
    """
    Export all accepted/added/modified polygons for one chip.
    Returns number of polygons written (0 if chip skipped).
    """
    # 3a. Fetch the annotation result(s) for this chip. A chip may be labeled
    # by multiple people via separate assignments; we take ALL qualifying
    # complete assignments and union their decisions. Downstream RE analysis
    # treats each polygon independently so duplicates are not a problem when
    # we later match predictions 1:1 via Hungarian.
    rows = (
        db.query(AnnotationResult, Assignment, Labeler)
        .join(Assignment, AnnotationResult.assignment_id == Assignment.id)
        .join(Labeler,    Assignment.labeler_id == Labeler.id)
        .filter(
            Assignment.chip_id == chip.id,
            Assignment.status == "complete",
            AnnotationResult.chip_verdict.in_(KEEP_VERDICTS),
        )
        .all()
    )
    if not rows:
        stats["no_label"] += 1
        return 0

    # 3b. Chip-level tag filter: if ANY complete AnnotationResult tags the
    # chip as cloud/ambiguous/land-edge/melange, drop the chip entirely.
    for result, _assn, _lab in rows:
        if should_drop_chip(result.tags):
            stats["dropped_by_tags"] += 1
            return 0
    # Keep the comma-joined tags of the first labeler result for output.
    tags_str = rows[0][0].tags or ""

    # 3c. Open the source TIF to recover affine and CRS.
    if not chip.tif_path or not os.path.exists(chip.tif_path):
        stats["no_tif"] += 1
        return 0
    with rasterio.open(chip.tif_path) as src:
        transform = src.transform
        crs       = src.crs

    # 3d. Collect qualifying polygon decisions from every result on this chip.
    records = []
    for result, assn, _lab in rows:
        decisions = (
            db.query(PolygonDecision)
            .filter(
                PolygonDecision.result_id == result.id,
                PolygonDecision.action.in_(KEEP_ACTIONS),
            )
            .all()
        )
        for dec in decisions:
            try:
                coords = json.loads(dec.pixel_coords)
            except Exception:
                stats["bad_json"] += 1
                continue
            poly = pixel_polygon_to_utm(coords, transform)
            if poly is None:
                stats["degenerate"] += 1
                continue
            area_m2 = float(poly.area)
            root_len = math.sqrt(area_m2) if area_m2 > 0 else 0.0
            if root_len < min_root_length_m:
                stats["below_size"] += 1
                continue
            records.append({
                "geometry":       poly,
                "class_name":     dec.class_name or "iceberg",
                "area_m2":        round(area_m2, 4),
                "root_length_m":  round(root_len, 4),
                "source_file":    Path(chip.filename).stem,
                "iceberg_id":     dec.id,
                "action":         dec.action,
                "tags":           tags_str,
                "labeler_id":     assn.labeler_id,
            })

    if not records:
        stats["empty_after_filter"] += 1
        return 0

    # 3e. Write the per-chip reference GPKG.
    region  = (chip.region or "").upper()
    sza_bin = chip.sza_bin or "unknown"
    out_dir = Path(out_root) / region / sza_bin / "reference" / "gpkgs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(chip.filename).stem
    out_path = out_dir / f"{stem}_reference.gpkg"

    gdf = gpd.GeoDataFrame(records, crs=crs)
    gdf.to_file(out_path, driver="GPKG")
    stats["polygons_written"] += len(records)
    stats["chips_written"]    += 1
    return len(records)


# 4. Driver ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--out_root", default=DEFAULT_OUT_ROOT,
                        help="Root of area_comparison/. Output tree: "
                             "{out_root}/{REGION}/{sza_bin}/reference/gpkgs/")
    parser.add_argument("--region",  default=None,
                        help="Filter: 'kq' or 'sk'. Default: all regions.")
    parser.add_argument("--sza_bin", default=None,
                        help="Filter: e.g. 'sza_65_70'. Default: all bins.")
    parser.add_argument("--min_root_length_m", type=float, default=40.0,
                        help="Minimum sqrt(area_m2) for a polygon to be kept "
                             "(default 40 m, matches plan.md cutoff).")
    args = parser.parse_args()

    db = SessionLocal()

    # 4a. Pull all chips that match the CLI filters.
    q = db.query(Chip)
    if args.region:
        q = q.filter(Chip.region == args.region.lower())
    if args.sza_bin:
        q = q.filter(Chip.sza_bin == args.sza_bin)
    chips = q.all()
    print(f"Inspecting {len(chips)} chips "
          f"(region={args.region or 'all'}, sza_bin={args.sza_bin or 'all'})")

    stats = {
        "chips_written":       0,
        "polygons_written":    0,
        "no_label":            0,
        "dropped_by_tags":     0,
        "no_tif":              0,
        "bad_json":            0,
        "degenerate":          0,
        "below_size":          0,
        "empty_after_filter":  0,
    }

    # 4b. Process each chip, emit one GPKG per chip that has surviving polygons.
    for i, chip in enumerate(chips, 1):
        export_chip(db, chip, args.out_root, args.min_root_length_m, stats)
        if i % 50 == 0:
            print(f"  {i}/{len(chips)} processed | "
                  f"chips_written={stats['chips_written']} "
                  f"polygons={stats['polygons_written']}")

    db.close()

    # 4c. Summary.
    print("\n" + "=" * 60)
    print("Reference export summary")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:22s} {v}")
    print(f"\nOutput tree: {args.out_root}/<REGION>/<sza_bin>/reference/gpkgs/")


if __name__ == "__main__":
    main()
