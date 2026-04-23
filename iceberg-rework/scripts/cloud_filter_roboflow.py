"""
cloud_filter_roboflow.py — Delete cloudy chips from Roboflow using QA60 cloud bitmask.

For each image in the Roboflow project:
  1. Parse scene stem and chip row/col from the PNG filename
  2. Find the parent S2 scene zip in sentinel2_downloads/
  3. Open QA60 band (60m cloud mask) from inside the zip via /vsizip/ (no extraction)
  4. Compute cloud pixel fraction within the chip's spatial extent
  5. Delete the image from Roboflow if cloud fraction > threshold
  6. Log all decisions to cloud_filter_log.csv

Chip filename format:  {stem}_r{row:04d}_c{col:04d}_B08.png
Parent zip:            sentinel2_downloads/{region}/{sza_bin}/{stem}.SAFE.zip
Cloud mask: MSK_CLASSI_B00.jp2 (L1C N0500+), bit 0 = opaque cloud, bit 1 = cirrus

Usage:
  python cloud_filter_roboflow.py \\
      --api_key    YOUR_API_KEY \\
      [--workspace iceberg-seg] \\
      [--project   iceberg-seg-experiment] \\
      [--chips_dir  /mnt/research/.../S2-iceberg-areas/chips] \\
      [--downloads_dir /mnt/research/.../sentinel2_downloads] \\
      [--threshold 0.10] \\
      [--dry_run]

Always dry_run first to verify counts before deleting.

Scene download threshold was 20% cloud cover (scene-wide). A 10% chip-level
threshold is appropriate: chips within a 20%-cloudy scene range from 0-100%
cloudy, and 10% catches the bad chips without over-filtering.

rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/cloud_filter_roboflow.py smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import argparse
import csv
import os
import re
import time
import zipfile
from glob import glob

import numpy as np
import rasterio
import rasterio.errors
import rasterio.windows
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.warp import transform_bounds
import requests

CLOUD_THRESHOLD  = 0.10
MSK_CLASSI_CLOUD = 0b11   # bit 0 = opaque cloud, bit 1 = cirrus (MSK_CLASSI_B00.jp2 in L1C N0500+)
API_BASE         = "https://api.roboflow.com"


# ─── Roboflow API ─────────────────────────────────────────────────────────────

def list_roboflow_images(workspace, project, api_key, unannotated_only=False):
    """
    Paginate through all images in the project. Returns list of {id, name} dicts.
    Uses POST https://api.roboflow.com/{workspace}/{project}/search (max 250 per page).
    unannotated_only=True filters to images not yet assigned to a dataset split.
    """
    images = []
    offset = 0
    limit  = 250   # API max
    url    = f"{API_BASE}/{workspace}/{project}/search"
    body   = {"offset": offset, "limit": limit, "fields": ["id", "name"]}
    if unannotated_only:
        body["in_dataset"] = False
    while True:
        body["offset"] = offset
        resp = requests.post(
            url,
            params={"api_key": api_key},
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        data  = resp.json()
        batch = data.get("results", [])
        if not batch:
            break
        images.extend(batch)
        total = data.get("total", "?")
        print(f"  fetched {len(images)}/{total} images...")
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(0.2)
    return images


def delete_roboflow_images(workspace, project, api_key, image_ids, dry_run=False):
    """
    Batch-delete images from Roboflow.
    DELETE https://api.roboflow.com/{workspace}/{project}/images
    Body: {"images": ["id1", "id2", ...]}
    Returns count of successfully deleted images.
    """
    if dry_run:
        return len(image_ids)
    url = f"{API_BASE}/{workspace}/{project}/images"
    try:
        resp = requests.delete(
            url,
            params={"api_key": api_key},
            json={"images": image_ids},
            timeout=60,
        )
        if resp.status_code == 204:
            return len(image_ids)
        resp.raise_for_status()
        return len(image_ids)
    except requests.exceptions.HTTPError:
        print(f"    [http error] batch delete: {resp.status_code} -- {resp.text[:200]}")
        return 0
    except Exception as e:
        print(f"    [error] batch delete: {e}")
        return 0


# ─── Chip / scene helpers ─────────────────────────────────────────────────────

# Matches: {stem}_r{row:04d}_c{col:04d}_B08.png  or  {stem}_r{row:04d}_c{col:04d}.tif
CHIP_RE = re.compile(r"^(.+)_r(\d+)_c(\d+)(?:_B08)?\.(?:png|tif)$")


def parse_chip_name(filename):
    """
    Parse (stem, row, col) from a chip PNG or TIF filename.
    Returns None if filename does not match the expected pattern.
    """
    m = CHIP_RE.match(os.path.basename(filename))
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def find_chip_tif(chips_dir, stem, row, col):
    """Glob recursively for the .tif chip. Returns path or None."""
    fname   = f"{stem}_r{row:04d}_c{col:04d}.tif"
    matches = glob(os.path.join(chips_dir, "**", fname), recursive=True)
    return matches[0] if matches else None


def find_parent_zip(downloads_dir, stem):
    """Glob recursively for the parent SAFE.zip. Returns path or None."""
    pattern = os.path.join(downloads_dir, "**", f"{stem}.SAFE.zip")
    matches  = glob(pattern, recursive=True)
    return matches[0] if matches else None


def find_qa60_entry(zip_path):
    """
    Return the internal zip entry path for the cloud classification mask, or None.
    S2 L1C N0500+ stores this as MSK_CLASSI_B00.jp2 in GRANULE/*/QI_DATA/.
    Bit 0 = opaque cloud, bit 1 = cirrus.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
    for n in names:
        if "MSK_CLASSI_B00.jp2" in n:
            return n
    return None


def get_cloud_fraction(zip_path, chip_bounds, chip_crs):
    """
    Open QA60 from inside the zip via /vsizip/ (no extraction),
    sample the window covering chip_bounds, and return cloud pixel fraction.

    chip_bounds : rasterio BoundingBox (left, bottom, right, top) in chip_crs
    chip_crs    : rasterio CRS of the chip

    Returns float in [0, 1], or None if QA60 is unavailable or out of bounds.
    """
    qa60_entry = find_qa60_entry(zip_path)
    if qa60_entry is None:
        return None

    qa60_vsi = f"/vsizip/{zip_path}/{qa60_entry}"
    try:
        with rasterio.open(qa60_vsi) as src:
            # Reproject chip bounds to QA60 CRS if they differ
            if src.crs and chip_crs and src.crs.to_epsg() != chip_crs.to_epsg():
                bounds = transform_bounds(chip_crs, src.crs,
                                          chip_bounds.left, chip_bounds.bottom,
                                          chip_bounds.right, chip_bounds.top)
            else:
                bounds = (chip_bounds.left, chip_bounds.bottom,
                          chip_bounds.right, chip_bounds.top)

            win = window_from_bounds(*bounds, transform=src.transform)

            # Clamp to raster extent; return None if chip doesn't overlap
            full = rasterio.windows.Window(0, 0, src.width, src.height)
            try:
                win = win.intersection(full)
            except rasterio.errors.WindowError:
                return None

            if win.width < 1 or win.height < 1:
                return None

            qa60 = src.read(1, window=win)

        if qa60.size == 0:
            return None

        cloud_px = int(((qa60.astype(np.uint8) & MSK_CLASSI_CLOUD) > 0).sum())
        return cloud_px / qa60.size

    except Exception as e:
        print(f"    [qa60 error] {e}")
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Delete cloudy chips from Roboflow using S2 QA60 cloud bitmask"
    )
    parser.add_argument("--workspace",    default="iceberg-seg",
        help="Roboflow workspace ID (default: iceberg-seg)")
    parser.add_argument("--project",      default="iceberg-seg-experiment",
        help="Roboflow project slug (default: iceberg-seg-experiment)")
    parser.add_argument("--api_key",      required=True, help="Roboflow API key")
    parser.add_argument("--chips_dir",
        default="/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/chips",
        help="Root chips directory (contains region/sza_bin/tifs/*.tif)")
    parser.add_argument("--downloads_dir",
        default="/mnt/research/v.gomezgilyaspik/students/smishra/sentinel2_downloads",
        help="Root sentinel2_downloads directory (contains region/sza_bin/*.SAFE.zip)")
    parser.add_argument("--threshold", type=float, default=CLOUD_THRESHOLD,
        help=f"Delete if QA60 cloud fraction exceeds this (default: {CLOUD_THRESHOLD})")
    parser.add_argument("--log_csv",
        default="/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/cloud_filter_log.csv",
        help="Output CSV log path (default: mnt/research path)")
    parser.add_argument("--delay", type=float, default=0.2,
        help="Seconds between Roboflow delete calls (default: 0.2)")
    parser.add_argument("--unannotated_only", action="store_true",
        help="Only process unannotated images (not yet assigned to a split)")
    parser.add_argument("--dry_run", action="store_true",
        help="Compute cloud fractions but do not delete from Roboflow")
    args = parser.parse_args()

    print(f"Listing images in {args.workspace}/{args.project}...")
    images = list_roboflow_images(args.workspace, args.project, args.api_key,
                                   unannotated_only=args.unannotated_only)
    print(f"Total images in project: {len(images)}\n")

    n_kept = n_no_zip = n_no_tif = n_no_qa60 = n_parse_err = 0
    to_delete = []   # list of {image_id, filename, ...} to batch-delete at end
    rows      = []   # CSV rows

    for i, img in enumerate(images):
        image_id = img.get("id", "")
        filename  = img.get("name", img.get("filename", ""))

        # Parse chip name
        parsed = parse_chip_name(filename)
        if parsed is None:
            print(f"  [{i+1:>5}/{len(images)}] SKIP (unparseable) {filename}")
            rows.append({"image_id": image_id, "filename": filename,
                         "cloud_fraction": "", "decision": "skip",
                         "note": "filename does not match chip pattern"})
            n_parse_err += 1
            continue

        stem, row, col = parsed
        tag = f"{stem[:50]}_r{row:04d}_c{col:04d}"

        # Find parent zip
        zip_path = find_parent_zip(args.downloads_dir, stem)
        if zip_path is None:
            print(f"  [{i+1:>5}/{len(images)}] NO ZIP  {tag}")
            rows.append({"image_id": image_id, "filename": filename,
                         "stem": stem, "row": row, "col": col,
                         "cloud_fraction": "", "decision": "skip",
                         "note": "parent SAFE.zip not found"})
            n_no_zip += 1
            continue

        # Find chip .tif to get spatial extent
        tif_path = find_chip_tif(args.chips_dir, stem, row, col)
        if tif_path is None:
            print(f"  [{i+1:>5}/{len(images)}] NO TIF  {tag}")
            rows.append({"image_id": image_id, "filename": filename,
                         "stem": stem, "row": row, "col": col,
                         "cloud_fraction": "", "decision": "skip",
                         "note": "chip .tif not found in chips_dir"})
            n_no_tif += 1
            continue

        with rasterio.open(tif_path) as src:
            chip_bounds = src.bounds
            chip_crs    = src.crs

        # Compute cloud fraction from QA60
        cloud_frac = get_cloud_fraction(zip_path, chip_bounds, chip_crs)
        if cloud_frac is None:
            print(f"  [{i+1:>5}/{len(images)}] NO QA60 {tag}")
            rows.append({"image_id": image_id, "filename": filename,
                         "stem": stem, "row": row, "col": col,
                         "cloud_fraction": "", "decision": "skip",
                         "note": "QA60 not found in zip or out of bounds"})
            n_no_qa60 += 1
            continue

        cloudy   = cloud_frac > args.threshold
        decision = "delete" if cloudy else "keep"
        dry_tag  = " [DRY RUN]" if (args.dry_run and cloudy) else ""
        print(f"  [{i+1:>5}/{len(images)}] cloud={cloud_frac:.3f}  {decision.upper()}{dry_tag}  {tag}")

        if cloudy:
            to_delete.append(image_id)
        else:
            n_kept += 1

        rows.append({"image_id": image_id, "filename": filename,
                     "stem": stem, "row": row, "col": col,
                     "cloud_fraction": f"{cloud_frac:.4f}",
                     "decision": decision, "note": ""})

    # Batch delete all flagged images in one API call
    print(f"\n{len(to_delete)} images flagged for deletion...")
    n_deleted = delete_roboflow_images(
        args.workspace, args.project, args.api_key, to_delete, dry_run=args.dry_run
    )

    # Write CSV log
    with open(args.log_csv, "w", newline="") as logf:
        writer = csv.DictWriter(logf, fieldnames=[
            "image_id", "filename", "stem", "row", "col",
            "cloud_fraction", "decision", "note",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'─'*60}")
    print(f"Images assessed   : {len(images)}")
    print(f"  deleted         : {n_deleted}"
          + (" (dry run -- not actually deleted)" if args.dry_run else ""))
    print(f"  kept            : {n_kept}")
    print(f"  no zip found    : {n_no_zip}")
    print(f"  no .tif found   : {n_no_tif}")
    print(f"  no QA60 in zip  : {n_no_qa60}")
    print(f"  parse errors    : {n_parse_err}")
    print(f"Log saved to      : {args.log_csv}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
