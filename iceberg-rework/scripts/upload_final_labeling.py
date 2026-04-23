"""
upload_final_labeling.py — Filter chips locally and upload pre-annotated batches to Roboflow.

Combines cloud filtering, IC filtering, Otsu pre-annotation, and batch upload.

Pipeline per chip:
  1. IC filter: skip if Otsu ic_frac >= 0.15 (sea-ice-dominated)
  2. Cloud filter: skip if MSK_CLASSI cloud fraction >= 0.02 (strict 2%)
  3. Run Otsu -> extract iceberg polygons
  4. Skip if no polygons (featureless ocean)
  5. Upload PNG + COCO annotation to Roboflow via SDK

Upload plan:
  - 20 batches x 150 chips = 3000 total
  - Shuffled with seed=42 for reproducibility
  - Excludes sza_lt65 (Fisser already has those)
  - Target project: final-labeling (workspace: iceberg-seg)
  - Reviewer: mishrashibali@gmail.com (set per batch via annotation jobs API)

MSK_CLASSI arrays are cached per scene zip (one zip open per unique scene).

Usage:
  python upload_final_labeling.py \\
      --api_key    YOUR_API_KEY \\
      [--workspace iceberg-seg] \\
      [--project   final-labeling] \\
      [--chips_dir /mnt/research/.../S2-iceberg-areas/chips] \\
      [--downloads_dir /mnt/research/.../sentinel2_downloads] \\
      [--n_batches 20] \\
      [--batch_size 150] \\
      [--reviewer  mishrashibali@gmail.com] \\
      [--dry_run]

rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/upload_final_labeling.py smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import argparse
import csv
import json
import os
import random
import re
import tempfile
import time
import zipfile
from glob import glob

import numpy as np
import rasterio
import rasterio.errors
import rasterio.windows
from rasterio.features import shapes as rio_shapes
from rasterio.transform import Affine
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.warp import transform_bounds
from skimage.filters import threshold_otsu
from shapely.geometry import shape
import requests

# ─── Constants ────────────────────────────────────────────────────────────────

CLOUD_THRESHOLD  = 0.01    # strict: <1% MSK_CLASSI cloud pixels per chip
IC_THRESHOLD     = 0.15    # skip chip if >15% of B08 pixels are above Otsu threshold
MIN_AREA_PX      = 4       # minimum polygon area in pixels
B08_IDX          = 2       # band index of B08 in chip stack (B04/B03/B08)
MSK_CLASSI_CLOUD = 0b11    # bit 0 = opaque cloud, bit 1 = cirrus
N_BATCHES        = 20
BATCH_SIZE       = 150
SEED             = 42
API_BASE         = "https://api.roboflow.com"

CHIP_RE = re.compile(r"^(.+)_r(\d+)_c(\d+)\.tif$")


# ─── Cloud mask (cached per scene zip) ────────────────────────────────────────

_cloud_cache = {}   # {zip_path: (array, transform, crs) or None}


def _load_cloud_mask(zip_path):
    if zip_path in _cloud_cache:
        return _cloud_cache[zip_path]
    entry = None
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if "MSK_CLASSI_B00.jp2" in name:
                entry = name
                break
    if entry is None:
        _cloud_cache[zip_path] = None
        return None
    try:
        with rasterio.open(f"/vsizip/{zip_path}/{entry}") as src:
            result = (src.read(1), src.transform, src.crs)
        _cloud_cache[zip_path] = result
        return result
    except Exception:
        _cloud_cache[zip_path] = None
        return None


def cloud_fraction(zip_path, chip_bounds, chip_crs):
    """Return cloud pixel fraction for chip extent, or None if mask unavailable."""
    mask_data = _load_cloud_mask(zip_path)
    if mask_data is None:
        return None
    array, transform, crs = mask_data
    try:
        if crs and chip_crs and crs.to_epsg() != chip_crs.to_epsg():
            bounds = transform_bounds(chip_crs, crs,
                                      chip_bounds.left, chip_bounds.bottom,
                                      chip_bounds.right, chip_bounds.top)
        else:
            bounds = (chip_bounds.left, chip_bounds.bottom,
                      chip_bounds.right, chip_bounds.top)
        win = window_from_bounds(*bounds, transform=transform)
        full = rasterio.windows.Window(0, 0, array.shape[1], array.shape[0])
        try:
            win = win.intersection(full)
        except rasterio.errors.WindowError:
            return None
        if win.width < 1 or win.height < 1:
            return None
        row_off = int(win.row_off)
        col_off = int(win.col_off)
        h = int(win.height)
        w = int(win.width)
        patch = array[row_off:row_off+h, col_off:col_off+w]
        if patch.size == 0:
            return None
        cloud_px = int(((patch.astype(np.uint8) & MSK_CLASSI_CLOUD) > 0).sum())
        return cloud_px / patch.size
    except Exception:
        return None


# ─── Otsu + IC filter ─────────────────────────────────────────────────────────

def otsu_filter_and_polygons(tif_path):
    """
    Returns (polygons, reason) where reason is None on success or a skip string.
    Also performs IC filter.
    """
    with rasterio.open(tif_path) as src:
        chip = src.read().astype(np.float32)

    if chip.shape[0] <= B08_IDX:
        return None, "too few bands"

    b08   = chip[B08_IDX]
    valid = b08[b08 > 0]
    if valid.size < 100:
        return None, "too few valid pixels"

    thresh  = float(threshold_otsu(valid))
    ic_frac = float((b08 >= thresh).mean())
    if ic_frac >= IC_THRESHOLD:
        return None, f"IC ({ic_frac:.2f})"

    mask = (b08 >= thresh).astype(np.uint8)
    polys = []
    for geom_dict, val in rio_shapes(mask, transform=Affine(1, 0, 0, 0, 1, 0)):
        if val == 0:
            continue
        geom = shape(geom_dict)
        if geom.is_empty or geom.area < MIN_AREA_PX:
            continue
        polys.append(geom)

    if not polys:
        return None, "no polygons"

    return polys, None


def to_coco_json(polygons, img_filename, w=256, h=256):
    anns = []
    for aid, poly in enumerate(polygons, 1):
        coords  = list(poly.exterior.coords)
        seg     = [round(v, 2) for xy in coords for v in xy]
        xs      = [c[0] for c in coords]
        ys      = [c[1] for c in coords]
        bx, by  = min(xs), min(ys)
        bw, bh  = max(xs) - bx, max(ys) - by
        anns.append({
            "id": aid, "image_id": 1, "category_id": 1,
            "segmentation": [seg],
            "bbox": [round(bx,2), round(by,2), round(bw,2), round(bh,2)],
            "area": round(float(poly.area), 2),
            "iscrowd": 0,
        })
    return {
        "images":      [{"id": 1, "file_name": img_filename, "width": w, "height": h}],
        "annotations": anns,
        "categories":  [{"id": 1, "name": "iceberg", "supercategory": "iceberg"}],
    }


# ─── Chip discovery ───────────────────────────────────────────────────────────

def find_parent_zip(downloads_dir, stem, region, sza_bin):
    path = os.path.join(downloads_dir, region, sza_bin, f"{stem}.SAFE.zip")
    return path if os.path.exists(path) else None


def find_png(chips_dir, region, sza_bin, stem, row, col):
    path = os.path.join(chips_dir, region, sza_bin, "pngs",
                        f"{stem}_r{row:04d}_c{col:04d}_B08.png")
    if os.path.exists(path):
        return path
    # fallback: try with actual row/col digit count
    pattern = os.path.join(chips_dir, region, sza_bin, "pngs",
                           f"{stem}_r*_c*_B08.png")
    for p in glob(pattern):
        m = re.match(r".*_r(\d+)_c(\d+)_B08\.png$", p)
        if m and int(m.group(1)) == row and int(m.group(2)) == col:
            return p
    return None


# ─── Roboflow job creation (assigns reviewer) ─────────────────────────────────

def create_review_job(workspace, project, api_key, batch_name, reviewer_email):
    """
    Create an annotation job for a batch with reviewer assigned.
    POST https://api.roboflow.com/{workspace}/{project}/jobs
    Returns True on success.
    """
    url  = f"{API_BASE}/{workspace}/{project}/jobs"
    body = {
        "name":      f"{batch_name}-review",
        "labelers":  [],
        "reviewers": [reviewer_email],
        "batch":     batch_name,
    }
    try:
        resp = requests.post(url, params={"api_key": api_key}, json=body, timeout=30)
        if resp.status_code in (200, 201):
            return True
        print(f"    [job warning] {batch_name}: {resp.status_code} -- {resp.text[:150]}")
        return False
    except Exception as e:
        print(f"    [job error] {batch_name}: {e}")
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Filter chips and upload pre-annotated batches to Roboflow final-labeling"
    )
    parser.add_argument("--workspace",    default="iceberg-seg")
    parser.add_argument("--project",      default="final-labeling")
    parser.add_argument("--api_key",      required=True)
    parser.add_argument("--chips_dir",
        default="/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/chips")
    parser.add_argument("--downloads_dir",
        default="/mnt/research/v.gomezgilyaspik/students/smishra/sentinel2_downloads")
    parser.add_argument("--log_csv",
        default="/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/upload_final_labeling_log.csv")
    parser.add_argument("--n_batches",    type=int,   default=N_BATCHES)
    parser.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--reviewer",     default="mishrashibali@gmail.com")
    parser.add_argument("--seed",         type=int,   default=SEED)
    parser.add_argument("--delay",        type=float, default=0.3)
    parser.add_argument("--dry_run",      action="store_true",
        help="Filter and log without uploading to Roboflow")
    args = parser.parse_args()

    target = args.n_batches * args.batch_size
    print(f"Target: {args.n_batches} batches x {args.batch_size} = {target} chips")
    print(f"Project: {args.workspace}/{args.project}  reviewer: {args.reviewer}\n")

    if not args.dry_run:
        try:
            from roboflow import Roboflow
            rf      = Roboflow(api_key=args.api_key)
            rf_proj = rf.workspace(args.workspace).project(args.project)
        except ImportError:
            print("ERROR: pip install roboflow")
            return

    # ── Discover all tif chips (exclude sza_lt65) ────────────────────────────
    print("Scanning chips directory...")
    all_tifs = []
    for tif in glob(os.path.join(args.chips_dir, "**", "*.tif"), recursive=True):
        parts = os.path.normpath(tif).split(os.sep)
        if "sza_lt65" in parts:
            continue
        # extract region and sza_bin from path
        try:
            chips_parts = os.path.normpath(args.chips_dir).split(os.sep)
            rel_parts   = parts[len(chips_parts):]   # [region, sza_bin, 'tifs', fname]
            if len(rel_parts) < 4 or rel_parts[2] != "tifs":
                continue
            region  = rel_parts[0]
            sza_bin = rel_parts[1]
        except Exception:
            continue
        m = CHIP_RE.match(parts[-1])
        if m is None:
            continue
        all_tifs.append((tif, region, sza_bin, m.group(1), int(m.group(2)), int(m.group(3))))

    print(f"Found {len(all_tifs)} chips (excluding sza_lt65)\n")

    # ── Filter ────────────────────────────────────────────────────────────────
    print("Filtering (IC + cloud + Otsu)...")
    passed   = []   # (tif_path, png_path, region, sza_bin, stem, row, col, n_polys, polys)
    n_ic = n_cloud = n_no_zip = n_no_png = n_empty = n_no_mask = 0

    for idx, (tif_path, region, sza_bin, stem, row, col) in enumerate(all_tifs):
        if (idx + 1) % 500 == 0:
            print(f"  [{idx+1:>6}/{len(all_tifs)}] passed={len(passed)} "
                  f"ic={n_ic} cloud={n_cloud} nozip={n_no_zip} empty={n_empty}")

        # IC filter + Otsu (cheap, just reads tif)
        polys, reason = otsu_filter_and_polygons(tif_path)
        if reason is not None:
            if reason.startswith("IC"):
                n_ic += 1
            else:
                n_empty += 1
            continue

        # Cloud filter (opens zip -- cached per scene)
        zip_path = find_parent_zip(args.downloads_dir, stem, region, sza_bin)
        if zip_path is None:
            n_no_zip += 1
            continue

        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            crs    = src.crs

        cf = cloud_fraction(zip_path, bounds, crs)
        if cf is None:
            n_no_mask += 1
            # no mask available -- skip to be safe
            continue
        if cf >= CLOUD_THRESHOLD:
            n_cloud += 1
            continue

        png_path = find_png(args.chips_dir, region, sza_bin, stem, row, col)
        if png_path is None:
            n_no_png += 1
            continue

        passed.append((tif_path, png_path, region, sza_bin, stem, row, col, len(polys), polys))

        if len(passed) >= target * 3:   # collect 3x target then stop (plenty to shuffle from)
            break

    print(f"\nFilter summary:")
    print(f"  passed        : {len(passed)}")
    print(f"  IC-filtered   : {n_ic}")
    print(f"  cloudy (>=2%) : {n_cloud}")
    print(f"  no zip        : {n_no_zip}")
    print(f"  no cloud mask : {n_no_mask}")
    print(f"  empty/other   : {n_empty}")

    if len(passed) < target:
        print(f"\nWARNING: only {len(passed)} chips passed filters, less than target {target}.")
        print("Consider relaxing --cloud_threshold or checking zip availability.")

    # ── Shuffle + select ──────────────────────────────────────────────────────
    random.seed(args.seed)
    random.shuffle(passed)
    selected = passed[:target]
    print(f"\nSelected {len(selected)} chips (seed={args.seed})\n")

    # ── Upload in batches ─────────────────────────────────────────────────────
    rows      = []
    n_ok = n_fail = 0
    batch_names_created = []

    for batch_idx in range(args.n_batches):
        batch_name  = f"batch{batch_idx + 1:02d}"
        batch_chips = selected[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
        if not batch_chips:
            break

        print(f"[{batch_name}] uploading {len(batch_chips)} chips...")

        for j, (tif_path, png_path, region, sza_bin, stem, row, col, n_polys, polys) in enumerate(batch_chips):
            dry_tag = " [DRY RUN]" if args.dry_run else ""
            print(f"  [{j+1:>3}/{len(batch_chips)}] {n_polys:>4} polys  {stem[:45]}_r{row}_c{col}{dry_tag}")

            ok = True
            if not args.dry_run:
                coco = to_coco_json(polys, os.path.basename(png_path))
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                    json.dump(coco, tmp)
                    tmp_path = tmp.name
                try:
                    rf_proj.single_upload(
                        image_path=png_path,
                        annotation_path=tmp_path,
                        annotation_labelmap={"iceberg": "iceberg"},
                        batch_name=batch_name,
                        overwrite=True,
                    )
                except Exception as e:
                    print(f"    [upload error] {e}")
                    ok = False
                finally:
                    os.unlink(tmp_path)
                time.sleep(args.delay)

            if ok:
                n_ok += 1
            else:
                n_fail += 1

            rows.append({
                "batch": batch_name, "region": region, "sza_bin": sza_bin,
                "stem": stem, "row": row, "col": col,
                "n_polygons": n_polys, "png": os.path.basename(png_path),
                "decision": "uploaded" if ok else "failed",
            })

        batch_names_created.append(batch_name)

    # ── Create annotation jobs with reviewer ──────────────────────────────────
    if not args.dry_run and batch_names_created:
        print(f"\nCreating review jobs (reviewer: {args.reviewer})...")
        for batch_name in batch_names_created:
            ok = create_review_job(args.workspace, args.project, args.api_key,
                                   batch_name, args.reviewer)
            status = "ok" if ok else "FAILED (assign manually in UI)"
            print(f"  {batch_name}: {status}")

    # ── Save log ──────────────────────────────────────────────────────────────
    with open(args.log_csv, "w", newline="") as logf:
        writer = csv.DictWriter(logf, fieldnames=[
            "batch", "region", "sza_bin", "stem", "row", "col",
            "n_polygons", "png", "decision",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'─'*60}")
    print(f"Uploaded ok   : {n_ok}" + (" (dry run)" if args.dry_run else ""))
    print(f"Upload failed : {n_fail}")
    print(f"Log saved to  : {args.log_csv}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
