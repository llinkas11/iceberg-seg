"""
annotate_roboflow_otsu.py — Pre-label unannotated Roboflow chips with Otsu segmentation.

For each unannotated image in the Roboflow project:
  1. Find the corresponding .tif chip in chips_dir
  2. Run Otsu threshold on B08 band → binary iceberg mask
  3. Extract polygon contours in pixel coordinates
  4. Save as COCO JSON annotation file (temp)
  5. Upload via roboflow SDK project.single_upload(image_path, annotation_path)

Uses roboflow Python SDK which handles annotation format internally.
Requires: pip install roboflow

IC filter: skip chips where >15% of B08 pixels exceed the Otsu threshold
(sea-ice-dominated scene, not open water with icebergs).

Usage:
  python annotate_roboflow_otsu.py \\
      --api_key    YOUR_API_KEY \\
      [--workspace iceberg-seg] \\
      [--project   iceberg-seg-experiment] \\
      [--chips_dir /mnt/research/.../S2-iceberg-areas/chips] \\
      [--min_area  4] \\
      [--ic_threshold 0.15] \\
      [--dry_run]

rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/annotate_roboflow_otsu.py smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import argparse
import csv
import json
import os
import re
import tempfile
import time
from glob import glob

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.features import shapes as rio_shapes
from skimage.filters import threshold_otsu
from shapely.geometry import shape
import requests

MIN_AREA_PX  = 4
IC_THRESHOLD = 0.15
B08_IDX      = 2
API_BASE     = "https://api.roboflow.com"

CHIP_RE = re.compile(r"^(.+)_r(\d+)_c(\d+)(?:_B08)?\.(?:png|tif)$")


# ─── Roboflow listing (raw API — SDK doesn't paginate search) ─────────────────

def list_unannotated_images(workspace, project, api_key):
    images = []
    offset = 0
    limit  = 250
    url    = f"{API_BASE}/{workspace}/{project}/search"
    body   = {"limit": limit, "in_dataset": False, "fields": ["id", "name"]}
    while True:
        body["offset"] = offset
        resp = requests.post(url, params={"api_key": api_key}, json=body, timeout=30)
        resp.raise_for_status()
        data  = resp.json()
        batch = data.get("results", [])
        if not batch:
            break
        images.extend(batch)
        total = data.get("total", "?")
        print(f"  fetched {len(images)}/{total} unannotated images...")
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(0.2)
    return images


# ─── Chip helpers ─────────────────────────────────────────────────────────────

def parse_chip_name(filename):
    m = CHIP_RE.match(os.path.basename(filename))
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def find_chip_tif(chips_dir, stem, row, col):
    fname = f"{stem}_r{row:04d}_c{col:04d}.tif"
    matches = glob(os.path.join(chips_dir, "**", fname), recursive=True)
    if matches:
        return matches[0]
    for p in glob(os.path.join(chips_dir, "**", f"{stem}_r*_c*.tif"), recursive=True):
        m = CHIP_RE.match(os.path.basename(p))
        if m and int(m.group(2)) == row and int(m.group(3)) == col:
            return p
    return None


def find_chip_png(chips_dir, stem, row, col):
    fname = f"{stem}_r{row:04d}_c{col:04d}_B08.png"
    matches = glob(os.path.join(chips_dir, "**", fname), recursive=True)
    if matches:
        return matches[0]
    for p in glob(os.path.join(chips_dir, "**", f"{stem}_r*_c*_B08.png"), recursive=True):
        m = CHIP_RE.match(os.path.basename(p))
        if m and int(m.group(2)) == row and int(m.group(3)) == col:
            return p
    return None


# ─── Otsu segmentation ────────────────────────────────────────────────────────

def otsu_polygons(tif_path, b08_idx=B08_IDX, min_area_px=MIN_AREA_PX, ic_threshold=IC_THRESHOLD):
    """
    Run Otsu on B08, return (list of shapely pixel-space polygons, reason_or_None).
    """
    with rasterio.open(tif_path) as src:
        chip = src.read().astype(np.float32)

    if chip.shape[0] <= b08_idx:
        return [], "too few bands"

    b08   = chip[b08_idx]
    valid = b08[b08 > 0]
    if valid.size < 100:
        return [], "too few valid pixels"

    thresh  = float(threshold_otsu(valid))
    ic_frac = float((b08 >= thresh).mean())
    if ic_frac > ic_threshold:
        return [], f"IC-filtered (ic_frac={ic_frac:.2f})"

    mask = (b08 >= thresh).astype(np.uint8)
    pixel_transform = Affine(1, 0, 0, 0, 1, 0)

    polys = []
    for geom_dict, val in rio_shapes(mask, transform=pixel_transform):
        if val == 0:
            continue
        geom = shape(geom_dict)
        if geom.is_empty or geom.area < min_area_px:
            continue
        polys.append(geom)

    return polys, None


def polygons_to_coco_json(polygons, img_filename, img_width=256, img_height=256, label="iceberg"):
    """
    Convert pixel-space shapely polygons to COCO instance segmentation JSON.
    """
    categories = [{"id": 1, "name": label, "supercategory": label}]
    images     = [{"id": 1, "file_name": img_filename, "width": img_width, "height": img_height}]
    annotations = []
    for ann_id, poly in enumerate(polygons, start=1):
        # COCO segmentation: list of [x1,y1,x2,y2,...] flattened rings
        coords  = list(poly.exterior.coords)
        seg     = [round(v, 2) for xy in coords for v in xy]
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        x, y     = min(x_coords), min(y_coords)
        w, h     = max(x_coords) - x, max(y_coords) - y
        annotations.append({
            "id":          ann_id,
            "image_id":    1,
            "category_id": 1,
            "segmentation": [seg],
            "bbox":         [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
            "area":         round(float(poly.area), 2),
            "iscrowd":      0,
        })
    return {"images": images, "annotations": annotations, "categories": categories}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-label unannotated Roboflow chips with Otsu segmentation"
    )
    parser.add_argument("--workspace",    default="iceberg-seg")
    parser.add_argument("--project",      default="iceberg-seg-experiment")
    parser.add_argument("--api_key",      required=True)
    parser.add_argument("--chips_dir",
        default="/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/chips")
    parser.add_argument("--log_csv",
        default="/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/otsu_annotation_log.csv")
    parser.add_argument("--b08_idx",      type=int,   default=B08_IDX)
    parser.add_argument("--min_area",     type=float, default=MIN_AREA_PX)
    parser.add_argument("--ic_threshold", type=float, default=IC_THRESHOLD)
    parser.add_argument("--delay",        type=float, default=0.3)
    parser.add_argument("--dry_run",      action="store_true")
    args = parser.parse_args()

    # Import roboflow SDK here so missing package gives a clear error
    if not args.dry_run:
        try:
            from roboflow import Roboflow
            rf      = Roboflow(api_key=args.api_key)
            rf_proj = rf.workspace(args.workspace).project(args.project)
        except ImportError:
            print("ERROR: roboflow package not installed. Run: pip install roboflow")
            return

    print(f"Listing unannotated images in {args.workspace}/{args.project}...")
    images = list_unannotated_images(args.workspace, args.project, args.api_key)
    print(f"Unannotated images: {len(images)}\n")

    n_annotated = n_skipped = n_no_tif = n_parse_err = n_empty = 0
    rows = []

    for i, img in enumerate(images):
        image_id = img.get("id", "")
        filename  = img.get("name", img.get("filename", ""))

        parsed = parse_chip_name(filename)
        if parsed is None:
            print(f"  [{i+1:>5}/{len(images)}] SKIP (unparseable) {filename}")
            rows.append({"image_id": image_id, "filename": filename,
                         "n_polygons": "", "decision": "skip",
                         "note": "filename does not match chip pattern"})
            n_parse_err += 1
            continue

        stem, row, col = parsed
        tag = f"{stem[:50]}_r{row}_c{col}"

        tif_path = find_chip_tif(args.chips_dir, stem, row, col)
        if tif_path is None:
            print(f"  [{i+1:>5}/{len(images)}] NO TIF  {tag}")
            rows.append({"image_id": image_id, "filename": filename,
                         "n_polygons": "", "decision": "skip", "note": "chip .tif not found"})
            n_no_tif += 1
            continue

        polys, reason = otsu_polygons(tif_path, args.b08_idx, args.min_area, args.ic_threshold)

        if reason is not None:
            print(f"  [{i+1:>5}/{len(images)}] SKIP ({reason})  {tag}")
            rows.append({"image_id": image_id, "filename": filename,
                         "n_polygons": 0, "decision": "skip", "note": reason})
            n_skipped += 1
            continue

        if not polys:
            print(f"  [{i+1:>5}/{len(images)}] EMPTY  {tag}")
            rows.append({"image_id": image_id, "filename": filename,
                         "n_polygons": 0, "decision": "empty", "note": "no polygons above min_area"})
            n_empty += 1
            continue

        dry_tag = " [DRY RUN]" if args.dry_run else ""
        print(f"  [{i+1:>5}/{len(images)}] {len(polys):>4} polygons  UPLOAD{dry_tag}  {tag}")

        ok = True
        if not args.dry_run:
            png_path = find_chip_png(args.chips_dir, stem, row, col)
            if png_path is None:
                print(f"    [no png] {tag}")
                rows.append({"image_id": image_id, "filename": filename,
                             "n_polygons": len(polys), "decision": "skip", "note": "png not found"})
                continue

            coco = polygons_to_coco_json(polys, os.path.basename(png_path))
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                json.dump(coco, tmp)
                tmp_path = tmp.name
            try:
                rf_proj.single_upload(
                    image_path=png_path,
                    annotation_path=tmp_path,
                    annotation_labelmap={"iceberg": "iceberg"},
                    batch_name="otsu-prelabel",
                    overwrite=True,
                )
            except Exception as e:
                print(f"    [upload error] {e}")
                ok = False
            finally:
                os.unlink(tmp_path)
            time.sleep(args.delay)

        if ok:
            n_annotated += 1
        rows.append({"image_id": image_id, "filename": filename,
                     "n_polygons": len(polys),
                     "decision": "annotated" if ok else "upload_failed",
                     "note": ""})

    with open(args.log_csv, "w", newline="") as logf:
        writer = csv.DictWriter(logf, fieldnames=["image_id", "filename", "n_polygons", "decision", "note"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'─'*60}")
    print(f"Images processed  : {len(images)}")
    print(f"  annotated       : {n_annotated}"
          + (" (dry run)" if args.dry_run else ""))
    print(f"  empty (no ice)  : {n_empty}")
    print(f"  IC-filtered     : {n_skipped}")
    print(f"  no .tif found   : {n_no_tif}")
    print(f"  parse errors    : {n_parse_err}")
    print(f"Log saved to      : {args.log_csv}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
