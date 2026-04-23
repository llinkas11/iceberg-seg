"""
export_manual_annotations_roboflow.py
Convert manual ground-truth masks (GeoTIFF) → COCO JSON + B08 PNGs for Roboflow.

Reads every *_ground_truth.tif in masks_dir, finds the matching image in imgs_dir,
and writes a COCO segmentation dataset ready for Roboflow import.

Usage:
  python export_manual_annotations_roboflow.py \
      --imgs_dir  S2UnetPlusPlus/imgs \
      --masks_dir S2UnetPlusPlus/masks \
      --out_dir   roboflow_manual_upload

Then zip roboflow_manual_upload/ and import into Roboflow as "COCO Segmentation".
"""

import os
import json
import glob
import argparse
import numpy as np
import cv2
import rasterio
from PIL import Image

CLASS_NAMES = {1: "iceberg"}   # binary: iceberg only (shadow merged)


def mask_to_polygons(mask_2d, class_id, min_area=50):
    """Extract COCO polygons for one class from an integer mask."""
    binary = (mask_2d == class_id).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        pts = cnt.squeeze()
        if pts.ndim < 2:          # single-point contour
            continue
        flat = pts.flatten().tolist()
        if len(flat) >= 6:        # need ≥ 3 points
            polys.append(flat)
    return polys


def tif_to_b08_png(img_path, b08_idx=2):
    """
    Read a 3-band GeoTIFF (B04/B03/B08) and return an 8-bit B08 grayscale array.
    b08_idx is 0-based (default 2 = third band).
    """
    with rasterio.open(img_path) as src:
        b08 = src.read(b08_idx + 1).astype(np.float32)   # rasterio bands are 1-indexed
    valid = b08[b08 > 0]
    if valid.size == 0:
        return np.zeros(b08.shape, dtype=np.uint8)
    lo, hi = np.percentile(valid, [2, 98])
    b08_u8 = np.clip((b08 - lo) / (hi - lo + 1e-6), 0, 1)
    return (b08_u8 * 255).astype(np.uint8)


def read_mask(mask_path):
    """Read a single-band integer GeoTIFF mask and return a 2-D numpy array."""
    with rasterio.open(mask_path) as src:
        return src.read(1).astype(np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_dir",  default="S2UnetPlusPlus/imgs",
                        help="Directory containing 3-band image .tif chips")
    parser.add_argument("--masks_dir", default="S2UnetPlusPlus/masks",
                        help="Directory containing *_ground_truth.tif masks")
    parser.add_argument("--out_dir",   default="roboflow_manual_upload")
    parser.add_argument("--b08_idx",   type=int, default=2,
                        help="0-based band index for B08 in the image chips (default 2)")
    parser.add_argument("--min_area",  type=int, default=50,
                        help="Minimum polygon area in pixels (default 50)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Collect all ground-truth masks
    mask_paths = sorted(glob.glob(os.path.join(args.masks_dir, "*_ground_truth.tif")))
    if not mask_paths:
        print(f"No *_ground_truth.tif files found in {args.masks_dir}")
        return

    coco = {
        "info": {"description": "Manual iceberg annotations — Roboflow pre-labels"},
        "categories": [
            {"id": cid, "name": name, "supercategory": "iceberg"}
            for cid, name in CLASS_NAMES.items()
        ],
        "images": [],
        "annotations": [],
    }

    ann_id   = 1
    img_id   = 0
    skipped  = 0

    for mask_path in mask_paths:
        mask_fname = os.path.basename(mask_path)                  # e.g. ...pB5_11_38__ground_truth.tif
        # Derive matching image name: strip trailing "_ground_truth"
        img_stem   = mask_fname.replace("_ground_truth.tif", ".tif")
        img_path   = os.path.join(args.imgs_dir, img_stem)

        if not os.path.exists(img_path):
            print(f"  [skip] no matching image for {mask_fname}")
            skipped += 1
            continue

        # Read image → B08 PNG
        b08_u8 = tif_to_b08_png(img_path, args.b08_idx)
        H, W   = b08_u8.shape

        out_fname = img_stem.replace(".tif", ".png")
        Image.fromarray(b08_u8).save(os.path.join(args.out_dir, out_fname))

        coco["images"].append({
            "id":        img_id,
            "file_name": out_fname,
            "width":     W,
            "height":    H,
        })

        # Read mask → polygons per class
        mask = read_mask(mask_path)

        for class_id in CLASS_NAMES:
            polys = mask_to_polygons(mask, class_id, args.min_area)
            for poly in polys:
                xs = poly[0::2]
                ys = poly[1::2]
                x0, y0 = min(xs), min(ys)
                bw, bh = max(xs) - x0, max(ys) - y0
                area = 0.5 * abs(sum(
                    xs[j] * ys[(j + 1) % len(xs)] - xs[(j + 1) % len(xs)] * ys[j]
                    for j in range(len(xs))
                ))
                coco["annotations"].append({
                    "id":          ann_id,
                    "image_id":    img_id,
                    "category_id": class_id,
                    "segmentation": [poly],
                    "area":        round(area, 1),
                    "bbox":        [x0, y0, bw, bh],
                    "iscrowd":     0,
                })
                ann_id += 1

        img_id += 1

    json_path = os.path.join(args.out_dir, "_annotations.coco.json")
    with open(json_path, "w") as f:
        json.dump(coco, f)

    print(f"Done — {img_id} images, {ann_id - 1} polygon annotations → {args.out_dir}/")
    if skipped:
        print(f"  ({skipped} masks skipped — no matching image found)")
    print(f"\nNext steps:")
    print(f"  zip -r roboflow_manual_upload.zip {args.out_dir}/")
    print(f"  Import into Roboflow as 'COCO Segmentation'")


if __name__ == "__main__":
    main()
