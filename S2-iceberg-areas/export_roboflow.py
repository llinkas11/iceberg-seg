"""
export_roboflow.py — Convert UNet++ predicted masks → COCO JSON + PNGs for Roboflow upload.

Packages the test-set chips (B08 grayscale PNGs) + model predictions as polygon
annotations in COCO segmentation format, ready to zip and upload to Roboflow.

Usage:
  python export_roboflow.py \
      --masks   predictions/s2_exp1/predicted_masks/predicted_masks.npy \
      --chips   S2UnetPlusPlus/train_validate_test/x_test.pkl \
      --out_dir roboflow_upload

Then zip roboflow_upload/ and import into Roboflow as a COCO segmentation dataset.
"""

import os
import json
import pickle
import argparse
import numpy as np
import cv2
from PIL import Image


DEFAULT_CLASS_MAP = {1: "iceberg"}   # class indices → labels


def parse_class_map(entries):
    class_map = {}
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"Invalid class map entry '{entry}'. Use <value>:<name>.")
        value_str, class_name = entry.split(":", 1)
        class_map[int(value_str)] = class_name.strip()
    return class_map


def merge_mask_values(mask, values):
    return np.isin(mask, list(values)).astype(np.uint8)


def mask_to_polygons(mask_2d, class_id, min_area=50):
    """Extract OpenCV polygons for a single class from an integer mask."""
    binary = (mask_2d == class_id).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        # Flatten to [x1,y1,x2,y2,...] — COCO segmentation format
        poly = cnt.squeeze().tolist()
        if isinstance(poly[0], int):   # single-point contour
            continue
        flat = [coord for pt in poly for coord in pt]
        if len(flat) >= 6:             # need at least 3 points
            polys.append(flat)
    return polys


def chip_to_b08_png(chip_chw, b08_idx=2):
    """Convert (3, H, W) float32 chip → 8-bit B08 grayscale PNG array."""
    b08 = chip_chw[b08_idx]
    valid = b08[b08 > 0]
    if valid.size == 0:
        return np.zeros(b08.shape, dtype=np.uint8)
    lo, hi = np.percentile(valid, [2, 98])
    b08_u8 = np.clip((b08 - lo) / (hi - lo + 1e-6), 0, 1)
    return (b08_u8 * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks",   required=True,
                        help="predicted_masks.npy from predict.py")
    parser.add_argument("--chips",   required=True,
                        help="x_test.pkl (or X_train.pkl / X_validation.pkl)")
    parser.add_argument("--out_dir", default="roboflow_upload")
    parser.add_argument("--min_area", type=int, default=50,
                        help="Minimum polygon area in pixels (default 50)")
    parser.add_argument("--class-map", nargs="+",
                        help="Mask value to class name, e.g. 1:iceberg. Defaults to iceberg-only.")
    parser.add_argument("--merge-values", nargs="*", type=int, default=None,
                        help="Optional mask values to merge into one output class, e.g. 3 4")
    parser.add_argument("--merge-name", default="iceberg",
                        help="Output class name when using --merge-values. Default: iceberg")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.merge_values:
        class_map = {1: args.merge_name}
        merge_values = set(args.merge_values)
    else:
        class_map = parse_class_map(args.class_map) if args.class_map else DEFAULT_CLASS_MAP
        merge_values = None

    masks = np.load(args.masks)                          # (N, H, W) int
    chips = pickle.load(open(args.chips, "rb"))          # (N, 3, H, W) float32
    assert len(masks) == len(chips), \
        f"Mask count {len(masks)} != chip count {len(chips)}"

    H, W = masks.shape[1], masks.shape[2]

    coco = {
        "info": {"description": "UNet++ iceberg predictions — pre-labels for Roboflow"},
        "categories": [{"id": cid, "name": name, "supercategory": "iceberg"}
                       for cid, name in class_map.items()],
        "images": [],
        "annotations": [],
    }

    ann_id = 1

    for i, (chip, mask) in enumerate(zip(chips, masks)):
        fname = f"chip_{i:03d}.png"
        if merge_values:
            mask = merge_mask_values(mask, merge_values)

        # Save B08 grayscale PNG
        b08_u8 = chip_to_b08_png(chip)
        Image.fromarray(b08_u8).save(os.path.join(args.out_dir, fname))

        coco["images"].append({
            "id": i,
            "file_name": fname,
            "width": W,
            "height": H,
        })

        # One annotation entry per polygon per class
        for class_id in class_map:
            polys = mask_to_polygons(mask, class_id, args.min_area)
            for poly in polys:
                xs = poly[0::2]
                ys = poly[1::2]
                x0, y0 = min(xs), min(ys)
                bw, bh = max(xs) - x0, max(ys) - y0
                area = 0.5 * abs(sum(
                    xs[j] * ys[(j+1) % len(xs)] - xs[(j+1) % len(xs)] * ys[j]
                    for j in range(len(xs))
                ))
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": i,
                    "category_id": class_id,
                    "segmentation": [poly],
                    "area": round(area, 1),
                    "bbox": [x0, y0, bw, bh],
                    "iscrowd": 0,
                })
                ann_id += 1

    json_path = os.path.join(args.out_dir, "_annotations.coco.json")
    with open(json_path, "w") as f:
        json.dump(coco, f)

    n_images = len(coco["images"])
    n_anns   = len(coco["annotations"])
    print(f"Saved {n_images} images + {n_anns} polygon annotations → {args.out_dir}/")
    print(f"  {json_path}")
    print(f"\nNext: zip {args.out_dir}/ and import into Roboflow as 'COCO Segmentation'")


if __name__ == "__main__":
    main()
