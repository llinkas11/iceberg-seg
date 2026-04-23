"""
add_small_icebergs.py — Add missed small iceberg annotations to a local COCO export.

Rules:
  - Only process images that already have at least one annotation.
  - Mask = existing annotation pixels + 1-pixel dilation. New components that
    overlap this mask have the overlapping pixels trimmed first; after trimming
    if the component is still in range it is kept.
  - Keep only new components with area > MIN_AREA (default 1) AND < MAX_AREA (default 8).
    Components >= MAX_AREA are discarded (they are large enough to have been caught
    by the human annotator — flagging them as a small missed iceberg is incorrect).

Threshold strategy (two-pass):
  Pass 1 — for each annotated image, compute mean B08 reflectance inside existing
            annotation masks, grouped by SZA bin.
  Pass 2 — use per-bin mean as the threshold.

Usage:
  # Full run
  python add_small_icebergs.py \\
      --coco_json  roboflow_export/train/_annotations.coco.json \\
      --chips_dir  /mnt/research/.../S2-iceberg-areas/chips \\
      --out_dir    viz_small_icebergs/

  # Test on specific chips only (omits --out_json so nothing is saved)
  python add_small_icebergs.py \\
      --coco_json  roboflow_export/train/_annotations.coco.json \\
      --chips_dir  /mnt/research/.../S2-iceberg-areas/chips \\
      --out_dir    viz_small_icebergs_test/ \\
      --test_stems S2B_MSIL1C_20221005T141949_N0510_R096_T24WWU_20240727T202953_r9472_c4864 \\
                   S2B_MSIL1C_20181003T141019_N0500_R053_T25WEQ_20230619T164213_r0512_c1792 \\
      --dry_run

rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/add_small_icebergs.py smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import argparse
import json
import os
import re
from collections import defaultdict
from glob import glob

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.features import shapes as rio_shapes
from scipy.ndimage import binary_dilation, label as nd_label
from PIL import Image, ImageDraw
from shapely.geometry import shape
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Constants ─────────────────────────────────────────────────────────────────

B08_IDX   = 2       # chips stacked as B04/B03/B08 by chip_sentinel2.py
CHIP_SIZE = 256
MIN_AREA  = 1       # pixels — strictly greater than this (so >= 2)
MAX_AREA  = 8       # pixels — strictly less than this (so <= 7)
SZA_BINS  = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]

CHIP_RE    = re.compile(r"^(.+)_r(\d+)_c(\d+)(?:_B08)?\.(?:png|tif)$")
RF_HASH_RE = re.compile(r"_png\.rf\.[A-Za-z0-9]+\.png$")


# ── File helpers ───────────────────────────────────────────────────────────────

def strip_rf_hash(filename):
    return RF_HASH_RE.sub(".png", filename)


def find_tif(chips_dir, stem, row, col):
    fname = f"{stem}_r{row:04d}_c{col:04d}.tif"
    matches = glob(os.path.join(chips_dir, "**", fname), recursive=True)
    if matches:
        return matches[0]
    for p in glob(os.path.join(chips_dir, "**", f"{stem}_r*_c*.tif"), recursive=True):
        m = CHIP_RE.match(os.path.basename(p))
        if m and int(m.group(2)) == row and int(m.group(3)) == col:
            return p
    return None


def tif_to_sza_bin(tif_path, chips_dir):
    rel   = os.path.relpath(tif_path, chips_dir)
    parts = rel.split(os.sep)
    for part in parts:
        if part in SZA_BINS:
            return part
    return "unknown"


# ── Annotation helpers ─────────────────────────────────────────────────────────

def rasterize_coco_anns(anns, width=CHIP_SIZE, height=CHIP_SIZE):
    """Rasterize existing COCO segmentation polygons → (H, W) bool mask."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for ann in anns:
        for seg in ann.get("segmentation", []):
            flat = [float(v) for v in seg] if isinstance(seg, list) else []
            if len(flat) < 6:
                continue
            coords = [(int(round(flat[i])), int(round(flat[i+1])))
                      for i in range(0, len(flat) - 1, 2)]
            draw.polygon(coords, fill=1)
    return np.array(mask, dtype=bool)


def find_new_components(bright_mask, existing_mask, min_area=MIN_AREA, max_area=MAX_AREA):
    """
    Given a thresholded bright_mask and existing_mask (existing annotations):
      1. Dilate existing_mask by 1px → buffer zone
      2. Trim bright pixels that fall inside the buffer from each component
      3. Keep components where min_area < area < max_area (after trimming)

    Returns list of (row, col) pixel index arrays, one per kept component.
    """
    buffer_mask = binary_dilation(existing_mask)

    # Label ALL bright components before trimming
    labeled, n_components = nd_label(bright_mask)
    kept_pixel_sets = []

    for comp_id in range(1, n_components + 1):
        comp_pixels = (labeled == comp_id)
        # Trim pixels that overlap the buffer
        trimmed = comp_pixels & ~buffer_mask
        area    = int(trimmed.sum())
        if min_area < area < max_area:
            kept_pixel_sets.append(trimmed)

    return kept_pixel_sets


def pixels_to_coco_ann(pixel_mask, ann_id, image_id, category_id=1):
    """Convert a boolean pixel mask to a COCO annotation via rasterio shapes."""
    pixel_transform = Affine(1, 0, 0, 0, 1, 0)
    polys = []
    for geom_dict, val in rio_shapes(pixel_mask.astype(np.uint8), transform=pixel_transform):
        if val == 0:
            continue
        geom = shape(geom_dict)
        if not geom.is_empty:
            polys.append(geom)

    if not polys:
        return None

    # Take largest polygon (should only be one after nd_label)
    poly   = max(polys, key=lambda g: g.area)
    coords = list(poly.exterior.coords)
    seg    = [round(v, 2) for xy in coords for v in xy]
    xs     = [c[0] for c in coords]
    ys     = [c[1] for c in coords]
    x, y   = min(xs), min(ys)
    w, h   = max(xs) - x, max(ys) - y
    return {
        "id":           ann_id,
        "image_id":     image_id,
        "category_id":  category_id,
        "segmentation": [seg],
        "bbox":         [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
        "area":         round(float(poly.area), 2),
        "iscrowd":      0,
    }


# ── Visualisation ─────────────────────────────────────────────────────────────

def b08_to_grey(b08):
    p2, p98 = np.percentile(b08, (2, 98))
    if p98 > p2:
        return np.clip((b08 - p2) / (p98 - p2), 0, 1)
    return np.zeros_like(b08)


def save_viz(grey, existing_anns, new_pixel_masks, out_path, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grey, cmap="gray", vmin=0, vmax=1)

    # existing annotations — gold outline
    for ann in existing_anns:
        for seg in ann.get("segmentation", []):
            flat = [float(v) for v in seg] if isinstance(seg, list) else []
            if len(flat) < 6:
                continue
            xs = flat[0::2]
            ys = flat[1::2]
            ax.plot(xs + [xs[0]], ys + [ys[0]], color="#ffd700", linewidth=0.8)

    # new small icebergs — cyan dots/outlines
    for pmask in new_pixel_masks:
        rows, cols = np.where(pmask)
        ax.scatter(cols, rows, s=4, c="#00ffff", marker="s", linewidths=0)

    gold_patch = mpatches.Patch(edgecolor="#ffd700", facecolor="none", label="existing")
    cyan_patch = mpatches.Patch(color="#00ffff", label=f"new ({len(new_pixel_masks)})")
    ax.legend(handles=[gold_patch, cyan_patch], loc="lower right",
              fontsize=7, framealpha=0.6)
    ax.set_title(title, fontsize=7)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Add missed small iceberg annotations via per-bin NIR threshold"
    )
    parser.add_argument("--coco_json",  required=True)
    parser.add_argument("--chips_dir",
        default="/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/chips")
    parser.add_argument("--out_json",   default=None,
        help="Output COCO JSON path (default: _annotations_updated.coco.json alongside input)")
    parser.add_argument("--out_dir",    default="viz_small_icebergs")
    parser.add_argument("--min_area",   type=float, default=MIN_AREA,
        help=f"Minimum component area in pixels, exclusive (default {MIN_AREA})")
    parser.add_argument("--max_area",   type=float, default=MAX_AREA,
        help=f"Maximum component area in pixels, exclusive (default {MAX_AREA})")
    parser.add_argument("--test_stems", nargs="+", default=None,
        help="Only process chips whose filename contains one of these stems (for testing)")
    parser.add_argument("--dry_run",    action="store_true",
        help="Run full logic and save viz PNGs, but do NOT write updated COCO JSON")
    parser.add_argument("--save_all_viz", action="store_true",
        help="Save viz for every annotated chip, not just chips with new annotations")
    args = parser.parse_args()

    if args.out_json is None:
        base = args.coco_json.replace("_annotations.coco.json", "_annotations_updated.coco.json")
        args.out_json = base if base != args.coco_json else args.coco_json + ".updated.json"

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load COCO JSON ────────────────────────────────────────────────────────
    with open(args.coco_json) as f:
        coco = json.load(f)

    anns_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    n_images    = len(coco["images"])
    n_no_ann    = sum(1 for img in coco["images"] if not anns_by_image[img["id"]])
    n_annotated = n_images - n_no_ann

    print(f"Total images       : {n_images}")
    print(f"  with annotations : {n_annotated}  ← only these are processed")
    print(f"  no annotations   : {n_no_ann}  ← skipped")
    if args.test_stems:
        print(f"  TEST MODE — only stems : {args.test_stems}")
    if args.dry_run:
        print("  DRY RUN — no JSON will be saved")
    print(f"  size filter      : {args.min_area} < area < {args.max_area} pixels")
    print()

    # ── Pass 1: per-bin B08 threshold from existing annotation masks ──────────
    print("Pass 1: computing per-bin B08 threshold from existing annotations...")
    bin_b08_vals = defaultdict(list)
    img_meta     = {}

    for img_info in coco["images"]:
        existing_anns = anns_by_image[img_info["id"]]
        if not existing_anns:
            continue

        filename = strip_rf_hash(img_info["file_name"])
        m        = CHIP_RE.match(os.path.basename(filename))
        if m is None:
            continue

        stem, row, col = m.group(1), int(m.group(2)), int(m.group(3))

        # In test mode, skip chips not matching requested stems
        if args.test_stems:
            chip_key = f"{stem}_r{row}_c{col}"
            if not any(ts in chip_key for ts in args.test_stems):
                continue

        tif_path = find_tif(args.chips_dir, stem, row, col)
        if tif_path is None:
            continue

        with rasterio.open(tif_path) as src:
            chip = src.read().astype(np.float32)
        b08 = chip[B08_IDX]

        existing_mask = rasterize_coco_anns(
            existing_anns,
            img_info.get("width", CHIP_SIZE),
            img_info.get("height", CHIP_SIZE),
        )
        sza_bin     = tif_to_sza_bin(tif_path, args.chips_dir)
        inside_vals = b08[existing_mask]
        if inside_vals.size > 0:
            bin_b08_vals[sza_bin].extend(inside_vals.tolist())

        img_meta[img_info["id"]] = {
            "stem": stem, "row": row, "col": col,
            "tif_path": tif_path, "sza_bin": sza_bin,
            "b08": b08, "existing_mask": existing_mask,
            "existing_anns": existing_anns, "img_info": img_info,
        }

    # Per-bin thresholds
    bin_thresholds = {}
    all_vals_flat  = []
    print()
    print(f"  {'Bin':<14}  {'Mean B08 (threshold)':>22}  {'N pixels':>10}")
    print(f"  {'─'*52}")
    for bin_name in SZA_BINS + ["unknown"]:
        if bin_name not in bin_b08_vals:
            continue
        vals   = bin_b08_vals[bin_name]
        thresh = float(np.mean(vals))
        bin_thresholds[bin_name] = thresh
        all_vals_flat.extend(vals)
        print(f"  {bin_name:<14}  {thresh:>22.4f}  {len(vals):>10,}")
    print()

    if not bin_thresholds:
        print("ERROR: no annotated chips with matching .tif files found. Check --chips_dir.")
        return

    global_thresh = float(np.mean(all_vals_flat))

    # ── Pass 2: find new small icebergs ───────────────────────────────────────
    print("Pass 2: finding new small icebergs...")
    existing_ids    = {ann["id"] for ann in coco.get("annotations", [])}
    next_ann_id     = max(existing_ids) + 1 if existing_ids else 1
    new_annotations = []
    n_with_new      = 0
    total_new       = 0
    width_str       = len(str(len(img_meta)))

    for idx, (image_id, meta) in enumerate(img_meta.items()):
        sza_bin   = meta["sza_bin"]
        threshold = bin_thresholds.get(sza_bin, global_thresh)

        b08           = meta["b08"]
        existing_mask = meta["existing_mask"]
        existing_anns = meta["existing_anns"]
        img_info      = meta["img_info"]

        bright_mask    = (b08 >= threshold)
        new_components = find_new_components(
            bright_mask, existing_mask, args.min_area, args.max_area
        )

        tag    = f"{meta['stem'][:40]}_r{meta['row']}_c{meta['col']}"
        status = f"{len(new_components):>4} new" if new_components else "  no new"
        print(f"  [{idx+1:>{width_str}}/{len(img_meta)}] {status}  "
              f"[{sza_bin}  thr={threshold:.4f}]  {tag}")

        for pmask in new_components:
            ann = pixels_to_coco_ann(pmask, next_ann_id, image_id)
            if ann is not None:
                new_annotations.append(ann)
                next_ann_id += 1

        if new_components:
            n_with_new += 1
            total_new  += len(new_components)

        if new_components or args.save_all_viz:
            grey     = b08_to_grey(b08)
            title    = (f"{meta['stem'][-50:]}\n"
                        f"r{meta['row']}_c{meta['col']}  [{sza_bin}  thr={threshold:.4f}]  "
                        f"exist={len(existing_anns)}  new={len(new_components)}")
            out_path = os.path.join(
                args.out_dir,
                f"{meta['stem']}_r{meta['row']:04d}_c{meta['col']:04d}.png"
            )
            save_viz(grey, existing_anns, new_components, out_path, title)

    # ── Save updated COCO JSON ────────────────────────────────────────────────
    if not args.dry_run:
        updated_coco = dict(coco)
        updated_coco["annotations"] = coco.get("annotations", []) + new_annotations
        with open(args.out_json, "w") as f:
            json.dump(updated_coco, f)
    else:
        print("\n[DRY RUN] JSON not saved.")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_no_tif = n_annotated - len(img_meta)
    print()
    print(f"{'─'*52}")
    print(f"Images total        : {n_images}")
    print(f"  skipped (no ann)  : {n_no_ann}")
    print(f"  skipped (no tif)  : {n_no_tif}")
    print(f"  processed         : {len(img_meta)}")
    print(f"  chips with new    : {n_with_new}")
    print(f"Total new polygons  : {total_new}")
    if not args.dry_run:
        print(f"Updated JSON        : {args.out_json}")
    print(f"Viz PNGs            : {args.out_dir}/  ({n_with_new} chips)")
    print(f"{'─'*52}")


if __name__ == "__main__":
    main()