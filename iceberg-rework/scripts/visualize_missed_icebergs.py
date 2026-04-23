"""
visualize_missed_icebergs.py — Find unannotated bright objects (40-500 m RL) in chips.

For each Roboflow chip WITH existing annotations, applies Otsu thresholding to
B08, finds connected components that do NOT overlap annotations, and flags those
in the 40-500 m root-length range as potential missed icebergs.

Each visualization shows the original B08 image (left) alongside the annotated
chip with existing icebergs in gold and missed candidates in cyan (right).

Chips with no annotations are skipped.

Usage:
  python scripts/visualize_missed_icebergs.py
  python scripts/visualize_missed_icebergs.py --n_viz 30
"""

import argparse
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from scipy.ndimage import label as cc_label
from skimage.filters import threshold_otsu
from glob import glob

# ── Paths ────────────────────────────────────────────────────────────────────
SMISHRA = "/mnt/research/v.gomezgilyaspik/students/smishra/rework"
LLINKAS = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"

FILTERED_COCO = os.path.join(LLINKAS, "data/annotations_filtered.coco.json")
CHIPS_ROOT = os.path.join(SMISHRA, "chips")

CHIP_RE = re.compile(r"^(.+)_r(\d+)_c(\d+)(?:_B08)?\.(?:png|tif)$")
RF_HASH_RE = re.compile(r"_png\.rf\.[A-Za-z0-9]+\.png$")

PIXEL_AREA_M2 = 100.0
B08_IDX = 2


def strip_rf_hash(filename):
    return RF_HASH_RE.sub(".png", filename)


_tif_index = None

def _build_tif_index(chips_root):
    global _tif_index
    if _tif_index is not None:
        return _tif_index
    _tif_index = {}
    for path in glob(os.path.join(chips_root, "**", "*.tif"), recursive=True):
        _tif_index[os.path.basename(path)] = path
    return _tif_index

def find_tif(chips_root, stem, row, col):
    idx = _build_tif_index(chips_root)
    fname = f"{stem}_r{row:04d}_c{col:04d}.tif"
    return idx.get(fname)


def rasterize_annotations(anns, w=256, h=256):
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for ann in anns:
        for seg in ann.get("segmentation", []):
            if isinstance(seg, list) and len(seg) >= 6:
                coords = [(int(round(seg[i])), int(round(seg[i+1])))
                           for i in range(0, len(seg) - 1, 2)]
                draw.polygon(coords, fill=1)
    return np.array(mask, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Visualize missed icebergs")
    parser.add_argument("--coco_json", default=FILTERED_COCO)
    parser.add_argument("--chips_root", default=CHIPS_ROOT)
    parser.add_argument("--min_rl", type=float, default=40.0)
    parser.add_argument("--max_rl", type=float, default=500.0)
    parser.add_argument("--n_viz", type=int, default=30)
    parser.add_argument("--viz_dir", default=os.path.join(LLINKAS, "viz/missed_icebergs"))
    parser.add_argument("--out_csv", default=os.path.join(LLINKAS, "reference/missed_icebergs_summary.csv"))
    args = parser.parse_args()

    min_px = int((args.min_rl ** 2) / PIXEL_AREA_M2)
    max_px = int((args.max_rl ** 2) / PIXEL_AREA_M2)

    with open(args.coco_json) as f:
        coco = json.load(f)

    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    print(f"Loaded {len(coco['images'])} images from filtered COCO")
    print(f"Looking for missed icebergs: {min_px}-{max_px} pixels ({args.min_rl}-{args.max_rl} m RL)")

    results = []
    viz_count = 0
    total_missed = 0
    total_images_with_missed = 0
    skipped_null = 0

    for img_info in coco["images"]:
        img_id = img_info["id"]
        anns = ann_by_img.get(img_id, [])

        # Skip chips with no annotations
        if not anns:
            skipped_null += 1
            continue

        filename = strip_rf_hash(img_info["file_name"])
        m = CHIP_RE.match(os.path.basename(filename))
        if not m:
            continue

        stem = m.group(1)
        row = int(m.group(2))
        col = int(m.group(3))

        tif_path = find_tif(args.chips_root, stem, row, col)
        if tif_path is None:
            continue

        with rasterio.open(tif_path) as src:
            bands = src.read()
        b08 = bands[B08_IDX].astype(np.float32)

        w = img_info.get("width", 256)
        h = img_info.get("height", 256)

        ann_mask = rasterize_annotations(anns, w, h)

        if b08.max() == b08.min():
            continue
        try:
            thresh = threshold_otsu(b08)
        except ValueError:
            continue

        bright_mask = (b08 > thresh).astype(np.uint8)
        non_annotated_bright = bright_mask.copy()
        non_annotated_bright[ann_mask > 0] = 0

        labels, n_comp = cc_label(non_annotated_bright)

        missed = []
        missed_mask = np.zeros_like(b08, dtype=bool)
        for comp_id in range(1, n_comp + 1):
            comp_mask = labels == comp_id
            comp_px = int(comp_mask.sum())
            if min_px <= comp_px <= max_px:
                missed.append(comp_px)
                total_missed += 1
                missed_mask |= comp_mask

        n_missed = len(missed)
        if n_missed > 0:
            total_images_with_missed += 1

        results.append({
            "chip_stem": f"{stem}_r{row:04d}_c{col:04d}",
            "n_existing": len(anns),
            "n_missed": n_missed,
            "missed_areas_px": missed,
        })

        # Visualize: side-by-side original B08 + annotated/missed overlay
        if viz_count < args.n_viz and n_missed > 0:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Left: original B08
            axes[0].imshow(b08, cmap="gray", vmin=np.percentile(b08, 2), vmax=np.percentile(b08, 98))
            axes[0].set_title("Original B08", fontsize=10)
            axes[0].set_axis_off()

            # Right: B08 with annotations (gold) and missed (cyan)
            axes[1].imshow(b08, cmap="gray", vmin=np.percentile(b08, 2), vmax=np.percentile(b08, 98))

            # Existing annotations in gold
            for ann in anns:
                for seg in ann.get("segmentation", []):
                    if isinstance(seg, list) and len(seg) >= 6:
                        pts = np.array(seg).reshape(-1, 2)
                        poly = plt.Polygon(pts, fill=False, edgecolor="gold", linewidth=1.2)
                        axes[1].add_patch(poly)

            # Missed candidates as cyan overlay
            if missed_mask.any():
                overlay = np.zeros((*b08.shape, 4), dtype=np.float32)
                overlay[missed_mask] = [0, 1, 1, 0.5]  # cyan with alpha
                axes[1].imshow(overlay)

            gold_patch = mpatches.Patch(edgecolor="gold", facecolor="none", linewidth=1.2,
                                        label=f"Annotated ({len(anns)})")
            cyan_patch = mpatches.Patch(facecolor="cyan", alpha=0.5, label=f"Missed ({n_missed})")
            axes[1].legend(handles=[gold_patch, cyan_patch], loc="upper right", fontsize=8)
            axes[1].set_title("Annotated + Missed Candidates", fontsize=10)
            axes[1].set_axis_off()

            chip_label = f"{stem}_r{row:04d}_c{col:04d}"
            fig.suptitle(chip_label, fontsize=9, y=0.02)
            fig.tight_layout(rect=[0, 0.03, 1, 1])

            os.makedirs(args.viz_dir, exist_ok=True)
            fig.savefig(os.path.join(args.viz_dir, f"{chip_label}_missed.png"),
                        dpi=120, bbox_inches="tight")
            plt.close(fig)
            viz_count += 1

    # Summary
    print(f"\n{'='*60}")
    print("MISSED ICEBERGS SUMMARY")
    print(f"{'='*60}")
    print(f"Images analyzed (with annotations): {len(results)}")
    print(f"Images skipped (null):              {skipped_null}")
    print(f"Images with missed objects:         {total_images_with_missed}")
    print(f"Total missed candidates:            {total_missed}")

    if total_missed > 0:
        all_areas = []
        for r in results:
            all_areas.extend(r["missed_areas_px"])
        arr = np.array(all_areas) * PIXEL_AREA_M2
        rls = np.sqrt(arr)
        print(f"Missed area range:                 {arr.min():.0f} - {arr.max():.0f} m2")
        print(f"Missed RL range:                   {rls.min():.1f} - {rls.max():.1f} m")
        print(f"Missed RL median:                  {np.median(rls):.1f} m")

    print(f"\nVisualizations saved: {args.viz_dir}/ ({viz_count} images)")

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_stem", "n_existing_annotations", "n_missed_candidates", "missed_areas_m2"])
        for r in results:
            areas_str = ";".join(str(int(a * PIXEL_AREA_M2)) for a in r["missed_areas_px"])
            writer.writerow([r["chip_stem"], r["n_existing"], r["n_missed"], areas_str])
    print(f"Summary CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
