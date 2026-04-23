"""
prepare_new_training_data.py — Merge existing pkl training data with new Roboflow COCO annotations.

Inputs:
  1. Existing pkl files in S2UnetPlusPlus/train_validate_test/ (323 Fisser sza_lt65 chips)
  2. Roboflow COCO JSON export (new sza_gt65 annotated chips)
  3. chips_dir to find matching .tif files for new chips

Output:
  New pkl files in --out_dir, ready for train.py:
    X_train.pkl, Y_train.pkl
    X_validation.pkl, Y_validation.pkl
    x_test.pkl, y_test.pkl

Split: 80% train / 10% val / 10% test
Test set is stratified across SZA bins so each bin is equally represented.
This test set is the held-out comparison set for UNet++ vs Otsu vs CRF.

Mask encoding:
  0 = ocean (background)
  1 = iceberg (from Roboflow polygon annotations)
  2 = shadow (only in existing Fisser chips — not in new annotations)

Usage:
  python prepare_new_training_data.py \\
      --coco_dir   /path/to/roboflow_export/ \\
      --chips_dir  /mnt/research/.../S2-iceberg-areas/chips \\
      --existing_pkl S2UnetPlusPlus/train_validate_test \\
      --out_dir    S2UnetPlusPlus/train_validate_test_v2 \\
      [--seed 42]

rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/prepare_new_training_data.py smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import argparse
import csv
import json
import os
import pickle
import re
import random
from collections import defaultdict
from glob import glob

import numpy as np
from PIL import Image, ImageDraw
import rasterio

CHIP_SIZE = 256
CHIP_RE   = re.compile(r"^(.+)_r(\d+)_c(\d+)(?:_B08)?\.(?:png|tif)$")
RF_HASH_RE = re.compile(r"_png\.rf\.[A-Za-z0-9]+\.png$")

SZA_BINS = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def strip_rf_hash(filename):
    """
    Strip Roboflow augmentation hash from exported filenames.
    e.g. stem_r0512_c4096_B08_png.rf.PEdeAmEq5S3a7MDM4f2O.png
         → stem_r0512_c4096_B08.png
    Roboflow replaces the final '.' before extension with '_png' and appends
    '.rf.{hash}.{ext}', so reversing that means replacing '_png.rf.HASH.png'
    with '.png'.
    """
    cleaned = RF_HASH_RE.sub(".png", filename)
    return cleaned


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def find_tif(chips_dir, stem, row, col):
    """Find TIF chip by stem + row + col, searching all region/sza_bin subdirs."""
    fname = f"{stem}_r{row:04d}_c{col:04d}.tif"
    matches = glob(os.path.join(chips_dir, "**", fname), recursive=True)
    if matches:
        return matches[0]
    # fallback for non-zero-padded row/col
    for p in glob(os.path.join(chips_dir, "**", f"{stem}_r*_c*.tif"), recursive=True):
        m = CHIP_RE.match(os.path.basename(p))
        if m and int(m.group(2)) == row and int(m.group(3)) == col:
            return p
    return None


def tif_to_sza_bin(tif_path, chips_dir):
    """Extract sza_bin from tif path relative to chips_dir."""
    rel = os.path.relpath(tif_path, chips_dir)
    parts = rel.split(os.sep)
    for part in parts:
        if part in SZA_BINS:
            return part
    return "unknown"


def _seg_to_flat(seg):
    """
    Normalize a COCO segmentation entry to a flat list of floats.
    Handles: list of numbers, list of strings, or a single string
    of space/comma-separated numbers (e.g. Roboflow 'c x1 y1 ...' format).
    """
    if isinstance(seg, str):
        # Drop any leading non-numeric tokens (e.g. 'c' in SVG-like exports)
        parts = seg.replace(",", " ").split()
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                pass
        return nums
    # Already a list — elements may still be strings
    return [float(v) for v in seg]


def polygons_to_mask(segmentations, width=CHIP_SIZE, height=CHIP_SIZE):
    """
    Rasterize COCO segmentation polygons to a 2D uint8 mask.
    segmentations: list of [x1,y1,x2,y2,...] flat lists (COCO format)
    Returns (H, W) uint8 array with 1 where iceberg, 0 elsewhere.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for seg in segmentations:
        flat = _seg_to_flat(seg)
        if len(flat) < 6:
            continue
        coords = [(int(round(flat[i])), int(round(flat[i+1]))) for i in range(0, len(flat) - 1, 2)]
        draw.polygon(coords, fill=1)
    return np.array(mask, dtype=np.uint8)


# ─── Load COCO annotations ────────────────────────────────────────────────────

def load_coco_chips(coco_dir, chips_dir):
    """
    Load all chips from a Roboflow COCO export directory.
    Roboflow exports train/_annotations.coco.json, valid/..., test/...
    Returns list of {X: (3,256,256) float32, Y: (256,256) uint8, sza_bin, stem}
    """
    chips = []
    splits_found = []

    for split_name in ["train", "valid", "test"]:
        json_path = os.path.join(coco_dir, split_name, "_annotations.coco.json")
        if not os.path.exists(json_path):
            # also try root level
            json_path = os.path.join(coco_dir, "_annotations.coco.json")
            if not os.path.exists(json_path):
                continue

        with open(json_path) as f:
            coco = json.load(f)

        # Build image_id → annotations map
        ann_by_image = defaultdict(list)
        for ann in coco.get("annotations", []):
            ann_by_image[ann["image_id"]].append(ann)

        splits_found.append(split_name)
        print(f"  {split_name}: {len(coco['images'])} images")

        for img_info in coco["images"]:
            filename = strip_rf_hash(img_info["file_name"])
            m = CHIP_RE.match(os.path.basename(filename))
            if m is None:
                print(f"    SKIP (unparseable): {filename}")
                continue

            stem = m.group(1)
            row  = int(m.group(2))
            col  = int(m.group(3))

            tif_path = find_tif(chips_dir, stem, row, col)
            if tif_path is None:
                print(f"    NO TIF: {filename}")
                continue

            # Load 3-band TIF
            with rasterio.open(tif_path) as src:
                X = src.read().astype(np.float32)   # (3, H, W)

            if X.shape != (3, CHIP_SIZE, CHIP_SIZE):
                print(f"    SHAPE MISMATCH {X.shape}: {filename}")
                continue

            # Rasterize annotations → mask
            anns  = ann_by_image[img_info["id"]]
            segs  = [s for ann in anns for s in ann.get("segmentation", [])]
            w, h  = img_info.get("width", CHIP_SIZE), img_info.get("height", CHIP_SIZE)
            mask  = polygons_to_mask(segs, width=w, height=h)

            sza_bin = tif_to_sza_bin(tif_path, chips_dir)

            chips.append({
                "X":        X,
                "Y":        mask,
                "sza_bin":  sza_bin,
                "stem":     stem,
                "chip_stem": f"{stem}_r{row:04d}_c{col:04d}",
                "tif_path": os.path.abspath(tif_path),
            })

    print(f"  Loaded {len(chips)} chips from COCO export (splits: {splits_found})")
    return chips


# ─── Load existing pkl chips ──────────────────────────────────────────────────

def load_existing_pkl(pkl_dir):
    """
    Load existing pkl arrays and wrap as chip dicts.
    All existing chips are assumed sza_lt65 (Fisser annotations).
    Masks are already multi-class (0=ocean, 1=iceberg, 2=shadow).
    """
    chips = []
    splits = [
        ("X_train.pkl",      "Y_train.pkl"),
        ("X_validation.pkl", "Y_validation.pkl"),
        ("x_test.pkl",       "y_test.pkl"),
    ]
    for x_file, y_file in splits:
        xp = os.path.join(pkl_dir, x_file)
        yp = os.path.join(pkl_dir, y_file)
        if not os.path.exists(xp) or not os.path.exists(yp):
            print(f"  Missing: {x_file} or {y_file} -- skipping")
            continue
        X = load_pkl(xp)   # (N, 3, 256, 256)
        Y = load_pkl(yp)   # (N, 1, 256, 256) or (N, 256, 256)
        if Y.ndim == 4:
            Y = Y[:, 0, :, :]   # (N, 256, 256)
        print(f"  {x_file}: {len(X)} chips")
        for i in range(len(X)):
            chips.append({
                "X":       X[i],           # (3, 256, 256) float32
                "Y":       Y[i],           # (256, 256) uint8/int
                "sza_bin": "sza_lt65",
                "stem":    f"fisser_{i:04d}",
            })
    print(f"  Total existing pkl chips: {len(chips)}")
    return chips


# ─── Stratified split ─────────────────────────────────────────────────────────

def stratified_split(chips, test_frac=0.10, val_frac=0.10, seed=42):
    """
    Split chips into train/val/test.
    Test set is drawn evenly from each SZA bin (stratified).
    Val set is drawn randomly from remaining.
    """
    rng = random.Random(seed)

    # Group by SZA bin
    by_bin = defaultdict(list)
    for i, c in enumerate(chips):
        by_bin[c["sza_bin"]].append(i)

    print("\nChips per SZA bin:")
    for b in SZA_BINS + ["unknown"]:
        if b in by_bin:
            print(f"  {b}: {len(by_bin[b])}")

    # Draw test set: equal number from each bin
    n_total     = len(chips)
    n_test      = max(1, round(n_total * test_frac))
    bins_present = [b for b in SZA_BINS if b in by_bin]
    n_per_bin   = max(1, n_test // len(bins_present))

    test_idx = []
    for b in bins_present:
        indices = by_bin[b].copy()
        rng.shuffle(indices)
        take = min(n_per_bin, len(indices))
        test_idx.extend(indices[:take])
        by_bin[b] = indices[take:]   # remove taken from pool

    test_idx = set(test_idx)
    remaining = [i for i in range(n_total) if i not in test_idx]
    rng.shuffle(remaining)

    n_val   = max(1, round(n_total * val_frac))
    val_idx = set(remaining[:n_val])
    train_idx = [i for i in remaining if i not in val_idx]

    print(f"\nSplit: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print("Test set SZA bin distribution:")
    test_bins = defaultdict(int)
    for i in test_idx:
        test_bins[chips[i]["sza_bin"]] += 1
    for b, n in sorted(test_bins.items()):
        print(f"  {b}: {n}")

    return list(train_idx), list(val_idx), list(test_idx)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge existing pkl + Roboflow COCO annotations into new training pkls"
    )
    parser.add_argument("--coco_dir",      required=True,
        help="Roboflow COCO export directory (contains train/valid/test subdirs)")
    parser.add_argument("--chips_dir",
        default="/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/chips",
        help="Root chips directory to find .tif files")
    parser.add_argument("--existing_pkl",
        default="S2UnetPlusPlus/train_validate_test",
        help="Directory containing existing X_train.pkl etc.")
    parser.add_argument("--out_dir",
        default="/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/train_validate_test_v2",
        help="Output directory for new pkl files")
    parser.add_argument("--test_frac",  type=float, default=0.10)
    parser.add_argument("--val_frac",   type=float, default=0.10)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load all chips ────────────────────────────────────────────────────────
    print("Loading existing pkl chips (Fisser sza_lt65)...")
    existing = load_existing_pkl(args.existing_pkl)

    print("\nLoading new Roboflow COCO chips...")
    new_chips = load_coco_chips(args.coco_dir, args.chips_dir)

    all_chips = existing + new_chips
    print(f"\nTotal chips: {len(all_chips)}")

    # ── Stratified split ──────────────────────────────────────────────────────
    train_idx, val_idx, test_idx = stratified_split(
        all_chips, args.test_frac, args.val_frac, args.seed
    )

    # ── Pack into arrays ──────────────────────────────────────────────────────
    def pack(indices):
        X = np.stack([all_chips[i]["X"] for i in indices], axis=0)   # (N,3,256,256)
        Y = np.stack([all_chips[i]["Y"] for i in indices], axis=0)   # (N,256,256)
        Y = Y[:, np.newaxis, :, :]                                    # (N,1,256,256)
        return X.astype(np.float32), Y.astype(np.int64)

    X_train, Y_train = pack(train_idx)
    X_val,   Y_val   = pack(val_idx)
    X_test,  Y_test  = pack(test_idx)

    print(f"\nArray shapes:")
    print(f"  X_train: {X_train.shape}  Y_train: {Y_train.shape}")
    print(f"  X_val:   {X_val.shape}    Y_val:   {Y_val.shape}")
    print(f"  X_test:  {X_test.shape}   Y_test:  {Y_test.shape}")

    # Class distribution in train
    flat = Y_train.flatten()
    total = flat.size
    for cls, name in [(0, "ocean"), (1, "iceberg"), (2, "shadow")]:
        n = int((flat == cls).sum())
        print(f"  train class {cls} ({name}): {n/total*100:.1f}%")

    # ── Save ──────────────────────────────────────────────────────────────────
    # train.py expects files in {data_dir}/train_validate_test/
    # and uses lowercase x_test.pkl / y_test.pkl
    split_dir = os.path.join(args.out_dir, "train_validate_test")
    os.makedirs(split_dir, exist_ok=True)

    save_pkl(X_train, os.path.join(split_dir, "X_train.pkl"))
    save_pkl(Y_train, os.path.join(split_dir, "Y_train.pkl"))
    save_pkl(X_val,   os.path.join(split_dir, "X_validation.pkl"))
    save_pkl(Y_val,   os.path.join(split_dir, "Y_validation.pkl"))
    save_pkl(X_test,  os.path.join(split_dir, "x_test.pkl"))
    save_pkl(Y_test,  os.path.join(split_dir, "y_test.pkl"))

    # ── Save split log ────────────────────────────────────────────────────────
    log_path = os.path.join(args.out_dir, "split_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "index", "stem", "chip_stem", "tif_path", "sza_bin"])
        writer.writeheader()
        for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            for idx in indices:
                chip = all_chips[idx]
                writer.writerow({
                    "split":     split_name,
                    "index":     idx,
                    "stem":      chip["stem"],
                    "chip_stem": chip.get("chip_stem", chip["stem"]),  # fisser chips keep stem
                    "tif_path":  chip.get("tif_path", ""),
                    "sza_bin":   chip["sza_bin"],
                })
    print(f"Split log saved to: {log_path}")
    print(f"\nSaved pkl files to: {split_dir}")
    print(f"\nRetrain without augmentation:")
    print(f"  python ~/S2-iceberg-areas/train.py --mode s2 \\")
    print(f"      --data_dir {args.out_dir} \\")
    print(f"      --out_dir /mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/runs/s2_v2_noaug")
    print(f"\nRetrain with augmentation (add --augment flag to train.py, or it augments by default):")
    print(f"  python ~/S2-iceberg-areas/train.py --mode s2 \\")
    print(f"      --data_dir {args.out_dir} \\")
    print(f"      --out_dir /mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/runs/s2_v2_aug")


if __name__ == "__main__":
    main()
