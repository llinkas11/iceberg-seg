"""
build_clean_dataset.py: Merge Fisser + Roboflow data into a versioned pkl split.

Default pipeline (canonical v4_clean):
  - Shadow (class 2) merged into iceberg (class 1): binary masks.
  - 40 m root-length cutoff (sources from filter_small_icebergs.py output).
  - IC chip-drop and IC pixel-mask in training (sources from filter_quality.py).
  - Met data attached as informational fields (no chip removal).

Three flags toggle preprocessing for the v4_raw companion + Phase A subsets:
  --skip_size_filter   Use raw COCO + raw Fisser pkls (no 40 m cutoff).
  --skip_ic_mask       Disable Fisser IC chip-drop AND training-time IC mask.
  --filter_sza_bin BIN Restrict the manifest to one SZA bin (sza_lt65, etc.).

Six canonical manifests are produced by combining these flags:
  v4_clean             (no flags)
  v4_clean_lt65        (--filter_sza_bin sza_lt65)
  v4_raw_lt65          (--skip_size_filter --skip_ic_mask --filter_sza_bin sza_lt65)
  v4_raw               (--skip_size_filter --skip_ic_mask)
  v4_clean_lt65_plus_nulls / v4_raw_lt65_plus_nulls
                       (built by build_lt65_nulls.py --merge_into_manifest)

Outputs (per --out_dir):
  manifest.json         Single dataset identity. Includes chips_sha; downstream
                        runs stamp this into eval CSVs.
  split_log.csv         Per-chip metadata (split, pkl_position, ic_masked, met).
  train_validate_test/  X_train.pkl, Y_train.pkl, X_validation.pkl,
                        Y_validation.pkl, x_test.pkl, y_test.pkl.

Output masks are binary: 0 = background, 1 = iceberg. Stratified 65/15/25
train/val/test split (test capped at 57 per SZA bin for cross-bin balance).

Usage:
  python scripts/build_clean_dataset.py                       # canonical v4_clean
  python scripts/build_clean_dataset.py --filter_sza_bin sza_lt65 \
      --out_dir data/v4_clean_lt65 --manifest_id v4_clean_lt65
"""

import argparse
import csv
import json
import os
import pickle
import random
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from glob import glob

import numpy as np
import rasterio
from rasterio.transform import Affine
from PIL import Image, ImageDraw
from scipy.ndimage import label as cc_label

from _method_common import get_git_sha, sha256_of_file, sha256_of_text

# ── Paths ────────────────────────────────────────────────────────────────────
SMISHRA = "/mnt/research/v.gomezgilyaspik/students/smishra/rework"
LLINKAS = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"

FILTERED_COCO = os.path.join(LLINKAS, "data/annotations_filtered.coco.json")
FISSER_FILTERED = os.path.join(LLINKAS, "data/fisser_filtered")
QUALITY_CSV = os.path.join(LLINKAS, "reference/fisser_quality_filter.csv")
MET_CSV = os.path.join(LLINKAS, "reference/met_data.csv")
CHIPS_ROOT = os.path.join(SMISHRA, "chips")

# Raw (pre-40m-filter) sources, used when --skip_size_filter is set.
# These are the inputs that filter_small_icebergs.py reads from.
RAW_COCO = os.path.join(SMISHRA, "data/roboflow_export/train/train/_annotations.coco.json")
RAW_FISSER = os.path.join(SMISHRA, "data/fisser_original/train_validate_test")

CHIP_SIZE = 256
PIXEL_AREA_M2 = 100.0
B08_THRESHOLD = 0.22   # Fisser's 0.12 + 0.10 DN offset (chip_sentinel2.py does not subtract offset)
IC_THRESHOLD = 0.15    # Fisser et al. (2024) IC cutoff
SZA_BINS = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]

CHIP_RE = re.compile(r"^(.+)_r(\d+)_c(\d+)(?:_B08)?\.(?:png|tif)$")
RF_HASH_RE = re.compile(r"_png\.rf\.[A-Za-z0-9]+\.png$")


def strip_rf_hash(fn):
    return RF_HASH_RE.sub(".png", fn)


_tif_index = None

def _build_tif_index(chips_root):
    """Build a filename -> path lookup once to avoid repeated recursive globs."""
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


def tif_to_sza_bin(tif_path, chips_root):
    rel = os.path.relpath(tif_path, chips_root)
    for part in rel.split(os.sep):
        if part in SZA_BINS:
            return part
    return "unknown"


def polygons_to_mask(segmentations, width=CHIP_SIZE, height=CHIP_SIZE):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for seg in segmentations:
        if isinstance(seg, list) and len(seg) >= 6:
            # Coerce coords to float first: raw Roboflow exports occasionally
            # mix string coords with ints in a single polygon (e.g. '170.07').
            coords = [(int(round(float(seg[i]))), int(round(float(seg[i + 1]))))
                       for i in range(0, len(seg) - 1, 2)]
            draw.polygon(coords, fill=1)
    return np.array(mask, dtype=np.uint8)


def load_quality_filter(csv_path):
    """Return set of global_index values that PASS IC filter."""
    passing = set()
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["pass_ic"] == "True":
                passing.add(int(row["global_index"]))
    return passing


SYNTHETIC_FISSER_DIR = os.path.join(LLINKAS, "data/raw_chips/fisser")

# The Fisser pickles do not carry CRS or transform. We write a 3-band GeoTIFF
# with a 10 m pixel transform and no CRS so that rasterio can open it, and so
# that polygonisation downstream produces areas in square metres. All Fisser
# chips share the same fake origin; only the pixel grid matters.
FISSER_FAKE_TRANSFORM = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 2560.0)


def write_synthetic_fisser_tif(chip_stem, X, out_dir=SYNTHETIC_FISSER_DIR):
    """
    Write a 3-band float32 GeoTIFF for a Fisser chip so downstream tooling
    has a real tif to open. X is (3, 256, 256) reflectance.
    Returns the absolute path to the written tif.
    """
    os.makedirs(out_dir, exist_ok=True)
    tif_path = os.path.join(out_dir, f"{chip_stem}.tif")
    if os.path.exists(tif_path):
        # Already written in a prior run. Skip re-writing to keep SHAs stable.
        return tif_path

    C, H, W = X.shape
    profile = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "count":     C,
        "height":    H,
        "width":     W,
        "crs":       None,
        "transform": FISSER_FAKE_TRANSFORM,
        "nodata":    None,
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(X.astype(np.float32))
    return tif_path


def load_met_data(csv_path):
    """Return dict of chip_stem -> met record."""
    met = {}
    if not os.path.exists(csv_path):
        return met
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            met[row["chip_stem"]] = row
    return met


# ---------------------------------------------------------------------------
# Manifest helpers (hashing + git SHA now live in _method_common)
# ---------------------------------------------------------------------------

def build_manifest_chip_rows(all_chips, train_idx, val_idx, test_idx):
    """
    Produce a list of chip records aligned to the final split + pkl_position
    ordering used when writing the pkls and split_log. Each record contains
    identity fields and a tif_sha for downstream hashing.
    """
    rows = []
    for split_name, indices in [
        ("train", list(train_idx)),
        ("val",   list(val_idx)),
        ("test",  list(test_idx)),
    ]:
        for pos, idx in enumerate(indices):
            c = all_chips[idx]
            tif_path = c.get("tif_path", "") or ""
            rows.append({
                "chip_stem":    c["chip_stem"],
                "stem":         c["stem"],
                "source":       c["source"],
                "sza_bin":      c["sza_bin"],
                "tif_path":     tif_path,
                "tif_sha":      sha256_of_file(tif_path),
                "n_icebergs":   int(c["n_icebergs"]),
                "has_iceberg":  bool(c["n_icebergs"] > 0),
                "ic_aware":     float(c.get("ic_aware", 0.0)),
                "wind_ms":      c.get("wind_ms", ""),
                "temp_c":       c.get("temp_c", ""),
                "split":        split_name,
                "pkl_position": pos,
            })
    return rows


def compute_chips_sha(chip_rows):
    """Stable hash over (chip_stem, tif_sha or '', split). Identity of the dataset."""
    tuples = sorted(
        (r["chip_stem"], r["tif_sha"] or "", r["split"]) for r in chip_rows
    )
    joined = "\n".join("|".join(t) for t in tuples)
    return sha256_of_text(joined)


def summarise_splits(chip_rows):
    """Count chips per split x (GT+ / GT0), and per SZA bin, for the manifest header."""
    counts_by_split = {}
    for split_name in ("train", "val", "test"):
        gtp = sum(1 for r in chip_rows if r["split"] == split_name and r["has_iceberg"])
        gt0 = sum(1 for r in chip_rows if r["split"] == split_name and not r["has_iceberg"])
        counts_by_split[split_name] = {"gt_positive": gtp, "gt_zero": gt0}
    return counts_by_split


def write_manifest(out_dir, manifest_id, args, chip_rows, chips_root):
    """
    Write manifest.json next to the pkls. This is the single source of truth
    for dataset identity; downstream training, inference, and evaluation all
    read it instead of the split_log.csv.
    """
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    chips_sha = compute_chips_sha(chip_rows)

    manifest = {
        "manifest_id":    manifest_id,
        "created_utc":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script":         os.path.basename(__file__),
        "git_sha":        get_git_sha(repo_dir),
        "chip_source":    chips_root,
        "split_policy": {
            "train_frac":  args.train_frac,
            "val_frac":    args.val_frac,
            "test_frac":   args.test_frac,
            "stratify_by": ["sza_bin"],
            "seed":        args.seed,
        },
        "filters": {
            "shadow_merge":      True,
            "root_length_min_m": 0 if args.skip_size_filter else 40,
            "ic_threshold":      None if args.skip_ic_mask else IC_THRESHOLD,
            "ic_mask_scope":     "none" if args.skip_ic_mask else "train_only",
            "b08_threshold_ic":  B08_THRESHOLD,
            "filter_sza_bin":    args.filter_sza_bin or "all",
            "skip_size_filter":  args.skip_size_filter,
            "skip_ic_mask":      args.skip_ic_mask,
        },
        "total_chips":     len(chip_rows),
        "counts_by_split": summarise_splits(chip_rows),
        "chips":           chip_rows,
        "chips_sha":       chips_sha,
    }

    out_path = os.path.join(out_dir, "manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return out_path, chips_sha


def main():
    parser = argparse.ArgumentParser(description="Build cleaned dataset with new split")
    parser.add_argument("--train_frac", type=float, default=0.65)
    parser.add_argument("--val_frac",   type=float, default=0.15)
    parser.add_argument("--test_frac",  type=float, default=0.25)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--out_dir",     default=os.path.join(LLINKAS, "data/v4_clean"))
    parser.add_argument("--manifest_id", default="v4_clean",
                        help="Manifest id written into manifest.json; should match --out_dir basename")
    parser.add_argument("--coco_json", default=None,
                        help="COCO annotations json. Defaults to filtered, or raw under --skip_size_filter.")
    parser.add_argument("--fisser_dir", default=None,
                        help="Fisser pkl dir. Defaults to filtered, or raw under --skip_size_filter.")
    parser.add_argument("--chips_root", default=CHIPS_ROOT)
    parser.add_argument("--quality_csv", default=QUALITY_CSV)
    parser.add_argument("--met_csv", default=MET_CSV)
    parser.add_argument("--skip_size_filter", action="store_true",
                        help="Use raw COCO + raw Fisser pkls. Disables the 40 m root-length cutoff.")
    parser.add_argument("--skip_ic_mask", action="store_true",
                        help="Disable both the Fisser IC chip-drop and the training-time IC pixel mask.")
    parser.add_argument("--filter_sza_bin", default=None, choices=SZA_BINS,
                        help="Restrict the dataset to a single SZA bin (e.g. sza_lt65).")
    args = parser.parse_args()

    # 1. Resolve sources based on --skip_size_filter
    if args.coco_json is None:
        args.coco_json = RAW_COCO if args.skip_size_filter else FILTERED_COCO
    if args.fisser_dir is None:
        args.fisser_dir = RAW_FISSER if args.skip_size_filter else FISSER_FILTERED

    print(f"COCO source : {args.coco_json}")
    print(f"Fisser dir  : {args.fisser_dir}")
    print(f"skip_size_filter = {args.skip_size_filter}")
    print(f"skip_ic_mask     = {args.skip_ic_mask}")
    print(f"filter_sza_bin   = {args.filter_sza_bin or 'all'}")

    rng = random.Random(args.seed)

    # 2. Load Fisser IC chip-drop list (skipped under --skip_ic_mask)
    if args.skip_ic_mask:
        passing_fisser = None
        print("Fisser IC chip-drop: SKIPPED (--skip_ic_mask)")
    else:
        passing_fisser = load_quality_filter(args.quality_csv)
        if passing_fisser is not None:
            print(f"Fisser IC filter: {len(passing_fisser)} chips pass")

    met = load_met_data(args.met_csv)

    # ── Load Roboflow COCO chips ─────────────────────────────────────────
    print("\nLoading Roboflow COCO chips (filtered annotations)...")
    with open(args.coco_json) as f:
        coco = json.load(f)

    # Filter to iceberg category only
    iceberg_cat_id = 2
    for cat in coco.get("categories", []):
        if cat["name"] == "iceberg":
            iceberg_cat_id = cat["id"]

    ann_by_img = {}
    for ann in coco["annotations"]:
        if ann.get("category_id") == iceberg_cat_id:
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

    coco_chips = []
    skipped = 0
    for img_info in coco["images"]:
        fn = strip_rf_hash(img_info["file_name"])
        m = CHIP_RE.match(os.path.basename(fn))
        if not m:
            skipped += 1
            continue

        stem = m.group(1)
        row = int(m.group(2))
        col = int(m.group(3))

        tif_path = find_tif(args.chips_root, stem, row, col)
        if tif_path is None:
            skipped += 1
            continue

        # The SZA bin is encoded in the tif path; resolve it before any heavy
        # work so --filter_sza_bin can short-circuit non-matching chips.
        sza_bin = tif_to_sza_bin(tif_path, args.chips_root)
        if args.filter_sza_bin and sza_bin != args.filter_sza_bin:
            continue

        with rasterio.open(tif_path) as src:
            X = src.read().astype(np.float32)

        if X.shape != (3, CHIP_SIZE, CHIP_SIZE):
            skipped += 1
            continue

        anns = ann_by_img.get(img_info["id"], [])
        segs = [s for ann in anns for s in ann.get("segmentation", [])]
        w = img_info.get("width", CHIP_SIZE)
        h = img_info.get("height", CHIP_SIZE)
        mask = polygons_to_mask(segs, w, h)

        chip_stem = f"{stem}_r{row:04d}_c{col:04d}"

        n_icebergs = 0
        if mask.sum() > 0:
            _, n_icebergs = cc_label(mask)

        met_rec = met.get(chip_stem, {})

        # Annotation-aware IC: fraction of non-annotated pixels above B08 threshold.
        # Only computed when the IC mask will actually consume it; skipping when
        # --skip_ic_mask is set saves a boolean pass over every chip.
        if args.skip_ic_mask:
            ic_aware = 0.0
        else:
            b08 = X[2]
            non_ann = (mask == 0)
            non_ann_count = int(non_ann.sum())
            bright_non_ann = int(((b08 >= B08_THRESHOLD) & non_ann).sum())
            ic_aware = bright_non_ann / non_ann_count if non_ann_count > 0 else 0.0

        coco_chips.append({
            "X": X, "Y": mask, "sza_bin": sza_bin, "stem": stem,
            "chip_stem": chip_stem, "tif_path": os.path.abspath(tif_path),
            "source": "roboflow", "n_icebergs": n_icebergs,
            "wind_ms": met_rec.get("wind_speed_10m", ""),
            "temp_c": met_rec.get("temp_2m", ""),
            "ic_aware": ic_aware,
        })

    print(f"  Loaded {len(coco_chips)} Roboflow chips (skipped {skipped})")

    # ── Load Fisser chips (filtered masks) ───────────────────────────────
    # Every Fisser chip is sza_lt65 by construction. When --filter_sza_bin
    # excludes lt65, the loader is a guaranteed no-op; short-circuit it.
    fisser_chips = []
    if args.filter_sza_bin and args.filter_sza_bin != "sza_lt65":
        print(f"\nSkipping Fisser loader: --filter_sza_bin={args.filter_sza_bin} "
              f"excludes all Fisser chips.")
    else:
        print("\nLoading Fisser chips (filtered masks)...")
        splits = [
            ("X_train.pkl", "Y_train.pkl"),
            ("X_validation.pkl", "Y_validation.pkl"),
            ("x_test.pkl", "y_test.pkl"),
        ]

        fisser_idx = 0
        for x_file, y_file in splits:
            xp = os.path.join(args.fisser_dir, x_file)
            yp = os.path.join(args.fisser_dir, y_file)
            if not os.path.exists(xp) or not os.path.exists(yp):
                continue
            with open(xp, "rb") as f:
                X = np.array(pickle.load(f))
            with open(yp, "rb") as f:
                Y = np.array(pickle.load(f))
            if Y.ndim == 4:
                Y = Y[:, 0, :, :]
            # Shadow merge: class 2 -> class 1. Idempotent: filtered fisser pkls
            # already had this applied by filter_small_icebergs.py; raw pkls do not.
            Y[Y == 2] = 1
            print(f"  {x_file}: {len(X)} chips")

            for i in range(len(X)):
                gidx = fisser_idx

                # Apply IC quality filter
                if passing_fisser is not None and gidx not in passing_fisser:
                    fisser_idx += 1
                    continue

                chip_stem = f"fisser_{gidx:04d}"
                met_rec = met.get(chip_stem, {})

                n_icebergs = 0
                if (Y[i] == 1).sum() > 0:
                    _, n_icebergs = cc_label((Y[i] == 1).astype(np.int32))

                # Annotation-aware IC, gated by --skip_ic_mask (see Roboflow branch).
                if args.skip_ic_mask:
                    ic_aware = 0.0
                else:
                    b08 = X[i][2]
                    ice_mask = (Y[i] == 1)
                    non_ann = ~ice_mask
                    non_ann_count = int(non_ann.sum())
                    bright_non_ann = int(((b08 >= B08_THRESHOLD) & non_ann).sum())
                    ic_aware = bright_non_ann / non_ann_count if non_ann_count > 0 else 0.0

                # Write a synthetic GeoTIFF so eval_methods.py can rasterise polygons
                # back to masks. Without this, the cached transform in the gt record
                # is None and the chip is silently dropped from the evaluation.
                tif_path = write_synthetic_fisser_tif(chip_stem, X[i])

                fisser_chips.append({
                    "X": X[i], "Y": Y[i], "sza_bin": "sza_lt65",
                    "stem": f"fisser_{gidx:04d}", "chip_stem": chip_stem,
                    "tif_path": tif_path, "source": "fisser", "n_icebergs": n_icebergs,
                    "wind_ms": met_rec.get("wind_speed_10m", ""),
                    "temp_c": met_rec.get("temp_2m", ""),
                    "ic_aware": ic_aware,
                })
                fisser_idx += 1

    print(f"  Loaded {len(fisser_chips)} Fisser chips (after IC filter)")

    # ── Merge all chips ──────────────────────────────────────────────────
    all_chips = coco_chips + fisser_chips
    print(f"\nTotal chips: {len(all_chips)}")

    # 3. Optional SZA-bin filter (e.g. lt65-only Phase A subsets)
    if args.filter_sza_bin:
        before = len(all_chips)
        all_chips = [c for c in all_chips if c["sza_bin"] == args.filter_sza_bin]
        print(f"Filtered to sza_bin={args.filter_sza_bin}: {before} -> {len(all_chips)} chips")

    # ── Stratified split ─────────────────────────────────────────────────
    by_bin = defaultdict(list)
    for i, c in enumerate(all_chips):
        by_bin[c["sza_bin"]].append(i)

    print(f"\nChips per SZA bin:")
    for b in SZA_BINS:
        print(f"  {b}: {len(by_bin.get(b, []))}")

    # Draw test set: equal per bin
    n_total = len(all_chips)
    n_test = max(1, round(n_total * args.test_frac))
    bins_present = [b for b in SZA_BINS if b in by_bin]
    n_test_per_bin = max(1, n_test // len(bins_present))

    test_idx = set()
    for b in bins_present:
        indices = by_bin[b].copy()
        rng.shuffle(indices)
        take = min(n_test_per_bin, len(indices))
        test_idx.update(indices[:take])
        by_bin[b] = indices[take:]

    remaining = [i for i in range(n_total) if i not in test_idx]
    rng.shuffle(remaining)

    # Draw val set
    n_val = max(1, round(n_total * args.val_frac))
    val_idx = set(remaining[:n_val])
    train_idx = [i for i in remaining if i not in val_idx]

    print(f"\nSplit: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # Print per-bin distribution
    for split_name, indices in [("train", train_idx), ("val", list(val_idx)), ("test", list(test_idx))]:
        bin_counts = defaultdict(int)
        null_count = 0
        for i in indices:
            bin_counts[all_chips[i]["sza_bin"]] += 1
            if all_chips[i]["n_icebergs"] == 0:
                null_count += 1
        print(f"  {split_name}: " + "  ".join(f"{b}={bin_counts.get(b,0)}" for b in SZA_BINS) +
              f"  null={null_count}/{len(indices)}")

    # ── Pack into arrays ─────────────────────────────────────────────────
    def pack(indices, apply_ic_mask=False):
        X_list = []
        Y_list = []
        n_masked = 0
        total_masked_px = 0
        total_preserved_px = 0
        for i in indices:
            x = all_chips[i]["X"].copy()
            y = all_chips[i]["Y"].copy()
            if y.ndim == 2:
                y = y[np.newaxis, :, :]

            # IC masking: for training chips with IC >= 15%, zero out bright
            # non-annotated pixels across all 3 bands
            if apply_ic_mask and all_chips[i].get("ic_aware", 0) >= IC_THRESHOLD:
                b08 = x[2]
                ann_mask = (y[0] == 1) if y.ndim == 3 else (y == 1)
                sea_ice = (b08 >= B08_THRESHOLD) & (~ann_mask)
                n_sea_ice = int(sea_ice.sum())
                if n_sea_ice > 0:
                    x[0][sea_ice] = 0.0
                    x[1][sea_ice] = 0.0
                    x[2][sea_ice] = 0.0
                    n_masked += 1
                    total_masked_px += n_sea_ice
                    total_preserved_px += int(ann_mask.sum())

            X_list.append(x)
            Y_list.append(y)

        X = np.stack(X_list, axis=0)
        Y = np.stack(Y_list, axis=0)
        if Y.ndim == 3:
            Y = Y[:, np.newaxis, :, :]

        if apply_ic_mask:
            print(f"  IC masking: {n_masked} chips masked, {total_masked_px:,} pixels zeroed, "
                  f"{total_preserved_px:,} iceberg pixels preserved")

        return X.astype(np.float32), Y.astype(np.int64)

    apply_train_ic = not args.skip_ic_mask
    if not apply_train_ic:
        print("\nIC pixel mask: SKIPPED for all splits (--skip_ic_mask)")
    X_train, Y_train = pack(train_idx, apply_ic_mask=apply_train_ic)
    X_val, Y_val = pack(list(val_idx), apply_ic_mask=False)
    X_test, Y_test = pack(list(test_idx), apply_ic_mask=False)

    print(f"\nArray shapes:")
    print(f"  X_train: {X_train.shape}  Y_train: {Y_train.shape}")
    print(f"  X_val:   {X_val.shape}    Y_val:   {Y_val.shape}")
    print(f"  X_test:  {X_test.shape}   Y_test:  {Y_test.shape}")

    # Class distribution (binary: iceberg only, shadow merged)
    flat = Y_train.flatten()
    total = flat.size
    for cls, name in [(0, "ocean"), (1, "iceberg")]:
        n = int((flat == cls).sum())
        print(f"  train class {cls} ({name}): {n/total*100:.1f}%")
    n_shadow = int((flat == 2).sum())
    if n_shadow > 0:
        print(f"  WARNING: {n_shadow} shadow pixels (class 2) still present, should be 0")

    # ── Save ─────────────────────────────────────────────────────────────
    split_dir = os.path.join(args.out_dir, "train_validate_test")
    os.makedirs(split_dir, exist_ok=True)

    def save_pkl(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    save_pkl(X_train, os.path.join(split_dir, "X_train.pkl"))
    save_pkl(Y_train, os.path.join(split_dir, "Y_train.pkl"))
    save_pkl(X_val,   os.path.join(split_dir, "X_validation.pkl"))
    save_pkl(Y_val,   os.path.join(split_dir, "Y_validation.pkl"))
    save_pkl(X_test,  os.path.join(split_dir, "x_test.pkl"))
    save_pkl(Y_test,  os.path.join(split_dir, "y_test.pkl"))

    # ── Save extended split log ──────────────────────────────────────────
    log_path = os.path.join(args.out_dir, "split_log.csv")
    fieldnames = [
        "split", "pkl_position", "stem", "chip_stem", "tif_path", "sza_bin",
        "source", "n_icebergs", "ic_aware", "ic_masked", "wind_ms", "temp_c"
    ]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, indices in [("train", train_idx), ("val", list(val_idx)), ("test", list(test_idx))]:
            for pos, idx in enumerate(indices):
                c = all_chips[idx]
                ic = c.get("ic_aware", 0)
                ic_masked = (
                    not args.skip_ic_mask
                    and split_name == "train"
                    and ic >= IC_THRESHOLD
                )
                writer.writerow({
                    "split": split_name,
                    "pkl_position": pos,
                    "stem": c["stem"],
                    "chip_stem": c["chip_stem"],
                    "tif_path": c.get("tif_path", ""),
                    "sza_bin": c["sza_bin"],
                    "source": c["source"],
                    "n_icebergs": c["n_icebergs"],
                    "ic_aware": f"{ic:.4f}",
                    "ic_masked": str(ic_masked),
                    "wind_ms": c.get("wind_ms", ""),
                    "temp_c": c.get("temp_c", ""),
                })

    print(f"\nSplit log: {log_path}")
    print(f"PKL files: {split_dir}/")

    # ── Write manifest.json ──────────────────────────────────────────────
    # This is the single identity record for the dataset. Everything else
    # downstream (training, inference, evaluation) stamps chips_sha into its
    # output CSVs so comparisons across runs are grounded in the same rows.
    chip_rows = build_manifest_chip_rows(all_chips, train_idx, val_idx, test_idx)
    manifest_path, chips_sha = write_manifest(
        args.out_dir, args.manifest_id, args, chip_rows, args.chips_root,
    )
    print(f"Manifest  : {manifest_path}")
    print(f"chips_sha : {chips_sha[:16]}...  ({len(chip_rows)} chips)")

    print(f"\nDone. Next: python scripts/balance_training.py")


if __name__ == "__main__":
    main()
