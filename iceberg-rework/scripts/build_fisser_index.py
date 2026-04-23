"""
build_fisser_index.py — One-time setup: map fisser_XXXX pkl indices → S2UnetPlusPlus/imgs/*.tif paths.

The Fisser sza_lt65 training data was stored as pkl arrays with sequential
global indices (fisser_0000, fisser_0001, …).  The original .tif chips live in
S2UnetPlusPlus/imgs/ but are named by scene/grid position, not by index.  This
script pixel-matches each pkl array to its corresponding .tif so that
prepare_test_chips_dir.py can symlink the real georeferenced chip.

How it works
------------
1. Load X_train.pkl, X_validation.pkl, x_test.pkl → X_all  (N × 3 × 256 × 256)
   (same concatenation order as load_existing_pkl() in prepare_new_training_data.py)
2. Load all *.tif from imgs_dir in sorted order.
3. For each tif, read its 3-band array and find the matching row in X_all by
   mean absolute error (MAE < tolerance). Record the match.
4. Write fisser_index.csv:
     global_index, tif_path
   Only matched chips are written; unmatched are reported as warnings.

Output
------
  fisser_index.csv  —  in --out_dir (default: same dir as --pkl_dir)

Usage
-----
  python build_fisser_index.py \\
      --pkl_dir   ~/S2-iceberg-areas/S2UnetPlusPlus/train_validate_test \\
      --imgs_dir  ~/S2-iceberg-areas/S2UnetPlusPlus/imgs \\
      --out_dir   ~/S2-iceberg-areas

  # On moosehead (paths are the same since S2UnetPlusPlus is checked in):
  python ~/S2-iceberg-areas/build_fisser_index.py \\
      --pkl_dir   ~/S2-iceberg-areas/S2UnetPlusPlus/train_validate_test \\
      --imgs_dir  ~/S2-iceberg-areas/S2UnetPlusPlus/imgs \\
      --out_dir   ~/S2-iceberg-areas

rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/build_fisser_index.py smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import argparse
import csv
import os
import pickle
from glob import glob

import numpy as np
import rasterio as rio


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Map fisser_XXXX pkl indices to S2UnetPlusPlus/imgs/*.tif paths"
    )
    parser.add_argument("--pkl_dir",
        default=os.path.expanduser("~/S2-iceberg-areas/S2UnetPlusPlus/train_validate_test"),
        help="Directory with X_train.pkl, X_validation.pkl, x_test.pkl")
    parser.add_argument("--imgs_dir",
        default=os.path.expanduser("~/S2-iceberg-areas/S2UnetPlusPlus/imgs"),
        help="Directory with Fisser chip .tif files")
    parser.add_argument("--out_dir",
        default=os.path.expanduser("~/S2-iceberg-areas"),
        help="Where to write fisser_index.csv")
    parser.add_argument("--tolerance", type=float, default=1e-4,
        help="MAE tolerance for pixel-matching (default: 1e-4)")
    args = parser.parse_args()

    # ── Load pkl arrays ───────────────────────────────────────────────────────
    parts = []
    for fname in ["X_train.pkl", "X_validation.pkl", "x_test.pkl"]:
        p = os.path.join(args.pkl_dir, fname)
        if not os.path.exists(p):
            print(f"WARNING: {fname} not found in {args.pkl_dir} — skipping")
            continue
        arr = load_pkl(p)
        if arr.ndim == 3:
            arr = arr[np.newaxis]   # single chip → (1, 3, H, W)
        parts.append(arr.astype(np.float32))
        print(f"  Loaded {fname}: {arr.shape[0]} chips")

    if not parts:
        print("ERROR: no pkl files loaded")
        return

    X_all = np.concatenate(parts, axis=0)   # (N, 3, H, W)
    N     = len(X_all)
    print(f"\nTotal pkl chips: {N}  (global indices 0–{N-1})")

    # ── Load tif files ────────────────────────────────────────────────────────
    tif_files = sorted(glob(os.path.join(args.imgs_dir, "*.tif")))
    if not tif_files:
        print(f"ERROR: no .tif files found in {args.imgs_dir}")
        return
    print(f"Tif files in imgs_dir: {len(tif_files)}")

    tif_arrays = []
    for path in tif_files:
        with rio.open(path) as src:
            arr = src.read().astype(np.float32)   # (bands, H, W)
        # Some Fisser tifs may have 1 or more bands; take first 3
        if arr.shape[0] < 3:
            tif_arrays.append(None)
        else:
            # Tifs store nodata as float NaN; pkls stored them as 0.0
            tif_arrays.append(np.nan_to_num(arr[:3], nan=0.0))

    # ── Pixel-match each tif to a pkl row ─────────────────────────────────────
    print(f"\nMatching {len(tif_files)} tifs to {N} pkl chips (tolerance={args.tolerance})…")

    # For speed: precompute per-tif mean vectors and compare in bulk where possible
    matched     = {}   # global_index → tif_path
    tif_matched = {}   # tif_path → global_index  (for collision detection)

    used_pkl    = set()

    for ti, (tif_path, tif_arr) in enumerate(zip(tif_files, tif_arrays)):
        if tif_arr is None:
            print(f"  SKIP (< 3 bands): {os.path.basename(tif_path)}")
            continue

        if tif_arr.shape[1] != X_all.shape[2] or tif_arr.shape[2] != X_all.shape[3]:
            print(f"  SKIP (shape mismatch {tif_arr.shape}): {os.path.basename(tif_path)}")
            continue

        # Compute MAE vs every unmatched pkl chip
        diff = np.abs(X_all - tif_arr[np.newaxis])   # (N, 3, H, W)
        mae  = diff.mean(axis=(1, 2, 3))              # (N,)

        best_i   = int(np.argmin(mae))
        best_mae = float(mae[best_i])

        name = os.path.basename(tif_path)
        if best_mae < args.tolerance:
            if best_i in used_pkl:
                print(f"  COLLISION at pkl[{best_i}]  tif={name}  mae={best_mae:.2e}")
            else:
                matched[best_i]           = os.path.abspath(tif_path)
                tif_matched[tif_path]     = best_i
                used_pkl.add(best_i)
                if (ti + 1) % 50 == 0:
                    print(f"  [{ti+1}/{len(tif_files)}] matched so far: {len(matched)}")
        else:
            print(f"  NO MATCH  tif={name}  best_mae={best_mae:.4f}  best_pkl_idx={best_i}")

    print(f"\nMatched {len(matched)} / {len(tif_files)} tifs to pkl indices")
    unmatched_pkl = [i for i in range(N) if i not in matched]
    if unmatched_pkl:
        print(f"Unmatched pkl indices ({len(unmatched_pkl)}): {unmatched_pkl[:20]}{'…' if len(unmatched_pkl)>20 else ''}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "fisser_index.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["global_index", "tif_path"])
        writer.writeheader()
        for idx in sorted(matched):
            writer.writerow({"global_index": idx, "tif_path": matched[idx]})

    print(f"\nSaved: {out_csv}  ({len(matched)} rows)")
    print(f"Next: python prepare_test_chips_dir.py --fisser_index {out_csv} …")


if __name__ == "__main__":
    main()