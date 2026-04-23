"""
build_test_index.py — Map x_test.pkl indices → actual .tif chip paths.

Pixel-matches each chip in x_test.pkl against the .tif files in chips_dir
to find the exact chip file used for each test sample.  Uses split_log.csv
to know which SZA bin each test chip belongs to, so the search is limited
to the right subdirectory (much faster than searching everything).

Fisser sza_lt65 chips are resolved via fisser_index.csv (already built by
build_fisser_index.py) — no pixel matching needed for those.

Output: test_index.csv
  Columns: pkl_position, global_index, sza_bin, chip_stem, tif_path
  - pkl_position : row in x_test.pkl  (0-based)
  - global_index : all_chips index from split_log.csv
  - chip_stem    : filename stem (without .tif) — used for source_file matching

Usage:
  python build_test_index.py \\
      --pkl_dir      /mnt/research/.../train_validate_test_v2/train_validate_test \\
      --split_log    /mnt/research/.../train_validate_test_v2/split_log.csv \\
      --chips_dir    /mnt/research/.../S2-iceberg-areas/chips \\
      --fisser_index ~/S2-iceberg-areas/fisser_index.csv \\
      --out_dir      ~/S2-iceberg-areas

rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/build_test_index.py smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import argparse
import csv
import os
import pickle
from glob import glob

import numpy as np
import rasterio as rio

RESEARCH = "/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas"


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_fisser_index(csv_path):
    """Load fisser_index.csv → {global_index: tif_path}."""
    index = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            index[int(row["global_index"])] = row["tif_path"]
    return index


def read_tif_array(path):
    """Read 3-band float32 array, replacing NaN with 0 (matches pkl storage)."""
    with rio.open(path) as src:
        arr = src.read().astype(np.float32)
    return np.nan_to_num(arr[:3], nan=0.0)


def find_match(target, tif_files, tolerance=1e-4):
    """
    Find the tif whose array best matches `target` (3×256×256 float32).
    Returns (tif_path, mae) or (None, inf) if nothing is within tolerance.
    """
    best_path = None
    best_mae  = float("inf")
    for path in tif_files:
        try:
            arr = read_tif_array(path)
        except Exception:
            continue
        if arr.shape != target.shape:
            continue
        mae = float(np.abs(arr - target).mean())
        if mae < best_mae:
            best_mae  = mae
            best_path = path
        if mae < tolerance:
            break   # close enough, stop early
    return best_path, best_mae


def main():
    parser = argparse.ArgumentParser(
        description="Map x_test.pkl rows to .tif chip paths via pixel matching"
    )
    parser.add_argument("--pkl_dir",
        default=os.path.join(RESEARCH, "train_validate_test_v2", "train_validate_test"),
        help="Directory with x_test.pkl")
    parser.add_argument("--split_log",
        default=os.path.join(RESEARCH, "train_validate_test_v2", "split_log.csv"),
        help="split_log.csv from prepare_new_training_data.py")
    parser.add_argument("--chips_dir",
        default=os.path.join(RESEARCH, "chips"),
        help="Root chips directory for sza_gt65 chips")
    parser.add_argument("--fisser_index",
        default=os.path.expanduser("~/S2-iceberg-areas/fisser_index.csv"),
        help="fisser_index.csv from build_fisser_index.py")
    parser.add_argument("--out_dir",
        default=os.path.expanduser("~/S2-iceberg-areas"))
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    # ── Load x_test.pkl ───────────────────────────────────────────────────────
    x_path = os.path.join(args.pkl_dir, "x_test.pkl")
    X_test = load_pkl(x_path).astype(np.float32)   # (N, 3, 256, 256)
    N = len(X_test)
    print(f"x_test.pkl: {N} chips")

    # ── Load split_log — test rows in order (same order as x_test.pkl) ────────
    test_rows = []
    with open(args.split_log, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                test_rows.append({
                    "global_index": int(row["index"]),
                    "stem":         row["stem"],
                    "sza_bin":      row["sza_bin"],
                })

    if len(test_rows) != N:
        print(f"WARNING: split_log has {len(test_rows)} test rows, pkl has {N} — using min")
        n = min(len(test_rows), N)
        test_rows = test_rows[:n]
        X_test    = X_test[:n]

    # ── Load fisser index for sza_lt65 ────────────────────────────────────────
    fisser_index = {}
    if os.path.exists(args.fisser_index):
        fisser_index = load_fisser_index(args.fisser_index)
        print(f"Fisser index: {len(fisser_index)} entries")
    else:
        print(f"WARNING: fisser_index not found: {args.fisser_index}")

    # ── Build per-bin tif file lists (avoid re-globbing for each chip) ────────
    print("\nIndexing chips_dir by SZA bin…")
    bin_tifs = {}
    for sza_bin in ["sza_65_70", "sza_70_75", "sza_gt75"]:
        bin_dir = os.path.join(args.chips_dir, "**")
        tifs = sorted(glob(os.path.join(args.chips_dir, "**", "*.tif"), recursive=True))
        # Filter to this bin
        bin_tifs[sza_bin] = [t for t in tifs if sza_bin in t]
        print(f"  {sza_bin}: {len(bin_tifs[sza_bin])} tif files")

    # ── Match each test chip ──────────────────────────────────────────────────
    print(f"\nMatching {N} test chips…")
    results = []
    n_ok = 0
    n_fail = 0

    for k, row in enumerate(test_rows):
        global_idx = row["global_index"]
        sza_bin    = row["sza_bin"]
        target     = X_test[k]   # (3, 256, 256)
        stem       = row["stem"]

        # ── Fisser chips: look up directly from fisser_index ─────────────────
        if stem.startswith("fisser_") or sza_bin == "sza_lt65":
            tif_path = fisser_index.get(global_idx)
            if tif_path and os.path.exists(tif_path):
                chip_stem = os.path.basename(tif_path).replace(".tif", "")
                print(f"  [{k:>3}/{N}] sza_lt65  FISSER  {chip_stem[:60]}")
                results.append({
                    "pkl_position": k,
                    "global_index": global_idx,
                    "sza_bin":      sza_bin,
                    "chip_stem":    chip_stem,
                    "tif_path":     tif_path,
                })
                n_ok += 1
            else:
                print(f"  [{k:>3}/{N}] sza_lt65  NOT IN FISSER INDEX  global_idx={global_idx}")
                n_fail += 1
            continue

        # ── sza_gt65 chips: pixel match ───────────────────────────────────────
        candidates = bin_tifs.get(sza_bin, [])
        if not candidates:
            print(f"  [{k:>3}/{N}] {sza_bin}  NO CANDIDATES")
            n_fail += 1
            continue

        # Speed up: filter candidates by scene stem (strip _r/_c suffix from tif names)
        scene_candidates = [t for t in candidates if stem in os.path.basename(t)]
        search_list = scene_candidates if scene_candidates else candidates

        best_path, best_mae = find_match(target, search_list, args.tolerance)

        if best_mae <= args.tolerance:
            chip_stem = os.path.basename(best_path).replace(".tif", "")
            print(f"  [{k:>3}/{N}] {sza_bin}  mae={best_mae:.2e}  {chip_stem[:55]}")
            results.append({
                "pkl_position": k,
                "global_index": global_idx,
                "sza_bin":      sza_bin,
                "chip_stem":    chip_stem,
                "tif_path":     os.path.abspath(best_path),
            })
            n_ok += 1
        else:
            print(f"  [{k:>3}/{N}] {sza_bin}  NO MATCH  best_mae={best_mae:.4f}  stem={stem[:50]}")
            n_fail += 1

    # ── Write output ──────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "test_index.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f,
            fieldnames=["pkl_position", "global_index", "sza_bin", "chip_stem", "tif_path"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'─'*50}")
    print(f"Matched : {n_ok} / {N}")
    print(f"Failed  : {n_fail} / {N}")
    print(f"Saved   : {out_csv}")
    print(f"\nNext: python prepare_test_chips_dir.py --test_index {out_csv}")


if __name__ == "__main__":
    main()