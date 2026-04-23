"""
prepare_test_chips_dir.py: Build a directory of test-set chip .tifs.

Preferred usage (manifest-driven, matches the current refactor):
  python prepare_test_chips_dir.py --manifest data/v4_clean/manifest.json

  The manifest is the single source of truth for which chips belong to the
  test split. tif_path is read directly from each chip row.

Alternate usage (legacy test_index.csv from build_test_index.py):
  python prepare_test_chips_dir.py --test_index ~/S2-iceberg-areas/test_index.csv

Fallback usage (if neither manifest nor test_index.csv is available):
  python prepare_test_chips_dir.py \\
      --split_log     /mnt/research/.../train_validate_test_v2/split_log.csv \\
      --fisser_index  ~/S2-iceberg-areas/fisser_index.csv
  (sza_gt65 chips will be missing unless split_log has chip_stem/tif_path columns)

Output structure:
  out_dir/
    sza_lt65/  → symlinks named {chip_stem}.tif
    sza_65_70/
    sza_70_75/
    sza_gt75/

Setup order (one-time):
  1. python build_fisser_index.py   → fisser_index.csv
  2. python build_test_index.py     → test_index.csv   (uses fisser_index.csv)
  3. python prepare_test_chips_dir.py --test_index test_index.csv

rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/prepare_test_chips_dir.py smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import argparse
import csv
import os
from collections import defaultdict
from glob import glob

from _method_common import load_manifest

SZA_BINS  = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]
RESEARCH  = "/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas"


def link_or_copy(src, dest, copy=False):
    if os.path.exists(dest):
        return
    if copy:
        import shutil
        shutil.copy2(src, dest)
    else:
        os.symlink(os.path.abspath(src), dest)


def main():
    parser = argparse.ArgumentParser(
        description="Symlink test-set chip .tifs into a per-bin directory"
    )
    # Preferred: manifest.json from build_clean_dataset.py
    parser.add_argument("--manifest",
        default=None,
        help="manifest.json from build_clean_dataset.py (preferred)")
    # Legacy: test_index.csv from build_test_index.py
    parser.add_argument("--test_index",
        default=os.path.expanduser("~/S2-iceberg-areas/test_index.csv"),
        help="test_index.csv from build_test_index.py (legacy)")
    # Fallback: split_log + fisser_index
    parser.add_argument("--split_log",
        default=os.path.join(RESEARCH, "train_validate_test_v2", "split_log.csv"),
        help="split_log.csv (fallback if test_index not available)")
    parser.add_argument("--fisser_index",
        default=os.path.expanduser("~/S2-iceberg-areas/fisser_index.csv"),
        help="fisser_index.csv from build_fisser_index.py (fallback)")
    parser.add_argument("--chips_dir",
        default=os.path.join(RESEARCH, "chips"),
        help="Root chips directory (fallback search)")
    parser.add_argument("--out_dir",
        default=os.path.join(RESEARCH, "test_chips"),
        help="Output directory for symlinked test chips")
    parser.add_argument("--copy", action="store_true",
        help="Copy files instead of symlinking")
    args = parser.parse_args()

    for b in SZA_BINS:
        os.makedirs(os.path.join(args.out_dir, b), exist_ok=True)

    # -- Preferred path: manifest.json ----------------------------------------
    if args.manifest and os.path.exists(args.manifest):
        manifest = load_manifest(args.manifest)
        print(f"Using manifest: {args.manifest}  id={manifest['manifest_id']}")
        print(f"Chips_sha: {manifest['chips_sha'][:16]}...")

        n_ok      = 0
        n_missing = 0
        for row in manifest["chips"]:
            if row.get("split") != "test":
                continue
            tif_path  = row.get("tif_path", "")
            chip_stem = row["chip_stem"]
            sza_bin   = row["sza_bin"]
            dest = os.path.join(args.out_dir, sza_bin, f"{chip_stem}.tif")
            if not tif_path or not os.path.exists(tif_path):
                print(f"  MISSING: {chip_stem} ({tif_path})")
                n_missing += 1
                continue
            link_or_copy(tif_path, dest, args.copy)
            n_ok += 1

        print(f"\n{'-'*50}")
        print(f"Linked : {n_ok}")
        print(f"Missing: {n_missing}")

    # -- Legacy path: test_index.csv ------------------------------------------
    elif os.path.exists(args.test_index):
        print(f"Using test_index.csv: {args.test_index}")
        n_ok = 0
        n_missing = 0
        with open(args.test_index, newline="") as f:
            for row in csv.DictReader(f):
                tif_path  = row["tif_path"]
                chip_stem = row["chip_stem"]
                sza_bin   = row["sza_bin"]
                dest = os.path.join(args.out_dir, sza_bin, f"{chip_stem}.tif")
                if not os.path.exists(tif_path):
                    print(f"  MISSING: {tif_path}")
                    n_missing += 1
                    continue
                link_or_copy(tif_path, dest, args.copy)
                n_ok += 1

        print(f"\n{'─'*50}")
        print(f"Linked : {n_ok}")
        print(f"Missing: {n_missing}")

    # ── Fallback path: split_log ──────────────────────────────────────────────
    else:
        print(f"test_index.csv not found, falling back to split_log")
        print(f"  (run build_test_index.py for the reliable path)\n")

        # Load fisser index
        fisser_index = {}
        if os.path.exists(args.fisser_index):
            with open(args.fisser_index, newline="") as f:
                for row in csv.DictReader(f):
                    fisser_index[int(row["global_index"])] = row["tif_path"]
            print(f"Fisser index: {len(fisser_index)} entries")

        test_chips = []
        with open(args.split_log, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == "test":
                    test_chips.append({
                        "stem":      row["stem"],
                        "chip_stem": row.get("chip_stem", row["stem"]),
                        "tif_path":  row.get("tif_path", ""),
                        "sza_bin":   row["sza_bin"],
                        "index":     int(row.get("index", -1)),
                    })

        print(f"Test chips in split_log: {len(test_chips)}")
        for b in SZA_BINS:
            n = sum(1 for c in test_chips if c["sza_bin"] == b)
            if n:
                print(f"  {b}: {n}")
        print()

        n_found = 0
        n_missing = 0
        n_fisser = 0
        n_fisser_skip = 0

        for chip in test_chips:
            stem      = chip["stem"]
            sza_bin   = chip["sza_bin"]
            dest      = os.path.join(args.out_dir, sza_bin, f"{chip['chip_stem']}.tif")

            if stem.startswith("fisser_"):
                global_idx = chip["index"]
                tif_path   = fisser_index.get(global_idx)
                if tif_path and os.path.exists(tif_path):
                    link_or_copy(tif_path, dest, args.copy)
                    n_fisser += 1
                else:
                    print(f"  FISSER NOT FOUND: global_idx={global_idx}")
                    n_fisser_skip += 1
                continue

            # sza_gt65: try tif_path from split_log, then search by chip_stem
            tif_path = chip["tif_path"]
            if not (tif_path and os.path.exists(tif_path)):
                matches = glob(os.path.join(args.chips_dir, "**",
                                            f"{chip['chip_stem']}.tif"), recursive=True)
                tif_path = matches[0] if matches else None

            if tif_path:
                link_or_copy(tif_path, dest, args.copy)
                n_found += 1
            else:
                print(f"  NOT FOUND: {chip['chip_stem']}")
                n_missing += 1

        print(f"{'─'*50}")
        print(f"sza_lt65 linked : {n_fisser}  (skipped: {n_fisser_skip})")
        print(f"sza_gt65 linked : {n_found}  (not found: {n_missing})")

    # Per-bin summary
    print(f"\nOutput: {args.out_dir}/")
    for b in SZA_BINS:
        n = len(glob(os.path.join(args.out_dir, b, "*.tif")))
        print(f"  {b}: {n} chips")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()