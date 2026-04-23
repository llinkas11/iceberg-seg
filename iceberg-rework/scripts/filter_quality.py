"""
filter_quality.py — IC filtering using Fisser's 10 km block method.

Implements Fisser et al. (2024) Section 3.2 "Ice filtering — Sentinel-2":
  1. Load full B08 band from parent .SAFE.zip (10980x10980 at 10m = ~110km tile)
  2. Divide into 10 km squared blocks (1000x1000 pixels)
  3. IC = percentage of pixels with B08 >= threshold per block
  4. Exclude blocks where IC >= 15%
  5. Flag chips that fall in excluded blocks

Threshold: 0.22 in our uncorrected reflectance space (equivalent to Fisser's
0.12 after subtracting the +1000 DN offset from processing baseline >=4.0;
chip_sentinel2.py does DN*1e-4 without offset subtraction, so all reflectances
are +0.10 high).

Applied uniformly to ALL chips (both Roboflow and Fisser-origin).

Usage:
  python scripts/filter_quality.py
  python scripts/filter_quality.py --threshold 0.22 --ic_threshold 0.15
"""

import argparse
import csv
import os
import re
import zipfile
from glob import glob

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds

# ── Paths ────────────────────────────────────────────────────────────────────
SMISHRA = "/mnt/research/v.gomezgilyaspik/students/smishra"
LLINKAS = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"

SPLIT_LOG = os.path.join(SMISHRA, "rework/data/split_log.csv")
PROVENANCE_CSV = os.path.join(LLINKAS, "reference/fisser_provenance_audit.csv")
DOWNLOADS_ROOT = os.path.join(SMISHRA, "sentinel2_downloads")
CHIPS_ROOT = os.path.join(SMISHRA, "rework/chips")

# DN offset: chip_sentinel2.py does DN*1e-4 without subtracting the +1000 DN
# baseline offset. Fisser's 0.12 threshold in corrected space = 0.22 in ours.
B08_THRESHOLD = 0.22
IC_THRESHOLD = 0.15
BLOCK_SIZE_PX = 1000  # 10 km at 10m resolution

SCENE_RE = re.compile(
    r"(S2[AB]_MSIL1C_\d{8}T\d{6}_N\d+_R\d+_T\w{5}_\d{8}T\d+)"
)


def find_safe_zip(scene_stem, downloads_root):
    """Find .SAFE.zip for a scene stem in the downloads directory."""
    for region in ["KQ", "SK"]:
        for sza in ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]:
            # Try .SAFE.zip
            zp = os.path.join(downloads_root, region, sza, scene_stem + ".SAFE.zip")
            if os.path.exists(zp):
                return zp
            # Try unzipped .SAFE directory
            sd = os.path.join(downloads_root, region, sza, scene_stem + ".SAFE")
            if os.path.isdir(sd):
                return sd
    return None


def find_b08_in_safe(safe_path):
    """Find the B08 IMG_DATA jp2 path inside a SAFE zip or directory."""
    if safe_path.endswith(".zip"):
        with zipfile.ZipFile(safe_path) as z:
            for name in z.namelist():
                if name.endswith("_B08.jp2") and "IMG_DATA" in name:
                    return f"/vsizip/{safe_path}/{name}"
    else:
        # Unzipped .SAFE directory
        matches = glob(os.path.join(safe_path, "GRANULE", "*", "IMG_DATA", "*_B08.jp2"))
        if matches:
            return matches[0]
    return None


def compute_block_ic(b08_path, threshold, block_size):
    """
    Load full B08 band and compute IC per 10km block.
    Returns dict mapping (block_row, block_col) -> ic_fraction.
    Also returns (transform, crs, H, W) for chip-to-block mapping.
    """
    with rasterio.open(b08_path) as src:
        # Read as DN, convert to reflectance (DN * 1e-4, no offset subtraction)
        raw = src.read(1).astype(np.float32)
        # Check if values are in DN range (>100) or already reflectance (<10)
        if raw.max() > 100:
            b08 = raw * 1e-4
        else:
            b08 = raw  # already converted
        transform = src.transform
        crs = src.crs
        H, W = b08.shape

    block_ic = {}
    for br in range(0, H, block_size):
        for bc in range(0, W, block_size):
            block = b08[br:br+block_size, bc:bc+block_size]
            if block.size == 0:
                continue
            ic = float((block >= threshold).sum()) / block.size
            block_ic[(br, bc)] = ic

    return block_ic, transform


def chip_to_block(chip_bounds, tile_transform, block_size):
    """
    Determine which 10km block a chip's center falls in.
    Returns (block_row, block_col) in pixel coordinates.
    """
    # Get chip center
    cx = (chip_bounds.left + chip_bounds.right) / 2
    cy = (chip_bounds.top + chip_bounds.bottom) / 2

    # Convert to pixel coordinates in the tile
    inv_transform = ~tile_transform
    px, py = inv_transform * (cx, cy)

    # Which block?
    block_row = int(py // block_size) * block_size
    block_col = int(px // block_size) * block_size

    return (block_row, block_col)


def main():
    parser = argparse.ArgumentParser(
        description="IC filtering using Fisser's 10km block method"
    )
    parser.add_argument("--threshold", type=float, default=B08_THRESHOLD,
                        help="B08 reflectance threshold (default: 0.22 = Fisser's 0.12 + DN offset)")
    parser.add_argument("--ic_threshold", type=float, default=IC_THRESHOLD,
                        help="IC block threshold (default: 0.15)")
    parser.add_argument("--block_size", type=int, default=BLOCK_SIZE_PX,
                        help="Block size in pixels (default: 1000 = 10km)")
    parser.add_argument("--split_log", default=SPLIT_LOG)
    parser.add_argument("--provenance_csv", default=PROVENANCE_CSV)
    parser.add_argument("--downloads_root", default=DOWNLOADS_ROOT)
    parser.add_argument("--chips_root", default=CHIPS_ROOT)
    parser.add_argument("--out_csv",
                        default=os.path.join(LLINKAS, "reference/ic_filter_10km.csv"))
    args = parser.parse_args()

    print(f"Fisser 10km block IC filter")
    print(f"  B08 threshold: {args.threshold} (= Fisser's 0.12 + 0.10 DN offset)")
    print(f"  IC threshold:  >= {args.ic_threshold} → exclude block")
    print(f"  Block size:    {args.block_size} px = {args.block_size * 10 / 1000:.0f} km")

    # ── Build chip → scene mapping ───────────────────────────────────────
    # From split_log: Roboflow chips
    chip_records = {}  # chip_stem → {scene_stem, tif_path, sza_bin, source, ...}

    with open(args.split_log) as f:
        for row in csv.DictReader(f):
            stem = row["stem"]
            chip_stem = row.get("chip_stem", stem)
            tif_path = row.get("tif_path", "")

            if stem.startswith("fisser_"):
                continue  # handle below

            m = SCENE_RE.search(stem)
            scene_stem = m.group(1) if m else None

            chip_records[chip_stem] = {
                "scene_stem": scene_stem,
                "tif_path": tif_path,
                "sza_bin": row["sza_bin"],
                "split": row["split"],
                "source": "roboflow",
            }

    # From provenance audit: Fisser chips
    if os.path.exists(args.provenance_csv):
        with open(args.provenance_csv) as f:
            for row in csv.DictReader(f):
                gidx = int(row["global_index"])
                chip_stem = f"fisser_{gidx:04d}"
                tif_path = row["tif_path"]
                basename = os.path.basename(tif_path)
                # Fisser tif names have format: S2X_MSIL1C_..._pB5_NN_NN_.tif
                # Scene stem is everything before _pB
                m = SCENE_RE.search(basename)
                scene_stem = m.group(1) if m else None

                chip_records[chip_stem] = {
                    "scene_stem": scene_stem,
                    "tif_path": tif_path,
                    "sza_bin": "sza_lt65",
                    "split": "",
                    "source": "fisser",
                }

    print(f"\nTotal chips to process: {len(chip_records)}")
    print(f"  Roboflow: {sum(1 for r in chip_records.values() if r['source'] == 'roboflow')}")
    print(f"  Fisser:   {sum(1 for r in chip_records.values() if r['source'] == 'fisser')}")

    # ── Group chips by scene ─────────────────────────────────────────────
    scenes = {}  # scene_stem → [chip_stem, ...]
    no_scene = []
    for chip_stem, rec in chip_records.items():
        ss = rec["scene_stem"]
        if ss:
            scenes.setdefault(ss, []).append(chip_stem)
        else:
            no_scene.append(chip_stem)

    print(f"Unique scenes: {len(scenes)}")
    if no_scene:
        print(f"Chips without parseable scene stem: {len(no_scene)}")

    # ── Process each scene: load B08, compute block IC, map chips ────────
    results = []  # per-chip results
    n_scenes_processed = 0
    n_scenes_no_zip = 0
    n_chips_pass = 0
    n_chips_fail = 0
    n_chips_no_zip = 0

    for scene_stem, chip_stems in sorted(scenes.items()):
        safe_path = find_safe_zip(scene_stem, args.downloads_root)

        if safe_path is None:
            # No SAFE file — can't compute IC
            n_scenes_no_zip += 1
            for cs in chip_stems:
                rec = chip_records[cs]
                results.append({
                    "chip_stem": cs,
                    "source": rec["source"],
                    "sza_bin": rec["sza_bin"],
                    "scene_stem": scene_stem,
                    "block_ic": "",
                    "pass_ic": "",
                    "note": "no_safe_file",
                })
                n_chips_no_zip += 1
            continue

        # Find B08 band
        b08_path = find_b08_in_safe(safe_path)
        if b08_path is None:
            n_scenes_no_zip += 1
            for cs in chip_stems:
                rec = chip_records[cs]
                results.append({
                    "chip_stem": cs,
                    "source": rec["source"],
                    "sza_bin": rec["sza_bin"],
                    "scene_stem": scene_stem,
                    "block_ic": "",
                    "pass_ic": "",
                    "note": "no_b08_in_safe",
                })
                n_chips_no_zip += 1
            continue

        # Compute block IC for this scene
        try:
            block_ic, tile_transform = compute_block_ic(
                b08_path, args.threshold, args.block_size
            )
        except Exception as e:
            for cs in chip_stems:
                rec = chip_records[cs]
                results.append({
                    "chip_stem": cs,
                    "source": rec["source"],
                    "sza_bin": rec["sza_bin"],
                    "scene_stem": scene_stem,
                    "block_ic": "",
                    "pass_ic": "",
                    "note": f"b08_read_error: {e}",
                })
                n_chips_no_zip += 1
            continue

        n_scenes_processed += 1
        n_blocks = len(block_ic)
        n_blocks_pass = sum(1 for ic in block_ic.values() if ic < args.ic_threshold)
        n_blocks_fail = n_blocks - n_blocks_pass
        usable_frac = n_blocks_pass / n_blocks if n_blocks > 0 else 0

        # Map each chip to its block
        for cs in chip_stems:
            rec = chip_records[cs]
            tif_path = rec["tif_path"]

            if not tif_path or not os.path.exists(tif_path):
                results.append({
                    "chip_stem": cs,
                    "source": rec["source"],
                    "sza_bin": rec["sza_bin"],
                    "scene_stem": scene_stem,
                    "block_ic": "",
                    "pass_ic": "",
                    "note": "chip_tif_missing",
                })
                n_chips_no_zip += 1
                continue

            try:
                with rasterio.open(tif_path) as src:
                    chip_bounds = src.bounds

                bkey = chip_to_block(
                    chip_bounds, tile_transform, args.block_size
                )
                ic_val = block_ic.get(bkey)

                if ic_val is None:
                    # Chip falls outside tile extent (edge case)
                    results.append({
                        "chip_stem": cs,
                        "source": rec["source"],
                        "sza_bin": rec["sza_bin"],
                        "scene_stem": scene_stem,
                        "block_ic": "",
                        "pass_ic": "",
                        "note": "chip_outside_tile",
                    })
                    n_chips_no_zip += 1
                    continue

                pass_ic = ic_val < args.ic_threshold
                if pass_ic:
                    n_chips_pass += 1
                else:
                    n_chips_fail += 1

                results.append({
                    "chip_stem": cs,
                    "source": rec["source"],
                    "sza_bin": rec["sza_bin"],
                    "scene_stem": scene_stem,
                    "block_ic": f"{ic_val:.4f}",
                    "pass_ic": str(pass_ic),
                    "note": f"block=({bkey[0]},{bkey[1]}) usable={usable_frac:.2f}",
                })
            except Exception as e:
                results.append({
                    "chip_stem": cs,
                    "source": rec["source"],
                    "sza_bin": rec["sza_bin"],
                    "scene_stem": scene_stem,
                    "block_ic": "",
                    "pass_ic": "",
                    "note": f"error: {e}",
                })
                n_chips_no_zip += 1

        if n_scenes_processed % 10 == 0:
            print(f"  [{n_scenes_processed:>4} scenes] pass={n_chips_pass} fail={n_chips_fail}")

    # Handle chips with no scene
    for cs in no_scene:
        rec = chip_records[cs]
        results.append({
            "chip_stem": cs,
            "source": rec["source"],
            "sza_bin": rec["sza_bin"],
            "scene_stem": "",
            "block_ic": "",
            "pass_ic": "",
            "note": "no_scene_stem",
        })
        n_chips_no_zip += 1

    # ── Write output ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fieldnames = ["chip_stem", "source", "sza_bin", "scene_stem",
                   "block_ic", "pass_ic", "note"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("FISSER 10KM BLOCK IC FILTER — RESULTS")
    print(f"{'='*65}")
    print(f"Method: B08 >= {args.threshold} in {args.block_size*10/1000:.0f}km blocks, IC >= {args.ic_threshold} → exclude")
    print(f"  (threshold 0.22 = Fisser's 0.12 + 0.10 DN offset correction)")
    print(f"")
    print(f"Scenes processed:     {n_scenes_processed}")
    print(f"Scenes without SAFE:  {n_scenes_no_zip}")
    print(f"")
    print(f"Chips passing IC:     {n_chips_pass}")
    print(f"Chips failing IC:     {n_chips_fail}")
    print(f"Chips unresolvable:   {n_chips_no_zip}")
    print(f"Total:                {len(results)}")

    # Per sza_bin breakdown
    by_bin = {}
    for r in results:
        b = r["sza_bin"]
        by_bin.setdefault(b, {"pass": 0, "fail": 0, "unknown": 0})
        if r["pass_ic"] == "True":
            by_bin[b]["pass"] += 1
        elif r["pass_ic"] == "False":
            by_bin[b]["fail"] += 1
        else:
            by_bin[b]["unknown"] += 1

    print(f"\n{'SZA Bin':<15} {'Pass':>6} {'Fail':>6} {'Unknown':>8} {'Total':>6}")
    print("-" * 45)
    for b in ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]:
        if b in by_bin:
            d = by_bin[b]
            total = d["pass"] + d["fail"] + d["unknown"]
            print(f"{b:<15} {d['pass']:>6} {d['fail']:>6} {d['unknown']:>8} {total:>6}")

    # Per source breakdown
    by_source = {}
    for r in results:
        s = r["source"]
        by_source.setdefault(s, {"pass": 0, "fail": 0, "unknown": 0})
        if r["pass_ic"] == "True":
            by_source[s]["pass"] += 1
        elif r["pass_ic"] == "False":
            by_source[s]["fail"] += 1
        else:
            by_source[s]["unknown"] += 1

    print(f"\n{'Source':<15} {'Pass':>6} {'Fail':>6} {'Unknown':>8}")
    print("-" * 40)
    for s in ["fisser", "roboflow"]:
        if s in by_source:
            d = by_source[s]
            print(f"{s:<15} {d['pass']:>6} {d['fail']:>6} {d['unknown']:>8}")

    print(f"\nOutput: {args.out_csv}")


if __name__ == "__main__":
    main()
