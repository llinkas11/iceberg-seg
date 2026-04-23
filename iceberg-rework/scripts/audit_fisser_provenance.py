"""
audit_fisser_provenance.py — Audit Fisser chip accessibility and provenance.

Maps each Fisser pkl chip to its source .tif file, parses acquisition date and
region, and computes ice-coverage fraction via Otsu thresholding on the B08 band.

Usage:
  python scripts/audit_fisser_provenance.py
  python scripts/audit_fisser_provenance.py --out_csv reference/fisser_provenance_audit.csv
"""

import argparse
import csv
import os
import re

import numpy as np
import rasterio
from skimage.filters import threshold_otsu

# ── Paths ────────────────────────────────────────────────────────────────────
SMISHRA = "/mnt/research/v.gomezgilyaspik/students/smishra"
FISSER_INDEX = os.path.join(SMISHRA, "rework/reference/fisser_index.csv")

# Regex to parse scene stem from Fisser tif filenames
# Example: S2B_MSIL1C_20200830T135739_N0500_R010_T25WER_20230321T193340_pB5_40_2_.tif
SCENE_RE = re.compile(
    r"(S2[AB])_MSIL1C_(\d{8}T\d{6})_N\d+_(R\d+)_(T\w{5})_\d{8}T\d+"
)

# Tile → region mapping (based on known UTM tiles for KQ and SK fjords)
TILE_REGION = {
    "T25WER": "KQ", "T25WES": "KQ", "T25WDR": "KQ", "T25WDS": "KQ",
    "T25WDQ": "KQ",
    "T24WWU": "SK", "T24WWT": "SK",
    "T22WES": "SK", "T22WDB": "SK", "T22WDC": "SK",
    "T21WXU": "SK", "T21WWU": "SK",
}

B08_IDX = 2  # Band index for NIR (B08) in the 3-band tif


def parse_scene_info(tif_path):
    """Extract satellite, date, orbit, tile, region from tif filename."""
    basename = os.path.basename(tif_path)
    m = SCENE_RE.search(basename)
    if not m:
        return None, None, None, None, None
    sat = m.group(1)
    dt_str = m.group(2)  # e.g. 20200830T135739
    tile = m.group(4)
    date_str = f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]}"
    hour_utc = int(dt_str[9:11])
    region = TILE_REGION.get(tile, "unknown")
    return date_str, hour_utc, tile, region, sat


def compute_ic_frac(tif_path):
    """Compute ice-coverage fraction via Otsu on B08 band."""
    try:
        with rasterio.open(tif_path) as src:
            b08 = src.read(B08_IDX + 1).astype(np.float32)  # 1-indexed
    except Exception as e:
        return None, str(e)

    # Skip constant images (all same value)
    if b08.max() == b08.min():
        return 0.0, "constant_image"

    try:
        thresh = threshold_otsu(b08)
    except ValueError:
        return 0.0, "otsu_failed"

    ic_frac = float((b08 > thresh).sum()) / b08.size
    return round(ic_frac, 4), "ok"


def main():
    parser = argparse.ArgumentParser(description="Audit Fisser chip provenance")
    parser.add_argument(
        "--fisser_index", default=FISSER_INDEX,
        help="Path to fisser_index.csv"
    )
    parser.add_argument(
        "--out_csv",
        default="/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/reference/fisser_provenance_audit.csv",
        help="Output CSV path"
    )
    args = parser.parse_args()

    # Load fisser index
    rows = []
    with open(args.fisser_index) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} entries from fisser_index.csv")

    results = []
    n_exist = 0
    n_parsed = 0

    for i, row in enumerate(rows):
        gidx = int(row["global_index"])
        tif_path = row["tif_path"]
        tif_exists = os.path.isfile(tif_path)

        if tif_exists:
            n_exist += 1

        date_str, hour_utc, tile, region, sat = parse_scene_info(tif_path)
        if date_str:
            n_parsed += 1

        ic_frac, ic_note = (None, "tif_missing")
        if tif_exists:
            ic_frac, ic_note = compute_ic_frac(tif_path)

        results.append({
            "global_index": gidx,
            "tif_path": tif_path,
            "tif_exists": tif_exists,
            "satellite": sat or "",
            "date": date_str or "",
            "hour_utc": hour_utc if hour_utc is not None else "",
            "tile": tile or "",
            "region": region or "",
            "ic_frac": ic_frac if ic_frac is not None else "",
            "ic_note": ic_note,
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            print(f"  [{i+1:>4}/{len(rows)}] tif_exists={n_exist}  parsed={n_parsed}")

    # Write output CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fieldnames = [
        "global_index", "tif_path", "tif_exists", "satellite", "date",
        "hour_utc", "tile", "region", "ic_frac", "ic_note"
    ]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    print(f"\n{'='*60}")
    print(f"FISSER PROVENANCE AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"Total Fisser chips:    {len(results)}")
    print(f"TIF files accessible:  {n_exist}")
    print(f"Dates parseable:       {n_parsed}")

    ic_vals = [r["ic_frac"] for r in results if isinstance(r["ic_frac"], float)]
    if ic_vals:
        ic_arr = np.array(ic_vals)
        n_fail_ic = int((ic_arr >= 0.15).sum())
        print(f"IC frac computed:      {len(ic_vals)}")
        print(f"IC >= 15% (fail):      {n_fail_ic}  ({n_fail_ic/len(ic_vals)*100:.1f}%)")
        print(f"IC < 15%  (pass):      {len(ic_vals) - n_fail_ic}")
        print(f"IC frac mean:          {ic_arr.mean():.4f}")
        print(f"IC frac max:           {ic_arr.max():.4f}")

    regions = {}
    for r in results:
        reg = r["region"] or "unknown"
        regions[reg] = regions.get(reg, 0) + 1
    print(f"\nRegion distribution:")
    for reg, cnt in sorted(regions.items()):
        print(f"  {reg}: {cnt}")

    print(f"\nOutput saved: {args.out_csv}")


if __name__ == "__main__":
    main()
