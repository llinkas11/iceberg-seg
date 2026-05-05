"""
sweep_ndwi_threshold.py — Sweep NDWI cutoff for the TR (NDWI) sensitivity branch.

Reads each Sentinel-2 chip once and applies threshold_masked_tifs._apply_thresholds
at every NDWI value in --ndwi_values. B08, IC, and min-area thresholds stay
fixed at the canonical defaults from threshold_masked_tifs.py.

Used to answer the script-check-README open question:
  "Is NDWI > 0 the right cutoff for icebergs in fjord water, or should it be
   slightly positive (e.g. 0.05) to exclude land-edge pixels?"

Note: Fisser and others (2024) do not use NDWI; this branch is a chip-level
land-edge safeguard added by this project.

Usage:
  python sweep_ndwi_threshold.py \\
      --chips_root  /mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/chips/KQ \\
      --sza_bins    sza_lt65,sza_65_70,sza_70_75,sza_gt75 \\
      --ndwi_values -0.05,0.00,0.05,0.10 \\
      --out_csv     /mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/sweeps/ndwi_threshold_sweep.csv

Output CSV columns:
  region, sza_bin, chip_stem, ndwi_threshold,
  band_skipped, ic_skipped, ic_frac, water_px, iceberg_px,
  n_polygons, total_area_m2

Each row is one (chip, ndwi_threshold) pair.

Deploy:
  rsync -av iceberg-rework/scripts/sweep_ndwi_threshold.py \\
            smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import argparse
import csv
import os
import warnings
from glob import glob

import numpy as np
import rasterio as rio

from threshold_masked_tifs import (
    NIR_THRESHOLD,
    MIN_AREA_M2,
    IC_THRESHOLD,
    _apply_thresholds,
)

warnings.filterwarnings("ignore")

DEFAULT_CHIPS_ROOT  = "/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/chips/KQ"
DEFAULT_OUT_CSV     = "/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/sweeps/ndwi_threshold_sweep.csv"
DEFAULT_SZA_BINS    = "sza_lt65,sza_65_70,sza_70_75,sza_gt75"
DEFAULT_NDWI_VALUES = "-0.05,0.00,0.05,0.10"
DEFAULT_CLOUD_THRESHOLD = 0.10


CSV_FIELDS = [
    "region", "sza_bin", "chip_stem", "ndwi_threshold",
    "band_skipped", "cloud_skipped", "cloud_fraction", "ic_skipped", "ic_frac",
    "water_px", "iceberg_px",
    "n_polygons", "total_area_m2",
]


def load_cloud_fractions(cloud_csv):
    """Read cloud_fractions CSV, return {chip_stem: cloud_fraction or None}."""
    if not cloud_csv:
        return {}
    out = {}
    with open(cloud_csv) as fh:
        for r in csv.DictReader(fh):
            if r["qa60_available"] == "True" and r["cloud_fraction"] != "":
                out[r["chip_stem"]] = float(r["cloud_fraction"])
            else:
                out[r["chip_stem"]] = None
    return out


def sweep(chips_root, sza_bins, ndwi_values, out_csv,
          cloud_fractions=None, cloud_threshold=DEFAULT_CLOUD_THRESHOLD,
          b03_idx=1, b08_idx=2):
    """Run an NDWI sensitivity sweep and write a tidy per-chip CSV.

    Inputs:
        chips_root      : region chips root containing <sza_bin>/tifs/*.tif.
        sza_bins        : list of SZA-bin subdirectory names.
        ndwi_values     : list of float NDWI thresholds to test.
        out_csv         : destination CSV path; parent dir is created.
        cloud_fractions : optional {chip_stem: cloud_fraction} dict; when present
                          chips with cloud_fraction > cloud_threshold are
                          recorded as cloud_skipped without iceberg detection.
        cloud_threshold : QA60 cloud-fraction cutoff (default 0.10).
    Output:
        Writes one CSV row per (chip, ndwi_threshold).
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    region = os.path.basename(chips_root.rstrip("/"))
    rows   = []
    cloud_fractions = cloud_fractions or {}

    for sza_bin in sza_bins:
        # 1. Glob chips for this SZA bin
        chip_glob = os.path.join(chips_root, sza_bin, "tifs", "*.tif")
        tif_files = sorted(glob(chip_glob))
        if not tif_files:
            print(f"[{sza_bin}] no chips at {chip_glob}")
            continue

        print(f"[{sza_bin}] {len(tif_files)} chips, NDWI in {ndwi_values}")

        for i, tif_path in enumerate(tif_files):
            stem = os.path.splitext(os.path.basename(tif_path))[0]
            cloud_frac = cloud_fractions.get(stem)
            base = {
                "region"        : region,
                "sza_bin"       : sza_bin,
                "chip_stem"     : stem,
                "cloud_fraction": "" if cloud_frac is None else cloud_frac,
                "cloud_skipped" : False,
                "band_skipped"  : False,
                "ic_skipped"    : False,
                "ic_frac"       : "",
                "water_px"      : 0,
                "iceberg_px"    : 0,
                "n_polygons"    : 0,
                "total_area_m2" : 0.0,
            }

            # 2. Cloud-skipped chips: emit one row per NDWI value, skip detection
            if cloud_frac is not None and cloud_frac > cloud_threshold:
                for ndwi_val in ndwi_values:
                    rows.append({**base, "ndwi_threshold": ndwi_val, "cloud_skipped": True})
                continue

            # 3. Read chip once per chip
            with rio.open(tif_path) as src:
                chip = src.read().astype(np.float32)
                meta = src.meta.copy()
            n_bands = chip.shape[0]

            # 4. Band-skipped chips: emit one row per NDWI value for accounting
            if n_bands <= max(b03_idx, b08_idx):
                for ndwi_val in ndwi_values:
                    rows.append({**base, "ndwi_threshold": ndwi_val, "band_skipped": True})
                continue

            b03         = chip[b03_idx]
            b08         = chip[b08_idx]
            transform   = meta["transform"]
            source_name = os.path.basename(tif_path)

            # 5. Apply thresholds at each NDWI value
            for ndwi_val in ndwi_values:
                res = _apply_thresholds(
                    b03,
                    b08,
                    transform,
                    source_name,
                    nir_threshold  = NIR_THRESHOLD,
                    ndwi_threshold = ndwi_val,
                    min_area_m2    = MIN_AREA_M2,
                    ic_threshold   = IC_THRESHOLD,
                )
                rows.append({
                    **base,
                    "ndwi_threshold": ndwi_val,
                    "ic_skipped"    : res["ic_skipped"],
                    "ic_frac"       : res["ic_frac"],
                    "water_px"      : res["water_px"],
                    "iceberg_px"    : res["iceberg_px"],
                    "n_polygons"    : res["n_polygons"],
                    "total_area_m2" : res["total_area_m2"],
                })

            # 6. Progress log
            if (i + 1) % 50 == 0:
                print(f"  [{sza_bin}] {i+1}/{len(tif_files)} chips done")

    # 6. Write tidy CSV
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep NDWI cutoff for the TR (NDWI) sensitivity branch."
    )
    parser.add_argument("--chips_root",   default=DEFAULT_CHIPS_ROOT,
                        help=f"Region chips root (default: {DEFAULT_CHIPS_ROOT})")
    parser.add_argument("--sza_bins",     default=DEFAULT_SZA_BINS,
                        help=f"Comma-separated SZA bins (default: {DEFAULT_SZA_BINS})")
    parser.add_argument("--ndwi_values",  default=DEFAULT_NDWI_VALUES,
                        help=f"Comma-separated NDWI thresholds (default: {DEFAULT_NDWI_VALUES})")
    parser.add_argument("--out_csv",      default=DEFAULT_OUT_CSV,
                        help=f"Tidy per-chip CSV (default: {DEFAULT_OUT_CSV})")
    parser.add_argument("--cloud_fractions_csv", default=None,
                        help="Per-chip cloud-fraction CSV from compute_chip_cloud_fractions.py; "
                             "when present, chips with cloud_fraction > --cloud_threshold are cloud_skipped")
    parser.add_argument("--cloud_threshold",     type=float, default=DEFAULT_CLOUD_THRESHOLD,
                        help=f"QA60 cloud-fraction cutoff (default: {DEFAULT_CLOUD_THRESHOLD})")
    parser.add_argument("--b03_idx",      type=int, default=1)
    parser.add_argument("--b08_idx",      type=int, default=2)
    args = parser.parse_args()

    sza_bins    = [s.strip() for s in args.sza_bins.split(",") if s.strip()]
    ndwi_values = [float(v.strip()) for v in args.ndwi_values.split(",") if v.strip()]
    cloud_fractions = load_cloud_fractions(args.cloud_fractions_csv)
    if cloud_fractions:
        n_above = sum(1 for v in cloud_fractions.values() if v is not None and v > args.cloud_threshold)
        print(f"Loaded {len(cloud_fractions)} cloud fractions; "
              f"{n_above} chips above threshold {args.cloud_threshold} will be cloud_skipped")

    sweep(
        chips_root      = args.chips_root,
        sza_bins        = sza_bins,
        ndwi_values     = ndwi_values,
        out_csv         = args.out_csv,
        cloud_fractions = cloud_fractions,
        cloud_threshold = args.cloud_threshold,
        b03_idx         = args.b03_idx,
        b08_idx         = args.b08_idx,
    )


if __name__ == "__main__":
    main()
