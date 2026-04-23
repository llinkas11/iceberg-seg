"""
rebin_downloads.py — Fix SZA bin assignments for already-downloaded scenes.

The original download used a rough monthly SZA estimate (October → 76°),
which put early-October scenes (real SZA ~70-75°) into sza_gt75 instead of sza_70_75.

This script:
  1. Walks sentinel2_downloads/{region}/{sza_bin}/*.zip
  2. Parses the exact acquisition UTC datetime from the filename
  3. Computes real solar zenith angle at the region centre using pysolar
  4. Moves the zip to the correct bin folder if it is wrong

Usage:
  python rebin_downloads.py --downloads_dir /mnt/research/v.gomezgilyaspik/students/smishra/sentinel2_downloads
"""

import os
import re
import shutil
import argparse
from datetime import datetime, timezone

try:
    from pysolar.solar import get_altitude
except ImportError:
    raise SystemExit("Install pysolar:  pip install pysolar")


# Region centre coordinates (WGS84) used for SZA calculation
REGION_CENTRES = {
    "KQ": (68.0, -32.0),   # Kangerlussuaq Fjord
    "SK": (66.0, -38.0),   # Sermilik Fjord
}

ALL_BINS = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]


def get_sza_bin(sza):
    if sza < 65:
        return "sza_lt65"
    elif sza < 70:
        return "sza_65_70"
    elif sza < 75:
        return "sza_70_75"
    else:
        return "sza_gt75"


def parse_acquisition_time(filename):
    """
    Extract UTC acquisition datetime from a Sentinel-2 product name.
    Format: S2X_MSIL1C_YYYYMMDDTHHMMSS_...
    """
    match = re.search(r"_(\d{8}T\d{6})_", filename)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def compute_sza(lat, lon, dt):
    """Return solar zenith angle in degrees."""
    altitude = get_altitude(lat, lon, dt)
    return 90.0 - altitude


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads_dir", required=True,
                        help="Root sentinel2_downloads directory")
    parser.add_argument("--region", default=None,
                        help="Only rebin this region (e.g. KQ or SK)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print moves without executing them")
    args = parser.parse_args()

    root = args.downloads_dir
    moved = 0
    skipped = 0
    errors = 0

    for region, (lat, lon) in REGION_CENTRES.items():
        if args.region and region != args.region:
            continue
        region_dir = os.path.join(root, region)
        if not os.path.isdir(region_dir):
            print(f"[skip] {region_dir} not found")
            continue

        for current_bin in ALL_BINS:
            bin_dir = os.path.join(region_dir, current_bin)
            if not os.path.isdir(bin_dir):
                continue

            for fname in os.listdir(bin_dir):
                if not fname.endswith(".zip"):
                    continue

                fpath = os.path.join(bin_dir, fname)
                dt = parse_acquisition_time(fname)
                if dt is None:
                    print(f"[error] Cannot parse date from: {fname}")
                    errors += 1
                    continue

                try:
                    sza = compute_sza(lat, lon, dt)
                except Exception as e:
                    print(f"[error] SZA calc failed for {fname}: {e}")
                    errors += 1
                    continue

                correct_bin = get_sza_bin(sza)

                if correct_bin == current_bin:
                    print(f"[ok]    {region}/{current_bin}/{fname}  SZA={sza:.1f}")
                    skipped += 1
                else:
                    dest_dir = os.path.join(region_dir, correct_bin)
                    dest_path = os.path.join(dest_dir, fname)
                    print(f"[move]  {region}/{current_bin} -> {correct_bin}  SZA={sza:.1f}  {fname}")
                    if not args.dry_run:
                        os.makedirs(dest_dir, exist_ok=True)
                        shutil.move(fpath, dest_path)
                    moved += 1

    print(f"\nDone.  moved={moved}  already-correct={skipped}  errors={errors}")
    if args.dry_run:
        print("(dry run — no files were moved)")


if __name__ == "__main__":
    main()