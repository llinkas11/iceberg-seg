"""
compute_chip_cloud_fractions.py — Compute per-chip cloud fraction from the
Sentinel-2 L1C MSK_CLASSI_B00 cloud bitmask.

For each chip under chips_root, locate the parent SAFE.zip in downloads_root,
open MSK_CLASSI_B00.jp2 (bit 0 = opaque cloud, bit 1 = cirrus) inside the zip
via /vsizip/, sample the window covering the chip's spatial extent, and write
the cloud-pixel fraction to a tidy CSV.

This artifact feeds sweep_ndwi_threshold.py as a per-chip cloud filter so the
NDWI sweep aggregates exclude cloud-contaminated chips (the IC chip filter
alone misses chips with cloud cover just under the 15% B08-bright threshold).

Reuses helpers from cloud_filter_roboflow.py. Batching: opens each scene's
QA60 raster once and reads windows for every chip from that scene, instead of
opening the zip 8410 times.

Usage:
  python compute_chip_cloud_fractions.py \\
      --chips_root      /mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/chips/KQ \\
      --downloads_root  /mnt/research/v.gomezgilyaspik/students/smishra/sentinel2_downloads/KQ \\
      --out_csv         /home/llinkas/S2-iceberg-areas/sweeps/cloud_fractions_kq.csv

Output CSV columns:
  region, sza_bin, chip_stem, cloud_fraction, qa60_available

cloud_fraction is empty when qa60_available is False.

Deploy:
  rsync -av iceberg-rework/scripts/compute_chip_cloud_fractions.py \\
            smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import argparse
import csv
import os
import re
import warnings
from collections import defaultdict
from glob import glob

import numpy as np
import rasterio
import rasterio.errors
import rasterio.windows
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.warp import transform_bounds

from cloud_filter_roboflow import (
    MSK_CLASSI_CLOUD,
    find_qa60_entry,
)

warnings.filterwarnings("ignore")

DEFAULT_CHIPS_ROOT     = "/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/chips/KQ"
DEFAULT_DOWNLOADS_ROOT = "/mnt/research/v.gomezgilyaspik/students/smishra/sentinel2_downloads/KQ"
DEFAULT_OUT_CSV        = "/home/llinkas/S2-iceberg-areas/sweeps/cloud_fractions_kq.csv"
DEFAULT_SZA_BINS       = "sza_lt65,sza_65_70,sza_70_75,sza_gt75"

CHIP_STEM_RE = re.compile(r"^(.+)_r(\d+)_c(\d+)$")


def parse_stem(chip_stem):
    """Split a chip filename stem into (scene_stem, row, col)."""
    m = CHIP_STEM_RE.match(chip_stem)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def index_zips(downloads_root):
    """Build a {scene_stem: zip_path} dict by globbing all SAFE.zip files once."""
    pattern = os.path.join(downloads_root, "**", "*.SAFE.zip")
    paths   = glob(pattern, recursive=True)
    index   = {}
    for p in paths:
        stem = os.path.basename(p).replace(".SAFE.zip", "")
        index[stem] = p
    return index


def cloud_fraction_for_chip(qa60_src, chip_bounds, chip_crs):
    """Sample MSK_CLASSI_B00 at chip_bounds; return cloud-pixel fraction in [0, 1].

    qa60_src is an already-open rasterio dataset on the L1C cloud-classification
    JPEG2000. Returns None if the chip falls outside the QA60 raster.
    """
    # 1. Reproject chip bounds to QA60 CRS if needed
    if qa60_src.crs and chip_crs and qa60_src.crs.to_epsg() != chip_crs.to_epsg():
        bounds = transform_bounds(
            chip_crs, qa60_src.crs,
            chip_bounds.left, chip_bounds.bottom,
            chip_bounds.right, chip_bounds.top,
        )
    else:
        bounds = (chip_bounds.left, chip_bounds.bottom, chip_bounds.right, chip_bounds.top)

    win = window_from_bounds(*bounds, transform=qa60_src.transform)

    # 2. Clamp to raster extent
    full = rasterio.windows.Window(0, 0, qa60_src.width, qa60_src.height)
    try:
        win = win.intersection(full)
    except rasterio.errors.WindowError:
        return None
    if win.width < 1 or win.height < 1:
        return None

    qa60 = qa60_src.read(1, window=win)
    if qa60.size == 0:
        return None

    # 3. Bit 0 (opaque cloud) OR bit 1 (cirrus) = cloud
    cloud_px = int(((qa60.astype(np.uint8) & MSK_CLASSI_CLOUD) > 0).sum())
    return cloud_px / qa60.size


def compute(chips_root, downloads_root, sza_bins, out_csv):
    """Iterate all chips under chips_root, compute cloud_fraction per chip."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    region = os.path.basename(chips_root.rstrip("/"))

    # 1. Build a one-shot index of {scene_stem: zip_path}
    zip_index = index_zips(downloads_root)
    print(f"Indexed {len(zip_index)} SAFE.zip files under {downloads_root}")

    # 2. Group chips by (sza_bin, scene_stem) so each zip opens once
    by_scene = defaultdict(list)
    for sza_bin in sza_bins:
        chip_glob = os.path.join(chips_root, sza_bin, "tifs", "*.tif")
        for tif in sorted(glob(chip_glob)):
            stem = os.path.splitext(os.path.basename(tif))[0]
            parsed = parse_stem(stem)
            if parsed is None:
                continue
            scene_stem, row, col = parsed
            by_scene[(sza_bin, scene_stem)].append((stem, tif))
    total_chips = sum(len(v) for v in by_scene.values())
    print(f"Found {total_chips} chips across {len(by_scene)} scenes")

    # 3. For each scene, open QA60 once, read window per chip
    rows = []
    n_no_zip   = 0
    n_no_qa60  = 0
    n_oob      = 0
    n_ok       = 0
    for i, ((sza_bin, scene_stem), chip_list) in enumerate(sorted(by_scene.items())):
        zip_path = zip_index.get(scene_stem)
        if zip_path is None:
            for stem, _ in chip_list:
                rows.append({
                    "region"          : region,
                    "sza_bin"         : sza_bin,
                    "chip_stem"       : stem,
                    "cloud_fraction"  : "",
                    "qa60_available"  : False,
                })
            n_no_zip += len(chip_list)
            print(f"  [{i+1}/{len(by_scene)}] {sza_bin}/{scene_stem[:50]}: NO ZIP ({len(chip_list)} chips)")
            continue

        qa60_entry = find_qa60_entry(zip_path)
        if qa60_entry is None:
            for stem, _ in chip_list:
                rows.append({
                    "region"          : region,
                    "sza_bin"         : sza_bin,
                    "chip_stem"       : stem,
                    "cloud_fraction"  : "",
                    "qa60_available"  : False,
                })
            n_no_qa60 += len(chip_list)
            print(f"  [{i+1}/{len(by_scene)}] {sza_bin}/{scene_stem[:50]}: NO QA60 ({len(chip_list)} chips)")
            continue

        qa60_vsi = f"/vsizip/{zip_path}/{qa60_entry}"
        try:
            with rasterio.open(qa60_vsi) as qa60_src:
                for stem, tif in chip_list:
                    with rasterio.open(tif) as chip_src:
                        bounds   = chip_src.bounds
                        chip_crs = chip_src.crs
                    frac = cloud_fraction_for_chip(qa60_src, bounds, chip_crs)
                    if frac is None:
                        rows.append({
                            "region"          : region,
                            "sza_bin"         : sza_bin,
                            "chip_stem"       : stem,
                            "cloud_fraction"  : "",
                            "qa60_available"  : False,
                        })
                        n_oob += 1
                    else:
                        rows.append({
                            "region"          : region,
                            "sza_bin"         : sza_bin,
                            "chip_stem"       : stem,
                            "cloud_fraction"  : round(frac, 4),
                            "qa60_available"  : True,
                        })
                        n_ok += 1
        except Exception as e:
            print(f"  [{i+1}/{len(by_scene)}] {sza_bin}/{scene_stem[:50]}: open failed {e}")
            for stem, _ in chip_list:
                rows.append({
                    "region"          : region,
                    "sza_bin"         : sza_bin,
                    "chip_stem"       : stem,
                    "cloud_fraction"  : "",
                    "qa60_available"  : False,
                })
            n_no_qa60 += len(chip_list)
            continue

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(by_scene)}] processed; ok={n_ok} oob={n_oob} no_qa60={n_no_qa60} no_zip={n_no_zip}")

    # 4. Write CSV
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["region", "sza_bin", "chip_stem", "cloud_fraction", "qa60_available"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_csv}")
    print(f"  ok       : {n_ok}")
    print(f"  out-of-bounds: {n_oob}")
    print(f"  no QA60  : {n_no_qa60}")
    print(f"  no ZIP   : {n_no_zip}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chips_root",     default=DEFAULT_CHIPS_ROOT)
    parser.add_argument("--downloads_root", default=DEFAULT_DOWNLOADS_ROOT)
    parser.add_argument("--sza_bins",       default=DEFAULT_SZA_BINS)
    parser.add_argument("--out_csv",        default=DEFAULT_OUT_CSV)
    args = parser.parse_args()

    sza_bins = [s.strip() for s in args.sza_bins.split(",") if s.strip()]
    compute(
        chips_root     = args.chips_root,
        downloads_root = args.downloads_root,
        sza_bins       = sza_bins,
        out_csv        = args.out_csv,
    )


if __name__ == "__main__":
    main()
