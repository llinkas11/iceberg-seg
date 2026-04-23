#!/usr/bin/env python3
"""
upload_roboflow_batches.py

Upload S2 chips (excluding sza_lt65) to an existing Roboflow project,
organized into 16 batches: batch1-batch15 (~200 images each), misc (remainder).

Chips are already rendered as PNGs in <bin>/pngs/ subdirectories — those are
uploaded directly (no tif conversion needed).

Usage on moosehead:
  conda activate iceberg-unet
  python ~/S2-iceberg-areas/upload_roboflow_batches.py \
      --api_key  YOUR_ROBOFLOW_API_KEY \
      --project  YOUR_PROJECT_ID \
      [--chips_dir /mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/chips] \
      [--images_per_batch 200] \
      [--dry_run]

Your Roboflow project ID is the slug in the URL:
  https://app.roboflow.com/<workspace>/<project-id>/

Always dry-run first to verify the batch plan.
"""

import argparse
import glob
import os
import random
import time

import requests

BATCH_NAMES = [f"batch{i}" for i in range(1, 16)] + ["misc"]
API_URL = "https://api.roboflow.com/dataset/{project}/upload"


def upload_image(png_path, project, api_key, batch_name, dry_run=False):
    """Upload one PNG to Roboflow. Returns True on success."""
    fname = os.path.basename(png_path)

    if dry_run:
        print(f"  [dry_run] {fname} -> {batch_name}")
        return True

    url = API_URL.format(project=project)
    params = {"api_key": api_key, "batch": batch_name, "name": fname}

    try:
        with open(png_path, "rb") as f:
            resp = requests.post(
                url,
                params=params,
                files={"file": (fname, f, "image/png")},
                timeout=30,
            )
        resp.raise_for_status()
        return True
    except requests.exceptions.HTTPError:
        print(f"  [http error] {fname}: {resp.status_code} -- {resp.text[:200]}")
        return False
    except Exception as e:
        print(f"  [upload error] {fname}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, help="Roboflow API key")
    parser.add_argument("--project", required=True,
                        help="Roboflow project ID (slug from URL)")
    parser.add_argument(
        "--chips_dir",
        default="/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/chips",
        help="Root chips directory (searches <bin>/pngs/ subdirs, excludes sza_lt65)",
    )
    parser.add_argument("--images_per_batch", type=int, default=200,
                        help="Images per named batch (default 200); remainder goes to misc")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default 42)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between uploads to avoid rate-limiting (default 0.3)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print plan and first few filenames without uploading")
    args = parser.parse_args()

    # Collect all PNGs from */pngs/ subdirectories, excluding sza_lt65
    all_pngs = []
    for png in glob.glob(os.path.join(args.chips_dir, "**", "pngs", "*.png"), recursive=True):
        parts = os.path.normpath(png).split(os.sep)
        if "sza_lt65" in parts:
            continue
        all_pngs.append(png)

    if not all_pngs:
        print(f"No PNGs found under {args.chips_dir}/*/pngs/ (excluding sza_lt65). Check the path.")
        return

    random.seed(args.seed)
    random.shuffle(all_pngs)

    # Assign to batches: batch1-15 get images_per_batch each, misc gets remainder
    n = args.images_per_batch
    batched = {}
    for i, name in enumerate(BATCH_NAMES[:-1]):   # batch1-batch15
        batched[name] = all_pngs[i * n : (i + 1) * n]
    batched["misc"] = all_pngs[15 * n : 16 * n]    # capped at images_per_batch

    total = sum(len(v) for v in batched.values())
    print(f"PNGs found (excluding sza_lt65): {len(all_pngs)}")
    print(f"Batch plan ({args.images_per_batch} per batch):")
    for name in BATCH_NAMES:
        print(f"  {name:10s}: {len(batched[name])} images")
    print(f"  {'TOTAL':10s}: {total}")
    print()

    if args.dry_run:
        print("[dry_run] No uploads will be made.")
        print("[dry_run] First 5 files assigned to batch1:")
        for p in batched["batch1"][:5]:
            print(f"  {p}")
        return

    # Upload
    ok = err = 0
    for batch_name in BATCH_NAMES:
        chips = batched[batch_name]
        if not chips:
            print(f"[{batch_name}] empty, skipping.")
            continue
        print(f"\n[{batch_name}] uploading {len(chips)} images...")
        for j, png in enumerate(chips):
            success = upload_image(png, args.project, args.api_key, batch_name)
            if success:
                ok += 1
            else:
                err += 1
            if (j + 1) % 25 == 0 or (j + 1) == len(chips):
                print(f"  {j+1}/{len(chips)} -- ok={ok} err={err}")
            time.sleep(args.delay)

    print(f"\nDone. {ok} uploaded successfully, {err} errors.")


if __name__ == "__main__":
    main()
