"""
Generate labeled preview panels for a batch of Roboflow example PNGs.

For each PNG in a Roboflow export folder, this script finds:
- the matching original color chip in S2UnetPlusPlus/imgs
- the matching mask in S2UnetPlusPlus/masks
- an optional matching locator PNG in a locator folder

It then creates a preview image with panels:
- locator map
- original color
- shadow
- iceberg
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create preview panels for Roboflow PNG examples.")
    parser.add_argument(
        "--pngs",
        nargs="+",
        required=True,
        help="One or more PNG files from a Roboflow export folder.",
    )
    parser.add_argument(
        "--imgs-dir",
        default="S2-iceberg-areas/S2UnetPlusPlus/imgs",
        help="Directory containing original .tif chips.",
    )
    parser.add_argument(
        "--masks-dir",
        default="S2-iceberg-areas/S2UnetPlusPlus/masks",
        help="Directory containing *_ground_truth.tif masks.",
    )
    parser.add_argument(
        "--out-dir",
        default="roboflow/roboflow_examples",
        help="Directory where preview PNGs will be saved.",
    )
    parser.add_argument(
        "--locator-dir",
        default=None,
        help="Optional directory containing pre-rendered locator PNGs named by chip stem.",
    )
    parser.add_argument(
        "--aoi-gpkg",
        default="S2-iceberg-areas/aois_greenland_area_distributions.gpkg",
        help="Optional AOI GeoPackage for building locator panels directly in Python.",
    )
    return parser.parse_args()


def find_locator_image(locator_dir: Path, stem: str) -> Path | None:
    candidates = [
        locator_dir / f"{stem}.png",
        locator_dir / f"{stem}_locator.png",
        locator_dir / f"{stem}__locator.png",
        locator_dir / f"{stem}_map.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    script_path = root / "roboflow" / "preview_mask_classes.py"
    imgs_dir = (root / args.imgs_dir).resolve()
    masks_dir = (root / args.masks_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    locator_dir = None
    if args.locator_dir:
        locator_dir = (root / args.locator_dir).resolve()
    aoi_gpkg = None
    if args.aoi_gpkg:
        aoi_gpkg = (root / args.aoi_gpkg).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for png_arg in args.pngs:
        png_path = Path(png_arg).expanduser()
        if not png_path.is_absolute():
            png_path = (root / png_path).resolve()
        else:
            png_path = png_path.resolve()

        stem = png_path.stem
        image_path = imgs_dir / f"{stem}.tif"
        mask_path = masks_dir / f"{stem}_ground_truth.tif"
        out_path = out_dir / f"{stem}_preview.png"
        locator_path = find_locator_image(locator_dir, stem) if locator_dir else None

        if not image_path.exists():
            print(f"skip: missing original image for {png_path.name}", file=sys.stderr)
            continue
        if not mask_path.exists():
            print(f"skip: missing mask for {png_path.name}", file=sys.stderr)
            continue

        cmd = [
            sys.executable,
            str(script_path),
            "--original-image",
            str(image_path),
            "--image",
            str(png_path),
            "--mask",
            str(mask_path),
            "--out",
            str(out_path),
            "--label-map",
            "2:shadow",
            "3:iceberg",
            "--show-values",
            "2",
            "3",
        ]
        if locator_path:
            cmd.extend(["--locator-image", str(locator_path)])
        elif aoi_gpkg and aoi_gpkg.exists():
            cmd.extend(["--aoi-gpkg", str(aoi_gpkg)])
        subprocess.run(cmd, check=True)
        print(f"made: {out_path}")


if __name__ == "__main__":
    main()
