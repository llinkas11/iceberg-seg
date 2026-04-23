"""
Inspect segmentation mask label values to verify class encoding before export.

This is useful for debugging Roboflow import issues when classes appear to be
swapped or mislabeled after conversion from raster masks to COCO polygons.

Example:
  python3 roboflow/check_mask_labels.py \
      --mask /path/to/example_ground_truth.tif

  python3 roboflow/check_mask_labels.py \
      --mask-dir /path/to/masks \
      --limit 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError as exc:
    raise SystemExit("rasterio is required for this script.") from exc


VALID_SUFFIXES = {".tif", ".tiff", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print unique mask label values and counts for segmentation masks."
    )
    parser.add_argument("--mask", help="Path to one mask file to inspect.")
    parser.add_argument("--mask-dir", help="Directory of mask files to inspect.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of mask files to inspect when using --mask-dir. Default: 5.",
    )
    return parser.parse_args()


def load_mask(mask_path: Path) -> np.ndarray:
    if mask_path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(mask_path) as src:
            return src.read(1)
    from PIL import Image

    return np.array(Image.open(mask_path))


def describe_mask(mask_path: Path) -> None:
    mask = load_mask(mask_path)
    values, counts = np.unique(mask, return_counts=True)
    print(f"\nMask: {mask_path}")
    print(f"Shape: {mask.shape}")
    print("Label counts:")
    for value, count in zip(values.tolist(), counts.tolist()):
        print(f"  value={value:<4} count={count}")


def list_masks(mask_dir: Path) -> list[Path]:
    masks = sorted(
        path for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )
    if not masks:
        raise FileNotFoundError(f"No mask files found in {mask_dir}")
    return masks


def main() -> None:
    args = parse_args()
    if not args.mask and not args.mask_dir:
        raise SystemExit("Provide either --mask or --mask-dir.")

    if args.mask:
        describe_mask(Path(args.mask).expanduser().resolve())
        return

    mask_dir = Path(args.mask_dir).expanduser().resolve()
    masks = list_masks(mask_dir)[: args.limit]
    print(f"Inspecting {len(masks)} mask file(s) from {mask_dir}")
    for mask_path in masks:
        describe_mask(mask_path)


if __name__ == "__main__":
    main()
