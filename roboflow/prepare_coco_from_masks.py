"""
Prepare a Roboflow-ready COCO segmentation package from image and mask folders.

This script is intended for the Step 3 annotation workflow:
  model predictions -> Roboflow upload -> SAM-assisted correction -> export

Example:
  python3 roboflow/prepare_coco_from_masks.py \
      --images-dir path/to/images \
      --masks-dir path/to/masks \
      --out-dir roboflow/upload_batch_01 \
      --class-map 1:iceberg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import rasterio
except ImportError:
    rasterio = None


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert predicted label masks into a COCO segmentation package."
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing source images to upload to Roboflow.",
    )
    parser.add_argument(
        "--masks-dir",
        required=True,
        help="Directory containing predicted label masks with matching basenames.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where the Roboflow upload package will be created.",
    )
    parser.add_argument(
        "--class-map",
        nargs="+",
        default=["1:iceberg"],
        help="Class mapping in the form <pixel_value>:<class_name>. Default: 1:iceberg",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=20.0,
        help="Minimum contour area in pixels to keep. Default: 20.",
    )
    parser.add_argument(
        "--mask-suffix",
        default="",
        help="Optional suffix appended to the image stem in mask filenames, e.g. '_pred'.",
    )
    parser.add_argument(
        "--merge-values",
        nargs="*",
        type=int,
        default=None,
        help="Optional mask values to merge into one output class, e.g. 3 4",
    )
    parser.add_argument(
        "--merge-name",
        default="iceberg",
        help="Output class name when using --merge-values. Default: iceberg",
    )
    parser.add_argument(
        "--include-values",
        nargs="*",
        type=int,
        default=None,
        help="Mask values to include directly in one merged output class, e.g. 3",
    )
    parser.add_argument(
        "--fill-values",
        nargs="*",
        type=int,
        default=None,
        help="Mask values whose enclosed regions should be filled into one merged output class, e.g. 4",
    )
    return parser.parse_args()


def parse_class_map(entries: list[str]) -> dict[int, str]:
    class_map: dict[int, str] = {}
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"Invalid class map entry '{entry}'. Use <pixel_value>:<class_name>.")
        pixel_value_str, class_name = entry.split(":", 1)
        pixel_value = int(pixel_value_str)
        class_name = class_name.strip()
        if not class_name:
            raise ValueError(f"Class name cannot be empty in '{entry}'.")
        class_map[pixel_value] = class_name
    return class_map


def merge_mask_values(mask: np.ndarray, values: set[int]) -> np.ndarray:
    return np.isin(mask, list(values)).astype(np.uint8)


def build_combined_mask(mask: np.ndarray, include_values: set[int], fill_values: list[int]) -> np.ndarray:
    combined = np.zeros(mask.shape, dtype=np.uint8)

    if include_values:
        combined[np.isin(mask, list(include_values))] = 1

    for value in fill_values:
        binary = (mask == value).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(combined, contours, contourIdx=-1, color=1, thickness=cv2.FILLED)

    return combined


def list_images(images_dir: Path) -> list[Path]:
    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return image_paths


def find_mask(mask_dir: Path, image_path: Path) -> Path:
    for suffix in VALID_SUFFIXES:
        candidate = mask_dir / f"{image_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No matching mask found for image '{image_path.name}' in {mask_dir}")


def find_mask_with_suffix(mask_dir: Path, image_path: Path, mask_suffix: str) -> Path:
    for suffix in VALID_SUFFIXES:
        candidate = mask_dir / f"{image_path.stem}{mask_suffix}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No matching mask found for image '{image_path.name}' with suffix '{mask_suffix}' in {mask_dir}"
    )


def load_mask(mask_path: Path) -> np.ndarray:
    if mask_path.suffix.lower() in {".tif", ".tiff"}:
        if rasterio is None:
            raise ImportError("rasterio is required to read TIFF masks.")
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
    else:
        mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.uint8)


def stretch_to_u8(array: np.ndarray) -> np.ndarray:
    valid = np.isfinite(array)
    if not np.any(valid):
        return np.zeros(array.shape, dtype=np.uint8)
    values = array[valid]
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = np.clip((array - lo) / (hi - lo), 0, 1)
    scaled[~valid] = 0
    return (scaled * 255).astype(np.uint8)


def load_image_for_upload(image_path: Path) -> tuple[Image.Image, int, int]:
    suffix = image_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        if rasterio is None:
            raise ImportError("rasterio is required to read TIFF images.")
        with rasterio.open(image_path) as src:
            data = src.read()
        if data.shape[0] >= 3:
            rgb = np.stack([stretch_to_u8(data[i].astype(np.float32)) for i in range(3)], axis=-1)
        else:
            gray = stretch_to_u8(data[0].astype(np.float32))
            rgb = np.stack([gray, gray, gray], axis=-1)
        image = Image.fromarray(rgb)
    else:
        image = Image.open(image_path).convert("RGB")

    width, height = image.size
    return image, width, height


def mask_to_polygons(mask: np.ndarray, class_id: int, min_area: float) -> list[list[float]]:
    binary = (mask == class_id).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        flattened = contour.reshape(-1, 2).astype(float).flatten().tolist()
        if len(flattened) >= 6:
            polygons.append(flattened)
    return polygons


def polygon_area(coords: list[float]) -> float:
    xs = coords[0::2]
    ys = coords[1::2]
    total = 0.0
    for idx in range(len(xs)):
        next_idx = (idx + 1) % len(xs)
        total += xs[idx] * ys[next_idx] - xs[next_idx] * ys[idx]
    return abs(total) * 0.5


def build_coco(class_map: dict[int, str]) -> dict:
    categories = [
        {"id": class_id, "name": class_name, "supercategory": "iceberg"}
        for class_id, class_name in sorted(class_map.items())
    ]
    return {
        "info": {"description": "Predicted iceberg masks prepared for Roboflow correction"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir).expanduser().resolve()
    masks_dir = Path(args.masks_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_images_dir = out_dir / "images"

    if args.include_values or args.fill_values:
        class_map = {1: args.merge_name}
        include_values = set(args.include_values or [])
        fill_values = list(args.fill_values or [])
        merge_values = None
    elif args.merge_values:
        class_map = {1: args.merge_name}
        merge_values = set(args.merge_values)
        include_values = set()
        fill_values = []
    else:
        class_map = parse_class_map(args.class_map)
        merge_values = None
        include_values = set()
        fill_values = []
    image_paths = list_images(images_dir)

    out_images_dir.mkdir(parents=True, exist_ok=True)
    coco = build_coco(class_map)

    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        if args.mask_suffix:
            mask_path = find_mask_with_suffix(masks_dir, image_path, args.mask_suffix)
        else:
            mask_path = find_mask(masks_dir, image_path)
        mask = load_mask(mask_path)
        if include_values or fill_values:
            mask = build_combined_mask(mask, include_values, fill_values)
        elif merge_values:
            mask = merge_mask_values(mask, merge_values)
        image, width, height = load_image_for_upload(image_path)

        if mask.shape[:2] != (height, width):
            raise ValueError(
                f"Mask/image size mismatch for '{image_path.name}': "
                f"image={(width, height)} mask={mask.shape[::-1]}"
            )

        target_image_name = f"{image_path.stem}.png"
        target_image_path = out_images_dir / target_image_name
        image.save(target_image_path)

        coco["images"].append(
            {
                "id": image_id,
                "file_name": f"images/{target_image_name}",
                "width": width,
                "height": height,
            }
        )

        for class_id in sorted(class_map):
            polygons = mask_to_polygons(mask, class_id, args.min_area)
            for polygon in polygons:
                xs = polygon[0::2]
                ys = polygon[1::2]
                x_min = min(xs)
                y_min = min(ys)
                bbox_width = max(xs) - x_min
                bbox_height = max(ys) - y_min

                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "segmentation": [polygon],
                        "area": round(polygon_area(polygon), 2),
                        "bbox": [round(x_min, 2), round(y_min, 2), round(bbox_width, 2), round(bbox_height, 2)],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "_annotations.coco.json").open("w", encoding="utf-8") as handle:
        json.dump(coco, handle)

    print(f"Prepared {len(coco['images'])} images and {len(coco['annotations'])} annotations.")
    print(f"Upload folder: {out_dir}")
    print("Next step: zip the folder and import it into Roboflow as COCO Segmentation.")


if __name__ == "__main__":
    main()
