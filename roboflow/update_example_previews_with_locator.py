"""
Replace the leftmost panel in existing Roboflow example previews with locator maps.

This keeps the preview width unchanged by swapping panel 1 with a resized
"locator map" panel while preserving the remaining panels as-is.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace the leftmost panel in preview PNGs with matching locator maps."
    )
    parser.add_argument(
        "--preview-dir",
        default="roboflow/roboflow_examples",
        help="Directory containing *_preview.png files.",
    )
    parser.add_argument(
        "--locator-dir",
        default="roboflow/locator_maps",
        help="Directory containing locator PNGs named by chip stem.",
    )
    parser.add_argument(
        "--label",
        default="locator map",
        help="Top-bar label to draw on the locator panel.",
    )
    parser.add_argument(
        "--panel-count",
        type=int,
        default=4,
        help="Number of equal-width panels expected in each preview image.",
    )
    return parser.parse_args()


def add_label(image: Image.Image, text: str) -> Image.Image:
    labeled = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(labeled)
    bar_height = 20
    draw.rectangle((0, 0, labeled.width, bar_height), fill=(0, 0, 0, 180))
    draw.text((6, 4), text, fill=(255, 255, 255, 255))
    return labeled


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


def build_locator_panel(locator_path: Path, size: tuple[int, int], label: str) -> Image.Image:
    locator = Image.open(locator_path).convert("RGBA")
    locator = ImageOps.fit(locator, size, method=Image.Resampling.LANCZOS)
    return add_label(locator, label)


def replace_left_panel(preview_path: Path, locator_path: Path, label: str, panel_count: int) -> bool:
    preview = Image.open(preview_path).convert("RGBA")
    if preview.width % panel_count != 0:
        print(f"skip: panel width not divisible in {preview_path.name}")
        return False

    panel_width = preview.width // panel_count
    locator_panel = build_locator_panel(locator_path, (panel_width, preview.height), label)
    remainder = preview.crop((panel_width, 0, preview.width, preview.height))

    updated = Image.new("RGBA", preview.size, (255, 255, 255, 255))
    updated.paste(locator_panel, (0, 0))
    updated.paste(remainder, (panel_width, 0))
    updated.convert("RGB").save(preview_path)
    return True


def main() -> None:
    args = parse_args()
    preview_dir = Path(args.preview_dir).expanduser().resolve()
    locator_dir = Path(args.locator_dir).expanduser().resolve()

    updated = 0
    missing = 0
    for preview_path in sorted(preview_dir.glob("*_preview.png")):
        stem = preview_path.stem.removesuffix("_preview")
        locator_path = find_locator_image(locator_dir, stem)
        if locator_path is None:
            missing += 1
            continue
        if replace_left_panel(preview_path, locator_path, args.label, args.panel_count):
            updated += 1
            print(f"updated: {preview_path.name}")

    print(f"done: updated={updated} missing_locator={missing}")


if __name__ == "__main__":
    main()
