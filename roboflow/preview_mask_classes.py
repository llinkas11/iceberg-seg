"""
Create a quick visual preview of an image chip and its mask classes.

This helps identify the semantic meaning of raster mask values before exporting
to Roboflow.

Example:
  python3 roboflow/preview_mask_classes.py \
      --image /path/to/chip.tif \
      --mask /path/to/chip_ground_truth.tif \
      --out roboflow/mask_preview.png
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import rasterio
except ImportError as exc:
    raise SystemExit("rasterio is required for this script.") from exc
from rasterio.warp import transform, transform_bounds


CLASS_COLORS = {
    1: (220, 220, 220),   # light gray
    2: (255, 215, 0),     # yellow
    3: (0, 255, 255),     # cyan
    4: (255, 140, 0),     # orange
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview an image chip with per-class mask overlays."
    )
    parser.add_argument("--image", required=True, help="Path to the source image chip.")
    parser.add_argument("--mask", required=True, help="Path to the mask file.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument(
        "--original-image",
        help="Optional path to the original color chip to show as an extra panel.",
    )
    parser.add_argument(
        "--aoi-gpkg",
        help="Optional path to AOI GeoPackage used to draw a geographic locator panel.",
    )
    parser.add_argument(
        "--locator-image",
        help="Optional pre-rendered locator image, e.g. exported from MATLAB.",
    )
    parser.add_argument(
        "--label-map",
        nargs="*",
        default=[],
        help="Optional display labels in the form <value>:<name>, e.g. 2:shadow 3:iceberg",
    )
    parser.add_argument(
        "--show-values",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of mask values to include in the preview, e.g. 2 3",
    )
    parser.add_argument(
        "--scale-crs",
        default=None,
        help="Projected CRS to use for meter-based scale calculations. If omitted, choose one from chip location.",
    )
    return parser.parse_args()


def add_scale_bar(
    image: Image.Image,
    length_px: int,
    label: str,
    *,
    margin: int = 12,
    bar_height: int = 5,
) -> Image.Image:
    out = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)
    x0 = margin
    y1 = out.height - margin
    x1 = x0 + length_px
    y0 = y1 - bar_height
    draw.rectangle((x0 - 6, y0 - 18, x1 + 6, y1 + 6), fill=(255, 255, 255, 200))
    draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, 255))
    draw.text((x0, y0 - 15), label, fill=(0, 0, 0, 255))
    return out


def add_corner_scale_bar(
    image: Image.Image,
    length_px: int,
    label: str,
    *,
    margin: int = 12,
    bar_height: int = 3,
    color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> Image.Image:
    out = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)
    text_bbox = draw.textbbox((0, 0), label)
    text_width = text_bbox[2] - text_bbox[0]
    total_width = max(text_width, length_px)
    x0 = out.width - margin - total_width
    y_text = margin
    y0 = y_text + 16
    x1 = x0 + length_px
    y1 = y0 + bar_height
    draw.text((x0, y_text), label, fill=color)
    draw.rectangle((x0, y0, x1, y1), fill=color)
    draw.line((x0, y0, x0, y1), fill=color, width=bar_height)
    draw.line((x1, y0, x1, y1), fill=color, width=bar_height)
    return out


def choose_scale_crs(image_path: Path) -> str | None:
    with rasterio.open(image_path) as src:
        if src.crs is None:
            return None
        bounds = src.bounds
        xs, ys = transform(
            src.crs,
            "EPSG:4326",
            [(bounds.left + bounds.right) / 2],
            [(bounds.bottom + bounds.top) / 2],
        )

    lon = xs[0]
    lat = ys[0]

    if lat >= 60:
        return "EPSG:3413"
    if lat <= -60:
        return "EPSG:3031"

    zone = int((lon + 180) // 6) + 1
    zone = max(1, min(zone, 60))
    if lat >= 0:
        return f"EPSG:{32600 + zone}"
    return f"EPSG:{32700 + zone}"


def get_scale_length_px(
    image_path: Path,
    scale_length_m: float = 100.0,
    scale_crs: str = "EPSG:5938",
) -> int | None:
    with rasterio.open(image_path) as src:
        if src.crs is None or src.width <= 0:
            return None
        try:
            min_x, _, max_x, _ = transform_bounds(src.crs, scale_crs, *src.bounds)
        except Exception:
            return None

        width_m = abs(max_x - min_x)
        if width_m <= 0:
            return None

        length_px = int(round(scale_length_m / width_m * src.width))
        return max(1, min(length_px, src.width))


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


def load_image_chip(image_path: Path) -> Image.Image:
    with rasterio.open(image_path) as src:
        data = src.read()
    if data.shape[0] >= 3:
        rgb = np.stack([stretch_to_u8(data[i].astype(np.float32)) for i in range(3)], axis=-1)
    else:
        gray = stretch_to_u8(data[0].astype(np.float32))
        rgb = np.stack([gray, gray, gray], axis=-1)
    return Image.fromarray(rgb)


def load_mask(mask_path: Path) -> np.ndarray:
    with rasterio.open(mask_path) as src:
        return src.read(1)


def mask_to_overlay(mask: np.ndarray, selected_value: int) -> Image.Image:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    color = CLASS_COLORS.get(selected_value, (255, 0, 255))
    where = mask == selected_value
    rgba[where, 0] = color[0]
    rgba[where, 1] = color[1]
    rgba[where, 2] = color[2]
    rgba[where, 3] = 150
    return Image.fromarray(rgba, mode="RGBA")


def add_label(
    image: Image.Image,
    text: str,
    scale_label: str | None = None,
    scale_length_px: int | None = None,
) -> Image.Image:
    labeled = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(labeled)
    bar_height = 20
    draw.rectangle((0, 0, labeled.width, bar_height), fill=(0, 0, 0, 180))
    draw.text((6, 4), text, fill=(255, 255, 255, 255))

    if scale_label and scale_length_px:
        label_bbox = draw.textbbox((0, 0), scale_label)
        label_width = label_bbox[2] - label_bbox[0]
        pad = 8
        x_text = labeled.width - label_width - pad
        y_text = 4
        # Draw a chip-specific scale line to the left of the scale text.
        max_line_len = max(1, x_text - 18)
        line_len = max(1, min(scale_length_px, max_line_len))
        line_x1 = x_text - 6 - line_len
        line_y = bar_height // 2
        draw.line((line_x1, line_y, line_x1 + line_len, line_y), fill=(255, 255, 255, 255), width=2)
        draw.line((line_x1, line_y - 4, line_x1, line_y + 4), fill=(255, 255, 255, 255), width=2)
        draw.line((line_x1 + line_len, line_y - 4, line_x1 + line_len, line_y + 4), fill=(255, 255, 255, 255), width=2)
        draw.text((x_text, y_text), scale_label, fill=(255, 255, 255, 255))

    return labeled


def parse_label_map(entries: list[str]) -> dict[int, str]:
    label_map = {}
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"Invalid label map entry '{entry}'. Use <value>:<name>.")
        value_str, name = entry.split(":", 1)
        label_map[int(value_str)] = name.strip()
    return label_map


def load_aoi_boxes(gpkg_path: Path):
    conn = sqlite3.connect(gpkg_path)
    query = """
        SELECT a.REGION, r.minx, r.maxx, r.miny, r.maxy
        FROM rtree_aois_greenland_area_distributions_geom AS r
        JOIN aois_greenland_area_distributions AS a
          ON a.fid = r.id
    """
    rows = conn.execute(query).fetchall()
    conn.close()
    return rows


def find_matching_regions(chip_bounds: tuple[float, float, float, float], aoi_rows) -> list[str]:
    chip_minx, chip_miny, chip_maxx, chip_maxy = chip_bounds
    chip_cx = (chip_minx + chip_maxx) / 2
    chip_cy = (chip_miny + chip_maxy) / 2
    matches: list[str] = []
    for region, box_minx, box_maxx, box_miny, box_maxy in aoi_rows:
        intersects = not (
            chip_maxx < box_minx
            or chip_minx > box_maxx
            or chip_maxy < box_miny
            or chip_miny > box_maxy
        )
        center_in_box = (
            box_minx <= chip_cx <= box_maxx
            and box_miny <= chip_cy <= box_maxy
        )
        if intersects or center_in_box:
            matches.append(str(region))
    return matches


def make_locator_panel(
    original_image_path: Path,
    aoi_gpkg_path: Path,
    scale_crs: str = "EPSG:5938",
) -> Image.Image:
    with rasterio.open(original_image_path) as src:
        bounds = src.bounds
        src_crs = src.crs
    chip_bounds = transform_bounds(src_crs, scale_crs, *bounds)
    xs_ll, ys_ll = transform(
        src_crs,
        "EPSG:4326",
        [(bounds.left + bounds.right) / 2],
        [(bounds.bottom + bounds.top) / 2],
    )
    center_lon = xs_ll[0]
    center_lat = ys_ll[0]
    chip_minx, chip_miny, chip_maxx, chip_maxy = chip_bounds

    aoi_rows = load_aoi_boxes(aoi_gpkg_path)
    matching_regions = find_matching_regions(chip_bounds, aoi_rows)
    xs = [row[1] for row in aoi_rows] + [row[2] for row in aoi_rows]
    ys = [row[3] for row in aoi_rows] + [row[4] for row in aoi_rows]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    pad_x = (max_x - min_x) * 0.08
    pad_y = (max_y - min_y) * 0.08
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    width = 420
    height = 420
    margin = 20
    canvas = Image.new("RGBA", (width, height), (233, 241, 236, 255))
    draw = ImageDraw.Draw(canvas)

    # Soft pseudo-map background with road-map-like grid.
    grid_color = (205, 214, 208, 255)
    major_grid_color = (184, 194, 187, 255)
    for x in range(margin, width - margin + 1, 40):
        draw.line((x, margin, x, height - margin), fill=grid_color, width=1)
    for y in range(margin, height - margin + 1, 40):
        draw.line((margin, y, width - margin, y), fill=grid_color, width=1)
    for x in range(margin, width - margin + 1, 120):
        draw.line((x, margin, x, height - margin), fill=major_grid_color, width=2)
    for y in range(margin, height - margin + 1, 120):
        draw.line((margin, y, width - margin, y), fill=major_grid_color, width=2)

    def project(x: float, y: float) -> tuple[int, int]:
        px = margin + (x - min_x) / (max_x - min_x) * (width - 2 * margin)
        py = height - margin - (y - min_y) / (max_y - min_y) * (height - 2 * margin)
        return int(px), int(py)

    for region, box_minx, box_maxx, box_miny, box_maxy in aoi_rows:
        x0, y1 = project(box_minx, box_miny)
        x1, y0 = project(box_maxx, box_maxy)
        is_match = str(region) in matching_regions
        draw.rectangle(
            [x0, y0, x1, y1],
            fill=(176, 203, 224, 220) if not is_match else (160, 213, 239, 235),
            outline=(78, 110, 129, 255) if not is_match else (220, 20, 60, 255),
            width=2 if not is_match else 3,
        )
        tx = (x0 + x1) // 2
        ty = (y0 + y1) // 2
        draw.text((tx - 14, ty - 6), str(region), fill=(27, 47, 58, 255))

    cx0, cy1 = project(chip_minx, chip_miny)
    cx1, cy0 = project(chip_maxx, chip_maxy)
    draw.rectangle([cx0, cy0, cx1, cy1], outline=(220, 20, 60, 255), width=4)
    draw.ellipse((cx0 - 4, cy0 - 4, cx0 + 4, cy0 + 4), fill=(220, 20, 60, 255))

    # Add a 10 km scale bar in map coordinates using the configured meter-based CRS.
    usable_width_m = max_x - min_x
    scale_length_m = 10000.0
    length_px = int(scale_length_m / usable_width_m * (width - 2 * margin))
    length_px = max(20, min(length_px, width // 3))
    canvas = add_scale_bar(canvas, length_px, "0-10 km", margin=20)

    # North arrow.
    arrow_x = width - 36
    arrow_y = 28
    draw.line((arrow_x, arrow_y + 18, arrow_x, arrow_y + 52), fill=(40, 40, 40, 255), width=3)
    draw.polygon(
        [(arrow_x, arrow_y), (arrow_x - 7, arrow_y + 18), (arrow_x + 7, arrow_y + 18)],
        fill=(40, 40, 40, 255),
    )
    draw.text((arrow_x - 4, arrow_y + 54), "N", fill=(40, 40, 40, 255))

    # Coordinate box.
    coord_text = f"{center_lat:.4f} N, {abs(center_lon):.4f} W"
    region_text = ", ".join(matching_regions) if matching_regions else "Unmatched fjord"
    draw.rounded_rectangle((18, height - 86, 238, height - 24), radius=8, fill=(255, 255, 255, 220))
    draw.text((28, height - 74), region_text, fill=(30, 30, 30, 255))
    draw.text((28, height - 52), coord_text, fill=(30, 30, 30, 255))

    return canvas


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    mask_path = Path(args.mask).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    label_map = parse_label_map(args.label_map)
    scale_crs = args.scale_crs or choose_scale_crs(image_path) or "EPSG:5938"

    base = load_image_chip(image_path).convert("RGBA")
    mask = load_mask(mask_path)
    values = [int(v) for v in np.unique(mask)]
    if args.show_values is not None and len(args.show_values) > 0:
        requested = set(args.show_values)
        values = [value for value in values if value in requested]

    panels = []
    if args.original_image:
        original_image_path = Path(args.original_image).expanduser().resolve()
        original_scale_length_px = get_scale_length_px(original_image_path, scale_crs=scale_crs)
        if args.locator_image:
            locator_image_path = Path(args.locator_image).expanduser().resolve()
            locator = Image.open(locator_image_path).convert("RGBA")
            panels.append(add_label(locator, "locator map"))
        if args.aoi_gpkg and not args.locator_image:
            aoi_gpkg_path = Path(args.aoi_gpkg).expanduser().resolve()
            locator = make_locator_panel(original_image_path, aoi_gpkg_path, scale_crs=scale_crs)
            panels.append(add_label(locator, "locator map"))
        original = load_image_chip(original_image_path).convert("RGBA")
        panels.append(
            add_label(
                original,
                "original color",
                scale_label="0-100 m" if original_scale_length_px else None,
                scale_length_px=original_scale_length_px,
            )
        )

    overlay_scale_length_px = get_scale_length_px(image_path, scale_crs=scale_crs)

    for value in values:
        overlay = Image.alpha_composite(base, mask_to_overlay(mask, value))
        display_label = label_map.get(value, f"value = {value}")
        scale_label = "0-100 m" if display_label.lower() == "iceberg" else None
        panels.append(
            add_label(
                overlay,
                display_label,
                scale_label=scale_label,
                scale_length_px=overlay_scale_length_px,
            )
        )

    total_width = sum(panel.width for panel in panels)
    max_height = max(panel.height for panel in panels)
    canvas = Image.new("RGBA", (total_width, max_height), (255, 255, 255, 255))

    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path)
    print(f"Saved preview: {out_path}")
    print(f"Mask values shown: {values}")


if __name__ == "__main__":
    main()
