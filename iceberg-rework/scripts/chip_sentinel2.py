"""
chip_sentinel2.py — Unzip .SAFE files → extract bands → tile to 256×256 chips

Produces two outputs per SAFE scene:
  chips/{region}/{sza_bin}/
    tifs/   <scene>_r{row}_c{col}.tif     — 3-band float32 [0,1], for predict_tifs.py
    pngs/   <scene>_r{row}_c{col}_B08.png — grayscale NIR preview, for Roboflow annotation

Usage:
  python chip_sentinel2.py \
      --safe_dir  sentinel2_downloads \
      --out_dir   chips \
      --aoi       aois_greenland_area_distributions.gpkg

NOTE on bands:
  Default is B04 (red), B03 (green), B08 (NIR) — the three 10m bands most
  likely used in the training data based on the research focus on NIR B08.
  If predictions look wrong, check which bands the training chips actually
  contain by running:
    python -c "import rasterio; src=rasterio.open('S2UnetPlusPlus/imgs/*.tif'); print(src.descriptions)"
  Then adjust --bands accordingly.
"""

import os
import glob
import zipfile
import argparse
import warnings
import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.mask import mask as rio_mask
from rasterio.transform import from_bounds
import geopandas as gpd
from PIL import Image

warnings.filterwarnings("ignore")

CHIP_SIZE   = 256        # pixels
SCALE       = 1e-4       # S2 L1C DN → TOA reflectance
MIN_VALID   = 0.05       # skip chips where >80% of pixels are below this (cloud/nodata)
DEFAULT_BANDS = ["B04", "B03", "B08"]   # red, green, NIR — adjust if needed


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_band_jp2(safe_dir, band_name):
    """Return path to the 10m jp2 for a given band name (e.g. 'B08') inside a SAFE dir."""
    pattern = os.path.join(safe_dir, "GRANULE", "*", "IMG_DATA", "R10m", f"*_{band_name}_10m.jp2")
    matches = glob.glob(pattern)
    if not matches:
        # Older SAFE format: IMG_DATA/*.jp2
        pattern2 = os.path.join(safe_dir, "GRANULE", "*", "IMG_DATA", f"*_{band_name}.jp2")
        matches = glob.glob(pattern2)
    return matches[0] if matches else None


def read_band(jp2_path):
    """Read a single jp2 band, return (data float32, meta)."""
    with rasterio.open(jp2_path) as src:
        data = src.read(1).astype(np.float32) * SCALE
        meta = src.meta.copy()
        meta.update(dtype="float32", count=1)
    return data, meta


def clip_to_aoi(data_stack, meta, aoi_geom):
    """Clip a (C, H, W) array to AOI polygon. Returns clipped array + updated meta."""
    import tempfile, uuid
    # Write to temp file then clip with rasterio.mask
    tmp = f"/tmp/chip_tmp_{uuid.uuid4().hex}.tif"
    h, w = data_stack.shape[1], data_stack.shape[2]
    meta_tmp = meta.copy()
    meta_tmp.update(driver="GTiff", dtype="float32", count=data_stack.shape[0])
    with rasterio.open(tmp, "w", **meta_tmp) as dst:
        dst.write(data_stack)
    with rasterio.open(tmp) as src:
        out_data, out_transform = rio_mask(src, [aoi_geom], crop=True, nodata=0)
        out_meta = src.meta.copy()
        out_meta.update(transform=out_transform,
                        height=out_data.shape[1],
                        width=out_data.shape[2])
    os.remove(tmp)
    return out_data, out_meta


def tile_array(data, meta, chip_size, stem, out_tif_dir, out_png_dir, b08_idx):
    """
    Slide a window over (C, H, W) data and save non-empty chips.
    Returns count of chips saved.
    """
    C, H, W = data.shape
    n_saved = 0
    transform = meta["transform"]
    crs = meta["crs"]

    for row in range(0, H - chip_size + 1, chip_size):
        for col in range(0, W - chip_size + 1, chip_size):
            chip = data[:, row:row+chip_size, col:col+chip_size]  # (C, 256, 256)

            # Skip chips that are mostly nodata / ocean with no signal
            if (chip[b08_idx] > MIN_VALID).mean() < 0.05:
                continue
            # Skip chips with large nodata regions (zeros from edges)
            if (chip == 0).all(axis=0).mean() > 0.3:
                continue

            # Compute chip's geotransform
            chip_transform = rasterio.transform.from_bounds(
                transform.c + col  * transform.a,
                transform.f + (row + chip_size) * transform.e,
                transform.c + (col + chip_size) * transform.a,
                transform.f + row  * transform.e,
                chip_size, chip_size
            )

            fname = f"{stem}_r{row:04d}_c{col:04d}"

            # ── Save 3-band GeoTIFF ──
            tif_path = os.path.join(out_tif_dir, f"{fname}.tif")
            chip_meta = {
                "driver": "GTiff", "dtype": "float32",
                "count": C, "height": chip_size, "width": chip_size,
                "crs": crs, "transform": chip_transform,
            }
            with rasterio.open(tif_path, "w", **chip_meta) as dst:
                dst.write(chip)

            # ── Save B08 grayscale PNG for Roboflow ──
            b08 = chip[b08_idx]
            # Stretch to 8-bit using 2nd–98th percentile
            lo, hi = np.percentile(b08[b08 > 0], [2, 98]) if (b08 > 0).any() else (0, 1)
            b08_u8 = np.clip((b08 - lo) / (hi - lo + 1e-6), 0, 1)
            b08_u8 = (b08_u8 * 255).astype(np.uint8)
            png_path = os.path.join(out_png_dir, f"{fname}_B08.png")
            Image.fromarray(b08_u8).save(png_path)

            n_saved += 1

    return n_saved


def unzip_safe(zip_path, out_dir):
    """Unzip a .SAFE.zip file. Returns path to the .SAFE directory."""
    with zipfile.ZipFile(zip_path, "r") as z:
        # Find the .SAFE root inside the zip
        names = z.namelist()
        safe_name = next((n.split("/")[0] for n in names if n.endswith(".SAFE") or ".SAFE/" in n), None)
        if safe_name is None:
            safe_name = names[0].split("/")[0]
        safe_path = os.path.join(out_dir, safe_name)
        if not os.path.exists(safe_path):
            print(f"      Unzipping {os.path.basename(zip_path)} ...")
            z.extractall(out_dir)
        return safe_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tile S2 SAFE files → 256×256 chips")
    parser.add_argument("--safe_dir", required=True,
                        help="Root of sentinel2_downloads/ (contains KQ/ SK/ subfolders)")
    parser.add_argument("--out_dir",  required=True,
                        help="Output root for chips/")
    parser.add_argument("--aoi",      default="aois_greenland_area_distributions.gpkg",
                        help="GeoPackage with AOI polygons to clip to (optional)")
    parser.add_argument("--bands",    nargs=3, default=DEFAULT_BANDS,
                        metavar=("B1","B2","B3"),
                        help=f"Three S2 band names to stack (default: {' '.join(DEFAULT_BANDS)})")
    args = parser.parse_args()

    # Load AOI if available
    aoi_geom = None
    if os.path.exists(args.aoi):
        aoi_gdf = gpd.read_file(args.aoi)
        print(f"Loaded AOI: {args.aoi}  ({len(aoi_gdf)} polygon(s))")
        aoi_union = aoi_gdf.geometry.union_all() if hasattr(aoi_gdf.geometry, "union_all") \
                    else aoi_gdf.geometry.unary_union
        aoi_geom = aoi_union
    else:
        print(f"No AOI file found at {args.aoi} — processing full scene extent")

    b08_idx = args.bands.index("B08") if "B08" in args.bands else 2
    print(f"Bands : {args.bands}  (B08 is index {b08_idx})")

    # Walk sentinel2_downloads/{region}/{sza_bin}/*.zip
    total_chips = 0
    zip_files = sorted(glob.glob(os.path.join(args.safe_dir, "*", "*", "*.zip")))
    if not zip_files:
        # Also look for already-unzipped SAFE dirs
        zip_files = sorted(glob.glob(os.path.join(args.safe_dir, "*", "*", "*.SAFE")))

    print(f"\nFound {len(zip_files)} SAFE scenes to process\n")

    for i, zpath in enumerate(zip_files):
        # Infer region and SZA bin from path
        parts = zpath.replace(args.safe_dir, "").strip("/").split("/")
        region  = parts[0] if len(parts) >= 3 else "unknown"
        sza_bin = parts[1] if len(parts) >= 3 else "unknown"
        basename = os.path.splitext(os.path.basename(zpath))[0].replace(".SAFE", "")

        out_tif_dir = os.path.join(args.out_dir, region, sza_bin, "tifs")
        out_png_dir = os.path.join(args.out_dir, region, sza_bin, "pngs")
        os.makedirs(out_tif_dir, exist_ok=True)
        os.makedirs(out_png_dir, exist_ok=True)

        print(f"[{i+1:>3}/{len(zip_files)}] {region}/{sza_bin}  {basename[:60]}")

        # Unzip if needed
        work_dir = os.path.dirname(zpath)
        if zpath.endswith(".zip"):
            try:
                safe_dir = unzip_safe(zpath, work_dir)
            except Exception as e:
                print(f"      ✗ Unzip failed: {e}")
                continue
        else:
            safe_dir = zpath

        # Find band jp2 files
        band_paths = {}
        for band in args.bands:
            bp = find_band_jp2(safe_dir, band)
            if bp is None:
                print(f"      ✗ Band {band} not found in {safe_dir} — skipping")
                break
            band_paths[band] = bp
        if len(band_paths) < 3:
            continue

        # Read and stack bands
        try:
            arrays = []
            meta = None
            for band in args.bands:
                arr, m = read_band(band_paths[band])
                arrays.append(arr)
                if meta is None:
                    meta = m
            data = np.stack(arrays, axis=0)   # (3, H, W) float32

            # Reproject AOI to scene CRS and clip
            if aoi_geom is not None:
                try:
                    scene_crs = meta["crs"]
                    aoi_reproj = aoi_gdf.to_crs(scene_crs).geometry.union_all() \
                                 if hasattr(aoi_gdf.geometry, "union_all") \
                                 else aoi_gdf.to_crs(scene_crs).geometry.unary_union
                    data, meta = clip_to_aoi(data, meta, aoi_reproj)
                except Exception as e:
                    print(f"      ⚠ AOI clip failed ({e}), using full scene")

            meta.update(count=3, dtype="float32")
            n = tile_array(data, meta, CHIP_SIZE, basename, out_tif_dir, out_png_dir, b08_idx)
            print(f"      → {n} chips saved")
            total_chips += n

        except Exception as e:
            print(f"      ✗ Processing failed: {e}")
            continue

    print(f"\n{'─'*50}")
    print(f"Total chips saved : {total_chips}")
    print(f"GeoTIFFs (model)  : {args.out_dir}/{{region}}/{{sza_bin}}/tifs/")
    print(f"PNGs (Roboflow)   : {args.out_dir}/{{region}}/{{sza_bin}}/pngs/")
    print(f"\nNext steps:")
    print(f"  1. Upload pngs/ to Roboflow → annotate icebergs → export masks")
    print(f"  2. Run predict_tifs.py on tifs/ to get UNet++ predictions")
    print(f"     python predict_tifs.py --checkpoint runs/s2_*/best_model.pth \\")
    print(f"         --imgs_dir chips/KQ/sza_gt75/tifs --out_dir georef_predictions/KQ_sza_gt75")


if __name__ == "__main__":
    main()
