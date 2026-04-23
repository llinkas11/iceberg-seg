"""
Georeferenced inference — runs the trained UNet++ on raw .tif chips and
produces outputs that preserve spatial coordinates.

For each .tif in imgs/:
  1. Read image + CRS / transform via rasterio
  2. Run UNet++ inference
  3. Save predicted mask as GeoTIFF (same CRS as input)
  4. Polygonize iceberg (and shadow) pixels → individual GeoPackage per chip

Final outputs:
  out_dir/
    geotiffs/    <chip_name>_pred.tif     — single-band class raster
    gpkgs/       <chip_name>_icebergs.gpkg — polygons per chip
    all_icebergs.gpkg                      — everything merged, one file

Usage:
  python predict_tifs.py \\
      --checkpoint  runs/s2_exp1/best_model.pth \\
      --imgs_dir    S2UnetPlusPlus/imgs \\
      --out_dir     georef_predictions/s2_exp1

  python predict_tifs.py \\
      --checkpoint  runs/s1_exp1/best_model.pth \\
      --imgs_dir    S1UnetPlusPlus/imgs \\
      --out_dir     georef_predictions/s1_exp1

Requires: rasterio, geopandas, shapely, segmentation_models_pytorch, torch
"""

import os
import argparse
import warnings
from glob import glob

import numpy as np
import torch
import rasterio as rio
from rasterio.features import shapes as rio_shapes
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd
import segmentation_models_pytorch as smp

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_args = ckpt["args"]
    mode        = saved_args["mode"]
    num_classes = 1 if mode == "s1" else 3

    model = smp.UnetPlusPlus(
        encoder_name    = saved_args["encoder"],
        encoder_weights = None,
        in_channels     = 3,
        classes         = num_classes,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded : {checkpoint_path}")
    print(f"Epoch  : {ckpt['epoch']}   val IoU : {ckpt['val_iou']:.4f}")
    return model, mode, num_classes


# ---------------------------------------------------------------------------
# Single-chip inference
# ---------------------------------------------------------------------------

def predict_chip(model, chip_np, num_classes, device):
    """
    chip_np : (3, H, W) float32 array, values already in [0, 1]
    Returns : (H, W) int numpy array of class labels
    """
    tensor = torch.from_numpy(chip_np).unsqueeze(0).to(device)  # (1, 3, H, W)
    with torch.no_grad():
        logits = model(tensor)
        if num_classes == 1:
            pred = (torch.sigmoid(logits) > 0.5).long().squeeze()  # (H, W)
        else:
            pred = torch.argmax(logits, dim=1).squeeze()            # (H, W)
    return pred.cpu().numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# Polygonize predicted mask → GeoDataFrame
# ---------------------------------------------------------------------------

def polygonize(pred_mask, src_meta, source_file, mode):
    """
    pred_mask   : (H, W) uint8 array
    src_meta    : rasterio metadata dict (contains crs, transform)
    source_file : original tif filename (stored as attribute)
    mode        : 's1' or 's2'

    Returns GeoDataFrame with iceberg (and shadow) polygons, or None if empty.
    """
    class_info = {1: "iceberg"}
    if mode == "s2":
        class_info[2] = "shadow"

    records = []
    for cls_id, cls_name in class_info.items():
        binary = (pred_mask == cls_id).astype(np.uint8)
        if binary.sum() == 0:
            continue

        # rasterio.features.shapes yields (geojson-geometry, pixel-value) pairs
        for geom_dict, val in rio_shapes(binary, transform=src_meta["transform"]):
            if val == 0:
                continue
            geom = shape(geom_dict)
            if geom.is_empty or geom.area == 0:
                continue
            records.append({
                "geometry"   : geom,
                "class_id"   : cls_id,
                "class_name" : cls_name,
                "area_m2"    : geom.area,
                "source_file": os.path.basename(source_file),
            })

    if not records:
        return None

    gdf = gpd.GeoDataFrame(records, crs=src_meta["crs"])
    return gdf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Georeferenced UNet++ inference on .tif chips"
    )
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to best_model.pth from train.py")
    parser.add_argument("--imgs_dir",    required=True,
                        help="Directory containing .tif image chips (e.g. S2UnetPlusPlus/imgs)")
    parser.add_argument("--out_dir",     required=True,
                        help="Output directory")
    parser.add_argument("--min_area_m2", type=float, default=100.0,
                        help="Minimum polygon area in m² to keep (default: 100 = 10×10 m)")
    args = parser.parse_args()

    tif_dir  = os.path.join(args.out_dir, "geotiffs")
    gpkg_dir = os.path.join(args.out_dir, "gpkgs")
    os.makedirs(tif_dir,  exist_ok=True)
    os.makedirs(gpkg_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mode, num_classes = load_model(args.checkpoint, device)

    tif_files = sorted(glob(os.path.join(args.imgs_dir, "*.tif")))
    print(f"\nFound {len(tif_files)} .tif files in {args.imgs_dir}")
    print(f"Min polygon area filter: {args.min_area_m2} m²\n")

    all_gdfs = []
    n_skipped = 0

    for i, tif_path in enumerate(tif_files):
        stem = os.path.splitext(os.path.basename(tif_path))[0]

        # ----------------------------------------------------------------
        # Read chip
        # ----------------------------------------------------------------
        with rio.open(tif_path) as src:
            chip   = src.read().astype(np.float32)   # (bands, H, W)
            meta   = src.meta.copy()

        if chip.shape[0] < 3:
            print(f"  [{i+1}/{len(tif_files)}] SKIP {stem} — only {chip.shape[0]} band(s)")
            n_skipped += 1
            continue

        chip = chip[:3]   # use first 3 bands

        # ----------------------------------------------------------------
        # Inference
        # ----------------------------------------------------------------
        pred_mask = predict_chip(model, chip, num_classes, device)  # (H, W) uint8

        # ----------------------------------------------------------------
        # Save predicted mask as GeoTIFF
        # ----------------------------------------------------------------
        out_tif = os.path.join(tif_dir, f"{stem}_pred.tif")
        pred_meta = meta.copy()
        pred_meta.update({"count": 1, "dtype": "uint8", "nodata": 255})
        with rio.open(out_tif, "w", **pred_meta) as dst:
            dst.write(pred_mask[np.newaxis, :, :])

        # ----------------------------------------------------------------
        # Polygonize
        # ----------------------------------------------------------------
        gdf = polygonize(pred_mask, meta, tif_path, mode)

        iceberg_count = 0
        if gdf is not None:
            # drop tiny fragments
            gdf = gdf[gdf["area_m2"] >= args.min_area_m2].copy()
            gdf = gdf.reset_index(drop=True)

            if len(gdf) > 0:
                out_gpkg = os.path.join(gpkg_dir, f"{stem}_icebergs.gpkg")
                gdf.to_file(out_gpkg, driver="GPKG")
                iceberg_count = len(gdf[gdf["class_name"] == "iceberg"])
                all_gdfs.append(gdf)

        print(f"  [{i+1:>4}/{len(tif_files)}] {stem[:60]}  "
              f"icebergs={iceberg_count}")

    # -----------------------------------------------------------------------
    # Merge all chips into one GeoPackage
    # -----------------------------------------------------------------------
    if all_gdfs:
        print(f"\nMerging {len(all_gdfs)} chip GeoPackages...")
        merged = gpd.GeoDataFrame(
            pd.concat(all_gdfs, ignore_index=True),
            crs=all_gdfs[0].crs
        )

        # Add unique ID and round area
        merged["iceberg_id"] = range(1, len(merged) + 1)
        merged["area_m2"]    = merged["area_m2"].round(2)

        # Summary stats
        icebergs = merged[merged["class_name"] == "iceberg"]
        print(f"\n{'─'*50}")
        print(f"Total polygons      : {len(merged)}")
        print(f"  Iceberg polygons  : {len(icebergs)}")
        if mode == "s2":
            shadows = merged[merged["class_name"] == "shadow"]
            print(f"  Shadow polygons   : {len(shadows)}")
        if len(icebergs) > 0:
            print(f"Iceberg area stats  :")
            print(f"  min  = {icebergs['area_m2'].min():.1f} m²  "
                  f"(√={np.sqrt(icebergs['area_m2'].min()):.1f} m)")
            print(f"  mean = {icebergs['area_m2'].mean():.1f} m²  "
                  f"(√={np.sqrt(icebergs['area_m2'].mean()):.1f} m)")
            print(f"  max  = {icebergs['area_m2'].max():.1f} m²  "
                  f"(√={np.sqrt(icebergs['area_m2'].max()):.1f} m)")
        print(f"{'─'*50}\n")

        out_merged = os.path.join(args.out_dir, "all_icebergs.gpkg")
        merged.to_file(out_merged, driver="GPKG")
        print(f"Merged GeoPackage   : {out_merged}")
    else:
        print("\nNo icebergs detected across all chips.")

    print(f"GeoTIFF masks       : {tif_dir}/")
    print(f"Per-chip GeoPackages: {gpkg_dir}/")
    if n_skipped:
        print(f"Skipped             : {n_skipped} chips (too few bands)")


if __name__ == "__main__":
    main()
