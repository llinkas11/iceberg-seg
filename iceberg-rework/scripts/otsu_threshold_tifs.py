"""
otsu_threshold_tifs.py: Apply per-chip Otsu threshold on B08 to S2 chip .tifs.

Threshold is computed unsupervised via skimage.filters.threshold_otsu on the
B08 band of each chip independently, so it adapts to local illumination/scene
conditions rather than using a fixed reflectance cutoff.

Mirrors the output format of threshold_tifs.py / predict_tifs.py so
compare_areas.py can load all methods from the same directory.

Usage:
  python otsu_threshold_tifs.py \\
      --chips_dir chips/KQ/sza_65_70/tifs \\
      --out_dir   area_comparison/KQ/sza_65_70

Output:
  out_dir/
    otsu_thresholding/
      all_icebergs_otsu.gpkg     : all iceberg polygons with area_m2, otsu_thresh
      pngs/
        <chip>_otsu.png          : 3-panel false-color RGB, B08 histogram, mask overlay

Note:
  --b08_idx is the 0-indexed band position of B08 in the chip stack.
  Default is 2 (i.e. bands stacked as B04/B03/B08 by chip_sentinel2.py).

  Chips where Otsu yields a threshold below --min_otsu_thresh are skipped
  (all-ocean chips with no bright targets; Otsu would fire on noise).
"""

import os
import argparse
from _method_common import (
    write_method_config, write_skipped_chips,
    SKIP_TOO_FEW_BANDS, SKIP_OTSU_FLOOR, SKIP_IC_BLOCK_FILTER,
)
import warnings
from glob import glob

import numpy as np
import rasterio as rio
from rasterio.features import shapes as rio_shapes
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd
from skimage.filters import threshold_otsu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

MIN_AREA_M2     = 100   # minimum polygon area in m² (~10×10 m)
IC_THRESHOLD    = 0.15  # skip chip if >15% of pixels exceed the Otsu threshold
MIN_OTSU_THRESH = 0.10  # skip chip if Otsu threshold < this (flat/featureless chips)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def percentile_stretch(band, lo=2, hi=98):
    """Stretch a 2D array to [0, 1] using percentile clipping."""
    p_lo, p_hi = np.percentile(band, [lo, hi])
    if p_hi == p_lo:
        return np.zeros_like(band, dtype=np.float32)
    return np.clip((band - p_lo) / (p_hi - p_lo), 0, 1).astype(np.float32)


def make_false_color(chip, b08_idx=2):
    """
    B04 to R, B03 to G, B08 to B, matches the UNet++ training chip rendering.
    Ocean appears dark blue (low NIR in blue channel); icebergs appear bright white.
    chip shape: (C, H, W)
    """
    red = percentile_stretch(chip[0])        # B04 → R
    grn = percentile_stretch(chip[1])        # B03 → G
    nir = percentile_stretch(chip[b08_idx])  # B08 → B
    return np.stack([red, grn, nir], axis=-1)  # (H, W, 3)


def save_png(stem, chip, b08, otsu_thresh, iceberg_mask, n_icebergs, out_path, b08_idx=2):
    """Save 3-panel diagnostic PNG for one chip."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # ── Panel 1: False-color RGB (NIR/R/G) ──────────────────────────────────
    fc = make_false_color(chip, b08_idx)
    axes[0].imshow(fc)
    axes[0].set_title("False color (NIR–R–G)", color="white", fontsize=11)
    axes[0].axis("off")

    # ── Panel 2: B08 histogram with Otsu threshold ──────────────────────────
    flat = b08.ravel()
    axes[1].hist(flat, bins=80, color="#4a9eff", alpha=0.8, edgecolor="none")
    axes[1].axvline(otsu_thresh, color="#ff6b6b", linewidth=2,
                    label=f"Otsu = {otsu_thresh:.4f}")
    axes[1].set_title("B08 histogram", color="white", fontsize=11)
    axes[1].set_xlabel("Reflectance", color="white", fontsize=9)
    axes[1].set_ylabel("Pixel count", color="white", fontsize=9)
    axes[1].tick_params(colors="white", labelsize=8)
    axes[1].legend(fontsize=9, facecolor="#2a2a4e", labelcolor="white",
                   edgecolor="#444")

    # shade the "iceberg" side of the threshold
    x_max = flat.max()
    axes[1].axvspan(otsu_thresh, x_max, alpha=0.15, color="#ff6b6b")

    # ── Panel 3: Mask overlay on B08 ────────────────────────────────────────
    b08_disp = percentile_stretch(b08)
    axes[2].imshow(b08_disp, cmap="gray")

    # overlay iceberg mask in semi-transparent cyan
    overlay = np.zeros((*iceberg_mask.shape, 4), dtype=np.float32)
    overlay[iceberg_mask == 1] = [0.0, 1.0, 0.9, 0.55]   # cyan, 55% opacity
    axes[2].imshow(overlay)

    patch = mpatches.Patch(color=(0.0, 1.0, 0.9, 0.8), label=f"Icebergs ({n_icebergs})")
    axes[2].legend(handles=[patch], fontsize=9, facecolor="#2a2a4e",
                   labelcolor="white", edgecolor="#444", loc="lower right")
    axes[2].set_title("Otsu mask on B08", color="white", fontsize=11)
    axes[2].axis("off")

    # ── Suptitle ─────────────────────────────────────────────────────────────
    fig.suptitle(
        f"{stem}\n"
        f"otsu_thresh={otsu_thresh:.4f}   icebergs={n_icebergs}",
        color="white", fontsize=10, y=1.01
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def apply_otsu(chips_dir, out_dir, b08_idx=2, min_area_m2=MIN_AREA_M2,
               ic_threshold=IC_THRESHOLD, min_otsu_thresh=MIN_OTSU_THRESH):

    png_dir = os.path.join(out_dir, "pngs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    tif_files = sorted(glob(os.path.join(chips_dir, "*.tif")))
    if not tif_files:
        print(f"No .tif files found in {chips_dir}")
        return

    print(f"Found {len(tif_files)} chips")
    print(f"  b08_idx={b08_idx}  ic_threshold={ic_threshold}  "
          f"min_otsu_thresh={min_otsu_thresh}")
    print(f"  Output dir : {out_dir}")
    print(f"  PNGs dir   : {png_dir}\n")

    all_gdfs = []
    skipped  = []

    for i, tif_path in enumerate(tif_files):
        stem = os.path.splitext(os.path.basename(tif_path))[0]

        with rio.open(tif_path) as src:
            chip = src.read().astype(np.float32)   # (C, H, W)
            meta = src.meta.copy()

        if chip.shape[0] <= b08_idx:
            print(f"  [{i+1:>4}/{len(tif_files)}] SKIP {stem}: only {chip.shape[0]} bands")
            skipped.append({"chip_stem": stem, "reason": SKIP_TOO_FEW_BANDS,
                            "n_bands": chip.shape[0]})
            continue

        b08 = np.nan_to_num(chip[b08_idx], nan=0.0)

        otsu_thresh = float(threshold_otsu(b08))

        if otsu_thresh < min_otsu_thresh:
            print(f"  [{i+1:>4}/{len(tif_files)}] FLAT {stem[:55]}  "
                  f"otsu={otsu_thresh:.4f}")
            skipped.append({"chip_stem": stem, "reason": SKIP_OTSU_FLOOR,
                            "otsu_thresh": f"{otsu_thresh:.4f}"})
            continue

        ic_frac = float((b08 >= otsu_thresh).mean())
        if ic_frac > ic_threshold:
            print(f"  [{i+1:>4}/{len(tif_files)}] IC   {stem[:55]}  "
                  f"otsu={otsu_thresh:.4f}  ic_frac={ic_frac:.2f}")
            skipped.append({"chip_stem": stem, "reason": SKIP_IC_BLOCK_FILTER,
                            "otsu_thresh": f"{otsu_thresh:.4f}",
                            "ic_frac":     f"{ic_frac:.4f}"})
            continue

        iceberg_mask = (b08 >= otsu_thresh).astype(np.uint8)

        # Polygonize
        records = []
        for geom_dict, val in rio_shapes(iceberg_mask, transform=meta["transform"]):
            if val == 0:
                continue
            geom = shape(geom_dict)
            if geom.is_empty or geom.area < min_area_m2:
                continue
            records.append({
                "geometry"    : geom,
                "class_id"    : 1,
                "class_name"  : "iceberg",
                "area_m2"     : round(geom.area, 2),
                "otsu_thresh" : round(otsu_thresh, 4),
                "source_file" : os.path.basename(tif_path),
            })

        print(f"  [{i+1:>4}/{len(tif_files)}] {stem[:55]}  "
              f"otsu={otsu_thresh:.4f}  icebergs={len(records)}")

        # Save PNG
        png_path = os.path.join(png_dir, f"{stem}_otsu.png")
        save_png(stem, chip, b08, otsu_thresh, iceberg_mask,
                 len(records), png_path, b08_idx)

        if records:
            gdf = gpd.GeoDataFrame(records, crs=meta["crs"])
            all_gdfs.append(gdf)

    # -----------------------------------------------------------------------
    # Save merged GeoPackage + provenance
    # -----------------------------------------------------------------------
    cfg_path = write_method_config(
        out_dir, "OT",
        params={
            "chips_dir":       os.path.abspath(chips_dir),
            "b08_idx":         b08_idx,
            "min_area_m2":     min_area_m2,
            "ic_threshold":    ic_threshold,
            "min_otsu_thresh": min_otsu_thresh,
        },
    )
    skip_path = write_skipped_chips(out_dir, skipped)

    n_skipped = sum(1 for r in skipped if r["reason"] == SKIP_TOO_FEW_BANDS)
    n_flat    = sum(1 for r in skipped if r["reason"] == SKIP_OTSU_FLOOR)
    n_ic      = sum(1 for r in skipped if r["reason"] == SKIP_IC_BLOCK_FILTER)

    if not all_gdfs:
        print("\nNo icebergs detected across all chips.")
        if n_flat:    print(f"  Flat chips skipped : {n_flat}")
        if n_ic:      print(f"  IC chips skipped   : {n_ic}")
        if n_skipped: print(f"  Too few bands      : {n_skipped}")
        print(f"Method config : {cfg_path}")
        print(f"Skipped chips : {skip_path}")
        return

    target_crs = all_gdfs[0].crs
    reprojected = [gdf.to_crs(target_crs) if gdf.crs != target_crs else gdf for gdf in all_gdfs]
    merged = gpd.GeoDataFrame(pd.concat(reprojected, ignore_index=True),
                               crs=target_crs)
    merged["iceberg_id"] = range(1, len(merged) + 1)

    icebergs = merged[merged["class_name"] == "iceberg"]
    print(f"\n{'-'*55}")
    print(f"Total iceberg polygons : {len(icebergs)}")
    if len(icebergs) > 0:
        print(f"  min  area = {icebergs['area_m2'].min():.1f} m2")
        print(f"  mean area = {icebergs['area_m2'].mean():.1f} m2")
        print(f"  max  area = {icebergs['area_m2'].max():.1f} m2")
        print(f"  Otsu thresh range : [{merged['otsu_thresh'].min():.4f}, "
              f"{merged['otsu_thresh'].max():.4f}]")
    if n_flat:    print(f"Flat chips skipped  : {n_flat}")
    if n_ic:      print(f"IC chips skipped    : {n_ic}")
    if n_skipped: print(f"Too few bands       : {n_skipped}")
    print(f"{'-'*55}")

    gpkg_path = os.path.join(out_dir, "all_icebergs.gpkg")
    merged.to_file(gpkg_path, driver="GPKG")

    print(f"\nSaved GeoPackage : {gpkg_path}")
    print(f"Method config    : {cfg_path}")
    print(f"Skipped chips    : {skip_path}")
    print(f"Saved PNGs       : {png_dir}/  ({len(tif_files) - n_skipped - n_flat - n_ic} files)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply per-chip Otsu threshold on B08 to S2 chip .tifs"
    )
    parser.add_argument("--chips_dir",       required=True,
                        help="Directory of .tif chip files")
    parser.add_argument("--out_dir",         required=True,
                        help="Output directory (gpkg + pngs/ subfolder)")
    parser.add_argument("--b08_idx",         type=int,   default=2,
                        help="0-indexed band position of B08 (default: 2 for B04/B03/B08)")
    parser.add_argument("--min_area",        type=float, default=MIN_AREA_M2,
                        help=f"Min iceberg area in m² (default: {MIN_AREA_M2})")
    parser.add_argument("--ic_threshold",    type=float, default=IC_THRESHOLD,
                        help=f"IC block filter fraction (default: {IC_THRESHOLD})")
    parser.add_argument("--min_otsu_thresh", type=float, default=MIN_OTSU_THRESH,
                        help=f"Skip chip if Otsu threshold < this (default: {MIN_OTSU_THRESH})")
    args = parser.parse_args()

    apply_otsu(
        args.chips_dir, args.out_dir, args.b08_idx,
        args.min_area, args.ic_threshold, args.min_otsu_thresh,
    )


if __name__ == "__main__":
    main()
