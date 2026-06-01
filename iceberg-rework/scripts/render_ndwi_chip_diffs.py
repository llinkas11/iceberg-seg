"""
render_ndwi_chip_diffs.py — Render side-by-side NDWI mask diffs for selected chips.

For each (sza_bin, chip_stem) input, produce a 4-panel PNG:
    [RGB]  [iceberg_mask @ NDWI=0]  [iceberg_mask @ NDWI=0.05]  [pixels lost when 0 -> 0.05]

Used to confirm whether area lost when tightening NDWI is land-edge (would lie
along shoreline) or inside icebergs (mixed water-ice pixels at high SZA).

Usage:
  python render_ndwi_chip_diffs.py \\
      --chips_root /mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/chips/KQ \\
      --out_dir    /home/llinkas/S2-iceberg-areas/sweeps/ndwi_chip_diffs \\
      --picks      sza_65_70:STEM,sza_gt75:STEM,sza_70_75:STEM,sza_lt65:STEM
"""

import argparse
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio as rio

from threshold_masked_tifs import (
    NIR_THRESHOLD,
    IC_THRESHOLD,
    MIN_AREA_M2,
)

warnings.filterwarnings("ignore")


def render_chip(tif_path, out_png, b04_idx=0, b03_idx=1, b08_idx=2,
                nir_threshold=NIR_THRESHOLD, ic_threshold=IC_THRESHOLD):
    """Render a 4-panel NDWI-diff figure for one chip."""
    with rio.open(tif_path) as src:
        chip = src.read().astype(np.float32)
    b04, b03, b08 = chip[b04_idx], chip[b03_idx], chip[b08_idx]

    # 1. RGB composite (B04 R, B03 G, B08 B): standard false-color for ice/water contrast
    rgb = np.clip(np.stack([b04, b03, b08], axis=-1) / 0.5, 0, 1)

    # 2. NDWI and iceberg masks at 0 and 0.05
    ndwi = (b03 - b08) / (b03 + b08 + 1e-6)

    def make_mask(ndwi_val):
        water = (ndwi > ndwi_val)
        return ((b08 >= nir_threshold) & water).astype(np.uint8)

    m0   = make_mask(0.00)
    m05  = make_mask(0.05)
    lost = (m0 == 1) & (m05 == 0)

    ic_frac = float((b08 >= nir_threshold).mean())

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB (B04 R, B03 G, B08 B)\nIC frac = {ic_frac:.2f}")

    axes[1].imshow(rgb)
    axes[1].imshow(np.ma.masked_where(m0 == 0, m0), cmap="autumn", alpha=0.6)
    axes[1].set_title(f"Iceberg mask, NDWI > 0\n{int(m0.sum())} px")

    axes[2].imshow(rgb)
    axes[2].imshow(np.ma.masked_where(m05 == 0, m05), cmap="autumn", alpha=0.6)
    axes[2].set_title(f"Iceberg mask, NDWI > 0.05\n{int(m05.sum())} px")

    axes[3].imshow(rgb)
    axes[3].imshow(np.ma.masked_where(~lost, lost), cmap="cool", alpha=0.7)
    axes[3].set_title(f"Pixels lost (NDWI 0 to 0.05)\n{int(lost.sum())} px")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(os.path.basename(tif_path), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}  (lost px = {int(lost.sum())})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chips_root", default="/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/chips/KQ")
    parser.add_argument("--out_dir",    default="/home/llinkas/S2-iceberg-areas/sweeps/ndwi_chip_diffs")
    parser.add_argument("--picks",      required=True,
                        help="Comma-separated 'sza_bin:chip_stem' pairs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Render each pick
    for spec in args.picks.split(","):
        sza_bin, stem = spec.split(":")
        tif = os.path.join(args.chips_root, sza_bin, "tifs", stem + ".tif")
        if not os.path.exists(tif):
            print(f"MISSING: {tif}")
            continue
        out_png = os.path.join(args.out_dir, f"{sza_bin}__{stem}.png")
        render_chip(tif, out_png)


if __name__ == "__main__":
    main()
