"""
make_fig_b08_std_skewness.py: Per-SZA-bin standard deviation and skewness of
iceberg-pixel B08 reflectance. Fisser 2024 Fig. 10 analog. The histogram
counterpart (Fig. 9 analog) already exists as `hist_iceberg_vs_neighborhood_b08`
in descriptive_stats.py.

Computes:
  For each SZA bin, pool all B08 pixel values inside annotated iceberg masks
  (Roboflow polygons + Fisser Y arrays, shadow merged into iceberg). Report
  np.std and scipy.stats.skew on the pooled distribution.

Reads:
  data/annotations_filtered.coco.json  (Roboflow)
  data/fisser_filtered/*.pkl           (Fisser)
  smishra/rework/data/split_log.csv    (chip stem -> sza bin map)
  smishra/rework/chips/**.tif          (Sentinel-2 B08)

Writes (via _fig_registry):
  viz/descriptive_stats/fig-archive/<ts>__b08_std_skewness_by_sza.png
  viz/descriptive_stats/figures.md (row appended or updated)

Usage:
  python scripts/make_fig_b08_std_skewness.py

Rsync after edit:
  rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/make_fig_b08_std_skewness.py \
      llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/
"""

import argparse
import csv
import json
import os
import pickle
import re
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from scipy.stats import skew

from _fig_registry import write as write_fig

SMISHRA = "/mnt/research/v.gomezgilyaspik/students/smishra/rework"
LLINKAS = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"

DEFAULT_COCO = os.path.join(LLINKAS, "data/annotations_filtered.coco.json")
DEFAULT_FISSER = os.path.join(LLINKAS, "data/fisser_filtered")
DEFAULT_SPLIT_LOG = os.path.join(SMISHRA, "data/split_log.csv")
DEFAULT_CHIPS_ROOT = os.path.join(SMISHRA, "chips")
DEFAULT_VIZ_DIR = os.path.join(LLINKAS, "viz/descriptive_stats")

SZA_ORDER = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]
SZA_LABELS = {"sza_lt65": "< 65", "sza_65_70": "65 to 70",
              "sza_70_75": "70 to 75", "sza_gt75": "> 75"}
# Approximate midpoint angle for each bin; used as x-axis position so the
# panel echoes Fisser 2024 Fig. 10's continuous-θ x-axis.
SZA_MIDPOINTS = {"sza_lt65": 60.0, "sza_65_70": 67.5,
                 "sza_70_75": 72.5, "sza_gt75": 78.0}

RF_HASH_RE = re.compile(r"_png\.rf\.[A-Za-z0-9]+\.png$")
CHIP_RE = re.compile(r"^(.+)_r(\d+)_c(\d+)(?:_B08)?\.(?:png|tif)$")


def strip_rf_hash(fn):
    return RF_HASH_RE.sub(".png", fn)


def load_split_log(path):
    """Map chip_stem -> sza_bin from the split log."""
    stem_to_sza = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            stem_to_sza[row.get("chip_stem", row["stem"])] = row["sza_bin"]
    return stem_to_sza


def collect_roboflow_pixels(coco_path, split_log_path, chips_root):
    """Return {sza_bin: list of np.ndarray of B08 pixel values inside polygons}."""
    with open(coco_path) as f:
        coco = json.load(f)
    stem_to_sza = load_split_log(split_log_path)

    img_sza = {}
    for img in coco["images"]:
        fn = strip_rf_hash(img["file_name"])
        m = CHIP_RE.match(os.path.basename(fn))
        if m:
            cs = f"{m.group(1)}_r{int(m.group(2)):04d}_c{int(m.group(3)):04d}"
            img_sza[img["id"]] = stem_to_sza.get(cs, "unknown")

    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    px_by_bin = {sza: [] for sza in SZA_ORDER}
    for img_info in coco["images"]:
        fn = strip_rf_hash(img_info["file_name"])
        m = CHIP_RE.match(os.path.basename(fn))
        if not m:
            continue
        stem, row, col = m.group(1), int(m.group(2)), int(m.group(3))
        sza = img_sza.get(img_info["id"], "unknown")
        if sza not in SZA_ORDER:
            continue
        anns = ann_by_img.get(img_info["id"], [])
        if not anns:
            continue
        fname = f"{stem}_r{row:04d}_c{col:04d}.tif"
        matches = glob(os.path.join(chips_root, "**", fname), recursive=True)
        if not matches:
            continue
        with rasterio.open(matches[0]) as src:
            b08 = src.read(3).astype(np.float32)
        mask_img = Image.new("L", (256, 256), 0)
        draw = ImageDraw.Draw(mask_img)
        for ann in anns:
            for seg in ann.get("segmentation", []):
                if isinstance(seg, list) and len(seg) >= 6:
                    coords = [(int(round(seg[i])), int(round(seg[i+1])))
                              for i in range(0, len(seg)-1, 2)]
                    draw.polygon(coords, fill=1)
        mask = np.array(mask_img)
        if mask.sum() > 0:
            px_by_bin[sza].append(b08[mask > 0])
    return px_by_bin


def collect_fisser_pixels(pkl_dir):
    """Return {sza_bin: list of np.ndarray of B08 pixel values inside masks}.
    All Fisser chips are sza_lt65; shadow class merged into iceberg."""
    px_by_bin = {sza: [] for sza in SZA_ORDER}
    for x_file, y_file in [("X_train.pkl", "Y_train.pkl"),
                            ("X_validation.pkl", "Y_validation.pkl"),
                            ("x_test.pkl", "y_test.pkl")]:
        x_path = os.path.join(pkl_dir, x_file)
        y_path = os.path.join(pkl_dir, y_file)
        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            continue
        with open(x_path, "rb") as f:
            X = np.array(pickle.load(f))
        with open(y_path, "rb") as f:
            Y = np.array(pickle.load(f))
        if Y.ndim == 4:
            Y = Y[:, 0, :, :]
        Y[Y == 2] = 1
        for i in range(len(Y)):
            b08 = X[i][2].astype(np.float32)
            mask = (Y[i] == 1)
            if mask.sum() > 0:
                px_by_bin["sza_lt65"].append(b08[mask])
    return px_by_bin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", default=DEFAULT_COCO)
    parser.add_argument("--fisser", default=DEFAULT_FISSER)
    parser.add_argument("--split_log", default=DEFAULT_SPLIT_LOG)
    parser.add_argument("--chips_root", default=DEFAULT_CHIPS_ROOT)
    parser.add_argument("--viz_dir", default=DEFAULT_VIZ_DIR)
    args = parser.parse_args()
    os.makedirs(args.viz_dir, exist_ok=True)

    # 1. Collect iceberg-pixel B08 values per SZA bin
    print("Collecting Roboflow iceberg pixels...")
    rf = collect_roboflow_pixels(args.coco, args.split_log, args.chips_root)
    print("Collecting Fisser iceberg pixels...")
    fs = collect_fisser_pixels(args.fisser)

    pooled = {sza: np.concatenate(rf[sza] + fs[sza]) if (rf[sza] or fs[sza]) else np.array([])
              for sza in SZA_ORDER}

    # 2. Compute std and skewness per bin
    stds = {sza: float(np.std(pooled[sza])) if pooled[sza].size else np.nan
            for sza in SZA_ORDER}
    skews = {sza: float(skew(pooled[sza])) if pooled[sza].size else np.nan
             for sza in SZA_ORDER}
    n_pixels = {sza: int(pooled[sza].size) for sza in SZA_ORDER}

    print("\nB08 reflectance inside iceberg masks (pooled across chips):")
    print(f"{'bin':<14}{'n_pixels':>12}{'mean':>10}{'std':>10}{'skewness':>12}")
    for sza in SZA_ORDER:
        m = float(np.mean(pooled[sza])) if pooled[sza].size else np.nan
        print(f"{sza:<14}{n_pixels[sza]:>12,}{m:>10.4f}{stds[sza]:>10.4f}{skews[sza]:>12.4f}")

    # 3. Plot 2-panel std + skewness vs SZA midpoint
    x = [SZA_MIDPOINTS[s] for s in SZA_ORDER]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    ax_std, ax_sk = axes

    ax_std.plot(x, [stds[s] for s in SZA_ORDER],
                marker="o", color="#1976D2", linewidth=2.0, markersize=9)
    ax_std.set_ylabel("B08 std")
    ax_std.set_title("a. Iceberg B08 standard deviation by SZA bin",
                     fontsize=11, fontweight="bold", loc="left")
    ax_std.grid(True, alpha=0.3, linestyle="--")
    ax_std.set_ylim(bottom=0)

    ax_sk.plot(x, [skews[s] for s in SZA_ORDER],
               marker="s", color="#D32F2F", linewidth=2.0, markersize=9)
    ax_sk.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_sk.set_xlabel(r"Solar zenith angle bin midpoint ($^{\circ}$)")
    ax_sk.set_ylabel("B08 skewness")
    ax_sk.set_title("b. Iceberg B08 skewness by SZA bin",
                    fontsize=11, fontweight="bold", loc="left")
    ax_sk.grid(True, alpha=0.3, linestyle="--")

    # Per-bin n_pixels annotation on top axis
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{SZA_LABELS[s]}\n($\\theta\\approx${SZA_MIDPOINTS[s]:.0f}$^{{\\circ}}$)"
                            for s in SZA_ORDER])
    for xi, sza in zip(x, SZA_ORDER):
        ax_std.annotate(f"n={n_pixels[sza]:,}\npixels",
                        xy=(xi, stds[sza]),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=8, color="#444")

    fig.suptitle("Iceberg B08 reflectance shape statistics by SZA",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # 4. Route through fig_registry
    archive = write_fig(
        fig,
        slug="b08_std_skewness_by_sza",
        caption=(
            "Per-SZA-bin standard deviation (a) and skewness (b) of iceberg-"
            "pixel B08 reflectance, pooled across all annotated icebergs in "
            "each bin (shadow merged into iceberg). Fisser (2024) Fig. 10 "
            "analog; the matched histogram view is "
            "`hist_iceberg_vs_neighborhood_b08`."
        ),
        out_dir=args.viz_dir,
    )
    plt.close(fig)
    print(f"\nWrote {archive}")


if __name__ == "__main__":
    main()
