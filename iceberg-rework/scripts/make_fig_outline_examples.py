"""
make_fig_outline_examples.py: Per-SZA-bin chip examples with reference and
predicted iceberg outlines overlaid. Fisser 2024 Fig. 13 analog. Visualizes
how predicted vs reference outlines diverge across the SZA range using our
best method (UNet_CRF) at three RE positions per bin: worst-positive,
near-zero, worst-negative.

Layout: 4 rows (one per SZA bin) x 3 columns (RE position).
Each panel: B08 grayscale + GT contour (cyan, dashed) + UNet_CRF contour
(bright purple, solid). Annotations: chip stem (truncated), per-pair
RE %, gt area.

Reads:
  <run>/per_iceberg/eval_per_iceberg.csv
  <manifest>/manifest.json + train_validate_test/y_test.pkl
  <run>/inference/<sza_bin>/UNet_CRF/...           (predicted polygons)
  <chip tif>                                        (B08 background)

Writes (via _fig_registry):
  <run>/per_iceberg/fig-archive/<ts>__outline_examples.png
  <run>/per_iceberg/figures.md (row appended or updated)
  <run>/per_iceberg/outline_examples_chips.csv  (selected chips per cell)

Usage:
  python scripts/make_fig_outline_examples.py \
      --run /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/runs/exp_baseline_v1/20260424_185158

Rsync after edit:
  rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/make_fig_outline_examples.py \
      llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from skimage.measure import find_contours

from _fig_registry import write as write_fig
from eval_methods import (
    SZA_ORDER,
    load_merged_gpkg,
    load_test_ground_truth_from_manifest,
)
from eval_per_iceberg import connected_components, load_pred_mask_from_gpkg

LLINKAS = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"

DEFAULT_RUN = os.path.join(LLINKAS, "runs/exp_baseline_v1/20260424_185158")
DEFAULT_MANIFEST = os.path.join(LLINKAS, "data/v4_clean/manifest.json")

METHOD = "UNet_CRF"
B08_BAND_INDEX = 3
MIN_GT_AREA_M2 = 5000.0   # focus on visually meaningful icebergs
CROP_PAD_PX = 20          # context margin around the matched-pair iceberg bbox. Tightened from 25 to make outlines legible, then loosened to 20 so contours are not clipped at panel edges and have visual breathing room.
RE_POSITIONS = ["worst_pos", "near_zero", "worst_neg"]
RE_POSITION_TITLES = {
    "worst_pos":  "Worst overestimation",
    "near_zero":  "Near-zero error",
    "worst_neg":  "Worst underestimation",
}

SZA_LABELS = {"sza_lt65": r"$\theta < 65^{\circ}$",
              "sza_65_70": r"$65^{\circ} \leq \theta < 70^{\circ}$",
              "sza_70_75": r"$70^{\circ} \leq \theta < 75^{\circ}$",
              "sza_gt75":  r"$\theta \geq 75^{\circ}$"}


def pick_chips(df, method, sza_bin):
    """Pick three rows per SZA bin: worst-positive, near-zero, worst-negative
    re_pct, restricted to gt_area_m2 >= MIN_GT_AREA_M2."""
    sub = df[(df["method"] == method) & (df["sza_bin"] == sza_bin)
             & (df["gt_area_m2"] >= MIN_GT_AREA_M2)].copy()
    if sub.empty:
        return {pos: None for pos in RE_POSITIONS}

    sub["abs_re"] = sub["re_pct"].abs()
    worst_pos_row = sub.loc[sub["re_pct"].idxmax()] if (sub["re_pct"] > 0).any() else None
    worst_neg_row = sub.loc[sub["re_pct"].idxmin()] if (sub["re_pct"] < 0).any() else None
    near_zero_row = sub.loc[sub["abs_re"].idxmin()]

    return {"worst_pos": worst_pos_row, "near_zero": near_zero_row,
            "worst_neg": worst_neg_row}


def short_stem(chip_stem, max_len=42):
    if len(chip_stem) <= max_len:
        return chip_stem
    return chip_stem[:max_len - 3] + "..."


def crop_window(gt_comp, pred_comp, chip_shape, pad=CROP_PAD_PX):
    """Compute (r0, r1, c0, c1) covering the union of gt + pred component
    bboxes plus pad pixels on each side, clipped to the chip dimensions.

    Falls back to the GT bbox alone when pred_comp is None."""
    H, W = chip_shape
    bboxes = [gt_comp["bbox"]]
    if pred_comp is not None:
        bboxes.append(pred_comp["bbox"])
    r0 = min(b[0].start for b in bboxes)
    r1 = max(b[0].stop for b in bboxes)
    c0 = min(b[1].start for b in bboxes)
    c1 = max(b[1].stop for b in bboxes)
    return (max(0, r0 - pad), min(H, r1 + pad),
            max(0, c0 - pad), min(W, c1 + pad))


def draw_panel(ax, b08, gt_comp, pred_comp, sza_bin, position, row):
    """Render one outline-overlay panel cropped to the matched-pair iceberg.

    Plots only the matched GT component (cyan dashed) and the matched
    predicted component (bright purple solid), not every iceberg in the
    chip.
    """
    ax.set_xticks([])
    ax.set_yticks([])

    if row is None or gt_comp is None:
        ax.text(0.5, 0.5, "no qualifying pair",
                transform=ax.transAxes, ha="center", va="center", fontsize=9)
        ax.set_facecolor("#f4f4f4")
        return

    chip_shape = b08.shape if b08 is not None else gt_comp["mask"].shape
    r0, r1, c0, c1 = crop_window(gt_comp, pred_comp, chip_shape)

    if b08 is not None:
        b08_crop = b08[r0:r1, c0:c1]
        vmin = float(np.percentile(b08_crop, 2))
        vmax = float(np.percentile(b08_crop, 98))
        ax.imshow(b08_crop, cmap="gray", vmin=vmin, vmax=vmax)

    # GT contour: cyan dashed (matched component only)
    gt_crop = gt_comp["mask"][r0:r1, c0:c1]
    for contour in find_contours(gt_crop.astype(np.float32), 0.5):
        ax.plot(contour[:, 1], contour[:, 0],
                color="#00BCD4", linewidth=1.6, linestyle="--", alpha=0.95)

    # Pred contour: bright purple solid (matched component only). Color
    # matches UNet_CRF in the headline data figures (Figs. 7-9).
    if pred_comp is not None:
        pred_crop = pred_comp["mask"][r0:r1, c0:c1]
        for contour in find_contours(pred_crop.astype(np.float32), 0.5):
            ax.plot(contour[:, 1], contour[:, 0],
                    color="#7C4DFF", linewidth=1.6, linestyle="-", alpha=0.95)

    # Title-strip annotation (chip stem omitted; reader-facing only).
    re_pct = float(row["re_pct"])
    gt_area = float(row["gt_area_m2"])
    pred_area = float(row["pred_area_m2"])

    text = (f"RE = {re_pct:+.1f}%\n"
            f"GT  = {gt_area/1e3:.1f}k m$^2$\n"
            f"Pred = {pred_area/1e3:.1f}k m$^2$")
    ax.text(0.02, 0.98, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=11, color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="black",
                      edgecolor="none", alpha=0.7))

    # 10 m scale bar (Sentinel-2 ground sample distance = 10 m / pixel).
    # Drawn in data (cropped-pixel) coords near the bottom-left so the
    # bar's length encodes a fixed ground distance regardless of crop size.
    H = b08_crop.shape[0] if b08 is not None else gt_crop.shape[0]
    bar_x0 = 1.0       # 1 px from left edge
    bar_y0 = H - 2.0   # 2 px from bottom edge (image coords: y grows downward)
    bar_w  = 1.0       # 1 px = 10 m at Sentinel-2 GSD
    bar_h  = 0.6
    ax.add_patch(mpatches.Rectangle(
        (bar_x0, bar_y0), bar_w, bar_h,
        facecolor="white", edgecolor="black", linewidth=0.8, zorder=10,
    ))
    ax.text(bar_x0 + bar_w + 0.5, bar_y0 + bar_h / 2,
            "10 m (1 px)", ha="left", va="center",
            fontsize=9, color="white",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="black",
                      edgecolor="none", alpha=0.75),
            zorder=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--method", default=METHOD,
                        help="Method whose predictions to overlay (default UNet_CRF).")
    args = parser.parse_args()

    pairs_path = os.path.join(args.run, "per_iceberg", "eval_per_iceberg.csv")
    df = pd.read_csv(pairs_path)
    if args.method not in df["method"].unique():
        raise ValueError(f"method {args.method} not in {pairs_path}")

    # 1. Pick three chips per SZA bin
    print(f"Picking exemplar chips per SZA bin for {args.method} ...")
    selections = {}
    for sza in SZA_ORDER:
        selections[sza] = pick_chips(df, args.method, sza)
        for pos, row in selections[sza].items():
            if row is not None:
                print(f"  {sza:<10} {pos:<11} chip={short_stem(str(row['chip_stem']))} "
                      f"RE={float(row['re_pct']):+.1f}% gt={float(row['gt_area_m2']):,.0f}")

    # 2. Persist chip selections
    sel_csv = os.path.join(args.run, "per_iceberg", "outline_examples_chips.csv")
    with open(sel_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sza_bin", "position", "chip_stem", "gt_area_m2",
                         "pred_area_m2", "re_pct", "iou"])
        for sza in SZA_ORDER:
            for pos in RE_POSITIONS:
                row = selections[sza][pos]
                if row is None:
                    writer.writerow([sza, pos, "", "", "", "", ""])
                else:
                    writer.writerow([
                        sza, pos, row["chip_stem"],
                        f"{row['gt_area_m2']:.0f}",
                        f"{row['pred_area_m2']:.0f}",
                        f"{row['re_pct']:+.3f}",
                        f"{row['iou']:.4f}",
                    ])
    print(f"\nWrote {sel_csv}")

    # 3. Load GT records once and index by chip_stem
    print("\nLoading manifest test split + y_test.pkl ...")
    gt_records = load_test_ground_truth_from_manifest(args.manifest)
    gt_by_stem = {r["chip_stem"]: r for r in gt_records}

    # 4. Pre-load merged gpkg per SZA bin (one read per bin instead of per chip)
    test_dir = os.path.join(args.run, "inference")
    merged_by_bin = {sza: load_merged_gpkg(test_dir, sza, args.method)
                     for sza in SZA_ORDER}

    # 5. Render 4x3 panel grid
    fig, axes = plt.subplots(len(SZA_ORDER), len(RE_POSITIONS),
                              figsize=(11.5, 14))

    for ri, sza in enumerate(SZA_ORDER):
        for ci, pos in enumerate(RE_POSITIONS):
            ax = axes[ri, ci]
            row = selections[sza][pos]
            if row is None:
                draw_panel(ax, None, None, None, sza, pos, None)
                continue

            chip = str(row["chip_stem"])
            gt_rec = gt_by_stem.get(chip)
            if gt_rec is None:
                draw_panel(ax, None, None, None, sza, pos, row)
                continue

            # B08 background
            b08 = None
            tif = gt_rec.get("tif_path")
            if tif and os.path.exists(tif):
                with rasterio.open(tif) as src:
                    if src.count >= B08_BAND_INDEX:
                        b08 = src.read(B08_BAND_INDEX).astype(np.float32)

            # Identify the specific matched components by their CSV-recorded
            # gt_idx and pred_idx; eval_per_iceberg's connected_components
            # returns components in cc_label order (no filtering, no sorting),
            # so the indices align with the per-pair table.
            gt_comps = connected_components(gt_rec["mask"])
            gt_idx = int(row["gt_idx"])
            gt_comp = gt_comps[gt_idx] if 0 <= gt_idx < len(gt_comps) else None

            pred_mask = load_pred_mask_from_gpkg(
                args.method, gt_rec, test_dir, merged_by_bin,
            )
            pred_comp = None
            if pred_mask is not None:
                pred_comps = connected_components(pred_mask)
                pred_idx = int(row["pred_idx"])
                if 0 <= pred_idx < len(pred_comps):
                    pred_comp = pred_comps[pred_idx]

            draw_panel(ax, b08, gt_comp, pred_comp, sza, pos, row)

        # Per-row label on the leftmost panel
        axes[ri, 0].set_ylabel(SZA_LABELS[sza], fontsize=15, fontweight="bold")

    # Column headers
    for ci, pos in enumerate(RE_POSITIONS):
        axes[0, ci].set_title(RE_POSITION_TITLES[pos],
                              fontsize=15, fontweight="bold", pad=8)

    # Figure-level legend
    handles = [
        plt.Line2D([0], [0], color="#00BCD4", linestyle="--", linewidth=2.0,
                   label="Reference (GT) outline"),
        plt.Line2D([0], [0], color="#7C4DFF", linestyle="-", linewidth=2.0,
                   label=f"{args.method} predicted outline"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=13,
               frameon=False, bbox_to_anchor=(0.5, 0.01))

    # No suptitle: figure caption (in the LaTeX figure environment) carries
    # the title; the on-image title was unreadable from a distance.
    fig.tight_layout(rect=[0, 0.04, 1, 1.0])

    archive = write_fig(
        fig,
        slug="outline_examples",
        caption=(
            "Per-SZA-bin chip examples with reference (cyan dashed) and "
            f"{args.method} predicted (bright purple) iceberg outlines, picked at "
            "three per-pair RE positions: worst-positive, near-zero, "
            "worst-negative (chips with reference root length below 70 m "
            "excluded). B08 NIR backdrop, 256x256 px chips. Fisser (2024) "
            "Fig. 13 analog."
        ),
        out_dir=os.path.join(args.run, "per_iceberg"),
    )
    plt.close(fig)
    print(f"Wrote {archive}")


if __name__ == "__main__":
    main()
