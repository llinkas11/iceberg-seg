"""
eval_per_iceberg.py — Per-iceberg evaluation metrics.

Computes MAE (mean absolute error of iceberg area), RERL (relative error for
root length), and contrast (B08 iceberg mean minus ocean mean) by matching
predicted icebergs to ground truth icebergs via connected component IoU.

Binary segmentation model: single class (iceberg). Shadow merged into iceberg.

Usage:
  python scripts/eval_per_iceberg.py \
      --pred_dir results/v3/UNet \
      --test_pkl data/v3_clean/train_validate_test/y_test.pkl \
      --test_index data/v3_clean/split_log.csv \
      --out_dir results/v3
"""

import argparse
import csv
import glob as glob_mod
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import label as cc_label

warnings.filterwarnings("ignore", category=UserWarning)

PIXEL_AREA_M2 = 100.0  # 10 m x 10 m
ICEBERG_CLASS = 1
SZA_BINS = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def connected_components(mask):
    """Return list of (component_mask, area_px) for iceberg class."""
    binary = (mask == ICEBERG_CLASS).astype(np.int32)
    labels, n = cc_label(binary)
    components = []
    for c in range(1, n + 1):
        comp = (labels == c)
        area = int(comp.sum())
        if area > 0:
            components.append({"mask": comp, "area_px": area, "area_m2": area * PIXEL_AREA_M2})
    return components


def compute_iou_matrix(gt_comps, pred_comps):
    """Compute IoU between all GT and predicted components."""
    n_gt = len(gt_comps)
    n_pred = len(pred_comps)
    iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i, gc in enumerate(gt_comps):
        for j, pc in enumerate(pred_comps):
            intersection = (gc["mask"] & pc["mask"]).sum()
            union = (gc["mask"] | pc["mask"]).sum()
            if union > 0:
                iou_matrix[i, j] = intersection / union
    return iou_matrix


def greedy_match(iou_matrix, min_iou=0.0):
    """
    Greedy matching: for each GT iceberg, assign the pred with highest IoU.
    Returns list of (gt_idx, pred_idx, iou). Unmatched GT/pred are not included.
    """
    matches = []
    used_pred = set()
    n_gt, n_pred = iou_matrix.shape

    # Sort GT by descending max IoU to prioritize strong matches
    gt_order = np.argsort(-iou_matrix.max(axis=1))

    for gi in gt_order:
        best_pi = -1
        best_iou = min_iou
        for pi in range(n_pred):
            if pi in used_pred:
                continue
            if iou_matrix[gi, pi] > best_iou:
                best_iou = iou_matrix[gi, pi]
                best_pi = pi
        if best_pi >= 0:
            matches.append((int(gi), int(best_pi), float(best_iou)))
            used_pred.add(best_pi)

    return matches


def main():
    parser = argparse.ArgumentParser(description="Per-iceberg evaluation metrics")
    parser.add_argument("--pred_dir", required=True,
                        help="Directory containing geotiffs/ with *_pred.tif prediction masks")
    parser.add_argument("--test_pkl", required=True,
                        help="Path to y_test.pkl ground truth")
    parser.add_argument("--test_x_pkl", default=None,
                        help="Path to x_test.pkl (for B08 contrast computation)")
    parser.add_argument("--test_index", required=True,
                        help="Path to split_log.csv or test_index.csv with pkl_position, sza_bin, chip_stem")
    parser.add_argument("--out_dir", default="results/v3",
                        help="Output directory for CSVs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load ground truth
    Y_test = np.array(load_pkl(args.test_pkl))
    if Y_test.ndim == 4:
        Y_test = Y_test[:, 0, :, :]
    print(f"Loaded {len(Y_test)} GT masks from {args.test_pkl}")

    X_test = None
    if args.test_x_pkl and os.path.exists(args.test_x_pkl):
        X_test = np.array(load_pkl(args.test_x_pkl))
        print(f"Loaded {len(X_test)} test images for contrast computation")

    # Load test index
    test_df = pd.read_csv(args.test_index)
    # Filter to test split if split_log
    if "split" in test_df.columns:
        test_df = test_df[test_df["split"] == "test"].reset_index(drop=True)
    test_df = test_df.sort_values("pkl_position").reset_index(drop=True)

    if len(test_df) != len(Y_test):
        print(f"WARNING: test_index has {len(test_df)} rows but y_test has {len(Y_test)} chips")
        n = min(len(test_df), len(Y_test))
        test_df = test_df.iloc[:n]
        Y_test = Y_test[:n]

    # Find prediction masks
    geotiff_dir = os.path.join(args.pred_dir, "geotiffs")
    if not os.path.isdir(geotiff_dir):
        print(f"ERROR: geotiffs/ not found in {args.pred_dir}")
        return

    # Per-iceberg records
    iceberg_records = []
    # Per-chip summary
    chip_records = []

    n_matched = 0
    n_gt_total = 0
    n_pred_total = 0
    n_fn = 0
    n_fp = 0

    for k in range(len(Y_test)):
        gt_mask = Y_test[k]
        sza_bin = test_df.iloc[k].get("sza_bin", "unknown")
        chip_stem = test_df.iloc[k].get("chip_stem", f"chip_{k}")

        # Find corresponding prediction
        pred_path = os.path.join(geotiff_dir, f"{chip_stem}_pred.tif")
        if not os.path.exists(pred_path):
            # Try without exact match
            candidates = glob_mod.glob(os.path.join(geotiff_dir, f"*{chip_stem}*_pred.tif"))
            if candidates:
                pred_path = candidates[0]
            else:
                chip_records.append({
                    "chip_stem": chip_stem, "sza_bin": sza_bin,
                    "n_gt": 0, "n_pred": 0, "n_matched": 0,
                    "n_fn": 0, "n_fp": 0, "note": "pred_not_found"
                })
                continue

        with rasterio.open(pred_path) as src:
            pred_mask = src.read(1)

        # Connected components
        gt_comps = connected_components(gt_mask)
        pred_comps = connected_components(pred_mask)

        n_gt = len(gt_comps)
        n_pred = len(pred_comps)
        n_gt_total += n_gt
        n_pred_total += n_pred

        # B08 for contrast
        b08 = None
        if X_test is not None:
            b08 = X_test[k][2]  # B08 band
            ocean_mask = (gt_mask == 0)
            ocean_mean = float(b08[ocean_mask].mean()) if ocean_mask.sum() > 100 else np.nan

        if n_gt == 0 and n_pred == 0:
            chip_records.append({
                "chip_stem": chip_stem, "sza_bin": sza_bin,
                "n_gt": 0, "n_pred": 0, "n_matched": 0,
                "n_fn": 0, "n_fp": 0, "note": "both_null"
            })
            continue

        if n_gt == 0:
            n_fp += n_pred
            chip_records.append({
                "chip_stem": chip_stem, "sza_bin": sza_bin,
                "n_gt": 0, "n_pred": n_pred, "n_matched": 0,
                "n_fn": 0, "n_fp": n_pred, "note": "gt_null"
            })
            continue

        if n_pred == 0:
            n_fn += n_gt
            chip_records.append({
                "chip_stem": chip_stem, "sza_bin": sza_bin,
                "n_gt": n_gt, "n_pred": 0, "n_matched": 0,
                "n_fn": n_gt, "n_fp": 0, "note": "pred_null"
            })
            continue

        # Match
        iou_mat = compute_iou_matrix(gt_comps, pred_comps)
        matches = greedy_match(iou_mat, min_iou=0.0)

        matched_gt = set(m[0] for m in matches)
        matched_pred = set(m[1] for m in matches)
        chip_fn = n_gt - len(matched_gt)
        chip_fp = n_pred - len(matched_pred)
        n_fn += chip_fn
        n_fp += chip_fp
        n_matched += len(matches)

        for gi, pi, iou in matches:
            gt_area = gt_comps[gi]["area_m2"]
            pred_area = pred_comps[pi]["area_m2"]
            gt_rl = np.sqrt(gt_area)
            pred_rl = np.sqrt(pred_area)

            mae = abs(pred_area - gt_area)
            rerl = abs(pred_rl - gt_rl) / gt_rl if gt_rl > 0 else np.nan

            contrast = np.nan
            if b08 is not None:
                pred_b08_mean = float(b08[pred_comps[pi]["mask"]].mean())
                contrast = pred_b08_mean - ocean_mean if not np.isnan(ocean_mean) else np.nan

            iceberg_records.append({
                "chip_stem": chip_stem,
                "sza_bin": sza_bin,
                "gt_area_m2": gt_area,
                "pred_area_m2": pred_area,
                "gt_rl_m": round(gt_rl, 1),
                "pred_rl_m": round(pred_rl, 1),
                "iou": round(iou, 4),
                "mae_m2": round(mae, 1),
                "rerl": round(rerl, 4) if not np.isnan(rerl) else "",
                "contrast": round(contrast, 4) if not np.isnan(contrast) else "",
            })

        chip_records.append({
            "chip_stem": chip_stem, "sza_bin": sza_bin,
            "n_gt": n_gt, "n_pred": n_pred, "n_matched": len(matches),
            "n_fn": chip_fn, "n_fp": chip_fp, "note": ""
        })

    # ── Save per-iceberg CSV ─────────────────────────────────────────
    iceberg_csv = os.path.join(args.out_dir, "eval_per_iceberg.csv")
    if iceberg_records:
        fieldnames = list(iceberg_records[0].keys())
        with open(iceberg_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(iceberg_records)

    # ── Save per-chip CSV ────────────────────────────────────────────
    chip_csv = os.path.join(args.out_dir, "eval_per_iceberg_chips.csv")
    if chip_records:
        fieldnames = list(chip_records[0].keys())
        with open(chip_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(chip_records)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("PER-ICEBERG EVALUATION SUMMARY")
    print(f"{'='*65}")
    print(f"GT icebergs:       {n_gt_total}")
    print(f"Pred icebergs:     {n_pred_total}")
    print(f"Matched pairs:     {n_matched}")
    print(f"False negatives:   {n_fn} (GT not matched)")
    print(f"False positives:   {n_fp} (pred not matched)")

    if iceberg_records:
        maes = [r["mae_m2"] for r in iceberg_records]
        rerls = [r["rerl"] for r in iceberg_records if r["rerl"] != ""]
        contrasts = [r["contrast"] for r in iceberg_records if r["contrast"] != ""]
        ious = [r["iou"] for r in iceberg_records]

        print(f"\nMatched iceberg metrics:")
        print(f"  MAE (area):     mean={np.mean(maes):,.0f} m2  median={np.median(maes):,.0f} m2")
        print(f"  RERL:           mean={np.mean(rerls):.4f}  median={np.median(rerls):.4f}" if rerls else "")
        print(f"  IoU:            mean={np.mean(ious):.4f}  median={np.median(ious):.4f}")
        if contrasts:
            print(f"  Contrast:       mean={np.mean(contrasts):.4f}  median={np.median(contrasts):.4f}")

        # Per SZA bin
        print(f"\nPer SZA bin:")
        print(f"{'Bin':<15} {'N matched':>10} {'MAE mean':>12} {'RERL mean':>12} {'IoU mean':>10}")
        print("-" * 62)
        for sza in SZA_BINS:
            bin_recs = [r for r in iceberg_records if r["sza_bin"] == sza]
            if not bin_recs:
                continue
            bin_mae = np.mean([r["mae_m2"] for r in bin_recs])
            bin_rerl = np.mean([r["rerl"] for r in bin_recs if r["rerl"] != ""])
            bin_iou = np.mean([r["iou"] for r in bin_recs])
            print(f"{sza:<15} {len(bin_recs):>10} {bin_mae:>12,.0f} {bin_rerl:>12.4f} {bin_iou:>10.4f}")

    print(f"\nPer-iceberg CSV: {iceberg_csv}")
    print(f"Per-chip CSV:    {chip_csv}")


if __name__ == "__main__":
    main()
