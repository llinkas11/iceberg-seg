"""
eval_per_iceberg.py: per-pair evaluation with Hungarian matching.

Produces the Fisser-comparable numbers the paper's model progression rests on:

  per-pair MAE on iceberg area (m2)
  per-pair MAE on iceberg root length (m)
  per-pair IoU (mean, median)
  detection stats: n_ref, n_pred, n_matched, match rate, mean IoU matched

Ground truth = connected components of the test split's y_test.pkl masks.
Predictions = rasterised iceberg polygons from each method's output gpkg.
Matching   = Hungarian on 1 - IoU with a post-filter IoU >= threshold.
             Greedy matching retained for A/B under --matcher greedy.

Usage (manifest-driven, preferred):
  python scripts/eval_per_iceberg.py \\
      --manifest data/v4_clean/manifest.json \\
      --test_dir results/baseline_v1/test \\
      --out_dir  results/baseline_v1/per_iceberg \\
      --methods  TR,OT,UNet,UNet_TR,UNet_OT,UNet_CRF

Legacy (y_test.pkl + test_index.csv, single-method):
  python scripts/eval_per_iceberg.py \\
      --test_pkl data/v4_clean/train_validate_test/y_test.pkl \\
      --test_index data/v4_clean/split_log.csv \\
      --pred_dir   results/v3/UNet \\
      --out_dir    results/v3/per_iceberg

Binary segmentation: single class (iceberg, shadow merged).
"""

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import label as cc_label
from scipy.optimize import linear_sum_assignment

from eval_methods import (
    ICEBERG_CLASS,
    SZA_ORDER,
    METHODS,
    filter_merged_to_chip,
    load_merged_gpkg,
    load_test_ground_truth,
    load_test_ground_truth_from_manifest,
    rasterize_gpkg,
    read_per_chip_gpkg,
)

PIXEL_AREA_M2 = 100.0  # 10 m x 10 m
DEFAULT_IOU_THRESHOLD = 0.3


def connected_components(mask):
    """
    Return list of {'mask', 'area_px', 'area_m2', 'bbox'} for the iceberg
    class. bbox is (rmin, rmax, cmin, cmax) and lets compute_iou_matrix skip
    disjoint-bounded component pairs without touching pixels.
    """
    binary = (mask == ICEBERG_CLASS).astype(np.int32)
    labels, n = cc_label(binary)
    components = []
    for c in range(1, n + 1):
        comp = (labels == c)
        area = int(comp.sum())
        if area == 0:
            continue
        rows = np.any(comp, axis=1)
        cols = np.any(comp, axis=0)
        rmin, rmax = int(rows.argmax()), int(len(rows) - rows[::-1].argmax())
        cmin, cmax = int(cols.argmax()), int(len(cols) - cols[::-1].argmax())
        components.append({
            "mask":    comp,
            "area_px": area,
            "area_m2": area * PIXEL_AREA_M2,
            "bbox":    (rmin, rmax, cmin, cmax),
        })
    return components


def _bboxes_overlap(a, b):
    """Half-open bbox intersection test, returns False when disjoint."""
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return not (ar1 <= br0 or br1 <= ar0 or ac1 <= bc0 or bc1 <= ac0)


def compute_iou_matrix(gt_comps, pred_comps):
    """
    Pairwise IoU, shape (n_gt, n_pred). Skips pairs whose bounding boxes do
    not overlap; for typical chips this cuts the inner loop by 5-10x.
    """
    n_gt, n_pred = len(gt_comps), len(pred_comps)
    iou = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i, gc in enumerate(gt_comps):
        gm, gbb = gc["mask"], gc["bbox"]
        for j, pc in enumerate(pred_comps):
            if not _bboxes_overlap(gbb, pc["bbox"]):
                continue
            inter = int((gm & pc["mask"]).sum())
            if inter == 0:
                continue
            union = int((gm | pc["mask"]).sum())
            iou[i, j] = inter / union
    return iou


def hungarian_match(iou_matrix, iou_threshold=DEFAULT_IOU_THRESHOLD):
    """
    Assign predictions to GT icebergs to maximise total IoU, then drop pairs
    below iou_threshold. Returns list of (gt_idx, pred_idx, iou).
    """
    n_gt, n_pred = iou_matrix.shape
    if n_gt == 0 or n_pred == 0:
        return []
    # linear_sum_assignment minimises cost; feed 1 - IoU so high IoU wins.
    cost = 1.0 - iou_matrix
    gt_idx, pred_idx = linear_sum_assignment(cost)
    matches = []
    for gi, pi in zip(gt_idx, pred_idx):
        score = float(iou_matrix[gi, pi])
        if score >= iou_threshold:
            matches.append((int(gi), int(pi), score))
    return matches


def greedy_match(iou_matrix, iou_threshold=0.0):
    """Kept for A/B comparison; same rule as the pre-refactor script."""
    matches = []
    used_pred = set()
    n_gt, n_pred = iou_matrix.shape
    if n_gt == 0 or n_pred == 0:
        return matches
    gt_order = np.argsort(-iou_matrix.max(axis=1))
    for gi in gt_order:
        best_pi, best_iou = -1, iou_threshold
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


def load_pred_mask_from_gpkg(method, gt_record, test_dir, merged_by_bin):
    """
    Resolve a method's polygons for one chip to a (H, W) uint8 mask on the
    chip's own transform. Returns None if no prediction is available.
    """
    sza_bin   = gt_record["sza_bin"]
    chip_stem = gt_record["chip_stem"]
    tif_path  = gt_record["tif_path"]
    transform = gt_record["transform"]
    crs       = gt_record["crs"]
    height    = gt_record["height"]
    width     = gt_record["width"]
    if transform is None:
        return None

    gdf = read_per_chip_gpkg(test_dir, sza_bin, method, chip_stem)
    if gdf is None:
        gdf = filter_merged_to_chip(merged_by_bin.get(sza_bin), chip_stem, tif_path)

    if gdf is not None and len(gdf) and crs and gdf.crs and gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    return rasterize_gpkg(gdf, transform, height, width)


def _chip_row(method, sza_bin, chip_stem, n_gt, n_pred, n_matched, note=""):
    return {
        "method":    method, "sza_bin": sza_bin, "chip_stem": chip_stem,
        "n_gt":      n_gt, "n_pred": n_pred, "n_matched": n_matched,
        "n_fn":      max(0, n_gt - n_matched), "n_fp": max(0, n_pred - n_matched),
        "note":      note,
    }


def _pair_row(method, sza_bin, chip_stem, gi, pi, gt_comp, pred_comp, iou):
    gt_area   = gt_comp["area_m2"]
    pred_area = pred_comp["area_m2"]
    gt_rl     = float(np.sqrt(gt_area))
    pred_rl   = float(np.sqrt(pred_area))
    abs_area  = abs(pred_area - gt_area)
    abs_rl    = abs(pred_rl   - gt_rl)
    re_pct    = 100.0 * (pred_area - gt_area) / gt_area if gt_area > 0 else float("nan")
    return {
        "method":          method,
        "sza_bin":         sza_bin,
        "chip_stem":       chip_stem,
        "gt_idx":          gi,
        "pred_idx":        pi,
        "gt_area_m2":      gt_area,
        "pred_area_m2":    pred_area,
        "gt_rl_m":         round(gt_rl,   2),
        "pred_rl_m":       round(pred_rl, 2),
        "iou":             round(iou, 4),
        "abs_area_err_m2": round(abs_area, 2),
        "abs_rootlen_err_m": round(abs_rl, 3),
        "sq_area_err_m2":  round(abs_area ** 2, 2),
        "re_pct":          round(re_pct, 3) if not np.isnan(re_pct) else "",
    }


def score_chip_pair(method, sza_bin, chip_stem, gt_comps, pred_mask,
                    matcher, iou_threshold):
    """
    Compute per-pair rows + the chip-level summary row for one chip.

    gt_comps is passed in pre-computed so the caller can hoist it out of any
    per-method loop (GT does not change with method).
    """
    pred_comps = connected_components(pred_mask) if pred_mask is not None else []
    n_gt, n_pred = len(gt_comps), len(pred_comps)

    if n_gt == 0 and n_pred == 0:
        return [], _chip_row(method, sza_bin, chip_stem, 0, 0, 0, note="both_null")
    if n_gt == 0:
        return [], _chip_row(method, sza_bin, chip_stem, 0, n_pred, 0, note="gt_null")
    if n_pred == 0:
        return [], _chip_row(method, sza_bin, chip_stem, n_gt, 0, 0, note="pred_null")

    match_func = hungarian_match if matcher == "hungarian" else greedy_match
    matches = match_func(compute_iou_matrix(gt_comps, pred_comps), iou_threshold)

    pair_rows = [
        _pair_row(method, sza_bin, chip_stem, gi, pi, gt_comps[gi], pred_comps[pi], iou)
        for gi, pi, iou in matches
    ]
    chip_row = _chip_row(method, sza_bin, chip_stem,
                         n_gt, n_pred, len(matches), note="")
    return pair_rows, chip_row


def eval_method_per_iceberg(method, gt_records, test_dir, matcher, iou_threshold):
    """Return (pair_rows, chip_rows, detection_stats) for one method."""
    merged_by_bin = {b: load_merged_gpkg(test_dir, b, method) for b in SZA_ORDER}

    pair_rows = []
    chip_rows = []
    n_gt_total = n_pred_total = n_matched_total = 0

    for rec in gt_records:
        sza_bin   = rec["sza_bin"]
        chip_stem = rec["chip_stem"]
        gt_comps  = rec["gt_comps"]  # hoisted from per-method path; see main()

        pred_mask = load_pred_mask_from_gpkg(method, rec, test_dir, merged_by_bin)
        if pred_mask is None and rec["transform"] is None:
            chip_rows.append(_chip_row(method, sza_bin, chip_stem, 0, 0, 0,
                                       note="no_chip_transform"))
            continue

        new_pairs, chip_row = score_chip_pair(
            method, sza_bin, chip_stem, gt_comps, pred_mask,
            matcher, iou_threshold,
        )
        pair_rows.extend(new_pairs)
        chip_rows.append(chip_row)
        n_gt_total      += chip_row["n_gt"]
        n_pred_total    += chip_row["n_pred"]
        n_matched_total += chip_row["n_matched"]

    detection = {
        "method":       method,
        "n_gt_total":   n_gt_total,
        "n_pred_total": n_pred_total,
        "n_matched":    n_matched_total,
        "match_rate":   round(n_matched_total / n_gt_total, 4) if n_gt_total   else 0.0,
        "precision":    round(n_matched_total / n_pred_total, 4) if n_pred_total else 0.0,
    }
    return pair_rows, chip_rows, detection


def build_pair_summary(pair_rows):
    """Mean / median per (method, sza_bin) on the per-pair metrics."""
    if not pair_rows:
        return pd.DataFrame()
    df = pd.DataFrame(pair_rows)
    df["sza_bin"] = pd.Categorical(df["sza_bin"], categories=SZA_ORDER, ordered=True)
    agg = (
        df.groupby(["method", "sza_bin"], observed=False)
        .agg(
            n_pairs                = ("iou", "count"),
            mean_iou               = ("iou", "mean"),
            median_iou             = ("iou", "median"),
            mean_abs_area_err_m2   = ("abs_area_err_m2",   "mean"),
            median_abs_area_err_m2 = ("abs_area_err_m2",   "median"),
            mean_abs_rootlen_err_m = ("abs_rootlen_err_m", "mean"),
            median_abs_rootlen_err_m = ("abs_rootlen_err_m", "median"),
            mean_sq_area_err_m2    = ("sq_area_err_m2", "mean"),
        )
        .reset_index()
    )
    for c in agg.columns:
        if agg[c].dtype.kind == "f":
            agg[c] = agg[c].round(4)
    return agg


def print_method_table(summary, metric_col, title):
    print(f"\n{title}")
    print(f"{'Method':<12} " + "".join(f"{b:>16}" for b in SZA_ORDER))
    print("-" * (12 + 16 * len(SZA_ORDER)))
    for m in METHODS:
        cells = []
        for b in SZA_ORDER:
            r = summary[(summary["method"] == m) & (summary["sza_bin"] == b)]
            if len(r) and not pd.isna(r[metric_col].values[0]):
                cells.append(f"{r[metric_col].values[0]:>16.4f}")
            else:
                cells.append(f"{'-':>16}")
        if any(c.strip() != "-" for c in cells):
            print(f"{m:<12} " + "".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=None,
                        help="manifest.json; if set, preferred over legacy test_pkl + test_index")
    parser.add_argument("--test_pkl", default=None,
                        help="Legacy: path to y_test.pkl")
    parser.add_argument("--test_index", default=None,
                        help="Legacy: path to split_log.csv or test_index.csv")
    parser.add_argument("--pkl_dir", default=None,
                        help="Optional override for the directory holding y_test.pkl (manifest mode)")
    parser.add_argument("--test_dir", default=None,
                        help="area_comparison/test/ root containing {sza_bin}/{method}/ outputs")
    parser.add_argument("--pred_dir", default=None,
                        help="Legacy single-method dir (bypasses --test_dir + --methods)")
    parser.add_argument("--methods", default=",".join(METHODS),
                        help="Comma-separated method names (default: all six)")
    parser.add_argument("--matcher", default="hungarian", choices=["hungarian", "greedy"])
    parser.add_argument("--iou_threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -- Load ground truth ----------------------------------------------------
    if args.manifest:
        gt_records = load_test_ground_truth_from_manifest(args.manifest, args.pkl_dir)
    elif args.test_pkl and args.test_index:
        gt_records = load_test_ground_truth(os.path.dirname(args.test_pkl), args.test_index)
    else:
        parser.error("pass either --manifest, or both --test_pkl and --test_index")

    # GT masks do not change with method; extract components once here and
    # hand the list off to every per-method scorer. Saves 5 x 228 cc_label
    # calls across the six-method sweep.
    for rec in gt_records:
        rec["gt_comps"] = connected_components(rec["mask"])

    if args.pred_dir:
        legacy_single_method(args, gt_records)
        return

    if not args.test_dir:
        parser.error("--test_dir is required unless --pred_dir is set")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        parser.error(f"unknown method(s): {unknown}; known: {METHODS}")

    all_pair_rows = []
    all_chip_rows = []
    detection_rows = []
    for m in methods:
        print(f"\nEvaluating {m}...")
        pair_rows, chip_rows, det = eval_method_per_iceberg(
            m, gt_records, args.test_dir, args.matcher, args.iou_threshold,
        )
        print(f"  matched pairs: {det['n_matched']}  "
              f"match_rate: {det['match_rate']:.3f}  "
              f"precision: {det['precision']:.3f}")
        all_pair_rows.extend(pair_rows)
        all_chip_rows.extend(chip_rows)
        detection_rows.append(det)

    # -- Save CSVs -----------------------------------------------------------
    pair_csv = os.path.join(args.out_dir, "eval_per_iceberg.csv")
    chip_csv = os.path.join(args.out_dir, "eval_per_iceberg_chips.csv")
    det_csv  = os.path.join(args.out_dir, "eval_per_iceberg_detection.csv")
    summ_csv = os.path.join(args.out_dir, "eval_per_iceberg_summary.csv")

    if all_pair_rows:
        pd.DataFrame(all_pair_rows).to_csv(pair_csv, index=False)
    if all_chip_rows:
        pd.DataFrame(all_chip_rows).to_csv(chip_csv, index=False)
    pd.DataFrame(detection_rows).to_csv(det_csv, index=False)

    summary = build_pair_summary(all_pair_rows)
    summary.to_csv(summ_csv, index=False)

    # -- Print tables --------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("PER-PAIR SUMMARY")
    print("=" * 65)
    print_method_table(summary, "mean_abs_area_err_m2",
                       "MAE on iceberg AREA (m2), mean over matched pairs")
    print_method_table(summary, "mean_abs_rootlen_err_m",
                       "MAE on iceberg ROOT LENGTH (m), mean over matched pairs")
    print_method_table(summary, "mean_iou",
                       "Mean IoU on matched pairs")

    print(f"\nOutputs in: {args.out_dir}/")
    print(f"  {os.path.basename(pair_csv)}")
    print(f"  {os.path.basename(chip_csv)}")
    print(f"  {os.path.basename(det_csv)}")
    print(f"  {os.path.basename(summ_csv)}")


def legacy_single_method(args, gt_records):
    """
    Back-compat path for the pre-Phase-B script shape: --pred_dir points at
    a single method's output dir with geotiffs/*_pred.tif. Treats the UNet
    method name and computes pair metrics only for that dir.
    """
    geotiff_dir = os.path.join(args.pred_dir, "geotiffs")
    if not os.path.isdir(geotiff_dir):
        print(f"ERROR: geotiffs/ not found under {args.pred_dir}")
        return

    pair_rows = []
    chip_rows = []
    for rec in gt_records:
        chip_stem = rec["chip_stem"]
        sza_bin   = rec["sza_bin"]

        pred_path = os.path.join(geotiff_dir, f"{chip_stem}_pred.tif")
        if not os.path.exists(pred_path):
            candidates = glob(os.path.join(geotiff_dir, f"*{chip_stem}*_pred.tif"))
            if not candidates:
                chip_rows.append(_chip_row("UNet", sza_bin, chip_stem, 0, 0, 0,
                                           note="pred_not_found"))
                continue
            pred_path = candidates[0]

        with rasterio.open(pred_path) as src:
            pred_mask = src.read(1)

        new_pairs, chip_row = score_chip_pair(
            "UNet", sza_bin, chip_stem, rec["gt_comps"], pred_mask,
            args.matcher, args.iou_threshold,
        )
        pair_rows.extend(new_pairs)
        chip_rows.append(chip_row)

    os.makedirs(args.out_dir, exist_ok=True)
    pair_csv = os.path.join(args.out_dir, "eval_per_iceberg.csv")
    chip_csv = os.path.join(args.out_dir, "eval_per_iceberg_chips.csv")
    if pair_rows:
        pd.DataFrame(pair_rows).to_csv(pair_csv, index=False)
    if chip_rows:
        pd.DataFrame(chip_rows).to_csv(chip_csv, index=False)
    print(f"Legacy single-method run complete: {pair_csv}")


if __name__ == "__main__":
    main()
