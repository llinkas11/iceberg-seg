"""
eval_methods.py: Evaluate all 6 segmentation methods against ground truth.

This version evaluates the exact 96 test chips using `test_index.csv`, which is the
definitive mapping from `y_test.pkl` row -> real chip tif path.

Why this matters:
  - `split_log.csv` only stores scene stems, which are not unique across chips
  - Fisser chips use synthetic stems like `fisser_0000` in the pkl files
  - the actual test chip filenames are the real Sentinel-2 chip stems in test_index.csv

Metrics (iceberg class only, binary: iceberg=1 vs not-iceberg=0):
  IoU         = TP / (TP + FP + FN)
  Precision   = TP / (TP + FP)
  Recall      = TP / (TP + FN)
  F1          = 2 * P * R / (P + R)
"""

import argparse
import csv
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.features import rasterize as rio_rasterize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _method_common import load_manifest

warnings.filterwarnings("ignore")

METHODS   = ["TR", "OT", "UNet", "UNet_TR", "UNet_OT", "UNet_CRF"]
SZA_ORDER = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]
SZA_LABELS = {
    "sza_lt65":  "< 65°",
    "sza_65_70": "65–70°",
    "sza_70_75": "70–75°",
    "sza_gt75":  "> 75°",
}
ICEBERG_CLASS = 1   # binary masks: 0=background, 1=iceberg (shadow merged)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_test_ground_truth(pkl_dir, test_index_path):
    """Return a list of chip records aligned to y_test.pkl row order."""
    y_path = os.path.join(pkl_dir, "y_test.pkl")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"y_test.pkl not found: {y_path}")

    Y_test = load_pkl(y_path)          # (N, 1, H, W) or (N, H, W)
    if Y_test.ndim == 4:
        Y_test = Y_test[:, 0, :, :]    # -> (N, H, W)

    test_index = pd.read_csv(test_index_path)
    required = {"pkl_position", "sza_bin", "chip_stem", "tif_path"}
    missing = required - set(test_index.columns)
    if missing:
        raise ValueError(f"test_index.csv missing columns: {sorted(missing)}")

    test_index = test_index.sort_values("pkl_position").reset_index(drop=True)
    if len(test_index) != len(Y_test):
        print(f"WARNING: test_index has {len(test_index)} rows but y_test.pkl has {len(Y_test)} chips")
        n = min(len(test_index), len(Y_test))
        test_index = test_index.iloc[:n].copy()
        Y_test = Y_test[:n]

    gt = []
    for k, row in test_index.iterrows():
        gt.append(_build_gt_record(
            int(row["pkl_position"]), row["sza_bin"], row["chip_stem"],
            row["tif_path"], Y_test[k],
        ))

    print(f"Ground truth loaded: {len(gt)} test chips")
    return gt


def _build_gt_record(pkl_position, sza_bin, chip_stem, tif_path, y_row):
    """
    Assemble one gt record and cache the chip's transform, crs, and shape on
    it so eval_method does not reopen the tif for each of the six methods.
    That trims ~1400 rio.open calls per evaluation run.
    """
    mask = (y_row == ICEBERG_CLASS).astype(np.uint8)
    transform, crs, height, width = None, None, None, None
    if tif_path and os.path.exists(tif_path):
        with rio.open(tif_path) as src:
            transform = src.transform
            crs       = src.crs
            height    = src.height
            width     = src.width
    return {
        "pkl_position": pkl_position,
        "sza_bin":      sza_bin,
        "chip_stem":    chip_stem,
        "tif_path":     tif_path,
        "mask":         mask,
        "transform":    transform,
        "crs":          crs,
        "height":       height,
        "width":        width,
    }


def load_test_ground_truth_from_manifest(manifest_path, pkl_dir=None):
    """
    Build the same gt list as load_test_ground_truth but reading chip
    identity from manifest.json. If pkl_dir is given, ground-truth masks are
    pulled from y_test.pkl; otherwise every chip's mask is read directly off
    its tif (pixel value > 0 on any band, heuristic, not used today; reserved
    for a future pure-geotiff-GT branch).
    """
    manifest = load_manifest(manifest_path)
    print(f"Manifest: {manifest['manifest_id']} "
          f"chips_sha={manifest['chips_sha'][:16]}...")

    test_rows = [r for r in manifest["chips"] if r.get("split") == "test"]
    test_rows.sort(key=lambda r: r["pkl_position"])

    if pkl_dir is None:
        pkl_dir = os.path.join(os.path.dirname(manifest_path), "train_validate_test")
    y_path = os.path.join(pkl_dir, "y_test.pkl")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"y_test.pkl not found: {y_path}")
    Y_test = load_pkl(y_path)
    if Y_test.ndim == 4:
        Y_test = Y_test[:, 0, :, :]

    if len(test_rows) != len(Y_test):
        print(f"WARNING: manifest has {len(test_rows)} test chips, y_test.pkl has {len(Y_test)}")
        n = min(len(test_rows), len(Y_test))
        test_rows = test_rows[:n]
        Y_test = Y_test[:n]

    gt = []
    for k, row in enumerate(test_rows):
        gt.append(_build_gt_record(
            int(row["pkl_position"]), row["sza_bin"], row["chip_stem"],
            row.get("tif_path", ""), Y_test[k],
        ))
    print(f"Ground truth loaded: {len(gt)} test chips (from manifest)")
    return gt


def rasterize_gpkg(gdf, transform, height=256, width=256):
    """
    Rasterize a GeoDataFrame of iceberg polygons to a binary (H, W) uint8 mask.
    Returns all-zeros mask if gdf is empty.
    """
    if gdf is None or gdf.empty:
        return np.zeros((height, width), dtype=np.uint8)

    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros((height, width), dtype=np.uint8)

    mask = rio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask


# Each 10 m pixel is 100 m2; used for chip-area error metrics.
PIXEL_AREA_M2 = 100.0


def compute_metrics(pred_mask, gt_mask, pixel_area_m2=PIXEL_AREA_M2):
    """
    Compute per-chip segmentation + area-error metrics for the iceberg class.

    Pixel metrics: IoU, precision, recall, F1. Area metrics: predicted and
    ground-truth total iceberg area in m2, plus the absolute and squared
    errors. The absolute and squared errors aggregate to MAE and MSE once
    averaged over chips; they are included here as per-chip signals so
    downstream filtering (e.g. GT-positive-only) reuses the same columns.

    Both masks are binary uint8 (1=iceberg, 0=not-iceberg). Returns a dict
    of floats; NaN where a denominator is 0.
    """
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    TP = float((pred & gt).sum())
    FP = float((pred & ~gt).sum())
    FN = float((~pred & gt).sum())

    iou       = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else float("nan")
    precision = TP / (TP + FP)      if (TP + FP) > 0      else float("nan")
    recall    = TP / (TP + FN)      if (TP + FN) > 0      else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else float("nan"))

    pred_area_m2    = float(pred.sum()) * pixel_area_m2
    gt_area_m2      = float(gt.sum())   * pixel_area_m2
    abs_area_err_m2 = abs(pred_area_m2 - gt_area_m2)
    sq_area_err_m2  = (pred_area_m2 - gt_area_m2) ** 2

    return {
        "iou":             iou,
        "precision":       precision,
        "recall":          recall,
        "f1":              f1,
        "pred_area_m2":    pred_area_m2,
        "gt_area_m2":      gt_area_m2,
        "abs_area_err_m2": abs_area_err_m2,
        "sq_area_err_m2":  sq_area_err_m2,
    }


def load_skipped_chips(method_out_dir):
    """Return the set of chip_stems this method explicitly skipped, if logged."""
    path = os.path.join(method_out_dir, "skipped_chips.csv")
    if not os.path.exists(path):
        return set()
    stems = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("chip_stem"):
                stems.add(row["chip_stem"])
    return stems


def load_merged_gpkg(test_dir, sza_bin, method):
    path = os.path.join(test_dir, sza_bin, method, "all_icebergs.gpkg")
    if not os.path.exists(path):
        return None
    return gpd.read_file(path)


def read_per_chip_gpkg(test_dir, sza_bin, method, chip_stem):
    path = os.path.join(test_dir, sza_bin, method, "gpkgs", f"{chip_stem}_icebergs.gpkg")
    if os.path.exists(path):
        return gpd.read_file(path)
    return None


def filter_merged_to_chip(gdf, chip_stem, tif_path):
    if gdf is None or gdf.empty:
        return gdf.iloc[0:0] if gdf is not None else None

    if "source_file" not in gdf.columns:
        return gdf.iloc[0:0]

    tif_name = os.path.basename(tif_path)
    candidates = {chip_stem, tif_name, f"{chip_stem}.tif"}
    mask = gdf["source_file"].isin(candidates)
    return gdf.loc[mask].copy()


def eval_method(method, test_dir, gt_records, skip_policy="count_as_false_negative"):
    """
    Evaluate one method across all SZA bins using exact chip identity.

    The method's skipped_chips.csv (per-SZA-bin) is consulted so the evaluator
    knows which chips the method refused to score. The skip_policy controls
    what happens for those chips:

      count_as_false_negative   pred mask is all zeros, metrics computed
                                normally. This is the default: a skip is
                                treated as a confident "no icebergs", which
                                hurts recall on GT-positive chips.
      exclude                   chip is dropped from per-method metrics,
                                but still logged in the per-chip CSV with
                                every metric set to None.

    Either way every skip appears in the output with a `was_skipped` flag and
    contributes to n_skipped in the summary.
    """
    results = []
    merged_by_bin = {b: load_merged_gpkg(test_dir, b, method) for b in SZA_ORDER}

    # Per-bin skip sets, read once.
    skipped_by_bin = {}
    for b in SZA_ORDER:
        method_dir = os.path.join(test_dir, b, method)
        skipped_by_bin[b] = load_skipped_chips(method_dir)

    for rec in gt_records:
        sza_bin = rec["sza_bin"]
        chip_stem = rec["chip_stem"]
        tif_path = rec["tif_path"]
        gt_mask = rec["mask"]

        transform = rec["transform"]
        crs       = rec["crs"]
        height    = rec["height"]
        width     = rec["width"]
        if transform is None:
            print(f"  WARNING: no chip .tif for {chip_stem} in {sza_bin}, skipping")
            continue

        was_skipped = chip_stem in skipped_by_bin.get(sza_bin, set())

        if was_skipped and skip_policy == "exclude":
            # Record the skip so n_skipped is non-zero, but null out metrics.
            results.append({
                "method":        method,
                "sza_bin":       sza_bin,
                "pkl_position":  rec["pkl_position"],
                "chip_stem":     chip_stem,
                "was_skipped":   True,
                "gt_pixels":     int(gt_mask.sum()),
                "pred_pixels":   0,
                "iou":           None, "precision": None,
                "recall":        None, "f1":        None,
                "pred_area_m2":  None, "gt_area_m2": None,
                "abs_area_err_m2": None, "sq_area_err_m2": None,
            })
            continue

        chip_gdf = read_per_chip_gpkg(test_dir, sza_bin, method, chip_stem)
        if chip_gdf is None:
            chip_gdf = filter_merged_to_chip(merged_by_bin.get(sza_bin), chip_stem, tif_path)

        if chip_gdf is not None and len(chip_gdf) and crs and chip_gdf.crs and chip_gdf.crs != crs:
            chip_gdf = chip_gdf.to_crs(crs)

        pred_mask = rasterize_gpkg(chip_gdf, transform, height, width)
        metrics   = compute_metrics(pred_mask, gt_mask)

        results.append({
            "method":       method,
            "sza_bin":      sza_bin,
            "pkl_position": rec["pkl_position"],
            "chip_stem":    chip_stem,
            "was_skipped":  was_skipped,
            "gt_pixels":    int(gt_mask.sum()),
            "pred_pixels":  int(pred_mask.sum()),
            **{k: round(v, 4) if not np.isnan(v) else None
               for k, v in metrics.items()},
        })

    return results


def plot_iou_heatmap(summary, out_dir):
    """Heatmap: rows = method, cols = SZA bin, values = mean IoU."""
    methods  = [m for m in METHODS if m in summary["method"].values]
    bins     = [b for b in SZA_ORDER if b in summary["sza_bin"].values]
    if not methods or not bins:
        return

    data = np.full((len(methods), len(bins)), np.nan)
    for i, m in enumerate(methods):
        for j, b in enumerate(bins):
            row = summary[(summary["method"] == m) & (summary["sza_bin"] == b)]
            if len(row):
                v = row["mean_iou"].values[0]
                if v is not None:
                    data[i, j] = float(v)

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(bins)), max(4, 0.8 * len(methods))))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean IoU")

    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([SZA_LABELS.get(b, b) for b in bins], fontsize=11)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=11)
    ax.set_xlabel("Solar Zenith Angle bin", fontsize=12)
    ax.set_title("Mean iceberg IoU vs ground truth: method × SZA bin",
                 fontsize=13, fontweight="bold")

    # Annotate cells
    for i in range(len(methods)):
        for j in range(len(bins)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=10, color="black" if 0.2 < v < 0.8 else "white",
                        fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "eval_iou_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_metric_bars(summary, metric, out_dir):
    """Grouped bar chart for a given metric across methods and SZA bins."""
    methods  = [m for m in METHODS if m in summary["method"].values]
    bins     = [b for b in SZA_ORDER if b in summary["sza_bin"].values]
    if not methods or not bins:
        return

    col = f"mean_{metric}"
    colors = ["#FB8C00", "#F4511E", "#1E88E5", "#5E35B1", "#00897B", "#43A047"]

    x      = np.arange(len(bins))
    bar_w  = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(max(8, 3 * len(bins)), 5))

    for j, (method, color) in enumerate(zip(methods, colors)):
        vals = []
        for b in bins:
            row = summary[(summary["method"] == method) & (summary["sza_bin"] == b)]
            vals.append(float(row[col].values[0]) if len(row) and row[col].values[0] is not None else 0.0)
        offset = (j - len(methods) / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals, bar_w, label=method, color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([SZA_LABELS.get(b, b) for b in bins], fontsize=11)
    ax.set_xlabel("Solar Zenith Angle bin", fontsize=12)
    ax.set_ylabel(f"Mean {metric.upper()}", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title(f"Mean {metric.upper()} vs ground truth by method and SZA bin",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"eval_{metric}_bars.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_summary_table(summary, metric="iou"):
    col = f"mean_{metric}"
    methods = [m for m in METHODS if m in summary["method"].values]
    bins    = [b for b in SZA_ORDER if b in summary["sza_bin"].values]

    col_w  = 14
    header = f"{'Method':<12}" + "".join(f"{SZA_LABELS.get(b, b):>{col_w}}" for b in bins)
    print(f"\n{'─'*len(header)}")
    print(f"Mean {metric.upper()} vs ground truth")
    print(header)
    print(f"{'─'*len(header)}")
    for m in methods:
        row = f"{m:<12}"
        for b in bins:
            r = summary[(summary["method"] == m) & (summary["sza_bin"] == b)]
            if len(r) and r[col].values[0] is not None:
                val = f"{r[col].values[0]:.3f}"
            else:
                val = "-"
            row += f"{val:>{col_w}}"
        print(row)
    print(f"{'─'*len(header)}")


def build_summary(df):
    """
    Collapse per-chip rows into per-(method, sza_bin) means. Adds a skip count
    and chip-area MAE/MSE alongside the pixel metrics so later tables can read
    MAE for apples-to-apples comparison with threshold-only papers.
    """
    metric_cols = ["iou", "precision", "recall", "f1",
                   "abs_area_err_m2", "sq_area_err_m2"]
    agg = (
        df.groupby(["method", "sza_bin"])[metric_cols]
        .mean()
        .round(4)
        .reset_index()
    )
    agg = agg.rename(columns={
        "iou":             "mean_iou",
        "precision":       "mean_precision",
        "recall":          "mean_recall",
        "f1":              "mean_f1",
        "abs_area_err_m2": "mae_area_m2",
        "sq_area_err_m2":  "mse_area_m2",
    })

    # n_chips is the number of rows contributing; n_skipped is the subset that
    # the method refused to score (under count_as_false_negative those still
    # contribute zero-pred metrics, under exclude they contribute nulls).
    counts = (
        df.groupby(["method", "sza_bin"])
        .agg(n_chips=("chip_stem", "count"),
             n_skipped=("was_skipped", "sum"))
        .reset_index()
    )
    summary = agg.merge(counts, on=["method", "sza_bin"], how="left")
    summary["sza_bin"] = pd.Categorical(summary["sza_bin"], categories=SZA_ORDER, ordered=True)
    summary = summary.sort_values(["method", "sza_bin"]).reset_index(drop=True)
    return summary


def main():
    RESEARCH = "/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas"
    parser = argparse.ArgumentParser(
        description="Evaluate 6 segmentation methods against ground truth (test set)"
    )
    parser.add_argument("--test_dir",
        default=os.path.join(RESEARCH, "area_comparison", "test"),
        help="area_comparison/test/, contains {sza_bin}/{METHOD}/all_icebergs.gpkg")
    parser.add_argument("--chips_dir",
        default=os.path.join(RESEARCH, "test_chips"),
        help="test_chips/, contains {sza_bin}/{stem}.tif (built by prepare_test_chips_dir.py)")
    parser.add_argument("--test_index",
        default=os.path.expanduser("~/S2-iceberg-areas/test_index.csv"),
        help="Legacy mapping from y_test.pkl row -> chip_stem/tif_path (built by build_test_index.py)")
    parser.add_argument("--manifest",
        default=None,
        help="manifest.json from build_clean_dataset.py; if set, overrides --test_index")
    parser.add_argument("--split_log",
        default=os.path.join(RESEARCH, "train_validate_test_v2", "split_log.csv"))
    parser.add_argument("--pkl_dir",
        default=os.path.join(RESEARCH, "train_validate_test_v2", "train_validate_test"),
        help="Directory with y_test.pkl")
    parser.add_argument("--out_dir",
        default=os.path.join(RESEARCH, "test_outputs"))
    parser.add_argument("--skipped_chip_policy",
        default="count_as_false_negative",
        choices=["count_as_false_negative", "exclude"],
        help="How to treat chips the method logged in skipped_chips.csv")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -- Load ground truth ----------------------------------------------------
    if args.manifest:
        print(f"Loading ground truth from manifest: {args.manifest}")
        gt = load_test_ground_truth_from_manifest(args.manifest, args.pkl_dir or None)
    else:
        print("Loading ground truth masks from y_test.pkl (legacy path)...")
        if not os.path.exists(args.test_index):
            raise FileNotFoundError(
                f"test_index.csv not found: {args.test_index}\n"
                "Run build_test_index.py first, or pass --manifest or --test_index explicitly."
            )
        gt = load_test_ground_truth(args.pkl_dir, args.test_index)

    # -- Evaluate each method -------------------------------------------------
    all_results = []
    for method in METHODS:
        print(f"\nEvaluating {method}...")
        results = eval_method(method, args.test_dir, gt,
                              skip_policy=args.skipped_chip_policy)
        if results:
            print(f"  {len(results)} chips evaluated")
            for r in results:
                iou = r['iou']
                print(f"  {r['sza_bin']}  {r['chip_stem'][:50]}  iou={iou:.3f}" if iou is not None else
                      f"  {r['sza_bin']}  {r['chip_stem'][:50]}  iou=-")
        else:
            print(f"  No results (all_icebergs.gpkg not found?)")
        all_results.extend(results)

    if not all_results:
        print("\nNo results, run run_all_methods.sh first.")
        return

    # ── Save per-chip results ─────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    df["sza_bin"] = pd.Categorical(df["sza_bin"], categories=SZA_ORDER, ordered=True)
    df = df.sort_values(["method", "sza_bin", "pkl_position"]).reset_index(drop=True)

    per_chip_path = os.path.join(args.out_dir, "eval_results.csv")
    df.to_csv(per_chip_path, index=False)
    print(f"\nPer-chip results → {per_chip_path}")

    # ── Summary per (method, sza_bin), all test chips ───────────────────────
    summary = build_summary(df)

    summary_path = os.path.join(args.out_dir, "eval_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary → {summary_path}")

    # ── Print tables ──────────────────────────────────────────────────────────
    for metric in ["iou", "precision", "recall"]:
        print_summary_table(summary, metric)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_iou_heatmap(summary, args.out_dir)
    for metric in ["iou", "precision", "recall"]:
        plot_metric_bars(summary, metric, args.out_dir)

    # ── Positive-GT-only summary ──────────────────────────────────────────────
    df_pos = df[df["gt_pixels"] > 0].copy()
    if len(df_pos):
        summary_pos = build_summary(df_pos)
        summary_pos_path = os.path.join(args.out_dir, "eval_summary_gt_positive_only.csv")
        summary_pos.to_csv(summary_pos_path, index=False)
        print(f"\nGT-positive-only summary → {summary_pos_path}")

        print(f"\n{'='*50}")
        print("GT-positive-only summaries (gt_pixels > 0)")
        for metric in ["iou", "precision", "recall"]:
            print_summary_table(summary_pos, metric)

        plot_iou_heatmap(summary_pos, args.out_dir)
        os.replace(
            os.path.join(args.out_dir, "eval_iou_heatmap.png"),
            os.path.join(args.out_dir, "eval_iou_heatmap_gt_positive_only.png")
        )
        print(f"Saved: {os.path.join(args.out_dir, 'eval_iou_heatmap_gt_positive_only.png')}")

        for metric in ["iou", "precision", "recall"]:
            plot_metric_bars(summary_pos, metric, args.out_dir)
            src = os.path.join(args.out_dir, f"eval_{metric}_bars.png")
            dst = os.path.join(args.out_dir, f"eval_{metric}_bars_gt_positive_only.png")
            os.replace(src, dst)
            print(f"Saved: {dst}")

        # Rebuild the all-chip plots so the original filenames still exist
        plot_iou_heatmap(summary, args.out_dir)
        for metric in ["iou", "precision", "recall"]:
            plot_metric_bars(summary, metric, args.out_dir)

    print(f"\n{'─'*50}")
    print(f"Outputs in: {args.out_dir}/")
    print(f"  eval_results.csv      , per-chip IoU/precision/recall/F1")
    print(f"  eval_summary.csv      , mean metrics per (method, sza_bin)")
    print(f"  eval_summary_gt_positive_only.csv")
    print(f"  eval_iou_heatmap.png  , IoU heatmap: method × SZA bin")
    print(f"  eval_iou_heatmap_gt_positive_only.png")
    print(f"  eval_iou_bars.png     , IoU bar chart")
    print(f"  eval_iou_bars_gt_positive_only.png")
    print(f"  eval_precision_bars.png")
    print(f"  eval_precision_bars_gt_positive_only.png")
    print(f"  eval_recall_bars.png")
    print(f"  eval_recall_bars_gt_positive_only.png")


if __name__ == "__main__":
    main()
