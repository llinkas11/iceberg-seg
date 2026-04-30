"""
make_fig_threshold_sweep.py: Per-pair relative-error sweep across B08 NIR
reflectance thresholds. Fisser 2024 Fig. 4 analog. Validates 0.22 (= Fisser
0.12 + 0.10 DN offset) as the lowest-spread threshold on our test split.

For each threshold T:
  pred_mask = (B08 >= T) on each test chip
  pred_comps = connected components with area >= 16 px (40 m root length)
  match against y_test.pkl GT components via Hungarian on 1 - IoU
  per-pair relative error = 100 * (pred_area - gt_area) / gt_area

Reads (HPC-only):
  data/v4_clean/manifest.json
  data/v4_clean/train_validate_test/y_test.pkl
  every test chip's tif (band 3 = B08)

Writes (via _fig_registry):
  reference/threshold_sweep/fig-archive/<ts>__threshold_sweep_re.png
  reference/threshold_sweep/figures.md
  reference/threshold_sweep/threshold_sweep.csv  (per-threshold aggregates)

Usage:
  python scripts/make_fig_threshold_sweep.py

Optional:
  --manifest data/v4_clean/manifest.json
  --thresholds 0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32

Rsync after edit:
  rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/make_fig_threshold_sweep.py \
      llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from _fig_registry import write as write_fig
from eval_methods import load_test_ground_truth_from_manifest
from eval_per_iceberg import (
    compute_iou_matrix,
    connected_components,
    hungarian_match,
)

LLINKAS = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"

DEFAULT_MANIFEST = os.path.join(LLINKAS, "data/v4_clean/manifest.json")
DEFAULT_OUT_DIR = os.path.join(LLINKAS, "reference/threshold_sweep")

MIN_AREA_PX = 16  # 40 m root length at 10 m resolution
B08_BAND_INDEX = 3
DEFAULT_THRESHOLDS = "0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32"
PROJECT_THRESHOLD = 0.22  # Fisser 0.12 + 0.10 DN offset, our canonical
FISSER_THRESHOLD_BASE = 0.12  # Fisser 2024 calibrated value (no DN offset)


def filter_components_by_area(components, min_area_px):
    """Drop components below the 40 m / 16 px floor."""
    return [c for c in components if c["area_px"] >= min_area_px]


def load_b08_bands(gt_records):
    """Pre-load B08 reflectance for every test chip; ~60 MB for 228 chips."""
    bands = {}
    skipped = []
    for rec in gt_records:
        tif = rec.get("tif_path")
        if not tif or not os.path.exists(tif):
            skipped.append(rec["chip_stem"])
            continue
        with rasterio.open(tif) as src:
            if src.count < B08_BAND_INDEX:
                skipped.append(rec["chip_stem"])
                continue
            b08 = src.read(B08_BAND_INDEX).astype(np.float32)
        bands[rec["chip_stem"]] = b08
    return bands, skipped


def sweep_threshold(threshold, gt_records, gt_comps_cache, b08_bands,
                    iou_threshold=0.3):
    """
    Apply threshold to every chip's B08, build pred_comps with the 40 m filter,
    Hungarian-match against pre-computed gt_comps, return per-pair re_pct list.
    """
    pair_re = []
    n_chips = 0
    n_pairs = 0
    for rec in gt_records:
        chip = rec["chip_stem"]
        b08 = b08_bands.get(chip)
        if b08 is None:
            continue
        n_chips += 1
        pred_mask = (b08 >= threshold).astype(np.uint8)
        pred_comps = filter_components_by_area(
            connected_components(pred_mask), MIN_AREA_PX,
        )
        gt_comps = gt_comps_cache[chip]
        if not gt_comps or not pred_comps:
            continue
        iou_mat = compute_iou_matrix(gt_comps, pred_comps)
        matches = hungarian_match(iou_mat, iou_threshold=iou_threshold)
        for gi, pi, _ in matches:
            gt_area = gt_comps[gi]["area_m2"]
            pred_area = pred_comps[pi]["area_m2"]
            if gt_area > 0:
                pair_re.append(100.0 * (pred_area - gt_area) / gt_area)
                n_pairs += 1
    return np.array(pair_re, dtype=np.float64), n_chips, n_pairs


def aggregate(re_array):
    if re_array.size == 0:
        return dict(mean=np.nan, p10=np.nan, p25=np.nan, p75=np.nan,
                    p90=np.nan, n=0)
    return dict(
        mean=float(np.mean(re_array)),
        p10=float(np.percentile(re_array, 10)),
        p25=float(np.percentile(re_array, 25)),
        p75=float(np.percentile(re_array, 75)),
        p90=float(np.percentile(re_array, 90)),
        n=int(re_array.size),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS,
                        help="Comma-separated B08 reflectance thresholds.")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]

    # 1. Load test GT records
    print("Loading manifest test split + y_test.pkl ...")
    gt_records = load_test_ground_truth_from_manifest(args.manifest)
    print(f"  {len(gt_records)} test chips loaded")

    # 2. Cache GT components per chip; identical across all thresholds
    print("Computing GT components per chip ...")
    gt_comps_cache = {}
    n_gt_total = 0
    for rec in gt_records:
        comps = filter_components_by_area(
            connected_components(rec["mask"]), MIN_AREA_PX,
        )
        gt_comps_cache[rec["chip_stem"]] = comps
        n_gt_total += len(comps)
    print(f"  {n_gt_total} GT components across {len(gt_records)} chips")

    # 3. Pre-load B08 once
    print("Loading B08 bands from chip tifs ...")
    b08_bands, skipped = load_b08_bands(gt_records)
    print(f"  {len(b08_bands)} chips loaded, {len(skipped)} skipped (no tif or B08)")
    if skipped:
        print(f"  first 5 skipped: {skipped[:5]}")

    # 4. Sweep thresholds
    print(f"\nSweeping thresholds: {thresholds}")
    rows = []
    aggregates = []
    for t in thresholds:
        pair_re, n_chips, n_pairs = sweep_threshold(
            t, gt_records, gt_comps_cache, b08_bands,
        )
        agg = aggregate(pair_re)
        agg["threshold"] = t
        agg["n_chips_with_pred"] = n_chips
        agg["n_pairs"] = n_pairs
        agg["iqr"] = agg["p75"] - agg["p25"]
        agg["p10p90_spread"] = agg["p90"] - agg["p10"]
        aggregates.append(agg)
        rows.append([t, n_pairs, agg["mean"], agg["p10"], agg["p25"],
                     agg["p75"], agg["p90"]])
        print(f"  T={t:.2f}: n_pairs={n_pairs:>5} "
              f"mean={agg['mean']:+7.1f}%  IQR={agg['iqr']:6.1f}%  "
              f"P10-P90={agg['p10p90_spread']:6.1f}%")

    # 5. Persist per-threshold aggregates
    csv_path = os.path.join(args.out_dir, "threshold_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "n_pairs", "mean_re_pct",
                         "p10_re_pct", "p25_re_pct", "p75_re_pct", "p90_re_pct"])
        for r in rows:
            writer.writerow([f"{r[0]:.4f}", r[1], f"{r[2]:.4f}",
                             f"{r[3]:.4f}", f"{r[4]:.4f}",
                             f"{r[5]:.4f}", f"{r[6]:.4f}"])
    print(f"\nWrote {csv_path}")

    # 6. Plot mean RE with IQR (P25-P75) and P10-P90 bands
    means = [a["mean"] for a in aggregates]
    p10s = [a["p10"] for a in aggregates]
    p25s = [a["p25"] for a in aggregates]
    p75s = [a["p75"] for a in aggregates]
    p90s = [a["p90"] for a in aggregates]

    # Find threshold with the smallest IQR; expected near 0.22
    iqr_min_idx = int(np.argmin([a["iqr"] for a in aggregates]))
    best_t = thresholds[iqr_min_idx]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.fill_between(thresholds, p10s, p90s, color="#1976D2", alpha=0.15,
                    label="P10-P90")
    ax.fill_between(thresholds, p25s, p75s, color="#1976D2", alpha=0.30,
                    label="IQR (P25-P75)")
    ax.plot(thresholds, means, marker="o", color="#0D47A1", linewidth=2.0,
            markersize=8, label="Mean RE")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(PROJECT_THRESHOLD, color="#D32F2F", linestyle=":",
               linewidth=1.5, alpha=0.85,
               label=f"Project threshold ({PROJECT_THRESHOLD:.2f})")
    ax.axvline(best_t, color="#388E3C", linestyle="-", linewidth=1.0,
               alpha=0.6,
               label=f"Min IQR at {best_t:.2f}")

    ax.set_xlabel(r"B08 reflectance threshold ($\rho_{NIR}$)")
    ax.set_ylabel(r"Per-pair relative error $RE_A$ (%)")
    ax.set_title("Per-pair area error vs B08 threshold "
                 "(Fisser 2024 Fig. 4 analog)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # Annotate the project threshold's mean RE
    proj_idx = thresholds.index(PROJECT_THRESHOLD) if PROJECT_THRESHOLD in thresholds else None
    if proj_idx is not None:
        proj = aggregates[proj_idx]
        ax.annotate(
            f"Mean = {proj['mean']:+.1f}%\nIQR = {proj['p75'] - proj['p25']:.1f}%\nn = {proj['n']:,}",
            xy=(PROJECT_THRESHOLD, proj["mean"]),
            xytext=(35, 60), textcoords="offset points",
            fontsize=8, color="#D32F2F",
            arrowprops=dict(arrowstyle="-", color="#D32F2F", lw=0.6),
        )

    fig.tight_layout()

    archive = write_fig(
        fig,
        slug="threshold_sweep_re",
        caption=(
            "Per-pair relative error in iceberg area as a function of the "
            "B08 NIR reflectance threshold, with the IQR (P25-P75) and "
            "P10-P90 bands shaded. Fisser (2024) Fig. 4 analog. The "
            "vertical dotted line at 0.22 marks our project threshold "
            "(Fisser 0.12 + 0.10 DN offset); the green line marks the "
            "threshold with the smallest IQR in this sweep."
        ),
        out_dir=args.out_dir,
    )
    plt.close(fig)
    print(f"Wrote {archive}")
    print(f"\nMin-IQR threshold: {best_t:.2f}; project threshold: {PROJECT_THRESHOLD:.2f}")


if __name__ == "__main__":
    main()
