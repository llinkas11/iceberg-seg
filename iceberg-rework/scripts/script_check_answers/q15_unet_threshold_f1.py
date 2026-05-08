"""
q15_unet_threshold_f1.py: empirical answer to script-check question 15.

Question (from script-check-README.md, threshold_probs.py):
  "Threshold = 0.22 on probability ... is reusing 0.22 the right choice, or
  should we calibrate the probability threshold separately, e.g. to the F1
  optimum on the validation set?"

What this script does:
  1. Load the v4_clean manifest, filter to the 137 val chips, pair them with
     the y_validation.pkl GT masks (collapse to binary via (y > 0)).
  2. Index every *_probs.tif under --probs_root by chip stem.
  3. For each val chip with both GT and probs, sweep tau in linspace(0.05,
     0.95, 19) and accumulate confusion-matrix counts (TP, FP, FN, TN) at
     pixel level.
  4. Report F1, IoU, precision, recall vs tau; mark argmax-F1 and tau=0.22
     for direct comparison with threshold_probs.py.
  5. Emit a per-tau CSV and two PNGs: an aggregate metric-vs-tau plot and a
     per-(region, SZA bin) F1 plot.

Inputs:
  --manifest    v4_clean manifest.json (HPC: data/v4_clean/manifest.json).
  --probs_root  Root containing every *_probs.tif emitted by predict_tifs.py.
  --val_pkl     Optional explicit path to the val GT pickle. By default the
                script probes common names alongside the manifest's pkl_dir.
  --split       Which manifest split to evaluate against ("val" or "test").
                As of 2026-05-05 the model-comparison runs only emit probs
                for the test chips, so the default is "test" with a sweep
                caveat documented in the README. When val probs become
                available, switch to --split val for an unbiased pick.
  --out_root    Parent directory; a slug subfolder is created under it.

Outputs (under <out_root>/q15_unet_threshold_f1/):
  <ts>__q15_unet_threshold_f1.csv               one row per tau
  <ts>__q15_unet_threshold_f1__overview.png     metric-vs-tau curves
  <ts>__q15_unet_threshold_f1__by_sza_region.png  F1-vs-tau per (region, sza)

Usage on moosehead:
  conda activate iceberg-unet
  python q15_unet_threshold_f1.py \
    --manifest /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/data/v4_clean/manifest.json \
    --probs_root /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/output/predictions_v4_clean

Deploy to moosehead:
  rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/script_check_answers/ \
    llinkas@moosehead.bowdoin.edu:~/iceberg-rework/scripts/script_check_answers/
"""

import argparse
import csv
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _method_common import load_manifest

from _common import (
    SZA_BINS, REGIONS,
    make_slug_dir, resolve_manifest_path, resolve_out_root, resolve_probs_root, stamp,
)


SLUG = "q15_unet_threshold_f1"
TAUS = np.linspace(0.05, 0.95, 19)
PRODUCTION_TAU = 0.22


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", default=str(resolve_manifest_path()),
                   help="v4_clean manifest.json path")
    p.add_argument("--probs_root", default=str(resolve_probs_root()),
                   help="Root containing *_probs.tif (recursively searched)")
    p.add_argument("--split", choices=["val", "test"], default="test",
                   help="Manifest split to evaluate against (default: test, "
                   "since val probs are not currently emitted)")
    p.add_argument("--val_pkl", default=None,
                   help="Override path to the GT pickle for the chosen split")
    p.add_argument("--out_root", default=str(resolve_out_root()),
                   help="Parent directory; a slug subfolder is created under it")
    return p.parse_args()


def find_gt_pkl(manifest_path, split, override):
    """Return a Path to the GT pickle for the chosen split, probing common names."""
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"--val_pkl not found: {p}")
        return p

    pkl_dir = Path(manifest_path).parent / "train_validate_test"
    if split == "val":
        names = ["y_validation.pkl", "Y_validation.pkl", "y_val.pkl"]
    else:
        names = ["y_test.pkl", "Y_test.pkl", "x_test.pkl".replace("x_", "y_")]
    for name in names:
        candidate = pkl_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No {split} pickle found under {pkl_dir} (tried {names}). "
        "Pass --val_pkl explicitly."
    )


def load_split_ground_truth(manifest_path, split, gt_pkl_path):
    """
    Build a list of {chip_stem, sza_bin, region, gt_mask} for every chip in
    the chosen manifest split. gt_mask is binary uint8 (1 = any non-ocean).
    """
    manifest = load_manifest(manifest_path)
    rows = [r for r in manifest["chips"] if r.get("split") == split]
    rows.sort(key=lambda r: r["pkl_position"])

    with open(gt_pkl_path, "rb") as f:
        Y = np.array(pickle.load(f))
    if Y.ndim == 4:
        Y = Y[:, 0, :, :]

    if len(rows) != len(Y):
        print(f"WARNING: manifest has {len(rows)} {split} chips, "
              f"{gt_pkl_path.name} has {len(Y)}. Truncating to min.")
        n = min(len(rows), len(Y))
        rows = rows[:n]
        Y = Y[:n]

    records = []
    for k, row in enumerate(rows):
        gt_mask = (Y[k] > 0).astype(np.uint8)  # collapse any non-ocean to iceberg
        region = row.get("region") or _region_from_stem(row.get("chip_stem", ""))
        records.append({
            "chip_stem": row["chip_stem"],
            "sza_bin":   row["sza_bin"],
            "region":    region,
            "gt_mask":   gt_mask,
        })
    return records


def _region_from_stem(stem):
    """Heuristic region inference from a Sentinel-2 chip stem (MGRS tile)."""
    # KQ chips fall in MGRS T25WDQ / T25WER / T24WXT etc.; SK in T24WVU.
    # The manifest typically already carries `region`, so this is a fallback.
    if "T24WVU" in stem or "T24WWU" in stem:
        return "SK"
    return "KQ"


def index_probs(probs_root):
    """Scan probs_root recursively and return {chip_stem: Path}."""
    index = {}
    for p in Path(probs_root).rglob("*_probs.tif"):
        stem = p.name[:-len("_probs.tif")]
        index[stem] = p
    return index


def main():
    args = parse_args()

    # 1. Resolve paths and load split GT
    out_dir = make_slug_dir(SLUG, args.out_root)
    gt_pkl_path = find_gt_pkl(args.manifest, args.split, args.val_pkl)
    print(f"Manifest: {args.manifest}")
    print(f"Split: {args.split}")
    print(f"GT pickle: {gt_pkl_path}")
    gt = load_split_ground_truth(args.manifest, args.split, gt_pkl_path)
    print(f"Loaded {len(gt)} {args.split} GT records")

    # 2. Index probs
    probs_idx = index_probs(args.probs_root)
    print(f"Indexed {len(probs_idx)} *_probs.tif under {args.probs_root}")

    # 3. Per-chip per-tau confusion-matrix accumulation
    aggregate = {tau: dict(tp=0, fp=0, fn=0, tn=0) for tau in TAUS}
    per_bin = {(r, s): {tau: dict(tp=0, fp=0, fn=0, tn=0)
                         for tau in TAUS}
               for r in REGIONS for s in SZA_BINS}
    chip_skips = []
    n_used = 0

    for rec in gt:
        stem = rec["chip_stem"]
        if stem not in probs_idx:
            chip_skips.append({"chip_stem": stem, "reason": "probs_missing"})
            continue
        with rasterio.open(probs_idx[stem]) as src:
            if src.count < 2:
                chip_skips.append({"chip_stem": stem, "reason": "too_few_prob_bands"})
                continue
            p_iceberg = src.read(2).astype(np.float32)
        if (p_iceberg.max() - p_iceberg.min()) < 0.01:
            chip_skips.append({"chip_stem": stem, "reason": "flat_prob"})
            continue

        gt_mask = rec["gt_mask"].astype(bool)
        for tau in TAUS:
            pred = p_iceberg >= tau
            tp = int((pred & gt_mask).sum())
            fp = int((pred & ~gt_mask).sum())
            fn = int((~pred & gt_mask).sum())
            tn = int((~pred & ~gt_mask).sum())
            aggregate[tau]["tp"] += tp
            aggregate[tau]["fp"] += fp
            aggregate[tau]["fn"] += fn
            aggregate[tau]["tn"] += tn
            key = (rec["region"], rec["sza_bin"])
            if key in per_bin:
                per_bin[key][tau]["tp"] += tp
                per_bin[key][tau]["fp"] += fp
                per_bin[key][tau]["fn"] += fn
                per_bin[key][tau]["tn"] += tn
        n_used += 1

    print(f"Used {n_used} chips; skipped {len(chip_skips)}")

    # 4. Compute metrics per tau and write CSV
    def metrics(c):
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1   = (2 * prec * rec / (prec + rec)
                if (prec + rec) > 0 else float("nan"))
        iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        return prec, rec, f1, iou

    csv_path = out_dir / f"{stamp(SLUG)}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou"])
        for tau in TAUS:
            c = aggregate[tau]
            prec, rec, f1, iou = metrics(c)
            writer.writerow([f"{tau:.4f}", c["tp"], c["fp"], c["fn"], c["tn"],
                             f"{prec:.6f}", f"{rec:.6f}", f"{f1:.6f}", f"{iou:.6f}"])
    print(f"CSV: {csv_path}")

    # 5. Print spot summary
    f1s = [metrics(aggregate[t])[2] for t in TAUS]
    best_idx = int(np.nanargmax(f1s))
    best_tau, best_f1 = TAUS[best_idx], f1s[best_idx]
    prod_idx = int(np.argmin(np.abs(TAUS - PRODUCTION_TAU)))
    prod_f1 = f1s[prod_idx]
    print(f"\nargmax-F1 tau = {best_tau:.3f}  (F1 = {best_f1:.4f})")
    print(f"production tau = {TAUS[prod_idx]:.3f}  (F1 = {prod_f1:.4f})")
    print(f"delta_F1 (best - production) = {best_f1 - prod_f1:+.4f}")

    # 6. Overview plot: F1 / IoU / P / R vs tau
    precs = [metrics(aggregate[t])[0] for t in TAUS]
    recs  = [metrics(aggregate[t])[1] for t in TAUS]
    ious  = [metrics(aggregate[t])[3] for t in TAUS]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(TAUS, f1s,   marker="o", color="#D32F2F", label="F1")
    ax.plot(TAUS, ious,  marker="s", color="#1976D2", label="IoU")
    ax.plot(TAUS, precs, marker="^", color="#388E3C", label="precision", alpha=0.7)
    ax.plot(TAUS, recs,  marker="v", color="#F57C00", label="recall",    alpha=0.7)
    ax.axvline(best_tau, color="#D32F2F", ls="--", lw=1.0, alpha=0.6)
    ax.axvline(PRODUCTION_TAU, color="#37474F", ls=":", lw=1.0, alpha=0.7)
    ax.text(best_tau, 0.02, f" argmax τ={best_tau:.2f}\n F1={best_f1:.3f}",
            color="#D32F2F", fontsize=8, va="bottom")
    ax.text(PRODUCTION_TAU, 0.95, f" prod τ=0.22\n F1={prod_f1:.3f}",
            color="#37474F", fontsize=8, va="top")
    ax.set_xlabel("Threshold τ on P(iceberg)")
    ax.set_ylabel("Pixel-level metric (val split)")
    ax.set_title(f"Q15: UNet+TR threshold sweep on v4_clean {args.split} (n={n_used})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower center", ncol=4, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    overview_path = out_dir / f"{stamp(SLUG, 'overview')}.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {overview_path}")

    # 7. Per-(region, sza_bin) F1 curves
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10
    color_idx = 0
    for r in REGIONS:
        for s in SZA_BINS:
            counts = per_bin[(r, s)]
            chip_count = sum(1 for rec in gt
                              if rec["region"] == r and rec["sza_bin"] == s)
            if chip_count == 0:
                continue
            f1_curve = [metrics(counts[t])[2] for t in TAUS]
            ax.plot(TAUS, f1_curve, marker="o", lw=1.2,
                    color=cmap(color_idx % 10),
                    label=f"{r} {s} (n={chip_count})")
            color_idx += 1
    ax.axvline(PRODUCTION_TAU, color="#37474F", ls=":", lw=1.0, alpha=0.7)
    ax.set_xlabel("Threshold τ on P(iceberg)")
    ax.set_ylabel("F1 (pixel-level)")
    ax.set_title("Q15: F1 vs τ split by (region, SZA bin)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")
    by_path = out_dir / f"{stamp(SLUG, 'by_sza_region')}.png"
    fig.savefig(by_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {by_path}")

    # 8. Skipped chips report
    if chip_skips:
        skip_path = out_dir / f"{stamp(SLUG, 'skipped')}.csv"
        with open(skip_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["chip_stem", "reason"])
            writer.writeheader()
            writer.writerows(chip_skips)
        print(f"Skipped: {skip_path}")


if __name__ == "__main__":
    main()
