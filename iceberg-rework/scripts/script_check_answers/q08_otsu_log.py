"""
q08_otsu_log.py: empirical answer to script-check question 8.

Question (from script-check-README.md, otsu_threshold_tifs.py):
  "Otsu on raw B08. No log-transform, no contrast stretch, no exclusion of
  saturated pixels. Otsu finds the threshold maximising inter-class variance
  on the chip's 256 x 256 pixel histogram. For ocean-dominated chips the
  histogram is highly skewed; does Otsu still give a sensible threshold, or
  should we be doing a log-transform first?"

What this script does:
  1. Glob every chip TIF.
  2. For each chip compute per-chip Otsu on raw B08 and on log(B08+eps)
     (log-transform the histogram before Otsu, then map the threshold back to
     reflectance space via expm1).
  3. For both thresholds, count post-threshold iceberg-pixel fraction so the
     reviewer can see whether the log-transform shifts how many pixels fall
     into the iceberg class for the same chip.
  4. Emit a per-chip CSV plus two PNGs: a scatter of raw-Otsu vs log-Otsu
     thresholds and a per-(region, SZA bin) bar chart of mean iceberg-pixel
     fraction under each variant.

Inputs:
  --chips_root   Directory containing <region>/<sza_bin>/tifs/*.tif.
  --b08_idx      Band index of B08 in the chip TIF (default 2).
  --eps          Floor added before log to avoid log(0) (default 1e-3).
  --out_root     Parent directory under which a slug folder is created.

Outputs (under <out_root>/q08_otsu_log/):
  <ts>__q08_otsu_log.csv               one row per chip
  <ts>__q08_otsu_log__overview.png     raw vs log Otsu scatter
  <ts>__q08_otsu_log__by_sza_region.png mean iceberg-fraction per bin
"""

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from skimage.filters import threshold_otsu

from _common import (
    SZA_BINS, REGIONS,
    list_chip_tifs, make_slug_dir, resolve_chips_root, resolve_out_root, stamp,
)


SLUG = "q08_otsu_log"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chips_root", default=str(resolve_chips_root()),
                   help="Root containing <region>/<sza_bin>/tifs/*.tif")
    p.add_argument("--b08_idx", type=int, default=2,
                   help="0-indexed band of B08 in the chip TIF")
    p.add_argument("--eps", type=float, default=1e-3,
                   help="Floor added before log to avoid log(0)")
    p.add_argument("--out_root", default=str(resolve_out_root()),
                   help="Parent directory; a slug subfolder is created under it")
    return p.parse_args()


def compute_per_chip(tif_path, b08_idx, eps):
    """
    Return a dict with raw-Otsu and log-Otsu thresholds (in reflectance space)
    plus the iceberg-pixel fraction under each. None if too few bands.
    """
    with rasterio.open(tif_path) as src:
        if src.count <= b08_idx:
            return None
        b08 = src.read(b08_idx + 1).astype(np.float32)

    flat = b08.ravel()
    if np.unique(flat).size < 2:
        return None

    # 1. Raw Otsu on B08
    raw_thresh = float(threshold_otsu(flat))

    # 2. Log-Otsu: log1p(B08) -> Otsu -> expm1 to map back to reflectance space.
    #    Add eps for numerical safety even though log1p handles 0; expm1 keeps
    #    the threshold comparable to the raw-Otsu value.
    log_b08 = np.log1p(np.maximum(flat, 0.0) + eps)
    if np.unique(log_b08).size < 2:
        return None
    log_thresh_log_space = float(threshold_otsu(log_b08))
    log_thresh = float(np.expm1(log_thresh_log_space) - eps)

    # 3. Iceberg-pixel fraction under each
    raw_frac = float((flat >= raw_thresh).mean())
    log_frac = float((flat >= log_thresh).mean())

    return {
        "raw_otsu":   raw_thresh,
        "log_otsu":   log_thresh,
        "raw_frac":   raw_frac,
        "log_frac":   log_frac,
    }


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    # 1. Resolve output dir and discover chips
    out_dir = make_slug_dir(SLUG, args.out_root)
    chip_rows = list_chip_tifs(args.chips_root)
    if not chip_rows:
        raise SystemExit(f"No chips found under {args.chips_root}")
    print(f"Found {len(chip_rows)} chips under {args.chips_root}")

    # 2. Per-chip raw + log Otsu
    rows = []
    skipped = 0
    for tif, region, sza_bin in chip_rows:
        res = compute_per_chip(tif, args.b08_idx, args.eps)
        if res is None:
            skipped += 1
            continue
        rows.append({
            "chip_stem":         tif.stem,
            "region":            region,
            "sza_bin":           sza_bin,
            "raw_otsu":          res["raw_otsu"],
            "log_otsu":          res["log_otsu"],
            "raw_iceberg_frac":  res["raw_frac"],
            "log_iceberg_frac":  res["log_frac"],
            "delta_thresh":      res["log_otsu"] - res["raw_otsu"],
            "delta_iceberg_frac": res["log_frac"] - res["raw_frac"],
        })

    if skipped:
        print(f"Skipped {skipped} chips (too few bands or constant pixels)")
    print(f"Computed Otsu pair on {len(rows)} chips")

    if not rows:
        raise SystemExit("No chips evaluated; nothing to plot.")

    # 3. Write per-chip CSV
    fieldnames = list(rows[0].keys())
    csv_path = out_dir / f"{stamp(SLUG)}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {csv_path}")

    # 4. Headline aggregates
    raw_thr = np.array([r["raw_otsu"] for r in rows], dtype=np.float64)
    log_thr = np.array([r["log_otsu"] for r in rows], dtype=np.float64)
    raw_frac = np.array([r["raw_iceberg_frac"] for r in rows], dtype=np.float64)
    log_frac = np.array([r["log_iceberg_frac"] for r in rows], dtype=np.float64)

    print(f"\nMean raw Otsu threshold:  {raw_thr.mean():.4f}  median {np.median(raw_thr):.4f}")
    print(f"Mean log Otsu threshold:  {log_thr.mean():.4f}  median {np.median(log_thr):.4f}")
    print(f"Mean delta (log - raw):   {(log_thr - raw_thr).mean():+.4f}")
    print(f"Mean iceberg-frac (raw):  {raw_frac.mean():.4%}")
    print(f"Mean iceberg-frac (log):  {log_frac.mean():.4%}")
    print(f"Mean delta iceberg-frac:  {(log_frac - raw_frac).mean():+.4%}")

    # 5. Overview scatter: raw vs log threshold per chip
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(raw_thr, log_thr, s=10, alpha=0.35, color="#37474F", edgecolor="none")
    lim = float(max(raw_thr.max(), log_thr.max(), 0.6))
    ax.plot([0, lim], [0, lim], color="#1976D2", lw=1.0, ls="--", alpha=0.8,
            label="y = x")
    ax.set_xlabel("Raw-Otsu threshold on B08")
    ax.set_ylabel("Log-Otsu threshold on B08 (mapped via expm1)")
    ax.set_title(f"Q8: raw vs log Otsu threshold per chip (n={len(rows)})")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")
    overview_path = out_dir / f"{stamp(SLUG, 'overview')}.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {overview_path}")

    # 6. Per-(region, SZA bin) mean iceberg-pixel fraction under each variant
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.35
    group_labels = []
    raw_vals = []
    log_vals = []
    for region in REGIONS:
        for sza_bin in SZA_BINS:
            sub = [row for row in rows if row["region"] == region and row["sza_bin"] == sza_bin]
            group_labels.append(f"{region}\n{sza_bin}\n(n={len(sub)})")
            if not sub:
                raw_vals.append(0.0); log_vals.append(0.0); continue
            raw_vals.append(float(np.mean([row["raw_iceberg_frac"] for row in sub])))
            log_vals.append(float(np.mean([row["log_iceberg_frac"] for row in sub])))
    x = np.arange(len(group_labels))
    ax.bar(x - width / 2, raw_vals, width=width, color="#37474F",
           label="raw Otsu (production)")
    ax.bar(x + width / 2, log_vals, width=width, color="#1976D2",
           label="log Otsu (proposed)")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("Mean iceberg-pixel fraction per chip")
    ax.set_title("Q8: mean iceberg-pixel fraction per (region, SZA bin) under raw vs log Otsu")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    by_path = out_dir / f"{stamp(SLUG, 'by_sza_region')}.png"
    fig.savefig(by_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {by_path}")


if __name__ == "__main__":
    main()
