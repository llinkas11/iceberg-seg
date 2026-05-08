"""
q01_ic_cutoff_sweep.py: empirical answer to script-check question 1.

Question (from script-check-README.md, threshold_tifs.py):
  "A chip is skipped if more than 15% of pixels exceed 0.22. Is this the right
  operating point for our SZA range?"

What this script does:
  1. Glob every chip TIF under chips_root for every (region, sza_bin).
  2. For each chip compute ic_frac = (B08 >= 0.22).mean() in offset-uncorrected
     reflectance, mirroring the production check in threshold_tifs.py.
  3. Tabulate skip-rate at IC cutoffs in {0.10, 0.15, 0.20, 0.25, 0.30}.
  4. Emit a per-chip CSV and two PNGs: an ECDF overview and a per-(region, sza)
     skip-rate bar chart.

Inputs:
  --chips_root   Directory containing <region>/<sza_bin>/tifs/*.tif.
  --b08_idx      Band index of B08 in the chip TIF (default 2 to match the
                 production scripts).
  --threshold    Reflectance threshold whose pixel-fraction is being tested
                 (default 0.22 to match threshold_tifs.py).
  --out_root     Parent directory under which a slug folder is created.

Outputs (under <out_root>/q01_ic_cutoff_sweep/):
  <ts>__q01_ic_cutoff_sweep.csv               one row per chip
  <ts>__q01_ic_cutoff_sweep__overview.png     ECDF of ic_frac with cutoff lines
  <ts>__q01_ic_cutoff_sweep__by_sza_region.png  skip-rate bars per (region, sza)

Usage (Mac smoke test):
  python iceberg-rework/scripts/script_check_answers/q01_ic_cutoff_sweep.py

Deploy to moosehead:
  rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/script_check_answers/ \
    llinkas@moosehead.bowdoin.edu:~/iceberg-rework/scripts/script_check_answers/
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from _common import (
    SZA_BINS, REGIONS,
    list_chip_tifs, make_slug_dir, parse_region_sza,
    resolve_chips_root, resolve_out_root, stamp,
)


SLUG = "q01_ic_cutoff_sweep"
CUTOFFS = [0.10, 0.15, 0.20, 0.25, 0.30]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chips_root", default=str(resolve_chips_root()),
                   help="Root containing <region>/<sza_bin>/tifs/*.tif")
    p.add_argument("--b08_idx", type=int, default=2,
                   help="0-indexed band of B08 in the chip TIF")
    p.add_argument("--threshold", type=float, default=0.22,
                   help="Reflectance threshold whose pixel-fraction is tested")
    p.add_argument("--out_root", default=str(resolve_out_root()),
                   help="Parent directory; a slug subfolder is created under it")
    return p.parse_args()


def compute_ic_frac(tif_path, b08_idx, threshold):
    """Return the fraction of pixels in band b08_idx with value >= threshold."""
    with rasterio.open(tif_path) as src:
        if src.count <= b08_idx:
            return None
        b08 = src.read(b08_idx + 1).astype(np.float32)
    return float((b08 >= threshold).mean())


def main():
    args = parse_args()

    # 1. Resolve output dir and discover chips
    out_dir = make_slug_dir(SLUG, args.out_root)
    chip_rows = list_chip_tifs(args.chips_root)
    if not chip_rows:
        raise SystemExit(f"No chips found under {args.chips_root}")
    print(f"Found {len(chip_rows)} chips under {args.chips_root}")

    # 2. Compute ic_frac per chip
    rows = []
    skipped = 0
    for tif, region, sza_bin in chip_rows:
        frac = compute_ic_frac(tif, args.b08_idx, args.threshold)
        if frac is None:
            skipped += 1
            continue
        rows.append({
            "chip_stem": tif.stem,
            "region":    region,
            "sza_bin":   sza_bin,
            "ic_frac":   frac,
        })
    if skipped:
        print(f"Skipped {skipped} chips with too few bands")
    print(f"Computed ic_frac for {len(rows)} chips")

    # 3. Write per-chip CSV
    csv_path = out_dir / f"{stamp(SLUG)}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chip_stem", "region", "sza_bin", "ic_frac"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {csv_path}")

    # 4. Aggregate skip-rate at each cutoff and print summary
    fracs = np.array([r["ic_frac"] for r in rows])
    print(f"\nSkip-rate at each IC cutoff (threshold={args.threshold}):")
    for c in CUTOFFS:
        rate = float((fracs > c).mean())
        print(f"  ic_frac > {c:.2f}: {rate:.3%}  ({int((fracs > c).sum())}/{len(fracs)})")

    # 5. Overview ECDF
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_fracs = np.sort(fracs)
    ecdf = np.arange(1, len(sorted_fracs) + 1) / len(sorted_fracs)
    ax.plot(sorted_fracs, ecdf, color="#37474F", lw=1.5)
    for c in CUTOFFS:
        ax.axvline(c, color="#D32F2F" if c == 0.15 else "#90A4AE",
                   lw=1.0, ls="--", alpha=0.8)
        ax.text(c, 0.02, f" {c:.2f}", color="#D32F2F" if c == 0.15 else "#37474F",
                fontsize=8, va="bottom")
    ax.set_xlabel(f"ic_frac = mean(B08 >= {args.threshold}) per chip")
    ax.set_ylabel("ECDF")
    ax.set_title(f"Q1: IC chip-rejection cutoff sweep (n={len(fracs)} chips)")
    ax.set_xlim(0, max(1.0, sorted_fracs.max() * 1.05))
    ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.3, linestyle="--")
    overview_path = out_dir / f"{stamp(SLUG, 'overview')}.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {overview_path}")

    # 6. Per-(region, sza_bin) skip-rate bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.15
    group_labels = []
    for r in REGIONS:
        for s in SZA_BINS:
            group_labels.append(f"{r}\n{s}")
    x = np.arange(len(group_labels))

    for i, c in enumerate(CUTOFFS):
        bar_vals = []
        for r in REGIONS:
            for s in SZA_BINS:
                bin_fracs = np.array([row["ic_frac"] for row in rows
                                      if row["region"] == r and row["sza_bin"] == s])
                if len(bin_fracs) == 0:
                    bar_vals.append(0.0)
                else:
                    bar_vals.append(float((bin_fracs > c).mean()))
        ax.bar(x + (i - 2) * width, bar_vals, width=width,
               label=f"cutoff={c:.2f}",
               color=plt.cm.viridis(i / max(1, len(CUTOFFS) - 1)))

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("Skip-rate")
    ax.set_title("Q1: Skip-rate per (region, SZA bin) at each IC cutoff")
    ax.legend(fontsize=8, ncol=len(CUTOFFS))
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    by_path = out_dir / f"{stamp(SLUG, 'by_sza_region')}.png"
    fig.savefig(by_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {by_path}")


if __name__ == "__main__":
    main()
