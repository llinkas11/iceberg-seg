"""
q16_flat_prob_distribution.py: empirical answer to script-check question 16.

Question (from script-check-README.md, otsu_probs.py):
  "otsu_probs.py skips chips where P(iceberg).max() - .min() < 0.01 ... is
  0.01 a reasonable flatness floor on softmax space?"

What this script does:
  1. Index every *_probs.tif under --probs_root and pair each with the chip's
     region + sza_bin via the v4_clean manifest.
  2. For each chip compute range_p = P(iceberg).max() - P(iceberg).min().
  3. Tabulate would-skip rate at flat-prob cutoffs in {0.005, 0.01, 0.02, 0.05}.
  4. Emit a per-chip CSV and two PNGs: a log-x histogram with cutoff lines
     and a per-(region, SZA bin) skip-rate bar chart.

Inputs:
  --manifest, --probs_root, --out_root.

Outputs (under <out_root>/q16_flat_prob_distribution/):
  <ts>__q16_flat_prob_distribution.csv
  <ts>__q16_flat_prob_distribution__overview.png
  <ts>__q16_flat_prob_distribution__by_sza_region.png

Usage on moosehead:
  conda activate iceberg-unet
  python q16_flat_prob_distribution.py \
    --manifest /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/data/v4_clean/manifest.json \
    --probs_root /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/output/predictions_v4_clean

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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _method_common import load_manifest

from _common import (
    SZA_BINS, REGIONS,
    make_slug_dir, resolve_manifest_path, resolve_out_root, resolve_probs_root, stamp,
)


SLUG = "q16_flat_prob_distribution"
CUTOFFS = [0.005, 0.01, 0.02, 0.05]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", default=str(resolve_manifest_path()),
                   help="v4_clean manifest.json path")
    p.add_argument("--probs_root", default=str(resolve_probs_root()),
                   help="Root containing *_probs.tif (recursively searched)")
    p.add_argument("--out_root", default=str(resolve_out_root()),
                   help="Parent directory; a slug subfolder is created under it")
    return p.parse_args()


def index_probs(probs_root):
    """Scan probs_root recursively and return {chip_stem: Path}."""
    index = {}
    for p in Path(probs_root).rglob("*_probs.tif"):
        stem = p.name[:-len("_probs.tif")]
        index[stem] = p
    return index


def main():
    args = parse_args()

    # 1. Resolve output dir, load manifest, index probs
    out_dir = make_slug_dir(SLUG, args.out_root)
    manifest = load_manifest(args.manifest)
    chip_meta = {r["chip_stem"]: (r.get("region") or _region_from_stem(r["chip_stem"]),
                                   r["sza_bin"])
                 for r in manifest["chips"]}
    probs_idx = index_probs(args.probs_root)
    print(f"Manifest chips: {len(chip_meta)}")
    print(f"Probs found: {len(probs_idx)}")

    # 2. For each prob TIF, compute range_p and join region/sza_bin
    rows = []
    too_few_bands = 0
    no_meta = 0
    for stem, path in sorted(probs_idx.items()):
        if stem not in chip_meta:
            no_meta += 1
            continue
        with rasterio.open(path) as src:
            if src.count < 2:
                too_few_bands += 1
                continue
            p_iceberg = src.read(2).astype(np.float32)
        # Mask out the predict_tifs.py nodata sentinel of -1.0 if present.
        valid = p_iceberg >= 0.0
        if not valid.any():
            continue
        vals = p_iceberg[valid]
        range_p = float(vals.max() - vals.min())
        region, sza_bin = chip_meta[stem]
        rows.append({
            "chip_stem":            stem,
            "region":               region,
            "sza_bin":              sza_bin,
            "range_p":              range_p,
            "would_skip_at_0p005":  int(range_p < 0.005),
            "would_skip_at_0p01":   int(range_p < 0.01),
            "would_skip_at_0p02":   int(range_p < 0.02),
            "would_skip_at_0p05":   int(range_p < 0.05),
        })
    if too_few_bands:
        print(f"Skipped {too_few_bands} probs files with <2 bands")
    if no_meta:
        print(f"Skipped {no_meta} probs files not in manifest")
    print(f"Recorded range_p for {len(rows)} chips")

    # 3. Write per-chip CSV
    csv_path = out_dir / f"{stamp(SLUG)}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows
                                else ["chip_stem", "region", "sza_bin", "range_p"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {csv_path}")

    if not rows:
        print("No probs found; nothing to plot.")
        return

    # 4. Print summary
    ranges = np.array([r["range_p"] for r in rows], dtype=np.float64)
    print("\nWould-skip rate at each flat-prob cutoff:")
    for c in CUTOFFS:
        n = int((ranges < c).sum())
        print(f"  range_p < {c:.3f}: {n / len(ranges):.3%}  ({n}/{len(ranges)})")

    # 5. Overview log-x histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    safe_min = max(ranges[ranges > 0].min(), 1e-6) if (ranges > 0).any() else 1e-6
    bins = np.logspace(np.log10(safe_min * 0.5), 0, 50)
    ax.hist(np.clip(ranges, safe_min * 0.5, None), bins=bins,
            color="#37474F", alpha=0.85)
    ax.set_xscale("log")
    for c in CUTOFFS:
        ax.axvline(c, color="#D32F2F" if c == 0.01 else "#90A4AE",
                   lw=1.0, ls="--", alpha=0.9)
        ax.text(c, ax.get_ylim()[1] * 0.95, f" {c:.3f}",
                color="#D32F2F" if c == 0.01 else "#37474F",
                fontsize=8, va="top")
    ax.set_xlabel("range_p = P(iceberg).max() - P(iceberg).min() per chip")
    ax.set_ylabel("Chip count")
    ax.set_title(f"Q16: per-chip P(iceberg) range distribution (n={len(ranges)})")
    ax.grid(True, alpha=0.3, linestyle="--")
    overview_path = out_dir / f"{stamp(SLUG, 'overview')}.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {overview_path}")

    # 6. Per-(region, sza_bin) skip-rate bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.18
    group_labels = [f"{r}\n{s}" for r in REGIONS for s in SZA_BINS]
    x = np.arange(len(group_labels))
    for i, c in enumerate(CUTOFFS):
        bar_vals = []
        for r in REGIONS:
            for s in SZA_BINS:
                bin_ranges = np.array([row["range_p"] for row in rows
                                        if row["region"] == r and row["sza_bin"] == s])
                bar_vals.append(float((bin_ranges < c).mean())
                                if len(bin_ranges) else 0.0)
        ax.bar(x + (i - 1.5) * width, bar_vals, width=width,
               label=f"cutoff={c:.3f}",
               color=plt.cm.viridis(i / max(1, len(CUTOFFS) - 1)))
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("Would-skip rate")
    ax.set_title("Q16: would-skip rate per (region, SZA bin) at each flat-prob cutoff")
    ax.legend(fontsize=8, ncol=len(CUTOFFS))
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    by_path = out_dir / f"{stamp(SLUG, 'by_sza_region')}.png"
    fig.savefig(by_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {by_path}")


def _region_from_stem(stem):
    """Heuristic region inference from a Sentinel-2 chip stem (MGRS tile)."""
    if "T24WVU" in stem or "T24WWU" in stem:
        return "SK"
    return "KQ"


if __name__ == "__main__":
    main()
