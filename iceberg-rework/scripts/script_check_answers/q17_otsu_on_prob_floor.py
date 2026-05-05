"""
q17_otsu_on_prob_floor.py: empirical answer to script-check question 17.

Question (from script-check-README.md, otsu_probs.py):
  "OT on B08 has a floor at 0.10 reflectance; OT on probability does not.
  Should OT on probability also have a floor (e.g. 0.5) to avoid carving
  icebergs out of low-confidence ocean?"

What this script does:
  1. Index every *_probs.tif under --probs_root and pair each with the chip's
     region + sza_bin via the v4_clean manifest.
  2. For each chip compute threshold_otsu(P(iceberg)) and the resulting
     iceberg pixel count at the bare Otsu threshold and at clipped
     thresholds max(otsu, floor) for floor in {0.3, 0.5, 0.7}.
  3. Report fraction of chips whose Otsu threshold falls below each floor and
     the per-(region, SZA bin) reduction in iceberg pixels under a 0.5 floor.
  4. Emit a per-chip CSV and two PNGs: a histogram of per-chip Otsu thresholds
     with floor lines and a per-(region, SZA bin) ice-pixel-delta bar chart.

Inputs:
  --manifest, --probs_root, --out_root.

Outputs (under <out_root>/q17_otsu_on_prob_floor/):
  <ts>__q17_otsu_on_prob_floor.csv
  <ts>__q17_otsu_on_prob_floor__overview.png
  <ts>__q17_otsu_on_prob_floor__by_sza_region.png

Usage on moosehead:
  conda activate iceberg-unet
  python q17_otsu_on_prob_floor.py \
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
from skimage.filters import threshold_otsu

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _method_common import load_manifest

from _common import (
    SZA_BINS, REGIONS,
    make_slug_dir, resolve_manifest_path, resolve_out_root, resolve_probs_root, stamp,
)


SLUG = "q17_otsu_on_prob_floor"
FLOORS = [0.3, 0.5, 0.7]


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


def _region_from_stem(stem):
    """Heuristic region inference from a Sentinel-2 chip stem (MGRS tile)."""
    if "T24WVU" in stem or "T24WWU" in stem:
        return "SK"
    return "KQ"


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

    # 2. Per-chip Otsu and pixel counts at each candidate floor
    rows = []
    too_few_bands = 0
    flat = 0
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
        valid = p_iceberg >= 0.0
        if not valid.any():
            continue
        vals = p_iceberg[valid]
        if (vals.max() - vals.min()) < 0.01:
            flat += 1
            continue
        otsu_t = float(threshold_otsu(vals))
        ice_px_otsu = int((p_iceberg >= otsu_t).sum())
        ice_px_by_floor = {}
        for f_floor in FLOORS:
            t_clipped = max(otsu_t, f_floor)
            ice_px_by_floor[f_floor] = int((p_iceberg >= t_clipped).sum())

        region, sza_bin = chip_meta[stem]
        rows.append({
            "chip_stem":          stem,
            "region":             region,
            "sza_bin":            sza_bin,
            "otsu_thresh":        otsu_t,
            "ice_px_otsu":        ice_px_otsu,
            "ice_px_floor_0p3":   ice_px_by_floor[0.3],
            "ice_px_floor_0p5":   ice_px_by_floor[0.5],
            "ice_px_floor_0p7":   ice_px_by_floor[0.7],
        })
    if too_few_bands:
        print(f"Skipped {too_few_bands} probs files with <2 bands")
    if flat:
        print(f"Skipped {flat} probs files failing the production flat-prob guard")
    if no_meta:
        print(f"Skipped {no_meta} probs files not in manifest")
    print(f"Recorded Otsu for {len(rows)} chips")

    # 3. Write per-chip CSV
    csv_path = out_dir / f"{stamp(SLUG)}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows
                                else ["chip_stem", "region", "sza_bin", "otsu_thresh"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {csv_path}")

    if not rows:
        print("No probs found; nothing to plot.")
        return

    # 4. Print summary
    threshes = np.array([r["otsu_thresh"] for r in rows], dtype=np.float64)
    print("\nFraction of chips with otsu < floor (clipping would activate):")
    for f_floor in FLOORS:
        n = int((threshes < f_floor).sum())
        print(f"  otsu < {f_floor:.1f}: {n / len(threshes):.3%}  ({n}/{len(threshes)})")

    print("\nTotal iceberg-pixel reduction under each floor (vs bare Otsu):")
    total_otsu = sum(r["ice_px_otsu"] for r in rows)
    for f_floor in FLOORS:
        col = f"ice_px_floor_0p{int(f_floor * 10)}"
        total_floor = sum(r[col] for r in rows)
        delta = total_otsu - total_floor
        print(f"  floor={f_floor:.1f}: kept {total_floor}/{total_otsu} px "
              f"(removed {delta}, {delta / max(1, total_otsu):.2%})")

    # 5. Overview histogram of Otsu thresholds with floor lines
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(threshes, bins=np.linspace(0, 1, 50), color="#37474F", alpha=0.85)
    for f_floor in FLOORS:
        ax.axvline(f_floor, color="#D32F2F" if f_floor == 0.5 else "#90A4AE",
                   lw=1.0, ls="--", alpha=0.9)
        n = int((threshes < f_floor).sum())
        ax.text(f_floor, ax.get_ylim()[1] * 0.95,
                f" {f_floor:.1f}\n n<= {n}",
                color="#D32F2F" if f_floor == 0.5 else "#37474F",
                fontsize=8, va="top")
    ax.set_xlabel("per-chip Otsu threshold on P(iceberg)")
    ax.set_ylabel("Chip count")
    ax.set_title(f"Q17: per-chip Otsu on P(iceberg) (n={len(threshes)})")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, linestyle="--")
    overview_path = out_dir / f"{stamp(SLUG, 'overview')}.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {overview_path}")

    # 6. Per-(region, sza_bin) iceberg-pixel reduction under floor=0.5
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.35
    group_labels = [f"{r}\n{s}" for r in REGIONS for s in SZA_BINS]
    x = np.arange(len(group_labels))
    bars_otsu = []
    bars_floor = []
    for r in REGIONS:
        for s in SZA_BINS:
            bin_rows = [row for row in rows
                        if row["region"] == r and row["sza_bin"] == s]
            bars_otsu.append(sum(row["ice_px_otsu"] for row in bin_rows))
            bars_floor.append(sum(row["ice_px_floor_0p5"] for row in bin_rows))
    ax.bar(x - width / 2, bars_otsu, width=width, color="#90A4AE",
           label="bare Otsu")
    ax.bar(x + width / 2, bars_floor, width=width, color="#D32F2F",
           label="max(Otsu, 0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("Total iceberg pixels (sum across chips)")
    ax.set_title("Q17: iceberg-pixel total per (region, SZA bin), bare Otsu vs 0.5 floor")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    by_path = out_dir / f"{stamp(SLUG, 'by_sza_region')}.png"
    fig.savefig(by_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {by_path}")


if __name__ == "__main__":
    main()
