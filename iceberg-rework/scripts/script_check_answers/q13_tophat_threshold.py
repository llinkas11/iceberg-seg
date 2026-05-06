"""
q13_tophat_threshold.py: empirical answer to script-check question 13.

Question (from script-check-README.md, tophat_recover.py):
  "Top-hat threshold. th_thresh = 0.05 reflectance units. Honestly chosen by
  eye on a few chips. If there is a more principled estimator (e.g.,
  chip-wise sigma of the top-hat response, or an Otsu of the response), I
  would adopt it."

What this script does:
  1. Iterate the v4_clean test_chips set.
  2. For each chip compute the top-hat response at the production disk(10),
     then derive three candidate thresholds: chip-wise sigma, 3*sigma, and
     an Otsu of the response (skipping chips with constant responses).
  3. Compare against the production fixed cutoff 0.05: report the per-chip
     threshold distribution, plus the resulting recovered-polygon counts at
     the per-chip Otsu threshold and at sigma-based thresholds.
  4. Emit a per-chip CSV plus two PNGs: a histogram of derived thresholds
     and a per-(SZA bin) bar chart of recovered counts at production vs
     Otsu thresholds.

Inputs:
  --chips_root, --base_root, --se_radius, --min_area_px, --out_root.

Outputs (under <out_root>/q13_tophat_threshold/):
  <ts>__q13_tophat_threshold.csv               one row per chip
  <ts>__q13_tophat_threshold__overview.png     threshold histogram
  <ts>__q13_tophat_threshold__by_sza_region.png  per-bin recovered counts
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
from skimage.measure import label
from skimage.morphology import disk, white_tophat

from _common import (
    SZA_BINS,
    make_slug_dir, resolve_out_root, stamp,
)


SLUG = "q13_tophat_threshold"
PIXEL_AREA_M2 = 100.0


def _hpc_v4_test_chips_root():
    candidates = [
        Path("/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/data/v4_clean/test_chips"),
        Path("/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/data/v4_clean/test_chips"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _hpc_base_root():
    candidates = [
        Path("/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/results/model_comparison_20260423_stage1_vs_baseline/area_comparison/baseline_v3_balanced_aug/test"),
        Path("/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/results/model_comparison_20260423_stage1_vs_baseline/area_comparison/baseline_v3_balanced_aug/test"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chips_root", default=str(_hpc_v4_test_chips_root()))
    p.add_argument("--base_root", default=str(_hpc_base_root()))
    p.add_argument("--se_radius", type=int, default=10,
                   help="Production disk radius (default 10 = 100 m)")
    p.add_argument("--min_area_px", type=int, default=16,
                   help="Min component size in pixels (40 m root length)")
    p.add_argument("--prod_thresh", type=float, default=0.05,
                   help="Production fixed top-hat threshold")
    p.add_argument("--out_root", default=str(resolve_out_root()),
                   help="Parent directory; a slug subfolder is created under it")
    return p.parse_args()


def list_v4_test_chips(chips_root):
    out = []
    for sza in SZA_BINS:
        d = Path(chips_root) / sza
        if not d.is_dir():
            continue
        for tif in sorted(d.glob("*.tif")):
            out.append((tif, sza, tif.stem))
    return out


def read_b08(tif_path):
    with rasterio.open(tif_path) as src:
        if src.count < 3:
            return None
        return src.read(3).astype(np.float32)


def read_base_mask(base_root, sza_bin, stem, ref_shape):
    p = Path(base_root) / sza_bin / "UNet" / "geotiffs" / f"{stem}_pred.tif"
    if p.exists():
        with rasterio.open(p) as src:
            return (src.read(1) > 0).astype(np.uint8)
    return None


def count_recovered(response, base_mask, thresh, min_area_px):
    if not np.isfinite(thresh):
        return 0, 0.0
    candidate = (response >= thresh).astype(np.uint8)
    if base_mask is not None:
        candidate = candidate & (base_mask == 0).astype(np.uint8)
    labels = label(candidate, connectivity=2)
    if labels.max() == 0:
        return 0, 0.0
    n = 0
    total_area = 0.0
    for li in range(1, labels.max() + 1):
        n_px = int((labels == li).sum())
        if n_px >= min_area_px:
            n += 1
            total_area += n_px * PIXEL_AREA_M2
    return n, total_area


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    out_dir = make_slug_dir(SLUG, args.out_root)
    chips = list_v4_test_chips(args.chips_root)
    print(f"Found {len(chips)} v4_clean test chips")
    print(f"se_radius: {args.se_radius}  prod_thresh: {args.prod_thresh}")

    rows = []
    n_skipped = 0
    for tif, sza, stem in chips:
        b08 = read_b08(tif)
        if b08 is None:
            n_skipped += 1
            continue
        base_mask = read_base_mask(args.base_root, sza, stem, b08.shape)
        response = white_tophat(b08, disk(args.se_radius))

        # Derived thresholds: chip-wise sigma, 3*sigma, Otsu of response.
        flat = response.ravel()
        sigma = float(flat.std())
        try:
            otsu_thresh = float(threshold_otsu(flat)) if np.unique(flat).size > 1 else float("nan")
        except Exception:
            otsu_thresh = float("nan")

        # Recovered counts at each candidate threshold.
        n_prod, area_prod = count_recovered(response, base_mask, args.prod_thresh, args.min_area_px)
        n_otsu, area_otsu = count_recovered(response, base_mask, otsu_thresh, args.min_area_px)
        n_3sig, area_3sig = count_recovered(response, base_mask, 3 * sigma, args.min_area_px)

        rows.append({
            "chip_stem":     stem,
            "sza_bin":       sza,
            "sigma":         sigma,
            "thresh_3sigma": 3 * sigma,
            "thresh_otsu":   otsu_thresh,
            "n_recovered_prod":  n_prod,
            "area_recovered_prod_m2": area_prod,
            "n_recovered_otsu":  n_otsu,
            "area_recovered_otsu_m2": area_otsu,
            "n_recovered_3sigma": n_3sig,
            "area_recovered_3sigma_m2": area_3sig,
            "has_base_mask": base_mask is not None,
        })

    if n_skipped:
        print(f"Skipped {n_skipped} chips (too few bands)")
    print(f"Evaluated {len(rows)} chips")

    if not rows:
        raise SystemExit("No rows; nothing to plot.")

    csv_path = out_dir / f"{stamp(SLUG)}.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {csv_path}")

    # 1. Headline distribution stats
    sigma_vals = np.array([r["sigma"] for r in rows], dtype=np.float64)
    otsu_vals  = np.array([r["thresh_otsu"] for r in rows], dtype=np.float64)
    valid_otsu = otsu_vals[np.isfinite(otsu_vals)]

    n_prod_total = sum(r["n_recovered_prod"] for r in rows)
    n_otsu_total = sum(r["n_recovered_otsu"] for r in rows)
    n_3sig_total = sum(r["n_recovered_3sigma"] for r in rows)
    print(f"\nMean chip-wise sigma:   {sigma_vals.mean():.5f}")
    print(f"Mean 3*sigma threshold: {(3 * sigma_vals).mean():.5f}")
    print(f"Mean Otsu threshold:    {valid_otsu.mean():.5f}  (median {np.median(valid_otsu):.5f})")
    print(f"Production fixed thresh: {args.prod_thresh:.4f}")
    print(f"\nRecovered polygon totals across {len(rows)} chips:")
    print(f"  production (0.05): {n_prod_total}")
    print(f"  per-chip Otsu:     {n_otsu_total}")
    print(f"  per-chip 3*sigma:  {n_3sig_total}")

    # 2. Overview: histogram of derived thresholds
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bins = np.linspace(0.0, max(0.2, valid_otsu.max() if valid_otsu.size else 0.2), 50)
    if valid_otsu.size:
        ax.hist(valid_otsu, bins=bins, color="#1976D2", alpha=0.6,
                label=f"Otsu of response (n={valid_otsu.size})")
    ax.hist(3 * sigma_vals, bins=bins, color="#388E3C", alpha=0.6,
            label=f"3 x sigma  (n={sigma_vals.size})")
    ax.axvline(args.prod_thresh, color="#D32F2F", lw=1.2, ls="--")
    ax.text(args.prod_thresh, ax.get_ylim()[1] * 0.95,
            f"  prod = {args.prod_thresh:.2f}", color="#D32F2F", fontsize=8, va="top")
    ax.set_xlabel("Top-hat threshold (reflectance units)")
    ax.set_ylabel("Chip count")
    ax.set_title(f"Q13: derived top-hat thresholds vs production "
                 f"(SE = disk({args.se_radius}))")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    overview_path = out_dir / f"{stamp(SLUG, 'overview')}.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {overview_path}")

    # 3. Per-(SZA bin) bars: recovered counts under each rule
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.27
    rule_keys = [("n_recovered_prod", "production 0.05", "#37474F"),
                 ("n_recovered_otsu", "per-chip Otsu", "#1976D2"),
                 ("n_recovered_3sigma", "per-chip 3 x sigma", "#388E3C")]
    for i, (k, lbl, color) in enumerate(rule_keys):
        vals = []
        for sza in SZA_BINS:
            sub = [r for r in rows if r["sza_bin"] == sza]
            vals.append(sum(r[k] for r in sub))
        x = np.arange(len(SZA_BINS))
        ax.bar(x + (i - 1) * width, vals, width=width, color=color, label=lbl)
    ax.set_xticks(np.arange(len(SZA_BINS)))
    ax.set_xticklabels(SZA_BINS, fontsize=9)
    ax.set_ylabel("Recovered polygon count (sum across test chips)")
    ax.set_title("Q13: recovered counts per SZA bin under candidate top-hat thresholds")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    by_path = out_dir / f"{stamp(SLUG, 'by_sza_region')}.png"
    fig.savefig(by_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {by_path}")


if __name__ == "__main__":
    main()
