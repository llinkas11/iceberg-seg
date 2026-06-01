"""
q04_ndwi_sweep.py: empirical answer to script-check question 4.

Question (from script-check-README.md, threshold_masked_tifs.py):
  "NDWI threshold = 0 (McFeeters 1996 default). NDWI = (B03 - B08) / (B03 +
  B08 + epsilon); pixels with NDWI > 0 are classified as open water. Fisser
  and others (2024) do not use NDWI; this branch is a chip-level land-edge
  safeguard added by this project. We keep NDWI > 0."

What this script does (audit + extension):
  1. Read iceberg-rework/sweeps/ndwi_threshold_sweep.csv if present and check
     whether it covers all (region, sza_bin) combos at the canonical NDWI
     thresholds {-0.05, 0.0, 0.05, 0.1, 0.2}.
  2. If any (region, sza_bin, threshold) combo is missing, run
     threshold_masked_tifs._apply_thresholds on the missing chips at the
     missing thresholds and append the new rows to the existing CSV (without
     touching the prior rows; reproducibility is preserved by the per-chip
     IDs).
  3. Emit a per-(region, sza_bin, threshold) summary CSV (km^2 per cell) and
     two PNGs: a per-region multi-panel km^2-vs-threshold curve and a
     per-(region, SZA bin) bar chart at threshold 0.0 (production) vs 0.2.

Inputs:
  --sweep_csv     Existing per-chip sweep CSV (default: iceberg-rework/sweeps/...).
  --chips_root    Chip root for any missing combos (default: shared chips root).
  --b03_idx       0-indexed B03 band (default 1).
  --b08_idx       0-indexed B08 band (default 2).
  --thresholds    Comma list of NDWI thresholds to ensure coverage on
                  (default -0.05,0.0,0.05,0.1,0.2).
  --out_root      Parent directory; a slug subfolder is created under it.

Outputs (under <out_root>/q04_ndwi_sweep/):
  <ts>__q04_ndwi_sweep.csv                     summary per (region, sza_bin, ndwi)
  <ts>__q04_ndwi_sweep__overview.png           km^2 vs ndwi per (region, SZA bin)
  <ts>__q04_ndwi_sweep__by_sza_region.png      per-bin bars at 0.0 vs 0.2
"""

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

# Make the iceberg-rework/scripts/ directory importable so we can reuse
# threshold_masked_tifs._apply_thresholds (the production NDWI-mask path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from threshold_masked_tifs import (  # noqa: E402
    _apply_thresholds,
    NIR_THRESHOLD,
    MIN_AREA_M2,
    IC_THRESHOLD,
)

from _common import (
    SZA_BINS, REGIONS,
    list_chip_tifs, make_slug_dir, resolve_chips_root, resolve_out_root, stamp,
)


SLUG = "q04_ndwi_sweep"
DEFAULT_THRESHOLDS = "-0.05,0.0,0.05,0.1,0.2"


def _hpc_sweep_default():
    """Best-effort default for the existing per-chip NDWI sweep CSV."""
    candidates = [
        Path("/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/sweeps/ndwi_threshold_sweep.csv"),
        Path("/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/sweeps/ndwi_threshold_sweep.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sweep_csv", default=str(_hpc_sweep_default()),
                   help="Existing per-chip NDWI sweep CSV (extended in place if needed)")
    p.add_argument("--chips_root", default=str(resolve_chips_root()),
                   help="Root containing <region>/<sza_bin>/tifs/*.tif")
    p.add_argument("--b03_idx", type=int, default=1, help="0-indexed B03 band")
    p.add_argument("--b08_idx", type=int, default=2, help="0-indexed B08 band")
    p.add_argument("--thresholds", default=DEFAULT_THRESHOLDS,
                   help="Comma list of NDWI thresholds to ensure coverage on")
    p.add_argument("--out_root", default=str(resolve_out_root()),
                   help="Parent directory; a slug subfolder is created under it")
    return p.parse_args()


def load_existing(sweep_csv):
    """Read sweep CSV, return list of dicts and a coverage set of (region,sza_bin,thr)."""
    if not Path(sweep_csv).exists():
        return [], set()
    rows = []
    coverage = set()
    with open(sweep_csv) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
            try:
                thr = round(float(r["ndwi_threshold"]), 4)
            except (KeyError, ValueError):
                continue
            coverage.add((r.get("region"), r.get("sza_bin"), thr))
    return rows, coverage


def append_rows(sweep_csv, header, new_rows):
    """Append new rows to the sweep CSV (keeping the existing header)."""
    if not new_rows:
        return
    with open(sweep_csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header,
                                extrasaction="ignore")
        for row in new_rows:
            writer.writerow(row)


def evaluate_chip(tif_path, b03_idx, b08_idx, thresholds):
    """Run _apply_thresholds at every NDWI threshold; return one row per threshold."""
    out_rows = []
    with rasterio.open(tif_path) as src:
        if src.count <= max(b03_idx, b08_idx):
            for thr in thresholds:
                out_rows.append({
                    "chip_stem":      Path(tif_path).stem,
                    "ndwi_threshold": thr,
                    "band_skipped":   True,
                    "ic_skipped":     False,
                    "ic_frac":        "",
                    "water_px":       0,
                    "iceberg_px":     0,
                    "n_polygons":     0,
                    "total_area_m2":  0.0,
                })
            return out_rows
        b03 = src.read(b03_idx + 1).astype(np.float32)
        b08 = src.read(b08_idx + 1).astype(np.float32)
        transform = src.transform
        source_name = os.path.basename(tif_path)

    for thr in thresholds:
        res = _apply_thresholds(
            b03, b08, transform, source_name,
            nir_threshold=NIR_THRESHOLD,
            ndwi_threshold=thr,
            min_area_m2=MIN_AREA_M2,
            ic_threshold=IC_THRESHOLD,
        )
        out_rows.append({
            "chip_stem":      Path(tif_path).stem,
            "ndwi_threshold": thr,
            "band_skipped":   False,
            "ic_skipped":     bool(res["ic_skipped"]),
            "ic_frac":        f"{res['ic_frac']:.4f}",
            "water_px":       int(res["water_px"]),
            "iceberg_px":     int(res["iceberg_px"]),
            "n_polygons":     int(res["n_polygons"]),
            "total_area_m2":  float(res["total_area_m2"]),
        })
    return out_rows


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    # 1. Resolve output dir
    out_dir = make_slug_dir(SLUG, args.out_root)
    thresholds = [round(float(t), 4) for t in args.thresholds.split(",")]
    print(f"NDWI thresholds in scope: {thresholds}")
    print(f"Sweep CSV: {args.sweep_csv}")

    # 2. Load existing rows
    existing_rows, coverage = load_existing(args.sweep_csv)
    print(f"Existing per-chip rows: {len(existing_rows)} "
          f"(unique (region, sza_bin, thr) combos: {len(coverage)})")

    # 3. Audit coverage and identify gaps
    chip_rows = list_chip_tifs(args.chips_root)
    chip_by_id = {(r, s, t.stem): t for (t, r, s) in chip_rows}
    missing_chip_thr = []  # list of (chip_path, region, sza_bin, [missing_thrs])
    chip_seen = {}
    for region, sza_bin, _ in {(r, s, "") for (_, r, s) in chip_rows}:
        chip_seen.setdefault((region, sza_bin), 0)

    # Build per-(region,sza_bin,stem) seen-thresholds map from existing rows.
    existing_per_chip = {}
    for r in existing_rows:
        try:
            thr = round(float(r["ndwi_threshold"]), 4)
        except (KeyError, ValueError):
            continue
        key = (r.get("region"), r.get("sza_bin"), r.get("chip_stem"))
        existing_per_chip.setdefault(key, set()).add(thr)

    for tif, region, sza_bin in chip_rows:
        key = (region, sza_bin, tif.stem)
        seen = existing_per_chip.get(key, set())
        missing = [t for t in thresholds if t not in seen]
        if missing:
            missing_chip_thr.append((tif, region, sza_bin, missing))

    print(f"Chips with at least one missing threshold: {len(missing_chip_thr)}")

    # 4. Run missing combos and append to the sweep CSV
    if missing_chip_thr:
        # Determine header from existing CSV or fall back to canonical.
        if Path(args.sweep_csv).exists():
            with open(args.sweep_csv) as fh:
                header = next(csv.reader(fh))
        else:
            header = [
                "region", "sza_bin", "chip_stem", "ndwi_threshold",
                "band_skipped", "ic_skipped", "ic_frac",
                "water_px", "iceberg_px", "n_polygons", "total_area_m2",
            ]
            with open(args.sweep_csv, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)

        new_rows = []
        for tif, region, sza_bin, missing in missing_chip_thr:
            chip_rows_out = evaluate_chip(tif, args.b03_idx, args.b08_idx, missing)
            for row in chip_rows_out:
                row["region"] = region
                row["sza_bin"] = sza_bin
                new_rows.append(row)
        print(f"Appending {len(new_rows)} new (chip, threshold) rows to {args.sweep_csv}")
        append_rows(args.sweep_csv, header, new_rows)

    # 5. Re-load (now-complete) sweep CSV and aggregate per (region, sza_bin, thr)
    rows, _ = load_existing(args.sweep_csv)
    print(f"Total per-chip rows in sweep CSV after audit: {len(rows)}")

    summary = {}  # {(region, sza_bin, thr): {"area_m2", "n_chips", "n_polys"}}
    for r in rows:
        try:
            thr = round(float(r["ndwi_threshold"]), 4)
        except (KeyError, ValueError):
            continue
        if thr not in thresholds:
            continue
        region = r.get("region")
        sza_bin = r.get("sza_bin")
        key = (region, sza_bin, thr)
        s = summary.setdefault(key, {"area_m2": 0.0, "n_chips": 0, "n_polys": 0,
                                     "ic_skipped": 0, "band_skipped": 0})
        s["n_chips"] += 1
        if str(r.get("band_skipped")).lower() in ("true", "1"):
            s["band_skipped"] += 1
            continue
        if str(r.get("ic_skipped")).lower() in ("true", "1"):
            s["ic_skipped"] += 1
            continue
        try:
            s["area_m2"] += float(r.get("total_area_m2") or 0)
            s["n_polys"] += int(r.get("n_polygons") or 0)
        except ValueError:
            pass

    # 6. Write summary CSV
    summary_path = out_dir / f"{stamp(SLUG)}.csv"
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["region", "sza_bin", "ndwi_threshold", "area_km2",
                         "n_polygons", "n_chips", "n_ic_skipped", "n_band_skipped"])
        for region in REGIONS:
            for sza_bin in SZA_BINS:
                for thr in thresholds:
                    key = (region, sza_bin, thr)
                    s = summary.get(key, {"area_m2": 0.0, "n_chips": 0, "n_polys": 0,
                                          "ic_skipped": 0, "band_skipped": 0})
                    writer.writerow([region, sza_bin, f"{thr:.4f}",
                                     f"{s['area_m2']/1e6:.4f}", s["n_polys"],
                                     s["n_chips"], s["ic_skipped"],
                                     s["band_skipped"]])
    print(f"Summary CSV: {summary_path}")

    # 7. Headline: total km^2 per threshold over the union of (region, SZA bin)
    overall = {thr: 0.0 for thr in thresholds}
    for (_, _, thr), s in summary.items():
        if thr in overall:
            overall[thr] += s["area_m2"] / 1e6
    base_thr = 0.0 if 0.0 in overall else thresholds[0]
    print("\nTotal iceberg area km^2 per NDWI threshold (all bins):")
    for thr in thresholds:
        delta = (overall[thr] - overall[base_thr]) / max(overall[base_thr], 1e-9)
        print(f"  thr={thr:>+5.2f}  area={overall[thr]:.2f} km^2  "
              f"delta vs thr={base_thr:.2f}: {delta:+.2%}")

    # 8. Overview: km^2 vs threshold per (region, SZA bin)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.cm.tab10
    color_idx = 0
    for region in REGIONS:
        for sza_bin in SZA_BINS:
            xs, ys = [], []
            for thr in thresholds:
                s = summary.get((region, sza_bin, thr))
                if not s or s["n_chips"] == 0:
                    continue
                xs.append(thr)
                ys.append(s["area_m2"] / 1e6)
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", lw=1.4,
                    color=cmap(color_idx % 10),
                    label=f"{region} {sza_bin}")
            color_idx += 1
    ax.axvline(0.0, color="#37474F", ls=":", lw=1.0, alpha=0.7)
    ax.text(0.0, ax.get_ylim()[1] * 0.95, " production NDWI=0",
            color="#37474F", fontsize=8, va="top")
    ax.set_xlabel("NDWI threshold")
    ax.set_ylabel("Total iceberg area (km^2)")
    ax.set_title(f"Q4: NDWI threshold sweep, area per (region, SZA bin)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")
    overview_path = out_dir / f"{stamp(SLUG, 'overview')}.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {overview_path}")

    # 9. Per-bin bar chart at 0.0 vs 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.35
    group_labels = []
    base_vals = []
    plus_vals = []
    for region in REGIONS:
        for sza_bin in SZA_BINS:
            group_labels.append(f"{region}\n{sza_bin}")
            s0 = summary.get((region, sza_bin, 0.0))
            s2 = summary.get((region, sza_bin, 0.2))
            base_vals.append((s0["area_m2"] / 1e6) if s0 else 0.0)
            plus_vals.append((s2["area_m2"] / 1e6) if s2 else 0.0)
    x = np.arange(len(group_labels))
    ax.bar(x - width / 2, base_vals, width=width, color="#37474F",
           label="NDWI > 0.0 (production)")
    ax.bar(x + width / 2, plus_vals, width=width, color="#1976D2",
           label="NDWI > 0.2 (tightened)")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("Total iceberg area (km^2)")
    ax.set_title("Q4: production vs tightened NDWI threshold per (region, SZA bin)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    by_path = out_dir / f"{stamp(SLUG, 'by_sza_region')}.png"
    fig.savefig(by_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {by_path}")


if __name__ == "__main__":
    main()
