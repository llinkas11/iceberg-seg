"""
summarize_ndwi_sweep.py — Reduce the per-chip NDWI sweep CSV to a per-SZA
summary table.

Inputs:
    --in_csv   : per-chip CSV from sweep_ndwi_threshold.py
    --out_csv  : per-(sza_bin, ndwi_threshold) summary CSV
Output columns:
    sza_bin, ndwi_threshold, n_chips, n_ic_skipped, n_band_skipped,
    n_with_detect, n_polygons, total_area_km2,
    median_chip_area_m2, mean_chip_area_m2, max_chip_area_m2

Also prints a compact text table grouped by sza_bin and an "ALL" row that
collapses across SZA bins.
"""

import argparse
import csv
import os
from collections import defaultdict
from statistics import median, mean


DEFAULT_IN_CSV  = "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/sweeps/ndwi_threshold_sweep.csv"
DEFAULT_OUT_CSV = "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/paper-writing/figure_review/results_figs_table_previews/ndwi_threshold_sweep_summary.csv"

SZA_ORDER = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]


def summarize(in_csv, out_csv):
    """Read per-chip CSV, aggregate, write summary CSV, and print a text table."""
    rows = list(csv.DictReader(open(in_csv)))

    # 1. Bucket by (sza_bin, ndwi_threshold)
    buckets = defaultdict(list)
    for r in rows:
        key = (r["sza_bin"], float(r["ndwi_threshold"]))
        buckets[key].append(r)

    # 2. Aggregate each bucket
    summary = []
    for (sza_bin, ndwi_val), bucket in sorted(buckets.items()):
        n_chips     = len(bucket)
        n_band      = sum(1 for r in bucket if r["band_skipped"] == "True")
        n_cloud     = sum(1 for r in bucket if r.get("cloud_skipped") == "True")
        n_ic        = sum(1 for r in bucket if r["ic_skipped"] == "True")
        non_skipped = [r for r in bucket
                       if r["band_skipped"] == "False"
                       and r.get("cloud_skipped", "False") == "False"
                       and r["ic_skipped"] == "False"]
        areas       = [float(r["total_area_m2"]) for r in non_skipped]
        polys       = [int(r["n_polygons"]) for r in non_skipped]
        n_detect    = sum(1 for r in non_skipped if int(r["n_polygons"]) > 0)
        total_area  = sum(areas)
        summary.append({
            "sza_bin"            : sza_bin,
            "ndwi_threshold"     : ndwi_val,
            "n_chips"            : n_chips,
            "n_cloud_skipped"    : n_cloud,
            "n_ic_skipped"       : n_ic,
            "n_band_skipped"     : n_band,
            "n_with_detect"      : n_detect,
            "n_polygons"         : sum(polys),
            "total_area_km2"     : round(total_area / 1e6, 6),
            "median_chip_area_m2": round(median(areas), 2) if areas else 0.0,
            "mean_chip_area_m2"  : round(mean(areas), 2) if areas else 0.0,
            "max_chip_area_m2"   : round(max(areas), 2) if areas else 0.0,
        })

    # 3. Write CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = list(summary[0].keys())
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {len(summary)} rows to {out_csv}\n")

    # 4. Print text table grouped by sza_bin, with ALL row
    region_summary = defaultdict(lambda: {
        "n_chips": 0, "n_cloud": 0, "n_ic": 0, "n_band": 0,
        "n_detect": 0, "n_poly": 0, "total_area": 0.0,
    })
    for s in summary:
        agg = region_summary[s["ndwi_threshold"]]
        agg["n_chips"]    += s["n_chips"]
        agg["n_cloud"]    += s["n_cloud_skipped"]
        agg["n_ic"]       += s["n_ic_skipped"]
        agg["n_band"]     += s["n_band_skipped"]
        agg["n_detect"]   += s["n_with_detect"]
        agg["n_poly"]     += s["n_polygons"]
        agg["total_area"] += s["total_area_km2"]

    for sza_bin in SZA_ORDER + ["ALL"]:
        print(f"=== {sza_bin} ===")
        print(f"  {'ndwi':>6} {'chips':>6} {'cloud_sk':>9} {'ic_sk':>6} {'detect':>7} {'polys':>7} {'km2':>10}")
        if sza_bin == "ALL":
            for ndwi_val in sorted(region_summary):
                a = region_summary[ndwi_val]
                print(f"  {ndwi_val:>6.2f} {a['n_chips']:>6} {a['n_cloud']:>9} {a['n_ic']:>6} {a['n_detect']:>7} {a['n_poly']:>7} {a['total_area']:>10.4f}")
        else:
            for s in summary:
                if s["sza_bin"] != sza_bin:
                    continue
                print(f"  {s['ndwi_threshold']:>6.2f} {s['n_chips']:>6} {s['n_cloud_skipped']:>9} {s['n_ic_skipped']:>6} {s['n_with_detect']:>7} {s['n_polygons']:>7} {s['total_area_km2']:>10.4f}")
        print()

    # 5. Headline area-shift table: pct change relative to NDWI=0 within each sza_bin
    print("=== % change in total iceberg km^2 vs NDWI=0 ===")
    print(f"  {'sza_bin':>10} {'-0.05':>8} {'0.00':>8} {'0.05':>8} {'0.10':>8}")
    by_sza = defaultdict(dict)
    for s in summary:
        by_sza[s["sza_bin"]][s["ndwi_threshold"]] = s["total_area_km2"]
    for sza_bin in SZA_ORDER:
        baseline = by_sza[sza_bin].get(0.0, 0.0)
        if baseline == 0:
            continue
        deltas = {k: 100 * (v - baseline) / baseline for k, v in by_sza[sza_bin].items()}
        print(f"  {sza_bin:>10} {deltas.get(-0.05, 0):>+7.2f}% {deltas.get(0.0, 0):>+7.2f}% {deltas.get(0.05, 0):>+7.2f}% {deltas.get(0.10, 0):>+7.2f}%")

    # 6. Region-wide pct change
    overall = defaultdict(float)
    for ndwi_val, agg in region_summary.items():
        overall[ndwi_val] = agg["total_area"]
    base = overall.get(0.0, 0.0)
    if base:
        print(f"  {'ALL':>10} ", end="")
        for k in [-0.05, 0.0, 0.05, 0.10]:
            v = overall.get(k, 0.0)
            d = 100 * (v - base) / base
            print(f"{d:>+7.2f}% ", end="")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv",  default=DEFAULT_IN_CSV)
    parser.add_argument("--out_csv", default=DEFAULT_OUT_CSV)
    args = parser.parse_args()
    summarize(args.in_csv, args.out_csv)


if __name__ == "__main__":
    main()
