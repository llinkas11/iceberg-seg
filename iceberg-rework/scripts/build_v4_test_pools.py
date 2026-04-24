"""
build_v4_test_pools.py: materialise the five test-chip pools that downstream
model variants will 2:1 pos:null sample from.

Pool definitions:

  og_szalt65    Fisser lt65 test chips exactly as v3 has them (no additional
                IC filter) + 29 our_lt65_null chips.
  sza_lt65      Same, but Fisser positives must pass our chip-level IC gate
                (B08 >= B08_THRESHOLD fraction < IC_THRESHOLD).
  sza_65_70     All v3 test chips in this bin (Roboflow source).
  sza_70_75     Same.
  sza_gt75      Same.

Fisser positives already passed Fisser's own cloud filter, so no QA60 gate is
applied here (Fisser chips have no matching SAFE zips in sentinel2_downloads).

For each pool, chips land under:

  data/v4_test_pools/<bin>/pos/<chip>.tif
  data/v4_test_pools/<bin>/null/<chip>.tif

and a single manifest is written to:

  reference/v4_test_pools.csv

Columns: bin_label, gt_label, chip_stem, source, n_icebergs, ic_frac,
         tif_src, tif_pool.

rsync:
  rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/build_v4_test_pools.py llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import rasterio as rio

from build_clean_dataset import B08_THRESHOLD, IC_THRESHOLD


FISSER_RAW_DIR = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/data/raw_chips/fisser"
OUR_LT65_NULL_SOURCE = "our_lt65_null"
NON_LT65_BINS = ("sza_65_70", "sza_70_75", "sza_gt75")


# 1. Helpers
def resolve_tif(row):
    """Return an absolute tif path for a split_log row."""
    if isinstance(row.tif_path, str) and row.tif_path and row.tif_path != "nan":
        return row.tif_path
    if row.source == "fisser":
        return os.path.join(FISSER_RAW_DIR, f"{row.stem}.tif")
    raise FileNotFoundError(f"no tif_path for row {row.stem} source={row.source}")


def compute_ic_frac(tif_path):
    """Chip-level B08 >= B08_THRESHOLD pixel fraction."""
    with rio.open(tif_path) as src:
        b08 = src.read(3).astype(np.float32)
    return float((b08 >= B08_THRESHOLD).mean())


def copy_chip(src, dst_dir, stem):
    """Copy tif into dst_dir with canonical '<stem>.tif' name."""
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"{stem}.tif")
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
    return dst


# 2. Pool builders
def rows_for_bin(split_log, bin_name):
    """v3-style test rows for a given sza_bin."""
    return split_log[(split_log.sza_bin == bin_name) & (split_log.split == "test")].copy()


def build_lt65_pools(v3_split, nulls_df, out_root):
    """
    og_szalt65: v3 lt65 test (56 pos + 1 null) + 29 our_lt65_null (30 null total).
    sza_lt65:   v3 lt65 test positives that pass chip-level IC<15% + 29 our_lt65_null.
    Returns a list of manifest rows.
    """
    rows = []
    lt65 = rows_for_bin(v3_split, "sza_lt65")

    for _, r in lt65.iterrows():
        tif_src = resolve_tif(r)
        ic_frac = compute_ic_frac(tif_src)
        gt_label = "pos" if r.n_icebergs > 0 else "null"
        chip_stem = r.chip_stem

        # og keeps every Fisser chip.
        og_dst = copy_chip(tif_src, os.path.join(out_root, "og_szalt65", gt_label), chip_stem)
        rows.append({
            "bin_label": "og_szalt65", "gt_label": gt_label,
            "chip_stem": chip_stem, "source": r.source,
            "n_icebergs": int(r.n_icebergs), "ic_frac": ic_frac,
            "tif_src": tif_src, "tif_pool": og_dst,
        })

        # sza_lt65 applies our IC gate to Fisser positives only; drops the Fisser
        # null because we're replacing it with our_lt65_null below.
        if gt_label == "pos" and ic_frac < IC_THRESHOLD:
            sza_dst = copy_chip(tif_src, os.path.join(out_root, "sza_lt65", "pos"), chip_stem)
            rows.append({
                "bin_label": "sza_lt65", "gt_label": "pos",
                "chip_stem": chip_stem, "source": r.source,
                "n_icebergs": int(r.n_icebergs), "ic_frac": ic_frac,
                "tif_src": tif_src, "tif_pool": sza_dst,
            })

    for _, r in nulls_df.iterrows():
        stem = f"lt65null_{r.region}_{r.stem}_r{int(r.row):04d}_c{int(r.col):04d}"
        ic_frac = float(r.ic_frac)
        for bin_label in ("og_szalt65", "sza_lt65"):
            dst = copy_chip(r.tif_path, os.path.join(out_root, bin_label, "null"), stem)
            rows.append({
                "bin_label": bin_label, "gt_label": "null",
                "chip_stem": stem, "source": OUR_LT65_NULL_SOURCE,
                "n_icebergs": 0, "ic_frac": ic_frac,
                "tif_src": r.tif_path, "tif_pool": dst,
            })
    return rows


def build_non_lt65_pool(v3_split, bin_name, out_root):
    """All v3 test chips for one non-lt65 bin go in, no refiltering."""
    rows = []
    for _, r in rows_for_bin(v3_split, bin_name).iterrows():
        tif_src = resolve_tif(r)
        ic_frac = compute_ic_frac(tif_src)
        gt_label = "pos" if r.n_icebergs > 0 else "null"
        dst = copy_chip(tif_src, os.path.join(out_root, bin_name, gt_label), r.chip_stem)
        rows.append({
            "bin_label": bin_name, "gt_label": gt_label,
            "chip_stem": r.chip_stem, "source": r.source,
            "n_icebergs": int(r.n_icebergs), "ic_frac": ic_frac,
            "tif_src": tif_src, "tif_pool": dst,
        })
    return rows


# 3. Main
def main():
    parser = argparse.ArgumentParser(description="Build v4 test-chip pools")
    parser.add_argument("--v3_split",
        default="/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/data/v3_clean/split_log.csv")
    parser.add_argument("--nulls_csv",
        default="/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/reference/lt65_nulls_selected.csv")
    parser.add_argument("--out_root",
        default="/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/data/v4_test_pools")
    parser.add_argument("--manifest",
        default="/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/reference/v4_test_pools.csv")
    args = parser.parse_args()

    v3 = pd.read_csv(args.v3_split)
    nulls = pd.read_csv(args.nulls_csv)

    if os.path.exists(args.out_root):
        shutil.rmtree(args.out_root)

    all_rows = []
    all_rows.extend(build_lt65_pools(v3, nulls, args.out_root))
    for bin_name in NON_LT65_BINS:
        all_rows.extend(build_non_lt65_pool(v3, bin_name, args.out_root))

    manifest = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    manifest.to_csv(args.manifest, index=False)

    print("\npool counts by (bin, gt):")
    print(manifest.groupby(["bin_label", "gt_label"]).size().unstack(fill_value=0))
    print(f"\nmanifest: {args.manifest}")
    print(f"pools:    {args.out_root}")


if __name__ == "__main__":
    main()
