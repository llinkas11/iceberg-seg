"""
build_v4_test_pools.py: materialise the four per-bin test-chip pools that
downstream model variants will 2:1 pos:null sample from.

Pool definitions (all test chips pass through raw: project convention is to
never IC-mask or re-filter test chips; training chips handle masking separately
in Step 6):

  sza_lt65     Fisser lt65 test positives as-is + 29 our_lt65_null chips.
  sza_65_70    All v3 test chips in this bin (Roboflow source).
  sza_70_75    Same.
  sza_gt75     Same.

For each pool, chips land under:

  data/v4_test_pools/<bin>/pos/<chip_stem>.tif
  data/v4_test_pools/<bin>/null/<chip_stem>.tif

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

from build_clean_dataset import B08_THRESHOLD, SYNTHETIC_FISSER_DIR, SZA_BINS
from build_lt65_nulls import OUR_LT65_NULL_SOURCE, SZA_LT65, null_stem
from prepare_test_chips_dir import link_or_copy


NON_LT65_BINS = tuple(b for b in SZA_BINS if b != SZA_LT65)


# 1. Helpers
def resolve_tif(row):
    """Return an absolute tif path for a split_log row."""
    if isinstance(row.tif_path, str) and row.tif_path and row.tif_path != "nan":
        return row.tif_path
    if row.source == "fisser":
        return os.path.join(SYNTHETIC_FISSER_DIR, f"{row.stem}.tif")
    raise FileNotFoundError(f"no tif_path for row {row.stem} source={row.source}")


def compute_ic_frac(tif_path):
    """Chip-level B08 >= B08_THRESHOLD pixel fraction."""
    with rio.open(tif_path) as src:
        b08 = src.read(3).astype(np.float32)
    return float((b08 >= B08_THRESHOLD).mean())


def materialise_chip(src, dst_dir, chip_stem):
    """Copy tif into dst_dir as '<chip_stem>.tif'; return the dest path."""
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"{chip_stem}.tif")
    link_or_copy(src, dst, copy=True)
    return dst


def manifest_row(bin_label, gt_label, chip_stem, source, n_icebergs,
                 ic_frac, tif_src, tif_pool):
    return {
        "bin_label": bin_label, "gt_label": gt_label, "chip_stem": chip_stem,
        "source": source, "n_icebergs": int(n_icebergs), "ic_frac": ic_frac,
        "tif_src": tif_src, "tif_pool": tif_pool,
    }


# 2. Pool builders
def rows_for_bin(split_log, bin_name):
    return split_log[(split_log.sza_bin == bin_name) & (split_log.split == "test")]


def build_lt65_pool(v3_split, nulls_df, out_root):
    """Fisser lt65 test positives as-is plus the 29 our_lt65_null chips."""
    rows = []
    lt65 = rows_for_bin(v3_split, SZA_LT65)
    assert (lt65.n_icebergs == 0).sum() <= 1, \
        "v3 lt65 test has more than one pre-existing null; builder must be updated"

    for _, r in lt65.iterrows():
        if r.n_icebergs == 0:
            continue  # Fisser null replaced by our_lt65_null below
        tif_src = resolve_tif(r)
        ic_frac = compute_ic_frac(tif_src)
        dst = materialise_chip(tif_src, os.path.join(out_root, SZA_LT65, "pos"), r.chip_stem)
        rows.append(manifest_row(SZA_LT65, "pos", r.chip_stem, r.source,
                                  r.n_icebergs, ic_frac, tif_src, dst))

    for _, r in nulls_df.iterrows():
        stem = null_stem(r.region, r.stem, r.row, r.col)
        dst = materialise_chip(r.tif_path, os.path.join(out_root, SZA_LT65, "null"), stem)
        rows.append(manifest_row(SZA_LT65, "null", stem, OUR_LT65_NULL_SOURCE,
                                  0, float(r.ic_frac), r.tif_path, dst))
    return rows


def build_non_lt65_pool(v3_split, bin_name, out_root):
    """All v3 test chips for one non-lt65 bin go in as-is. ic_frac is not
    computed here (purely informational, and reading every tif over NFS is
    the dominant cost). Set to None; the manifest column is still present.
    """
    rows = []
    for _, r in rows_for_bin(v3_split, bin_name).iterrows():
        tif_src = resolve_tif(r)
        gt_label = "pos" if r.n_icebergs > 0 else "null"
        dst = materialise_chip(tif_src, os.path.join(out_root, bin_name, gt_label), r.chip_stem)
        rows.append(manifest_row(bin_name, gt_label, r.chip_stem, r.source,
                                  r.n_icebergs, None, tif_src, dst))
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
    all_rows.extend(build_lt65_pool(v3, nulls, args.out_root))
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
