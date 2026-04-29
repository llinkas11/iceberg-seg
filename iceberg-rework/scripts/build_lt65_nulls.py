"""
build_lt65_nulls.py: find GT-zero (null) chips for the sza_lt65 test bin.

Scans every pre-chipped lt65 tif under chips_dir/{KQ,SK}/sza_lt65/tifs, applies
the same quality gates the project already uses for positives, and keeps only
chips that contain no iceberg-sized bright object. The accepted pool is ranked
and subsampled KQ:SK proportional to the Fisser GT-positive region distribution
(~20:80). Output:

  reference/lt65_null_candidates.csv   every scanned chip + decision
  reference/lt65_nulls_selected.csv    the 29 picks (6 KQ + 23 SK)
  viz/lt65_nulls_qc/contact_sheet.png  29 thumbnails for manual QA

Two run modes:

  Discovery (default):
    python build_lt65_nulls.py
  Scans, selects 29 chips, and writes the candidates / selected CSVs and the QA
  contact sheet. Idempotent.

  Merge into a base manifest:
    python build_lt65_nulls.py \
        --merge_into_manifest data/v4_clean_lt65/manifest.json \
        --merge_out_dir       data/v4_clean_lt65_plus_nulls \
        --merge_manifest_id   v4_clean_lt65_plus_nulls
  Skips the scan; reads the existing lt65_nulls_selected.csv and grafts those
  29 chips into the base manifest's TRAIN split. Val and test pass through
  byte-stable; chips_sha is recomputed. Used by Phase A's A1, A3-A9
  experiments.

Optional --build_split (legacy) writes data/v4_clean_lt65_balanced/ by copying
v3_clean and rebalancing the lt65 TEST bin to 28 positives + 29 nulls. Kept
for back-compat; superseded by --merge_into_manifest for Phase A purposes.

Quality gates (all must pass):
  1. Cloud: QA60 (opaque|cirrus) fraction < 1% within chip bounds.
  2. Ice coverage: fraction of pixels with B08 >= B08_THRESHOLD is < 15%.
  3. No iceberg: Otsu on B08 yields zero connected components >= 16 px (40 m
     root length), and B08 p95 < B08_THRESHOLD as a safety net.

Sea ice, cloud-edge, land, and open water all pass provided gates 1-3 hold.

rsync:
  rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/build_lt65_nulls.py llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/
"""

import argparse
import csv
import functools
import json
import os
import pickle
import shutil
from collections import Counter
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from scipy.ndimage import label as cc_label
from skimage.filters import threshold_otsu

from _method_common import SKIP_TOO_FEW_BANDS, get_git_sha, sha256_of_file
from build_clean_dataset import (
    B08_THRESHOLD,
    CHIP_SIZE,
    IC_THRESHOLD,
    compute_chips_sha,
    summarise_splits,
)

# cloud_filter_roboflow pulls in `requests`, which isn't always installed in the
# inference venv. Merge mode never touches it; defer the import to the
# discovery code path so the merge CLI works on any env that has rasterio +
# pandas + numpy.
def _import_discovery_deps():
    global find_qa60_entry, get_cloud_fraction, parse_chip_name
    global make_false_color, percentile_stretch
    from cloud_filter_roboflow import (  # noqa: F401
        find_qa60_entry,
        get_cloud_fraction,
        parse_chip_name,
    )
    from otsu_threshold_tifs import (  # noqa: F401
        make_false_color,
        percentile_stretch,
    )


# 1. Constants
CLOUD_THRESHOLD   = 0.01
MIN_ICEBERG_PX    = 16      # 40 m root length on 10 m pixels
OTSU_FLOOR        = 0.10    # below this Otsu threshold is noise; skip the CC test
SZA_LT65          = "sza_lt65"
SEED              = 20260423
TARGET_KQ         = 6
TARGET_SK         = 23
TARGET_V4_POS     = 28      # lt65 test positives to retain from v3
REGIONS           = ("KQ", "SK")

PROVENANCE_CSV    = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/reference/fisser_provenance_audit.csv"

OUR_LT65_NULL_SOURCE = "our_lt65_null"


def null_stem(region, base_stem, row, col):
    """Canonical split_log `stem` for a generated lt65 null chip."""
    return f"lt65null_{region}_{base_stem}_r{int(row):04d}_c{int(col):04d}"


# 2. Zip index (built once, scanned 4k+ times)
def build_zip_index(downloads_dir):
    """Map SAFE-zip scene stem -> absolute zip path. Walks downloads_dir once."""
    index = {}
    for root, _, files in os.walk(downloads_dir):
        for f in files:
            if f.endswith(".SAFE.zip"):
                index[f[:-len(".SAFE.zip")]] = os.path.join(root, f)
    return index


@functools.lru_cache(maxsize=1024)
def _qa60_entry_cached(zip_path):
    """Memoise find_qa60_entry; SAFE zips are reused across many chips."""
    return find_qa60_entry(zip_path)


# 3. Scanning
def scan_chip(tif_path, zip_index):
    """Score a single lt65 chip against the null gates. Returns a row dict."""
    parsed = parse_chip_name(os.path.basename(tif_path))
    if parsed is None:
        return {"tif_path": tif_path, "decision": "skip", "note": "unparseable_filename"}
    stem, chip_row, chip_col = parsed

    with rio.open(tif_path) as src:
        if src.count < 3:
            return {"tif_path": tif_path, "stem": stem, "row": chip_row, "col": chip_col,
                    "decision": "skip", "note": SKIP_TOO_FEW_BANDS}
        b08 = src.read(3).astype(np.float32)
        chip_bounds = src.bounds
        chip_crs    = src.crs

    dark_frac = float((b08 < 0.05).mean())
    ic_frac   = float((b08 >= B08_THRESHOLD).mean())

    result = {
        "tif_path": tif_path, "stem": stem, "row": chip_row, "col": chip_col,
        "region": None, "dark_frac": dark_frac, "ic_frac": ic_frac,
        "b08_p95": None, "cloud_frac": None, "otsu_thresh": None, "max_cc_px": None,
        "decision": "", "note": "",
    }

    # 3a. IC gate (cheapest, skips roughly half the pool on these scenes)
    if ic_frac >= IC_THRESHOLD:
        result["decision"], result["note"] = "reject", "ic_frac>=IC_THRESHOLD"
        return result

    # 3b. Cloud gate (opens zip; needs QA60)
    zip_path = zip_index.get(stem)
    if zip_path is None:
        result["decision"], result["note"] = "skip", "no_safe_zip"
        return result
    qa60_entry = _qa60_entry_cached(zip_path)
    if qa60_entry is None:
        result["decision"], result["note"] = "skip", "no_qa60"
        return result
    cloud_frac = get_cloud_fraction(zip_path, chip_bounds, chip_crs)
    if cloud_frac is None:
        result["decision"], result["note"] = "skip", "no_qa60"
        return result
    result["cloud_frac"] = cloud_frac
    if cloud_frac >= CLOUD_THRESHOLD:
        result["decision"], result["note"] = "reject", "cloud_frac>=CLOUD_THRESHOLD"
        return result

    # 3c. Iceberg gate (only compute p95 + Otsu for chips that already passed IC+cloud)
    b08_p95 = float(np.percentile(b08, 95))
    result["b08_p95"] = b08_p95
    if b08_p95 >= B08_THRESHOLD:
        result["decision"], result["note"] = "reject", "b08_p95>=B08_THRESHOLD"
        return result
    try:
        otsu_t = float(threshold_otsu(b08))
    except ValueError:
        otsu_t = 0.0
    result["otsu_thresh"] = otsu_t
    if otsu_t >= OTSU_FLOOR:
        _, max_cc_px = _largest_component(b08 > otsu_t)
        result["max_cc_px"] = max_cc_px
        if max_cc_px >= MIN_ICEBERG_PX:
            result["decision"], result["note"] = "reject", "otsu_cc>=MIN_ICEBERG_PX"
            return result

    result["decision"] = "accept"
    return result


def _largest_component(mask):
    """Return (n_labels, max_component_size_in_pixels) using cc_label + bincount."""
    if not mask.any():
        return 0, 0
    labels, n = cc_label(mask)
    if n == 0:
        return 0, 0
    sizes = np.bincount(labels.ravel())
    return n, int(sizes[1:].max())


def scan_region(region, chips_dir, zip_index):
    """Yield a scan-row dict for every lt65 tif under chips_dir/<region>/sza_lt65/tifs."""
    tif_dir = os.path.join(chips_dir, region, SZA_LT65, "tifs")
    if not os.path.isdir(tif_dir):
        return
    for name in sorted(os.listdir(tif_dir)):
        if not name.endswith(".tif"):
            continue
        row = scan_chip(os.path.join(tif_dir, name), zip_index)
        row["region"] = region
        yield row


# 4. Selection
def pick_nulls(candidates_df, target_kq=TARGET_KQ, target_sk=TARGET_SK, seed=SEED):
    """Rank accepted candidates per region by B08 p95 asc, dark_frac desc; pick top N."""
    rng = np.random.default_rng(seed)
    accepted = candidates_df[candidates_df.decision == "accept"].copy()
    accepted["tie"] = rng.random(len(accepted))
    accepted = accepted.sort_values(["b08_p95", "dark_frac", "tie"],
                                    ascending=[True, False, True])
    picks = []
    for region, target in zip(REGIONS, (target_kq, target_sk)):
        sub = accepted[accepted.region == region].head(target)
        if len(sub) < target:
            print(f"  [warn] only {len(sub)} accepted in {region}, target was {target}")
        picks.append(sub)
    return pd.concat(picks, ignore_index=True).drop(columns=["tie"])


# 5. Contact sheet
# Per-chip percentile stretch amplifies sensor noise on near-uniform open-water
# chips and saturates snow/land. Use a fixed reflectance cap so all 29 tiles
# share the same brightness scale.
FIXED_STRETCH_MAX = 0.30


def _fixed_false_color(chip, vmax=FIXED_STRETCH_MAX):
    """Map B04/B03/B08 reflectance to [0,1] linearly with shared vmax."""
    rgb = np.stack([chip[0], chip[1], chip[2]], axis=-1)
    return np.clip(rgb / vmax, 0.0, 1.0)


def write_contact_sheet(selected_df, out_png):
    """6-wide grid of selected nulls (B04/B03/B08, fixed reflectance scale)."""
    n = len(selected_df)
    if n == 0:
        return
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2),
                              facecolor="#1a1a2e")
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
        ax.set_facecolor("#1a1a2e")
    for i, (_, r) in enumerate(selected_df.iterrows()):
        ax = axes[i // cols, i % cols]
        with rio.open(r.tif_path) as src:
            chip = src.read([1, 2, 3]).astype(np.float32)
        ax.imshow(_fixed_false_color(chip))
        ax.set_title(f"{r.region} p95={r.b08_p95:.2f}", color="white", fontsize=8)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, facecolor="#1a1a2e")
    plt.close(fig)


# 6. v4 split
def build_v4_split(v3_dir, v4_dir, selected_df, seed=SEED):
    """
    Copy v3_clean to v4_clean_lt65_balanced and rebalance the lt65 test bin
    to TARGET_V4_POS positives + len(selected_df) nulls. v3 stays untouched.
    Positive subsample is proportional to the v3 lt65 test KQ:SK split.
    Both the pkl arrays and split_log.csv are rebuilt in the same order so
    pkl_position stays aligned to x_test / y_test row index.
    """
    rng = np.random.default_rng(seed)

    # 6a. Copy v3 into v4
    if os.path.exists(v4_dir):
        shutil.rmtree(v4_dir)
    shutil.copytree(v3_dir, v4_dir)

    split_path = os.path.join(v4_dir, "split_log.csv")
    split = pd.read_csv(split_path)

    # 6b. Identify which lt65 test positives to keep (proportional KQ:SK)
    lt65_test_pos = split[(split.sza_bin == SZA_LT65) &
                          (split.split == "test") &
                          (split.n_icebergs > 0)].copy()
    if len(lt65_test_pos) == 0:
        raise RuntimeError("v3 split_log has no lt65 test positives; nothing to subsample")
    lt65_test_pos["region"] = _region_from_stem(lt65_test_pos.stem)
    kq_target = int(round(TARGET_V4_POS * (lt65_test_pos.region == "KQ").sum()
                          / len(lt65_test_pos)))
    sk_target = TARGET_V4_POS - kq_target
    keep_idx = []
    for region, target in zip(REGIONS, (kq_target, sk_target)):
        pool = lt65_test_pos[lt65_test_pos.region == region]
        keep_idx.extend(rng.choice(pool.index,
                                    size=min(target, len(pool)),
                                    replace=False))
    drop_indices = set(split[(split.sza_bin == SZA_LT65) &
                              (split.split == "test")].index) - set(keep_idx)

    # 6c. Rebuild pkl arrays and split_log in identical pkl order
    tvt_dir = os.path.join(v4_dir, "train_validate_test")
    with open(os.path.join(tvt_dir, "x_test.pkl"), "rb") as f:
        x_test = pickle.load(f)
    with open(os.path.join(tvt_dir, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)

    test_rows_pkl_order = split[split.split == "test"].sort_values("pkl_position")
    keep_mask = ~test_rows_pkl_order.index.isin(drop_indices)
    kept_x = x_test[keep_mask]
    kept_y = y_test[keep_mask]
    kept_test_rows = test_rows_pkl_order[keep_mask]

    null_x = np.zeros((len(selected_df), 3, CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    null_y = np.zeros((len(selected_df), 1, CHIP_SIZE, CHIP_SIZE), dtype=np.int64)
    null_rows = []
    for i, (_, r) in enumerate(selected_df.iterrows()):
        with rio.open(r.tif_path) as src:
            null_x[i] = src.read([1, 2, 3]).astype(np.float32)
        chip_stem = f"{r.stem}_r{int(r.row):04d}_c{int(r.col):04d}"
        null_rows.append({
            "split": "test", "pkl_position": -1,
            "stem": null_stem(r.region, r.stem, r.row, r.col),
            "chip_stem": chip_stem, "tif_path": r.tif_path,
            "sza_bin": SZA_LT65, "source": OUR_LT65_NULL_SOURCE,
            "n_icebergs": 0, "ic_aware": r.ic_frac, "ic_masked": False,
            "wind_ms": "", "temp_c": "",
        })
    x_test_v4 = np.concatenate([kept_x, null_x], axis=0)
    y_test_v4 = np.concatenate([kept_y, null_y], axis=0)
    with open(os.path.join(tvt_dir, "x_test.pkl"), "wb") as f:
        pickle.dump(x_test_v4, f)
    with open(os.path.join(tvt_dir, "y_test.pkl"), "wb") as f:
        pickle.dump(y_test_v4, f)

    new_split = pd.concat([
        split[split.split != "test"],
        kept_test_rows,
        pd.DataFrame(null_rows),
    ], ignore_index=True)
    new_split = _renumber_pkl_positions(new_split)
    new_split.to_csv(split_path, index=False)

    return {
        "v4_test_total": int((new_split.split == "test").sum()),
        "v4_lt65_test_pos": int(((new_split.split == "test") &
                                  (new_split.sza_bin == SZA_LT65) &
                                  (new_split.n_icebergs > 0)).sum()),
        "v4_lt65_test_null": int(((new_split.split == "test") &
                                   (new_split.sza_bin == SZA_LT65) &
                                   (new_split.n_icebergs == 0)).sum()),
        "x_test_shape": x_test_v4.shape,
    }


# 6.5 Merge nulls into a base manifest (training-time injection)
def merge_nulls_into_manifest(base_manifest_path, out_dir, manifest_id,
                              selected_df):
    """
    Append lt65 GT0 chips to the TRAIN split of a base manifest.

    Pkls: X_train/Y_train get the new null chips concatenated at the end;
    val and test pkls are copied byte-stable. split_log.csv and manifest.json
    are written from scratch with chips_sha recomputed.

    Args:
        base_manifest_path: Path to manifest.json built by build_clean_dataset.
        out_dir: Destination directory; pkls / split_log / manifest written here.
        manifest_id: String to record as manifest_id in the new manifest.
        selected_df: DataFrame from reference/lt65_nulls_selected.csv with
            columns tif_path, stem, region, row, col, ic_frac.

    Returns:
        (out_manifest_path, chips_sha) where chips_sha is the new manifest's
        full hex digest (caller may want the prefix for logs).

    Side effects:
        Writes <out_dir>/manifest.json, <out_dir>/split_log.csv, and six pkls
        under <out_dir>/train_validate_test/. Creates directories as needed.

    Used by Phase A experiments A1, A3-A9 which need GT0 chips in training.
    Differs from build_v4_split: that function rebalances v3's lt65 TEST bin.
    """

    # 1. Load base manifest and (only) the train pkls
    with open(base_manifest_path) as f:
        base = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(base_manifest_path))
    tvt = os.path.join(base_dir, "train_validate_test")

    with open(os.path.join(tvt, "X_train.pkl"), "rb") as f:
        x_train = pickle.load(f)
    with open(os.path.join(tvt, "Y_train.pkl"), "rb") as f:
        y_train = pickle.load(f)

    # 2. Materialise null arrays (X from raw S2 tifs, Y all zeros)
    n_nulls = len(selected_df)
    null_x = np.zeros((n_nulls, 3, CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    null_y = np.zeros((n_nulls, 1, CHIP_SIZE, CHIP_SIZE), dtype=np.int64)
    null_chip_records = []
    train_chips = [c for c in base["chips"] if c["split"] == "train"]
    n_train_base = len(train_chips)

    for i, (_, r) in enumerate(selected_df.iterrows()):
        with rio.open(r.tif_path) as src:
            null_x[i] = src.read([1, 2, 3]).astype(np.float32)
        chip_stem = f"{r.stem}_r{int(r.row):04d}_c{int(r.col):04d}"
        null_chip_records.append({
            "chip_stem":    chip_stem,
            "stem":         null_stem(r.region, r.stem, r.row, r.col),
            "source":       OUR_LT65_NULL_SOURCE,
            "sza_bin":      SZA_LT65,
            "tif_path":     r.tif_path,
            "tif_sha":      sha256_of_file(r.tif_path),
            "n_icebergs":   0,
            "has_iceberg":  False,
            "ic_aware":     float(r.ic_frac),
            "wind_ms":      "",
            "temp_c":       "",
            "split":        "train",
            "pkl_position": n_train_base + i,
        })

    # 3. Rewrite train pkls with appended nulls; val/test copied byte-stable.
    # shutil.copy beats load-then-pickle: avoids a ~400 MB round-trip in RAM
    # per merge and preserves the base pkl mtime as a provenance hint.
    out_tvt = os.path.join(out_dir, "train_validate_test")
    os.makedirs(out_tvt, exist_ok=True)

    x_train_new = np.concatenate([x_train, null_x], axis=0)
    y_train_new = np.concatenate([y_train, null_y], axis=0)
    with open(os.path.join(out_tvt, "X_train.pkl"), "wb") as f:
        pickle.dump(x_train_new, f)
    with open(os.path.join(out_tvt, "Y_train.pkl"), "wb") as f:
        pickle.dump(y_train_new, f)
    for name in ("X_validation.pkl", "Y_validation.pkl", "x_test.pkl", "y_test.pkl"):
        shutil.copy(os.path.join(tvt, name), os.path.join(out_tvt, name))

    # 4. Compose new chip list: base train + nulls + base val + base test
    val_chips = [c for c in base["chips"] if c["split"] == "val"]
    test_chips = [c for c in base["chips"] if c["split"] == "test"]
    new_chips = train_chips + null_chip_records + val_chips + test_chips

    # 5. Write split_log.csv
    log_path = os.path.join(out_dir, "split_log.csv")
    fieldnames = [
        "split", "pkl_position", "stem", "chip_stem", "tif_path", "sza_bin",
        "source", "n_icebergs", "ic_aware", "ic_masked", "wind_ms", "temp_c",
    ]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in new_chips:
            writer.writerow({
                "split":        c["split"],
                "pkl_position": c["pkl_position"],
                "stem":         c.get("stem", ""),
                "chip_stem":    c["chip_stem"],
                "tif_path":     c.get("tif_path", ""),
                "sza_bin":      c["sza_bin"],
                "source":       c["source"],
                "n_icebergs":   c["n_icebergs"],
                "ic_aware":     f"{float(c.get('ic_aware', 0)):.4f}",
                "ic_masked":    "False",  # nulls never IC-masked; base IC state preserved by source
                "wind_ms":      c.get("wind_ms", ""),
                "temp_c":       c.get("temp_c", ""),
            })

    # 6. Write manifest.json with recomputed chips_sha
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    chips_sha = compute_chips_sha(new_chips)
    manifest = {
        "manifest_id":      manifest_id,
        "created_utc":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script":           os.path.basename(__file__),
        "git_sha":          get_git_sha(repo_dir),
        "chip_source":      base.get("chip_source", ""),
        "split_policy":     base.get("split_policy", {}),
        "filters":          dict(base.get("filters", {}), nulls_merged=True),
        "base_manifest_id": base["manifest_id"],
        "base_chips_sha":   base["chips_sha"],
        "n_added_nulls":    n_nulls,
        "total_chips":      len(new_chips),
        "counts_by_split":  summarise_splits(new_chips),
        "chips":            new_chips,
        "chips_sha":        chips_sha,
    }

    out_manifest = os.path.join(out_dir, "manifest.json")
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    return out_manifest, chips_sha


@functools.lru_cache(maxsize=1)
def _provenance_idx_to_region():
    prov = pd.read_csv(PROVENANCE_CSV)
    return {f"fisser_{int(r.global_index):04d}": r.region for _, r in prov.iterrows()}


def _region_from_stem(stems):
    """Map Fisser stems (fisser_NNNN) to region (KQ/SK) via the provenance audit."""
    idx_to_region = _provenance_idx_to_region()
    return stems.map(idx_to_region)


def _renumber_pkl_positions(split_df):
    """Re-assign pkl_position within each split so it matches the pkl array order."""
    split_df = split_df.copy()
    for s in ("train", "val", "test"):
        mask = split_df.split == s
        split_df.loc[mask, "pkl_position"] = np.arange(int(mask.sum()))
    return split_df


# 7. Main
def main():
    parser = argparse.ArgumentParser(description="Build sza_lt65 null chips and merge into a base manifest")
    parser.add_argument("--chips_dir",
        default="/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/chips")
    parser.add_argument("--downloads_dir",
        default="/mnt/research/v.gomezgilyaspik/students/smishra/sentinel2_downloads")
    parser.add_argument("--out_dir",
        default="/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework")
    parser.add_argument("--v3_dir",
        default="/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/data/v3_clean")
    parser.add_argument("--v4_dir",
        default="/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/data/v4_clean_lt65_balanced")
    parser.add_argument("--build_split", action="store_true",
        help="(legacy) After selecting nulls, also write v4_clean_lt65_balanced (v3-based test-time injection).")
    parser.add_argument("--merge_into_manifest", default=None,
        help="Path to a base manifest.json. If set, skips discovery and grafts the 29 lt65 nulls "
             "from --nulls_csv into the base manifest's TRAIN split, writing a new manifest at "
             "--merge_out_dir.")
    parser.add_argument("--merge_out_dir", default=None,
        help="Output directory for the merged manifest. Required if --merge_into_manifest is set.")
    parser.add_argument("--merge_manifest_id", default=None,
        help="manifest_id to record in the merged manifest. Required if --merge_into_manifest is set.")
    parser.add_argument("--nulls_csv", default=None,
        help="Path to lt65_nulls_selected.csv. Defaults to reference/lt65_nulls_selected.csv "
             "under --out_dir.")
    args = parser.parse_args()

    # 7a. Merge-only mode: skip discovery, use existing CSV
    if args.merge_into_manifest is not None:
        if args.merge_out_dir is None or args.merge_manifest_id is None:
            parser.error("--merge_into_manifest requires --merge_out_dir and --merge_manifest_id")
        nulls_csv = args.nulls_csv or os.path.join(
            args.out_dir, "reference/lt65_nulls_selected.csv"
        )
        if not os.path.exists(nulls_csv):
            parser.error(f"nulls CSV not found: {nulls_csv}. Run discovery first (no --merge flags).")
        selected = pd.read_csv(nulls_csv)
        print(f"merging {len(selected)} nulls into {args.merge_into_manifest}")
        out_manifest, chips_sha = merge_nulls_into_manifest(
            args.merge_into_manifest,
            args.merge_out_dir,
            args.merge_manifest_id,
            selected,
        )
        print(f"  manifest : {out_manifest}")
        print(f"  chips_sha: {chips_sha[:16]}...")
        return

    # 7b. Discovery mode: full scan + select + (optional) v3-based v4 build
    _import_discovery_deps()

    # Build the SAFE-zip index once
    print(f"indexing SAFE zips under {args.downloads_dir} ...")
    zip_index = build_zip_index(args.downloads_dir)
    print(f"  indexed {len(zip_index)} SAFE zips")

    # Scan
    rows = []
    for region in REGIONS:
        print(f"scanning {region}/{SZA_LT65}/tifs ...")
        for i, r in enumerate(scan_region(region, args.chips_dir, zip_index)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {region}: {i + 1} chips scanned", flush=True)
    df = pd.DataFrame(rows)

    # Persist every scan row
    cand_path = os.path.join(args.out_dir, "reference/lt65_null_candidates.csv")
    os.makedirs(os.path.dirname(cand_path), exist_ok=True)
    df.to_csv(cand_path, index=False)
    print("\ngate counts by (region, decision):")
    for key, n in sorted(Counter(zip(df.region, df.decision)).items()):
        print(f"  {key}: {n}")
    print("\nrejection notes by (region, note):")
    for key, n in sorted(Counter(zip(df.region, df.note)).items()):
        if key[1]:
            print(f"  {key}: {n}")

    # Select 29 nulls
    selected = pick_nulls(df)
    sel_path = os.path.join(args.out_dir, "reference/lt65_nulls_selected.csv")
    selected.to_csv(sel_path, index=False)
    print(f"\nselected {len(selected)} null chips:")
    print(selected.groupby("region").size().to_string())

    # Contact sheet
    qc_path = os.path.join(args.out_dir, "viz/lt65_nulls_qc/contact_sheet.png")
    write_contact_sheet(selected, qc_path)
    print(f"\ncontact sheet: {qc_path}")
    print(f"candidates CSV: {cand_path}")
    print(f"selected CSV:   {sel_path}")

    # Legacy v4 test-time split (optional)
    if args.build_split:
        print(f"\nbuilding v4 split at {args.v4_dir} ...")
        stats = build_v4_split(args.v3_dir, args.v4_dir, selected)
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
