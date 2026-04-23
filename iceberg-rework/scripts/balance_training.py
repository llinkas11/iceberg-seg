"""
balance_training.py — Balance the training set with SZA-aware staged sampling.

Stage 1 balances GT0 vs GT+ within each SZA bin to a max ratio.
Stage 2 optionally balances positive chips by iceberg-size bin within each SZA bin.

Only modifies training data. Validation and test sets are copied unchanged.

Usage:
  python scripts/balance_training.py
  python scripts/balance_training.py --balance_positive_area_bins
"""

import argparse
import csv
import os
import pickle
import random
import shutil

import numpy as np
from scipy.ndimage import label as cc_label

# -- Paths --------------------------------------------------------------------
LLINKAS = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"
CLEAN_DIR = os.path.join(LLINKAS, "data/v3_clean")

PIXEL_AREA_M2 = 100.0
SZA_BINS = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]
AREA_BINS = ["rl_40_100", "rl_100_300", "rl_300_plus"]
GT_GROUPS = ["GT0", "GT+"]


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)



def get_chip_stats(Y):
    """Compute per-chip stats needed for SZA-aware balancing."""
    stats = []
    for i in range(len(Y)):
        mask = Y[i, 0, :, :] if Y.ndim == 4 else Y[i]
        iceberg = (mask == 1).astype(np.int32)
        if iceberg.sum() == 0:
            stats.append({
                "has_iceberg": False,
                "max_rl": 0.0,
                "n_icebergs": 0,
                "area_bin": "null",
            })
            continue

        labels, n_comp = cc_label(iceberg)
        comp_sizes = np.bincount(labels.ravel())
        max_px = int(comp_sizes[1:].max()) if n_comp > 0 else 0
        max_rl = np.sqrt(max_px * PIXEL_AREA_M2)
        stats.append({
            "has_iceberg": True,
            "max_rl": max_rl,
            "n_icebergs": n_comp,
            "area_bin": assign_area_bin(max_rl),
        })
    return stats



def assign_area_bin(max_rl):
    """Assign chip to an area bin based on largest iceberg root length."""
    if max_rl < 100:
        return "rl_40_100"
    if max_rl < 300:
        return "rl_100_300"
    return "rl_300_plus"



def load_train_metadata(clean_dir, Y_train):
    """Load split_log rows aligned to Y_train row order."""
    split_log = os.path.join(clean_dir, "split_log.csv")
    rows = []
    with open(split_log, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") == "train":
                rows.append(row)

    rows.sort(key=lambda row: int(row["pkl_position"]))
    if len(rows) != len(Y_train):
        raise ValueError(
            f"Training split_log rows ({len(rows)}) do not match Y_train length ({len(Y_train)})"
        )
    return rows



def summarize_gt_counts(metadata, indices):
    counts = {sza: {group: 0 for group in GT_GROUPS} for sza in SZA_BINS}
    for idx in indices:
        row = metadata[idx]
        group = "GT+" if row["has_iceberg"] else "GT0"
        counts[row["sza_bin"]][group] += 1
    return counts



def summarize_area_counts(metadata, indices):
    counts = {sza: {area_bin: 0 for area_bin in AREA_BINS} for sza in SZA_BINS}
    for idx in indices:
        row = metadata[idx]
        if row["has_iceberg"]:
            counts[row["sza_bin"]][row["area_bin"]] += 1
    return counts



def print_gt_summary(title, counts):
    print(f"\n{title}")
    for sza in SZA_BINS:
        row = counts.get(sza, {})
        gt0 = row.get("GT0", 0)
        gtpos = row.get("GT+", 0)
        print(f"  {sza:<10} GT0={gt0:>4}  GT+={gtpos:>4}")



def print_area_summary(title, counts):
    print(f"\n{title}")
    for sza in SZA_BINS:
        row = counts.get(sza, {})
        print(
            f"  {sza:<10} rl_40_100={row.get('rl_40_100', 0):>4}  "
            f"rl_100_300={row.get('rl_100_300', 0):>4}  "
            f"rl_300_plus={row.get('rl_300_plus', 0):>4}"
        )



def replicate_indices(indices, target_count, rng):
    """Oversample by deterministic replication plus random fill."""
    if not indices:
        return []
    reps = target_count // len(indices)
    remainder = target_count % len(indices)
    out = indices * reps
    if remainder:
        out.extend(rng.sample(indices, remainder))
    return out



def rebalance_area_bins(indices_by_bin, target_ratio, min_bin_count, rng):
    """Apply Fisser-style balancing to present positive area bins only."""
    present = {name: idx for name, idx in indices_by_bin.items() if idx}
    counts = {name: len(idx) for name, idx in present.items()}
    n_min = min(counts.values())
    n_target_low = max(n_min, min_bin_count)
    n_target_high = max(int(target_ratio * n_min), int(target_ratio * n_target_low))

    balanced = {}
    actions = {}
    for name, idx in present.items():
        n = len(idx)
        if n <= n_target_low:
            balanced[name] = replicate_indices(idx, n_target_low, rng)
            actions[name] = "oversampled" if len(balanced[name]) > n else "unchanged"
        elif n > n_target_high:
            balanced[name] = rng.sample(idx, n_target_high)
            actions[name] = "undersampled"
        else:
            balanced[name] = idx.copy()
            actions[name] = "unchanged"

    return balanced, actions



def main():
    parser = argparse.ArgumentParser(description="Balance training set with SZA-aware stages")
    parser.add_argument("--clean_dir", default=CLEAN_DIR)
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for the balanced dataset (defaults to a non-baseline variant path)",
    )
    parser.add_argument("--ratio", type=float, default=2.0, help="Max ratio (default 2.0)")
    parser.add_argument(
        "--min_minority_for_gt_balance",
        type=int,
        default=5,
        help="Skip stage-1 GT balancing in an SZA bin if minority count is below this value",
    )
    parser.add_argument(
        "--balance_positive_area_bins",
        action="store_true",
        help="Optionally rebalance positive area bins within each SZA bin",
    )
    parser.add_argument(
        "--min_bin_count_for_area_balance",
        type=int,
        default=3,
        help="Minimum count required in every present area bin to run stage 2",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out_dir is None:
        variant_name = "v3_balanced_sza_stage1_stage2" if args.balance_positive_area_bins else "v3_balanced_sza_stage1"
        args.out_dir = os.path.join(LLINKAS, "data", variant_name)

    rng = random.Random(args.seed)
    split_dir = os.path.join(args.clean_dir, "train_validate_test")

    X_train = np.array(load_pkl(os.path.join(split_dir, "X_train.pkl")))
    Y_train = np.array(load_pkl(os.path.join(split_dir, "Y_train.pkl")))
    print(f"Training set: {len(X_train)} chips")

    stats = get_chip_stats(Y_train)
    split_rows = load_train_metadata(args.clean_dir, Y_train)

    metadata = []
    for idx, row in enumerate(split_rows):
        metadata.append({
            "orig_index": idx,
            "sza_bin": row.get("sza_bin", "unknown"),
            "has_iceberg": stats[idx]["has_iceberg"],
            "area_bin": stats[idx]["area_bin"],
            "max_rl": stats[idx]["max_rl"],
            "n_icebergs": stats[idx]["n_icebergs"],
        })

    original_indices = list(range(len(metadata)))
    print_gt_summary("Original training GT counts by SZA:", summarize_gt_counts(metadata, original_indices))

    report_rows = []
    stage1_indices = []

    for sza in SZA_BINS:
        sza_indices = [i for i, row in enumerate(metadata) if row["sza_bin"] == sza]
        gt0_indices = [i for i in sza_indices if not metadata[i]["has_iceberg"]]
        gtpos_indices = [i for i in sza_indices if metadata[i]["has_iceberg"]]

        orig_counts = {"GT0": len(gt0_indices), "GT+": len(gtpos_indices)}
        minority = min(orig_counts.values())
        majority_group = "GT0" if orig_counts["GT0"] >= orig_counts["GT+"] else "GT+"

        if minority < args.min_minority_for_gt_balance:
            kept_gt0 = gt0_indices.copy()
            kept_gtpos = gtpos_indices.copy()
            skip_reason = (
                f"minority count {minority} below min_minority_for_gt_balance="
                f"{args.min_minority_for_gt_balance}"
            )
            print(f"  Stage 1 skip for {sza}: {skip_reason}")
            for group_name, kept in [("GT0", kept_gt0), ("GT+", kept_gtpos)]:
                report_rows.append({
                    "stage": "stage1_gt_balance",
                    "sza_bin": sza,
                    "group_name": group_name,
                    "original_count": orig_counts[group_name],
                    "balanced_count": len(kept),
                    "action": "skipped",
                    "skipped_reason": skip_reason,
                })
        else:
            kept_gt0 = gt0_indices.copy()
            kept_gtpos = gtpos_indices.copy()
            majority_count = max(orig_counts.values())
            target_majority = int(args.ratio * minority)
            skip_reason = ""
            if majority_count > target_majority:
                if majority_group == "GT0":
                    kept_gt0 = rng.sample(gt0_indices, target_majority)
                    action_map = {"GT0": "undersampled", "GT+": "unchanged"}
                else:
                    kept_gtpos = rng.sample(gtpos_indices, target_majority)
                    action_map = {"GT0": "unchanged", "GT+": "undersampled"}
            else:
                action_map = {"GT0": "unchanged", "GT+": "unchanged"}

            report_rows.append({
                "stage": "stage1_gt_balance",
                "sza_bin": sza,
                "group_name": "GT0",
                "original_count": orig_counts["GT0"],
                "balanced_count": len(kept_gt0),
                "action": action_map["GT0"],
                "skipped_reason": skip_reason,
            })
            report_rows.append({
                "stage": "stage1_gt_balance",
                "sza_bin": sza,
                "group_name": "GT+",
                "original_count": orig_counts["GT+"],
                "balanced_count": len(kept_gtpos),
                "action": action_map["GT+"],
                "skipped_reason": skip_reason,
            })

        stage1_indices.extend(kept_gt0)
        stage1_indices.extend(kept_gtpos)

    print_gt_summary("Post-stage-1 GT counts by SZA:", summarize_gt_counts(metadata, stage1_indices))

    final_indices = stage1_indices.copy()
    if args.balance_positive_area_bins:
        print_area_summary(
            "Original positive area-bin counts by SZA after stage 1:",
            summarize_area_counts(metadata, stage1_indices),
        )

        stage2_indices = []
        for sza in SZA_BINS:
            sza_stage1 = [i for i in stage1_indices if metadata[i]["sza_bin"] == sza]
            gt0_indices = [i for i in sza_stage1 if not metadata[i]["has_iceberg"]]
            gtpos_indices = [i for i in sza_stage1 if metadata[i]["has_iceberg"]]

            indices_by_bin = {
                area_bin: [i for i in gtpos_indices if metadata[i]["area_bin"] == area_bin]
                for area_bin in AREA_BINS
            }
            present = {name: idx for name, idx in indices_by_bin.items() if idx}
            present_counts = {name: len(idx) for name, idx in present.items()}

            if len(present) < 2:
                skip_reason = "fewer than two present positive area bins after stage 1"
                print(f"  Stage 2 skip for {sza}: {skip_reason}")
                for area_bin in AREA_BINS:
                    orig = len(indices_by_bin[area_bin])
                    report_rows.append({
                        "stage": "stage2_positive_area_balance",
                        "sza_bin": sza,
                        "group_name": area_bin,
                        "original_count": orig,
                        "balanced_count": orig,
                        "action": "skipped",
                        "skipped_reason": skip_reason,
                    })
                stage2_indices.extend(gt0_indices)
                stage2_indices.extend(gtpos_indices)
                continue

            min_present = min(present_counts.values())
            if min_present < args.min_bin_count_for_area_balance:
                skip_reason = (
                    f"present area-bin count {min_present} below min_bin_count_for_area_balance="
                    f"{args.min_bin_count_for_area_balance}"
                )
                print(f"  Stage 2 skip for {sza}: {skip_reason}")
                for area_bin in AREA_BINS:
                    orig = len(indices_by_bin[area_bin])
                    report_rows.append({
                        "stage": "stage2_positive_area_balance",
                        "sza_bin": sza,
                        "group_name": area_bin,
                        "original_count": orig,
                        "balanced_count": orig,
                        "action": "skipped",
                        "skipped_reason": skip_reason,
                    })
                stage2_indices.extend(gt0_indices)
                stage2_indices.extend(gtpos_indices)
                continue

            balanced_bins, action_map = rebalance_area_bins(
                indices_by_bin,
                target_ratio=args.ratio,
                min_bin_count=args.min_bin_count_for_area_balance,
                rng=rng,
            )
            candidate_gtpos = []
            for area_bin in AREA_BINS:
                candidate_gtpos.extend(balanced_bins.get(area_bin, indices_by_bin[area_bin]))

            stage1_gt_balance_ok = len(gt0_indices) <= (args.ratio * len(gtpos_indices))
            if stage1_gt_balance_ok and len(gt0_indices) > (args.ratio * len(candidate_gtpos)):
                skip_reason = (
                    f"stage-2 area-bin balancing would break GT0:GT+ <= {args.ratio}:1 "
                    f"preserved by stage 1"
                )
                print(f"  Stage 2 skip for {sza}: {skip_reason}")
                for area_bin in AREA_BINS:
                    orig = len(indices_by_bin[area_bin])
                    report_rows.append({
                        "stage": "stage2_positive_area_balance",
                        "sza_bin": sza,
                        "group_name": area_bin,
                        "original_count": orig,
                        "balanced_count": orig,
                        "action": "skipped",
                        "skipped_reason": skip_reason,
                    })
                stage2_indices.extend(gt0_indices)
                stage2_indices.extend(gtpos_indices)
                continue

            stage2_indices.extend(gt0_indices)
            for area_bin in AREA_BINS:
                orig = len(indices_by_bin[area_bin])
                new_idx = balanced_bins.get(area_bin, indices_by_bin[area_bin])
                stage2_indices.extend(new_idx)
                report_rows.append({
                    "stage": "stage2_positive_area_balance",
                    "sza_bin": sza,
                    "group_name": area_bin,
                    "original_count": orig,
                    "balanced_count": len(new_idx),
                    "action": action_map.get(area_bin, "unchanged" if orig > 0 else "absent"),
                    "skipped_reason": "",
                })

        final_indices = stage2_indices
        print_area_summary(
            "Post-stage-2 positive area-bin counts by SZA:",
            summarize_area_counts(metadata, final_indices),
        )
    else:
        print("\nStage 2 disabled: positive area-bin balancing not applied")

    rng.shuffle(final_indices)
    X_balanced = X_train[final_indices]
    Y_balanced = Y_train[final_indices]

    print(f"\nTotal training chips: {len(X_train)} -> {len(final_indices)}")
    print(f"Balanced arrays: X={X_balanced.shape}  Y={Y_balanced.shape}")

    flat = Y_balanced.flatten()
    total = flat.size
    for cls, name in [(0, "ocean"), (1, "iceberg")]:
        n = int((flat == cls).sum())
        print(f"  class {cls} ({name}): {n / total * 100:.1f}%")

    out_split = os.path.join(args.out_dir, "train_validate_test")
    os.makedirs(out_split, exist_ok=True)

    save_pkl(X_balanced, os.path.join(out_split, "X_train.pkl"))
    save_pkl(Y_balanced, os.path.join(out_split, "Y_train.pkl"))

    for fname in ["X_validation.pkl", "Y_validation.pkl", "x_test.pkl", "y_test.pkl"]:
        src = os.path.join(split_dir, fname)
        dst = os.path.join(out_split, fname)
        shutil.copy2(src, dst)
        print(f"  Copied: {fname}")

    src_log = os.path.join(args.clean_dir, "split_log.csv")
    dst_log = os.path.join(args.out_dir, "split_log.csv")
    shutil.copy2(src_log, dst_log)

    report_path = os.path.join(args.out_dir, "balance_report.csv")
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "sza_bin",
                "group_name",
                "original_count",
                "balanced_count",
                "action",
                "skipped_reason",
            ],
        )
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)

    print(f"\nBalance report: {report_path}")
    print(f"Balanced data:  {out_split}/")
    print(f"\nNext: python scripts/train.py --mode s2 --data_dir {args.out_dir} --out_dir model/v3_balanced_aug")


if __name__ == "__main__":
    main()
