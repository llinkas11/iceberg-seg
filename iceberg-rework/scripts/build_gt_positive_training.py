#!/usr/bin/env python3
"""Build a training-only GT-positive dataset variant.

Validation and test splits are copied unchanged. Only training chips with zero
iceberg pixels are removed.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil

import numpy as np
import pandas as pd


def load_pkl(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_pkl(path: str, obj) -> None:
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def has_iceberg_mask(y: np.ndarray) -> np.ndarray:
    if y.ndim == 4:
        return (y[:, 0] == 1).reshape(len(y), -1).any(axis=1)
    return (y == 1).reshape(len(y), -1).any(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    clean_split = os.path.join(args.clean_dir, "train_validate_test")
    out_split = os.path.join(args.out_dir, "train_validate_test")
    os.makedirs(out_split, exist_ok=True)

    x_train = load_pkl(os.path.join(clean_split, "X_train.pkl"))
    y_train = load_pkl(os.path.join(clean_split, "Y_train.pkl"))
    keep = has_iceberg_mask(y_train)

    save_pkl(os.path.join(out_split, "X_train.pkl"), x_train[keep])
    save_pkl(os.path.join(out_split, "Y_train.pkl"), y_train[keep])

    for name in ["X_validation.pkl", "Y_validation.pkl", "x_test.pkl", "y_test.pkl"]:
        shutil.copy2(os.path.join(clean_split, name), os.path.join(out_split, name))

    split_log_path = os.path.join(args.clean_dir, "split_log.csv")
    split_log = pd.read_csv(split_log_path)
    split_log.to_csv(os.path.join(args.out_dir, "split_log.csv"), index=False)

    train_rows = split_log[split_log["split"].eq("train")].copy()
    if len(train_rows) != len(y_train):
        raise ValueError(f"split_log train rows {len(train_rows)} != Y_train rows {len(y_train)}")
    train_rows = train_rows.sort_values("pkl_position").reset_index(drop=True)
    train_rows["has_iceberg"] = keep

    kept_rows = train_rows[train_rows["has_iceberg"]].copy()
    counts = (
        train_rows.groupby(["sza_bin", "has_iceberg"], observed=False)
        .size()
        .unstack(fill_value=0)
        .rename(columns={False: "GT0_original", True: "GTpos_original"})
    )
    kept_counts = kept_rows["sza_bin"].value_counts().rename("GTpos_kept")
    report = counts.join(kept_counts, how="left").fillna(0).astype(int).reset_index()
    report["GT0_kept"] = 0
    report = report[["sza_bin", "GT0_original", "GTpos_original", "GT0_kept", "GTpos_kept"]]
    report.to_csv(os.path.join(args.out_dir, "gt_positive_training_report.csv"), index=False)

    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("# GT-Positive-Only Training Dataset\n\n")
        handle.write("Built from `data/v3_clean`.\n\n")
        handle.write("- `X_train.pkl` / `Y_train.pkl`: only training chips with one or more iceberg pixels.\n")
        handle.write("- Validation and test PKLs are copied unchanged from `data/v3_clean`.\n")
        handle.write("- `split_log.csv` is copied unchanged for provenance.\n\n")
        handle.write(f"Training chips before: `{len(y_train)}`\n")
        handle.write(f"Training chips after: `{int(keep.sum())}`\n")
        handle.write(f"Removed GT0 training chips: `{int((~keep).sum())}`\n")

    print(f"Training chips before: {len(y_train)}")
    print(f"Training chips after : {int(keep.sum())}")
    print(f"Removed GT0 chips    : {int((~keep).sum())}")
    print(report.to_string(index=False))
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
