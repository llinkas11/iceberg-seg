"""
Audit the 40m filter and IC-mask impact on Fisser lt65 chips.

Inputs:
  - data/v4_clean/split_log.csv (chip_stem -> split, source, ic_masked, n_icebergs)
  - smishra/rework/data/fisser_original/train_validate_test/Y_*.pkl (pre-40m)
  - data/fisser_filtered/Y_*.pkl (post-40m)

Maps chip_stem 'fisser_NNNN' (N in [0,398)) to its position in the original
Fisser pkl scheme: train [0,323), val [323,362), test [362,398).
"""

import csv
import pickle
import numpy as np
from skimage.measure import label

RAW = "/mnt/research/v.gomezgilyaspik/students/smishra/rework/data/fisser_original/train_validate_test"
FILT = "data/fisser_filtered"
MIN_AREA_PX = 16

# Original Fisser pkl scheme: train 323, val 39, test 36
TRAIN_N = 323
VAL_N = 39


def locate_raw(stem_idx):
    """Return (pkl_name, position) for a fisser chip_stem index in original Fisser pkls."""
    if stem_idx < TRAIN_N:
        return ("Y_train.pkl", stem_idx)
    if stem_idx < TRAIN_N + VAL_N:
        return ("Y_validation.pkl", stem_idx - TRAIN_N)
    return ("y_test.pkl", stem_idx - TRAIN_N - VAL_N)


# 1. Load all six pkls once
raw_pkls = {n: pickle.load(open(f"{RAW}/{n}", "rb"))
            for n in ["Y_train.pkl", "Y_validation.pkl", "y_test.pkl"]}
filt_pkls = {n: pickle.load(open(f"{FILT}/{n}", "rb"))
             for n in ["Y_train.pkl", "Y_validation.pkl", "y_test.pkl"]}

# 2. Identify Fisser lt65 chips in v4_clean
rows = list(csv.DictReader(open("data/v4_clean/split_log.csv")))
fisser_lt65 = [r for r in rows if r["source"] == "fisser" and r["sza_bin"] == "sza_lt65"]
print(f"Fisser lt65 chips total: {len(fisser_lt65)}")

# 3. Per-chip component count before vs after the 40m filter
totals = {"before": 0, "after": 0, "chips_changed": 0, "chips_zeroed": 0}
sample_changed = []
sample_zeroed = []

for r in fisser_lt65:
    stem_idx = int(r["chip_stem"].split("_")[-1])
    pkl_name, pos = locate_raw(stem_idx)

    m_raw = (raw_pkls[pkl_name][pos] > 0).astype(np.uint8)
    m_filt = (filt_pkls[pkl_name][pos] > 0).astype(np.uint8)

    n_raw = int(label(m_raw, connectivity=2).max())
    n_filt = int(label(m_filt, connectivity=2).max())

    totals["before"] += n_raw
    totals["after"] += n_filt
    if n_raw != n_filt:
        totals["chips_changed"] += 1
        if len(sample_changed) < 5:
            sample_changed.append((r["chip_stem"], n_raw, n_filt))
    if n_raw > 0 and n_filt == 0:
        totals["chips_zeroed"] += 1
        sample_zeroed.append((r["chip_stem"], n_raw))

# 4. Print 40m summary
print()
print("=== 40 m filter impact on Fisser lt65 (all splits) ===")
print(f"  Components before filter: {totals['before']}")
print(f"  Components after  filter: {totals['after']}")
removed = totals["before"] - totals["after"]
pct = removed / max(1, totals["before"]) * 100
print(f"  Components removed:       {removed} ({pct:.1f}%)")
print(f"  Chips with >=1 component removed: {totals['chips_changed']} / {len(fisser_lt65)}")
print(f"  Chips reduced to 0 icebergs:      {totals['chips_zeroed']}")
if sample_changed:
    print(f"  Sample changed (stem, before, after): {sample_changed}")
if sample_zeroed:
    print(f"  Sample zeroed (stem, before): {sample_zeroed[:5]}")

# 5. IC-mask summary (training chips only)
n_train = sum(1 for r in fisser_lt65 if r["split"] == "train")
n_train_ic = sum(1 for r in fisser_lt65 if r["split"] == "train" and r["ic_masked"] == "True")
print()
print("=== IC-mask impact on Fisser lt65 ===")
print(f"  Training Fisser lt65 chips: {n_train}")
print(f"  Training chips IC-masked:   {n_train_ic} ({n_train_ic/max(1,n_train)*100:.1f}%)")
print(f"  Val/test Fisser lt65 chips: {len(fisser_lt65) - n_train} (never masked)")
