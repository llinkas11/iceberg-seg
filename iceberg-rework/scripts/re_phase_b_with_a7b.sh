#!/bin/bash
# re_phase_b_with_a7b.sh: re-run Phase B's six-method sweep using the A7b
# checkpoint (A1 manifest + size oversample + augmentation, the new higher-SZA
# champion from T1b) on the v4_clean test split for all four SZA bins.
# Companion to re_phase_b_with_a0.sh / re_phase_b_with_a1.sh; together they
# populate the T3 backbone comparison and feed T4's pipeline recommendation.
#
# Why A7b: T1b / T2-revised showed A7b (= A8b == A9b by collapse) wins the
# higher-SZA aggregate (mean IoU 0.531, mean MAE 27.24 m) over A1 (0.499 /
# 28.01 m) and A0 (0.490 / 33.33 m). Phase B has not yet been run on A7b;
# this populates the missing row so T3's cross-bin recommendation can move
# from A1 + UNet_CRF (current) to A7b + UNet_CRF if the new backbone agrees.
#
# Usage (on moosehead, inside iceberg-unet venv):
#   bash scripts/re_phase_b_with_a7b.sh
#
# rsync to moosehead from local:
#   rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/re_phase_b_with_a7b.sh \
#       llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/

set -euo pipefail

# 1. Repo + env paths
REPO_DIR="${ROOT:-/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework}"
PY="${PY:-/home/llinkas/.venvs/iceberg-unet312/bin/python}"
MANIFEST="${REPO_DIR}/data/v4_clean/manifest.json"
EXP_ID="exp_A7b_a1_size_aug"
TS="20260505_211329"
CKPT="${REPO_DIR}/runs/${EXP_ID}/${TS}/model/best_model.pth"
OUT_BASE="${REPO_DIR}/runs/${EXP_ID}/${TS}/re_phase_b_v4_clean"

# 2. Sanity check
[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest missing: $MANIFEST" >&2; exit 2; }
[[ -f "$CKPT" ]]     || { echo "ERROR: checkpoint missing: $CKPT" >&2; exit 2; }

mkdir -p "$OUT_BASE/test"

# 3. Inference: six methods x four bins.
#    FORCE=1 routes around run_methods.sh's dataset-drift guard. A7b trained
#    on v4_raw_lt65_plus_nulls; eval here uses v4_clean to match the chip set
#    used in T1, T1b, and T3.
echo "==================================================="
echo "Phase B re-run with A7b backbone"
echo "Checkpoint : $CKPT"
echo "Manifest   : $MANIFEST"
echo "Out base   : $OUT_BASE"
echo "==================================================="

PY="$PY" FORCE=1 bash "${REPO_DIR}/scripts/run_methods.sh" \
    --manifest   "$MANIFEST" \
    --checkpoint "$CKPT" \
    --out_base   "$OUT_BASE/test"

# 4. Per-iceberg eval across all six methods, all four bins
PER_ICE_DIR="${OUT_BASE}/per_iceberg"
mkdir -p "$PER_ICE_DIR"
"$PY" "${REPO_DIR}/scripts/eval_per_iceberg.py" \
    --manifest "$MANIFEST" \
    --test_dir "${OUT_BASE}/test" \
    --out_dir  "$PER_ICE_DIR" \
    --methods  TR,OT,UNet,UNet_TR,UNet_OT,UNet_CRF

# 5. Chip-level eval
EVAL_DIR="${OUT_BASE}/evaluation"
mkdir -p "$EVAL_DIR"
"$PY" "${REPO_DIR}/scripts/eval_methods.py" \
    --manifest "$MANIFEST" \
    --test_dir "${OUT_BASE}/test" \
    --out_dir  "$EVAL_DIR"

echo ""
echo "==================================================="
echo "A7b Phase B re-run complete. Outputs:"
echo "  per_iceberg : ${PER_ICE_DIR}/eval_per_iceberg_summary.csv"
echo "  evaluation  : ${EVAL_DIR}/eval_summary.csv"
echo "==================================================="
