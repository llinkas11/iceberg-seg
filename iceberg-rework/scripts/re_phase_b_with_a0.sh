#!/bin/bash
# re_phase_b_with_a0.sh: re-run Phase B's six-method sweep using the A0
# checkpoint (Fisser preprocessing, no GT-zero chips, no IC mask) on the
# v4_clean test split for all four SZA bins. Companion to
# re_phase_b_with_a1.sh; together they populate T3 (A0 vs A1 Phase B).
#
# Why: original Phase B was lt65-only (the published headline 8.18 m MAE).
# To compare A0 vs A1 fairly across all SZA bins, A0 needs the same
# v4_clean test-split sweep that A1 just got.
#
# Usage (on moosehead, inside iceberg-unet venv):
#   bash scripts/re_phase_b_with_a0.sh
#
# rsync to moosehead from local:
#   rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/re_phase_b_with_a0.sh \
#       llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/

set -euo pipefail

# 1. Repo + env paths
REPO_DIR="${ROOT:-/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework}"
PY="${PY:-/home/llinkas/.venvs/iceberg-unet312/bin/python}"
MANIFEST="${REPO_DIR}/data/v4_clean/manifest.json"
EXP_ID="exp_A0_fisser_lt65_original"
TS="20260428_094028"
CKPT="${REPO_DIR}/runs/${EXP_ID}/${TS}/model/best_model.pth"
OUT_BASE="${REPO_DIR}/runs/${EXP_ID}/${TS}/re_phase_b_v4_clean"

# 2. Sanity check
[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest missing: $MANIFEST" >&2; exit 2; }
[[ -f "$CKPT" ]]     || { echo "ERROR: checkpoint missing: $CKPT" >&2; exit 2; }

mkdir -p "$OUT_BASE/test"

# 3. Inference: six methods x four bins.
#    FORCE=1 routes around run_methods.sh's dataset-drift guard. A0 trained
#    on v4_raw_lt65; eval here uses v4_clean to match the chip set used in
#    T1 and the published Phase B numbers.
echo "==================================================="
echo "Phase B re-run with A0 backbone"
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

# 5. Chip-level eval (same six methods)
EVAL_DIR="${OUT_BASE}/evaluation"
mkdir -p "$EVAL_DIR"
"$PY" "${REPO_DIR}/scripts/eval_methods.py" \
    --manifest "$MANIFEST" \
    --test_dir "${OUT_BASE}/test" \
    --out_dir  "$EVAL_DIR"

echo ""
echo "==================================================="
echo "A0 Phase B re-run complete. Outputs:"
echo "  per_iceberg : ${PER_ICE_DIR}/eval_per_iceberg_summary.csv"
echo "  evaluation  : ${EVAL_DIR}/eval_summary.csv"
echo "==================================================="
