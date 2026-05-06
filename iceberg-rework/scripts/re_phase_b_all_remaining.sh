#!/bin/bash
# re_phase_b_all_remaining.sh: Phase B six-method sweep for the seven
# distinct Phase A backbones not covered by re_phase_b_with_a{0,1,7b}.sh.
# Backbone list lives in _exps_remaining.sh (sourced below) along with the
# documentation of the empirical 1x3 collapse that reduces the original 15
# remaining backbones to 7 distinct training sets.
#
# Usage (on moosehead, inside iceberg-unet venv):
#   bash scripts/re_phase_b_all_remaining.sh
#
# rsync to moosehead from local:
#   rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/re_phase_b_all_remaining.sh \
#       llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/

set -euo pipefail

# 1. Repo + env
REPO_DIR="${ROOT:-/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework}"
PY="${PY:-/home/llinkas/.venvs/iceberg-unet312/bin/python}"
MANIFEST="${REPO_DIR}/data/v4_clean/manifest.json"

# 2. (exp_id, training_timestamp) pairs sourced from _exps_remaining.sh
source "$(dirname "$0")/_exps_remaining.sh"
EXPS=("${EXPS_REMAINING[@]}")

# 3. Sanity check
[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest missing: $MANIFEST" >&2; exit 2; }

echo "==================================================="
echo "Phase B re-run for ${#EXPS[@]} remaining backbones"
echo "Manifest : $MANIFEST"
echo "==================================================="

# 4. Per-backbone loop: run_methods.sh + eval_per_iceberg.py + eval_methods.py
for entry in "${EXPS[@]}"; do
    EXP_ID="${entry%|*}"
    TS="${entry#*|}"
    CKPT="${REPO_DIR}/runs/${EXP_ID}/${TS}/model/best_model.pth"
    OUT_BASE="${REPO_DIR}/runs/${EXP_ID}/${TS}/re_phase_b_v4_clean"

    echo ""
    echo ">>> ${EXP_ID} (ts=${TS})"
    if [[ ! -f "$CKPT" ]]; then
        echo "    SKIP: checkpoint missing: $CKPT"
        continue
    fi
    if [[ -f "${OUT_BASE}/per_iceberg/eval_per_iceberg_summary.csv" ]]; then
        echo "    SKIP: per_iceberg summary already exists at ${OUT_BASE}"
        continue
    fi

    mkdir -p "$OUT_BASE/test"

    # 5. Six-method inference
    PY="$PY" FORCE=1 bash "${REPO_DIR}/scripts/run_methods.sh" \
        --manifest   "$MANIFEST" \
        --checkpoint "$CKPT" \
        --out_base   "$OUT_BASE/test"

    # 6. Per-iceberg eval (six base methods)
    mkdir -p "${OUT_BASE}/per_iceberg"
    "$PY" "${REPO_DIR}/scripts/eval_per_iceberg.py" \
        --manifest "$MANIFEST" \
        --test_dir "${OUT_BASE}/test" \
        --out_dir  "${OUT_BASE}/per_iceberg" \
        --methods  TR,OT,UNet,UNet_TR,UNet_OT,UNet_CRF

    # 7. Chip-level eval
    mkdir -p "${OUT_BASE}/evaluation"
    "$PY" "${REPO_DIR}/scripts/eval_methods.py" \
        --manifest "$MANIFEST" \
        --test_dir "${OUT_BASE}/test" \
        --out_dir  "${OUT_BASE}/evaluation"
done

echo ""
echo "==================================================="
echo "Phase B remaining backbones complete."
echo "==================================================="
