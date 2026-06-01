#!/bin/bash
# re_eval_phase_a_all_sza.sh: re-evaluate every Phase A UNet++ checkpoint
# (A0..A9) on the v4_clean test split for ALL four SZA bins, including bins
# the model never saw at training time. UNet method only (no TR/OT/CRF) to
# keep Slurm wallclock manageable.
#
# Why: original Phase A evals were lt65-only because the training manifests
# (v4_raw_lt65*, v4_clean_lt65*) contained no higher-SZA chips. To populate
# a per-bin x per-experiment comparison table for the paper we need the
# trained checkpoints scored against sza_65_70, sza_70_75, sza_gt75.
#
# Manifest choice: v4_clean is the only all-SZA manifest with the same
# 57-chip-per-bin stratified test split used by baseline_v1 and Phase B.
# Using one manifest across all 10 checkpoints makes cells directly
# comparable (same test chips for every experiment in every bin).
#
# Note: this routes around the dataset-drift guard in run_methods.sh by
# calling predict_tifs.py and eval_per_iceberg.py directly. The guard exists
# to prevent accidental cross-manifest inference; here it is intentional.
#
# Usage (on moosehead, inside iceberg-unet conda env):
#   bash scripts/re_eval_phase_a_all_sza.sh
#
# rsync to moosehead from local:
#   rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/re_eval_phase_a_all_sza.sh \
#       llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/

set -euo pipefail

# 1. Repo + env paths
REPO_DIR="${ROOT:-/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework}"
PY="${PY:-/home/llinkas/.venvs/iceberg-unet312/bin/python}"
MANIFEST="${REPO_DIR}/data/v4_clean/manifest.json"
TEST_CHIPS_ROOT="${REPO_DIR}/data/v4_clean/test_chips"
BINS=(sza_lt65 sza_65_70 sza_70_75 sza_gt75)

# 2. Audit-confirmed (exp_id, checkpoint_timestamp) pairs.
#    Hard-coded so the eval is reviewable and reproducible; auto-picking the
#    latest timestamp would silently shift if a stray training run lands here.
declare -a EXPS=(
    "exp_A0_fisser_lt65_original|20260428_094028"
    "exp_A1_fisser_lt65_plus_nulls|20260429_234146"
    "exp_A2_our_lt65|20260428_094654"
    "exp_A3_our_lt65_plus_nulls|20260429_234518"
    "exp_A4_our_lt65_plus_nulls_aug|20260429_234835"
    "exp_A5_our_lt65_plus_nulls_aug_2pos|20260430_001810"
    "exp_A6_our_lt65_plus_nulls_aug_adaptive|20260430_002155"
    "exp_A7_our_lt65_plus_nulls_aug_size|20260430_002545"
    "exp_A8_our_lt65_plus_nulls_aug_2pos_size|20260430_002942"
    "exp_A9_our_lt65_plus_nulls_aug_adaptive_size|20260430_003348"
)

# 3. Sanity check inputs once before the loop
[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest missing: $MANIFEST" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "ERROR: python missing: $PY" >&2; exit 2; }
for B in "${BINS[@]}"; do
    n=$(ls "$TEST_CHIPS_ROOT/$B"/*.tif 2>/dev/null | wc -l | tr -d ' ')
    [[ "$n" -gt 0 ]] || { echo "ERROR: no chips in $TEST_CHIPS_ROOT/$B" >&2; exit 2; }
done

echo "==================================================="
echo "Phase A re-eval (all SZA, UNet only, manifest=v4_clean)"
echo "Experiments  : ${#EXPS[@]}"
echo "Bins         : ${BINS[*]}"
echo "==================================================="

# 4. Outer loop: one Phase A checkpoint at a time
for entry in "${EXPS[@]}"; do
    EXP_ID="${entry%|*}"
    TS="${entry#*|}"
    CKPT="${REPO_DIR}/runs/${EXP_ID}/${TS}/model/best_model.pth"
    OUT_BASE="${REPO_DIR}/runs/${EXP_ID}/${TS}/re_eval_v4_clean"

    echo ""
    echo ">>> ${EXP_ID} (ts=${TS})"
    if [[ ! -f "$CKPT" ]]; then
        echo "    SKIP: checkpoint missing: $CKPT"
        continue
    fi

    # 5. Inference loop: one bin at a time, UNet only
    for B in "${BINS[@]}"; do
        OUT_UNET="${OUT_BASE}/test/${B}/UNet"
        if [[ -f "${OUT_UNET}/all_icebergs.gpkg" ]]; then
            echo "    [${B}] UNet already done, skip"
            continue
        fi
        mkdir -p "$OUT_UNET"
        echo "    [${B}] predict_tifs.py -> ${OUT_UNET}"
        "$PY" "${REPO_DIR}/scripts/predict_tifs.py" \
            --checkpoint "$CKPT" \
            --imgs_dir   "${TEST_CHIPS_ROOT}/${B}" \
            --out_dir    "$OUT_UNET" \
            --save_probs
    done

    # 6. Per-iceberg evaluation (Hungarian matching) against v4_clean GT
    PER_ICE_DIR="${OUT_BASE}/per_iceberg"
    mkdir -p "$PER_ICE_DIR"
    echo "    eval_per_iceberg.py -> ${PER_ICE_DIR}"
    "$PY" "${REPO_DIR}/scripts/eval_per_iceberg.py" \
        --manifest "$MANIFEST" \
        --test_dir "${OUT_BASE}/test" \
        --out_dir  "$PER_ICE_DIR" \
        --methods  UNet
done

echo ""
echo "==================================================="
echo "Re-eval complete. Per-experiment per_iceberg CSVs at:"
echo "  ${REPO_DIR}/runs/exp_A*/<ts>/re_eval_v4_clean/per_iceberg/eval_per_iceberg_summary.csv"
echo "==================================================="
