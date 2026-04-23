#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"
SCRIPTS="${ROOT}/scripts"

PREV_ROOT="${ROOT}/results/model_comparison_20260423_stage1_vs_baseline"
OUT_ROOT="${ROOT}/results/model_comparison_20260423_matched_seed42"

PKL_DIR="${ROOT}/data/v3_clean/train_validate_test"
SPLIT_LOG="${ROOT}/data/v3_clean/split_log.csv"
TEST_INDEX="${OUT_ROOT}/test_index.csv"
CHIPS="${OUT_ROOT}/test_chips"

PY_SYS="/mnt/local/python3.12/bin/python"
PY_VENV="/home/llinkas/.venvs/iceberg-unet312/bin/python"

BASELINE_CKPT="${ROOT}/model/v3_clean_matched_seed42_aug_20260423/best_model.pth"
STAGE1_CKPT="${ROOT}/model/v3_balanced_sza_stage1_matched_seed42_aug_20260423/best_model.pth"

SZA_BINS=("sza_lt65" "sza_65_70" "sza_70_75" "sza_gt75")
MODEL_RUNS=("matched_baseline_v3_clean" "matched_stage1_sza_gt_balance")
MODEL_CKPTS=("${BASELINE_CKPT}" "${STAGE1_CKPT}")

mkdir -p "${OUT_ROOT}" "${ROOT}/logs/model_compare"

if [ ! -f "${TEST_INDEX}" ]; then
    cp "${PREV_ROOT}/test_index.csv" "${TEST_INDEX}"
fi
if [ ! -d "${CHIPS}" ]; then
    ln -sfn "${PREV_ROOT}/test_chips" "${CHIPS}"
fi

echo "Output root: ${OUT_ROOT}"
echo "Baseline ckpt:  ${BASELINE_CKPT}"
echo "Stage-1 ckpt:   ${STAGE1_CKPT}"
"${PY_VENV}" - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
PY

for i in "${!MODEL_RUNS[@]}"; do
    RUN_NAME="${MODEL_RUNS[$i]}"
    CKPT="${MODEL_CKPTS[$i]}"
    TEST_DIR="${OUT_ROOT}/area_comparison/${RUN_NAME}/test"
    EVAL_DIR="${OUT_ROOT}/eval_outputs/${RUN_NAME}"

    echo "============================================================"
    echo "Model run: ${RUN_NAME}"
    echo "Checkpoint: ${CKPT}"
    echo "============================================================"

    for BIN in "${SZA_BINS[@]}"; do
        BIN_CHIPS="${CHIPS}/${BIN}"
        OUT_BASE="${TEST_DIR}/${BIN}"
        N_CHIPS=$(find "${BIN_CHIPS}" -maxdepth 1 -name "*.tif" | wc -l)

        echo "---- ${RUN_NAME} :: ${BIN} (${N_CHIPS} chips) ----"
        "${PY_SYS}" "${SCRIPTS}/threshold_tifs.py" \
            --chips_dir "${BIN_CHIPS}" --out_dir "${OUT_BASE}/TR"
        "${PY_SYS}" "${SCRIPTS}/otsu_threshold_tifs.py" \
            --chips_dir "${BIN_CHIPS}" --out_dir "${OUT_BASE}/OT"
        "${PY_VENV}" "${SCRIPTS}/predict_tifs.py" \
            --checkpoint "${CKPT}" --imgs_dir "${BIN_CHIPS}" \
            --out_dir "${OUT_BASE}/UNet" --save_probs --device cuda
        "${PY_SYS}" "${SCRIPTS}/threshold_probs.py" \
            --probs_dir "${OUT_BASE}/UNet/probs" --out_dir "${OUT_BASE}/UNet_TR"
        "${PY_SYS}" "${SCRIPTS}/otsu_probs.py" \
            --probs_dir "${OUT_BASE}/UNet/probs" --out_dir "${OUT_BASE}/UNet_OT"
        "${PY_SYS}" "${SCRIPTS}/densecrf_tifs.py" \
            --probs_dir "${OUT_BASE}/UNet/probs" --chips_dir "${BIN_CHIPS}" \
            --out_dir "${OUT_BASE}/UNet_CRF"
    done

    echo "Evaluating ${RUN_NAME}..."
    "${PY_SYS}" "${SCRIPTS}/eval_methods.py" \
        --test_dir "${TEST_DIR}" \
        --chips_dir "${CHIPS}" \
        --test_index "${TEST_INDEX}" \
        --pkl_dir "${PKL_DIR}" \
        --out_dir "${EVAL_DIR}"
done

echo "Writing matched comparison summary..."
"${PY_SYS}" "${SCRIPTS}/compare_model_eval.py" \
    --out_root "${OUT_ROOT}" \
    --split_log "${SPLIT_LOG}" \
    --baseline_dir "${OUT_ROOT}/eval_outputs/matched_baseline_v3_clean" \
    --stage1_dir "${OUT_ROOT}/eval_outputs/matched_stage1_sza_gt_balance" \
    --baseline_label "matched_baseline_v3_clean" \
    --stage1_label "matched_stage1_sza_gt_balance" \
    --training_summary "${ROOT}/model/v3_balanced_sza_stage1_matched_seed42_aug_20260423/run_summary.md"

echo "Complete: ${OUT_ROOT}/summary_sheet.md"
