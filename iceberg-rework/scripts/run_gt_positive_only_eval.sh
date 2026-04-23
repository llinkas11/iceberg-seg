#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"
SCRIPTS="${ROOT}/scripts"
SOURCE_COMPARISON="${ROOT}/results/model_comparison_20260423_stage1_vs_baseline"
OUT_ROOT="${ROOT}/results/model_comparison_20260423_no_null_vs_baseline"
PKL_DIR="${ROOT}/data/v3_clean/train_validate_test"
SPLIT_LOG="${ROOT}/data/v3_clean/split_log.csv"
TEST_INDEX="${SOURCE_COMPARISON}/test_index.csv"
CHIPS="${SOURCE_COMPARISON}/test_chips"

PY_SYS="/mnt/local/python3.12/bin/python"
PY_VENV="/home/llinkas/.venvs/iceberg-unet312/bin/python"

RUN_NAME="gt_positive_only_train"
CKPT="${ROOT}/model/v3_train_gt_positive_only_aug_20260423/best_model.pth"
TEST_DIR="${OUT_ROOT}/area_comparison/${RUN_NAME}/test"
EVAL_DIR="${OUT_ROOT}/eval_outputs/${RUN_NAME}"
BASELINE_EVAL="${SOURCE_COMPARISON}/eval_outputs/baseline_v3_balanced_aug"
BASELINE_COPY="${OUT_ROOT}/eval_outputs/baseline_v3_balanced_aug"

SZA_BINS=("sza_lt65" "sza_65_70" "sza_70_75" "sza_gt75")

mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/eval_outputs"
cp -a "${BASELINE_EVAL}" "${BASELINE_COPY}"
cp -f "${TEST_INDEX}" "${OUT_ROOT}/test_index.csv"

echo "Output root: ${OUT_ROOT}"
echo "Checkpoint: ${CKPT}"
"${PY_VENV}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

for BIN in "${SZA_BINS[@]}"; do
    BIN_CHIPS="${CHIPS}/${BIN}"
    OUT_BASE="${TEST_DIR}/${BIN}"
    N_CHIPS=$(find "${BIN_CHIPS}" -maxdepth 1 -name "*.tif" | wc -l)

    echo "---------------- ${RUN_NAME} :: ${BIN} (${N_CHIPS} chips) ----------------"
    "${PY_SYS}" "${SCRIPTS}/threshold_tifs.py" \
        --chips_dir "${BIN_CHIPS}" \
        --out_dir "${OUT_BASE}/TR"

    "${PY_SYS}" "${SCRIPTS}/otsu_threshold_tifs.py" \
        --chips_dir "${BIN_CHIPS}" \
        --out_dir "${OUT_BASE}/OT"

    "${PY_VENV}" "${SCRIPTS}/predict_tifs.py" \
        --checkpoint "${CKPT}" \
        --imgs_dir "${BIN_CHIPS}" \
        --out_dir "${OUT_BASE}/UNet" \
        --save_probs \
        --device cuda

    "${PY_SYS}" "${SCRIPTS}/threshold_probs.py" \
        --probs_dir "${OUT_BASE}/UNet/probs" \
        --out_dir "${OUT_BASE}/UNet_TR"

    "${PY_SYS}" "${SCRIPTS}/otsu_probs.py" \
        --probs_dir "${OUT_BASE}/UNet/probs" \
        --out_dir "${OUT_BASE}/UNet_OT"

    "${PY_SYS}" "${SCRIPTS}/densecrf_tifs.py" \
        --probs_dir "${OUT_BASE}/UNet/probs" \
        --chips_dir "${BIN_CHIPS}" \
        --out_dir "${OUT_BASE}/UNet_CRF"
done

echo "Evaluating ${RUN_NAME}..."
"${PY_SYS}" "${SCRIPTS}/eval_methods.py" \
    --test_dir "${TEST_DIR}" \
    --chips_dir "${CHIPS}" \
    --test_index "${TEST_INDEX}" \
    --pkl_dir "${PKL_DIR}" \
    --out_dir "${EVAL_DIR}"

echo "Writing comparison summary..."
"${PY_SYS}" "${ROOT}/scripts/compare_model_eval.py" \
    --out_root "${OUT_ROOT}" \
    --split_log "${SPLIT_LOG}" \
    --baseline_dir "${BASELINE_COPY}" \
    --stage1_dir "${EVAL_DIR}" \
    --baseline_label "baseline_v3_balanced_aug" \
    --stage1_label "${RUN_NAME}" \
    --training_summary "${ROOT}/model/v3_train_gt_positive_only_aug_20260423/run_summary.md"

"${PY_SYS}" "${ROOT}/scripts/make_figure21_iou_gt_positive_comparison.py" \
    --baseline-summary "${BASELINE_COPY}/eval_summary_gt_positive_only.csv" \
    --stage1-summary "${EVAL_DIR}/eval_summary_gt_positive_only.csv" \
    --out-png "${OUT_ROOT}/figure_21_iou_heatmap_gt_positive_only_comparison.png" \
    --out-csv "${OUT_ROOT}/figure_21_iou_heatmap_gt_positive_only_values.csv"

echo "Complete. Summary:"
echo "${OUT_ROOT}/summary_sheet.md"
