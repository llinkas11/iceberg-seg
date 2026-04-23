#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"
SCRIPTS="${ROOT}/scripts"
OUT_ROOT="${ROOT}/results/model_comparison_20260423_stage1_vs_baseline"
PKL_DIR="${ROOT}/data/v3_clean/train_validate_test"
SPLIT_LOG="${ROOT}/data/v3_clean/split_log.csv"
FISSER_INDEX="/mnt/research/v.gomezgilyaspik/students/smishra/rework/reference/fisser_index.csv"
TEST_INDEX="${OUT_ROOT}/test_index.csv"
CHIPS="${OUT_ROOT}/test_chips"

PY_SYS="/mnt/local/python3.12/bin/python"
PY_VENV="/home/llinkas/.venvs/iceberg-unet312/bin/python"

BASELINE_CKPT="/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/v3_rework/model/v3_balanced_aug/best_model.pth"
STAGE1_CKPT="${ROOT}/model/v3_balanced_sza_stage1_aug_20260423/best_model.pth"

SZA_BINS=("sza_lt65" "sza_65_70" "sza_70_75" "sza_gt75")
MODEL_RUNS=("baseline_v3_balanced_aug" "stage1_sza_gt_balance")
MODEL_CKPTS=("${BASELINE_CKPT}" "${STAGE1_CKPT}")

mkdir -p "${OUT_ROOT}" "${ROOT}/logs/model_compare"

echo "Output root: ${OUT_ROOT}"
echo "System python: $(${PY_SYS} -V)"
echo "Venv python: $(${PY_VENV} -V)"
echo "CUDA check:"
"${PY_VENV}" - <<'PY'
import torch
print("  torch:", torch.__version__)
print("  cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device:", torch.cuda.get_device_name(0))
PY

echo "Building test_index from copied v3_clean split_log..."
"${PY_SYS}" - <<PY
import os
import pickle

import numpy as np
import pandas as pd
import rasterio as rio

split_log = "${SPLIT_LOG}"
fisser_index_path = "${FISSER_INDEX}"
pkl_dir = "${PKL_DIR}"
out_path = "${TEST_INDEX}"

log = pd.read_csv(split_log)
test = log[log["split"].eq("test")].copy()
test = test.sort_values("pkl_position").reset_index(drop=True)

fisser = pd.read_csv(fisser_index_path)
fisser_map = dict(zip(fisser["global_index"].astype(int), fisser["tif_path"]))

with open(os.path.join(pkl_dir, "x_test.pkl"), "rb") as handle:
    x_test = pickle.load(handle).astype(np.float32)

rows = []
for _, row in test.iterrows():
    pkl_position = int(row["pkl_position"])
    stem = str(row["stem"])
    sza_bin = row["sza_bin"]
    source = str(row.get("source", ""))

    if source == "fisser" or sza_bin == "sza_lt65" or stem.startswith("fisser_"):
        global_index = int(stem.split("_")[1])
        tif_path = fisser_map[global_index]
        chip_stem = os.path.splitext(os.path.basename(tif_path))[0]
    else:
        global_index = pkl_position
        tif_path = row["tif_path"]
        chip_stem = row["chip_stem"]

    if not isinstance(tif_path, str) or not os.path.exists(tif_path):
        raise FileNotFoundError(f"Missing tif for pkl_position={pkl_position}: {tif_path}")

    if len(rows) < 8 or pkl_position in {0, 48, 49, 52}:
        with rio.open(tif_path) as src:
            arr = np.nan_to_num(src.read().astype(np.float32)[:3])
        target = np.nan_to_num(x_test[pkl_position])
        mae = float(np.abs(arr - target).mean())
        if mae > 1e-6:
            raise ValueError(
                f"Pixel mismatch at pkl_position={pkl_position}, stem={stem}, mae={mae}"
            )

    rows.append({
        "pkl_position": pkl_position,
        "global_index": global_index,
        "sza_bin": sza_bin,
        "chip_stem": chip_stem,
        "tif_path": tif_path,
    })

out = pd.DataFrame(rows).sort_values("pkl_position").reset_index(drop=True)
if len(out) != len(x_test):
    raise ValueError(f"test_index rows {len(out)} != x_test rows {len(x_test)}")

os.makedirs(os.path.dirname(out_path), exist_ok=True)
out.to_csv(out_path, index=False)
print(out["sza_bin"].value_counts().sort_index().to_string())
print(f"Saved {len(out)} rows -> {out_path}")
PY

echo "Preparing symlinked test chip directories..."
"${PY_SYS}" "${SCRIPTS}/prepare_test_chips_dir.py" \
    --test_index "${TEST_INDEX}" \
    --out_dir "${CHIPS}"

for i in "${!MODEL_RUNS[@]}"; do
    RUN_NAME="${MODEL_RUNS[$i]}"
    CKPT="${MODEL_CKPTS[$i]}"
    TEST_DIR="${OUT_ROOT}/area_comparison/${RUN_NAME}/test"
    EVAL_DIR="${OUT_ROOT}/eval_outputs/${RUN_NAME}"

    echo "============================================================"
    echo "Model run: ${RUN_NAME}"
    echo "Checkpoint: ${CKPT}"
    echo "Test dir: ${TEST_DIR}"
    echo "Eval dir: ${EVAL_DIR}"
    echo "============================================================"

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
done

echo "Writing comparison summary..."
"${PY_SYS}" "${ROOT}/scripts/compare_model_eval.py" \
    --out_root "${OUT_ROOT}" \
    --split_log "${SPLIT_LOG}" \
    --baseline_dir "${OUT_ROOT}/eval_outputs/baseline_v3_balanced_aug" \
    --stage1_dir "${OUT_ROOT}/eval_outputs/stage1_sza_gt_balance" \
    --baseline_label "baseline_v3_balanced_aug" \
    --stage1_label "stage1_sza_gt_balance" \
    --training_summary "${ROOT}/model/v3_balanced_sza_stage1_aug_20260423/run_summary.md"

echo "Complete. Summary:"
echo "${OUT_ROOT}/summary_sheet.md"
