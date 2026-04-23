#!/bin/bash
#SBATCH --job-name=ice_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/matched/%x_%j.out
#SBATCH --error=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/matched/%x_%j.err

set -euo pipefail

: "${DATA_SUBDIR:?set DATA_SUBDIR (dir under iceberg-rework/data/)}"
: "${OUT_SUBDIR:?set OUT_SUBDIR (dir under iceberg-rework/model/)}"
: "${SEED:=42}"
: "${EPOCHS:=100}"
: "${BATCH:=16}"
: "${LR:=0.0001}"
: "${WORKERS:=4}"

ROOT="/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"
DATA_DIR="${ROOT}/data/${DATA_SUBDIR}"
OUT_DIR="${ROOT}/model/${OUT_SUBDIR}"
LOG_DIR="${ROOT}/logs/matched"
PY="/home/llinkas/.venvs/iceberg-unet312/bin/python"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

"${PY}" "${ROOT}/scripts/train.py" \
    --mode s2 \
    --data_dir "${DATA_DIR}" \
    --out_dir "${OUT_DIR}" \
    --encoder resnet34 \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --lr "${LR}" \
    --workers "${WORKERS}" \
    --seed "${SEED}"

"${PY}" "${ROOT}/scripts/summarize_training_run.py" \
    --job-id "${SLURM_JOB_ID}" \
    --run-dir "${OUT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --stdout-log "${LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
    --stderr-log "${LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
