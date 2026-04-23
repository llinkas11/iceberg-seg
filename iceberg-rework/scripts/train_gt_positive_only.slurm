#!/bin/bash
#SBATCH --job-name=ice_gtpos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/gt_positive_only/ice_gtpos_%j.out
#SBATCH --error=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/gt_positive_only/ice_gtpos_%j.err

set -euo pipefail

ROOT="/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"
DATA_DIR="${ROOT}/data/v3_train_gt_positive_only"
OUT_DIR="${ROOT}/model/v3_train_gt_positive_only_aug_20260423"
PY="/home/llinkas/.venvs/iceberg-unet312/bin/python"

mkdir -p "${OUT_DIR}" "${ROOT}/logs/gt_positive_only"

"${PY}" "${ROOT}/scripts/train.py" \
    --mode s2 \
    --data_dir "${DATA_DIR}" \
    --out_dir "${OUT_DIR}" \
    --encoder resnet34 \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.0001 \
    --workers 4

"${PY}" "${ROOT}/scripts/summarize_training_run.py" \
    --job-id "${SLURM_JOB_ID}" \
    --run-dir "${OUT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --stdout-log "${ROOT}/logs/gt_positive_only/ice_gtpos_${SLURM_JOB_ID}.out" \
    --stderr-log "${ROOT}/logs/gt_positive_only/ice_gtpos_${SLURM_JOB_ID}.err"
