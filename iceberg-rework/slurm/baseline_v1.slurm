#!/bin/bash
# baseline_v1.slurm: train + infer + evaluate the canonical baseline.
#
# Runs scripts/run_experiment.py with --exp exp_baseline_v1 under
# ICEBERG_EXPERIMENT=1 so train.py's seed guard fires and the resulting
# training_config.json records the experiment-mode flag.
#
# Submit with:
#   sbatch slurm/baseline_v1.slurm
#
# Outputs land at iceberg-rework/runs/exp_baseline_v1/<timestamp>/.

#SBATCH --job-name=ice_baseline_v1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/baseline/%x_%j.out
#SBATCH --error=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/baseline/%x_%j.err

set -euo pipefail

ROOT="/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"
PY="/home/llinkas/.venvs/iceberg-unet312/bin/python"
LOG_DIR="${ROOT}/logs/baseline"

mkdir -p "${LOG_DIR}"

cd "${ROOT}"

export ICEBERG_EXPERIMENT=1
export PY

echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "GPUs      : $(nvidia-smi --query-gpu=name --format=csv,noheader || echo none)"
echo "ICEBERG_EXPERIMENT=${ICEBERG_EXPERIMENT}"
echo ""

"${PY}" "${ROOT}/scripts/run_experiment.py" \
    --exp    exp_baseline_v1 \
    --stages manifest,train,infer,evaluate

echo ""
echo "baseline_v1 run complete. Find outputs under runs/exp_baseline_v1/."
