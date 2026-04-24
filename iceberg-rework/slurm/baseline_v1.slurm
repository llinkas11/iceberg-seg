#!/bin/bash
# baseline_v1.slurm: train + infer + evaluate the canonical baseline.
#
# Runs scripts/run_experiment.py with --exp exp_baseline_v1 under
# ICEBERG_EXPERIMENT=1. Output dir: runs/exp_baseline_v1/<timestamp>/.
#
# If the run fails mid-pipeline, the most recent run dir is printed in the
# log; resume via slurm/baseline_v1_resume.slurm with RUN_TS=<that-stamp>.

#SBATCH --job-name=ice_baseline_v1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/baseline/%x_%j.out
#SBATCH --error=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/baseline/%x_%j.err

source "$(dirname "$0")/_common.sh"
mkdir -p "${ROOT}/logs/baseline"

set +e
"${PY}" "${ROOT}/scripts/run_experiment.py" \
    --exp    exp_baseline_v1 \
    --stages manifest,train,infer,evaluate
status=$?
set -e

if [[ $status -ne 0 ]]; then
    LATEST=$(ls -1d "${ROOT}/runs/exp_baseline_v1"/2* 2>/dev/null | tail -1)
    echo ""
    echo "baseline_v1 failed (exit ${status}). Most recent run dir: ${LATEST:-<none>}"
    echo "To resume from after a successful train, run:"
    echo "  RUN_TS=$(basename "${LATEST:-RUN_TS_HERE}") sbatch ${ROOT}/slurm/baseline_v1_resume.slurm"
    exit $status
fi

echo ""
echo "baseline_v1 complete. Outputs under runs/exp_baseline_v1/."
