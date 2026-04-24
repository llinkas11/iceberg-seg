#!/bin/bash
#SBATCH --job-name=ice_baseline_v1_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/baseline/%x_%j.out
#SBATCH --error=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/baseline/%x_%j.err

# Resume infer + evaluate for the already-trained baseline_v1 checkpoint.
# Skips manifest + train; reuses runs/exp_baseline_v1/20260424_185158/model/.

set -euo pipefail
ROOT="/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"
PY="/home/llinkas/.venvs/iceberg-unet312/bin/python"
RUN_DIR="${ROOT}/runs/exp_baseline_v1/20260424_185158"

cd "${ROOT}"
export ICEBERG_EXPERIMENT=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Resuming run: ${RUN_DIR}"
echo ""

# run_experiment.py expects an existing run dir to skip stages cleanly. Since
# its --runs_root + auto-timestamp would create a NEW dir, run the infer +
# evaluate scripts directly against the existing checkpoint.

CHECKPOINT="${RUN_DIR}/model/best_model.pth"
MANIFEST="${ROOT}/data/v4_clean/manifest.json"
INFER_DIR="${RUN_DIR}/inference"
EVAL_DIR="${RUN_DIR}/evaluation"

# Clear any partial inference output from the failed run so the methods do
# not skip work they already wrote and we get a clean per-method audit.
rm -rf "${INFER_DIR}"
mkdir -p "${EVAL_DIR}"

bash "${ROOT}/scripts/run_methods.sh" \
    --manifest   "${MANIFEST}" \
    --checkpoint "${CHECKPOINT}" \
    --out_base   "${INFER_DIR}"

"${PY}" "${ROOT}/scripts/eval_methods.py" \
    --manifest             "${MANIFEST}" \
    --test_dir             "${INFER_DIR}" \
    --out_dir              "${EVAL_DIR}" \
    --skipped_chip_policy  count_as_false_negative

"${PY}" "${ROOT}/scripts/eval_per_iceberg.py" \
    --manifest      "${MANIFEST}" \
    --test_dir      "${INFER_DIR}" \
    --out_dir       "${RUN_DIR}/per_iceberg" \
    --matcher       hungarian \
    --iou_threshold 0.3

echo ""
echo "Resume complete. Outputs:"
echo "  inference:    ${INFER_DIR}"
echo "  evaluation:   ${EVAL_DIR}"
echo "  per-iceberg:  ${RUN_DIR}/per_iceberg"
