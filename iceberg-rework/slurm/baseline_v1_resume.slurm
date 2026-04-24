#!/bin/bash
# baseline_v1_resume.slurm: resume infer + evaluate against an existing
# run dir whose train stage already finished.
#
# Submit with:
#   sbatch slurm/baseline_v1_resume.slurm                          # uses default RUN_TS
#   RUN_TS=20260424_185158 sbatch slurm/baseline_v1_resume.slurm   # explicit
#
# RUN_TS is the timestamp directory under runs/exp_baseline_v1/ that holds
# the trained checkpoint. If unset, falls back to the original 2026-04-24
# baseline run.

#SBATCH --job-name=ice_baseline_v1_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/baseline/%x_%j.out
#SBATCH --error=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/baseline/%x_%j.err

source "$(dirname "$0")/_common.sh"

RUN_TS="${RUN_TS:-20260424_185158}"
RUN_DIR="${ROOT}/runs/exp_baseline_v1/${RUN_TS}"
CHECKPOINT="${RUN_DIR}/model/best_model.pth"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found at ${CHECKPOINT}" >&2
    echo "       Set RUN_TS=<timestamp_dir> and re-submit." >&2
    exit 2
fi

echo "Resuming run: ${RUN_DIR}"

MANIFEST="${ROOT}/data/v4_clean/manifest.json"
INFER_DIR="${RUN_DIR}/inference"
EVAL_DIR="${RUN_DIR}/evaluation"

# Wipe partial inference output so each method gets a clean per-method audit.
rm -rf "${INFER_DIR}"
mkdir -p "${EVAL_DIR}"

bash "${ROOT}/scripts/run_methods.sh" \
    --manifest   "${MANIFEST}" \
    --checkpoint "${CHECKPOINT}" \
    --out_base   "${INFER_DIR}"

# eval_methods + eval_per_iceberg read the same gpkgs but write to disjoint
# dirs, so run them concurrently. Saves one Python startup + most of the
# wall time of whichever finishes first.
"${PY}" "${ROOT}/scripts/eval_methods.py" \
    --manifest             "${MANIFEST}" \
    --test_dir             "${INFER_DIR}" \
    --out_dir              "${EVAL_DIR}" \
    --skipped_chip_policy  count_as_false_negative &

"${PY}" "${ROOT}/scripts/eval_per_iceberg.py" \
    --manifest      "${MANIFEST}" \
    --test_dir      "${INFER_DIR}" \
    --out_dir       "${RUN_DIR}/per_iceberg" \
    --matcher       hungarian \
    --iou_threshold 0.3 &

wait

echo ""
echo "Resume complete. Outputs:"
echo "  inference:    ${INFER_DIR}"
echo "  evaluation:   ${EVAL_DIR}"
echo "  per-iceberg:  ${RUN_DIR}/per_iceberg"
