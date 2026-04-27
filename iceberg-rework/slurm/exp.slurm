#!/bin/bash
# exp.slurm: generic experiment runner. Submit any experiment in
# configs/experiments/ by passing EXP_ID via --export.
#
# Usage:
#   EXP_ID=exp_ablation_no_nulls sbatch slurm/exp.slurm
#   EXP_ID=exp_A4_our_lt65_plus_nulls_aug sbatch slurm/exp.slurm
#
# Or with --export so the env var only lives for the job:
#   sbatch --export=EXP_ID=exp_B2_method_unet slurm/exp.slurm
#
# Optional env vars:
#   STAGES   comma-separated stages (default: manifest,train,infer,evaluate)
#   SEED     training seed (default 42; set in baseline_v1.yaml)
#
# Logs land at logs/<EXP_ID>/<job_name>_<job_id>.{out,err}.

#SBATCH --job-name=ice_exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/exp/%x_%j.out
#SBATCH --error=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/exp/%x_%j.err

# Slurm copies this script to /var/spool/slurmd/<job>/, so $(dirname "$0")
# would not find _common.sh. Source by absolute path instead.
source /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/slurm/_common.sh

if [[ -z "${EXP_ID:-}" ]]; then
    echo "ERROR: EXP_ID env var is required." >&2
    echo "       Submit with: EXP_ID=exp_<id> sbatch slurm/exp.slurm" >&2
    echo "       Or:          sbatch --export=EXP_ID=exp_<id> slurm/exp.slurm" >&2
    exit 2
fi

EXP_YAML="${ROOT}/configs/experiments/${EXP_ID}.yaml"
if [[ ! -f "${EXP_YAML}" ]]; then
    echo "ERROR: experiment yaml not found: ${EXP_YAML}" >&2
    exit 2
fi

STAGES="${STAGES:-manifest,train,infer,evaluate}"
mkdir -p "${ROOT}/logs/exp" "${ROOT}/runs/${EXP_ID}"

echo "Experiment: ${EXP_ID}"
echo "Stages    : ${STAGES}"
echo ""

set +e
"${PY}" "${ROOT}/scripts/run_experiment.py" \
    --exp    "${EXP_ID}" \
    --stages "${STAGES}"
status=$?
set -e

if [[ $status -ne 0 ]]; then
    LATEST=$(ls -1d "${ROOT}/runs/${EXP_ID}"/2* 2>/dev/null | tail -1)
    echo ""
    echo "${EXP_ID} failed (exit ${status}). Most recent run dir: ${LATEST:-<none>}"
    if [[ -n "${LATEST:-}" ]]; then
        echo "To resume from after a successful train, run:"
        echo "  EXP_ID=${EXP_ID} RUN_TS=$(basename "${LATEST}") STAGES=infer,evaluate sbatch ${ROOT}/slurm/exp.slurm"
    fi
    exit $status
fi

echo ""
echo "${EXP_ID} complete. Outputs under runs/${EXP_ID}/."
