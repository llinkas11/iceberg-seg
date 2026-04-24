#!/bin/bash
# _common.sh: shared bash preamble for iceberg-rework slurm scripts.
#
# Source this from any slurm wrapper after its #SBATCH header. Sets:
#   ROOT  /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework
#   PY    /home/llinkas/.venvs/iceberg-unet312/bin/python
#   ICEBERG_EXPERIMENT=1   so train.py's seed guard fires
# Also: cd into ROOT and echo a one-line job banner.
#
# Slurm requires #SBATCH directives at the top of the submitted file, so
# this file cannot replace those; only the bash logic below them.

set -euo pipefail

ROOT="/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"
PY="/home/llinkas/.venvs/iceberg-unet312/bin/python"

cd "${ROOT}"
export ICEBERG_EXPERIMENT=1
export PY ROOT

echo "Job ID    : ${SLURM_JOB_ID}"
echo "Job name  : ${SLURM_JOB_NAME}"
echo "Node      : $(hostname)"
echo "GPUs      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "ICEBERG_EXPERIMENT=${ICEBERG_EXPERIMENT}"
echo ""
