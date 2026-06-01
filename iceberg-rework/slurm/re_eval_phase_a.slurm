#!/bin/bash
# re_eval_phase_a.slurm: Slurm wrapper for re_eval_phase_a_all_sza.sh.
#
# Submits one job that loops all 10 Phase A checkpoints over all 4 SZA bins
# under v4_clean. UNet-only, so wallclock is dominated by 10 * 4 * 57 chips
# of GPU inference (~20-30 minutes total). 4h budget for headroom.
#
# Usage:
#   sbatch slurm/re_eval_phase_a.slurm
#
# rsync to moosehead from local:
#   rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/slurm/re_eval_phase_a.slurm \
#       llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/slurm/

#SBATCH --job-name=ice_re_eval_A
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/exp/%x_%j.out
#SBATCH --error=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/exp/%x_%j.err

# 1. Shared preamble (sets ROOT, PY, set -euo pipefail, prints banner)
source /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/slurm/_common.sh

mkdir -p "${ROOT}/logs/exp"

# 2. Run the loop
bash "${ROOT}/scripts/re_eval_phase_a_all_sza.sh"
