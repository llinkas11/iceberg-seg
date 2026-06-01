#!/bin/bash
# re_eval_phase_a_a1variants.slurm: Slurm wrapper for the A1-anchored
# re-eval (8 experiments x 4 SZA bins, UNet only). GPU partition because
# predict_tifs.py runs UNet inference. ~14 min wallclock expected (mirrors
# Slurm 60293 which covered A0..A9).
#
# Usage:
#   sbatch slurm/re_eval_phase_a_a1variants.slurm
#
# rsync to moosehead from local:
#   rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/slurm/re_eval_phase_a_a1variants.slurm \
#       llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/slurm/

#SBATCH --job-name=ice_re_eval_A_a1var
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
bash "${ROOT}/scripts/re_eval_phase_a_a1variants.sh"
