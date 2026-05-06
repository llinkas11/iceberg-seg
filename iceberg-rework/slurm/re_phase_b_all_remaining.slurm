#!/bin/bash
# re_phase_b_all_remaining.slurm: Slurm wrapper for re_phase_b_all_remaining.sh.
#
# Loops 7 distinct backbones; each Phase B sweep takes ~7 min (A0/A1/A7b
# priors), so total estimate ~50-60 min. 4h budget for headroom.
#
# Usage:
#   sbatch slurm/re_phase_b_all_remaining.slurm
#
# rsync to moosehead from local:
#   rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/slurm/re_phase_b_all_remaining.slurm \
#       llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/slurm/

#SBATCH --job-name=ice_phase_b_remain
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
bash "${ROOT}/scripts/re_phase_b_all_remaining.sh"
