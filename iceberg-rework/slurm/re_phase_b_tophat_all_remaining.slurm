#!/bin/bash
# re_phase_b_tophat_all_remaining.slurm: Slurm wrapper for the TH addition
# across all seven remaining backbones. Submit with --dependency=afterok:<phase_b_jobid>
# so it waits for re_phase_b_all_remaining to succeed.
#
# Per-backbone TH cost ~10 min; seven backbones ~70 min total. 3h budget.
# CPU partition (no GPU needed for top-hat or eval scripts).
#
# Usage:
#   sbatch --dependency=afterok:<jobid> slurm/re_phase_b_tophat_all_remaining.slurm
#
# rsync to moosehead from local:
#   rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/slurm/re_phase_b_tophat_all_remaining.slurm \
#       llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/slurm/

#SBATCH --job-name=ice_phase_b_th_remain
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/exp/%x_%j.out
#SBATCH --error=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/exp/%x_%j.err

# 1. Shared preamble (sets ROOT, PY, set -euo pipefail, prints banner)
source /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/slurm/_common.sh

mkdir -p "${ROOT}/logs/exp"

# 2. Run the TH-only loop
bash "${ROOT}/scripts/re_phase_b_tophat_all_remaining.sh"
