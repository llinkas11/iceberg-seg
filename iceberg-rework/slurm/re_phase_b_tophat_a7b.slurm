#!/bin/bash
# re_phase_b_tophat_a7b.slurm: Slurm wrapper for re_phase_b_tophat_a7b.sh.
#
# Adds the six top-hat companions to the A7b Phase B re-run and re-evaluates
# per-iceberg + chip-level metrics across all 12 methods. CPU-bound (top-hat
# + eval), so this targets `main` rather than `gpu`. 1h budget; expected
# 5-10 min.
#
# Usage:
#   sbatch slurm/re_phase_b_tophat_a7b.slurm
#
# rsync to moosehead from local:
#   rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/slurm/re_phase_b_tophat_a7b.slurm \
#       llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/slurm/

#SBATCH --job-name=ice_phase_b_th_a7b
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/exp/%x_%j.out
#SBATCH --error=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/exp/%x_%j.err

# 1. Shared preamble (sets ROOT, PY, set -euo pipefail, prints banner)
source /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/slurm/_common.sh

mkdir -p "${ROOT}/logs/exp"

# 2. Run the top-hat addition + re-eval
bash "${ROOT}/scripts/re_phase_b_tophat_a7b.sh"
