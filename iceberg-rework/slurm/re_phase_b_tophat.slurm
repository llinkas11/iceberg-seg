#!/bin/bash
# re_phase_b_tophat.slurm: Slurm wrapper for re_phase_b_tophat_only.sh.
#
# Adds the six top-hat companions to the existing A0 and A1 Phase B
# re-runs and re-evaluates per-iceberg + chip-level metrics across all
# 12 methods. CPU-bound (top-hat + eval), so this targets the `main`
# partition rather than `gpu`. 2h budget; expected ~10-20 min.
#
# Usage:
#   sbatch slurm/re_phase_b_tophat.slurm
#
# rsync to moosehead from local:
#   rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/slurm/re_phase_b_tophat.slurm \
#       llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/slurm/

#SBATCH --job-name=ice_phase_b_th
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/exp/%x_%j.out
#SBATCH --error=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/exp/%x_%j.err

# 1. Shared preamble (sets ROOT, PY, set -euo pipefail, prints banner)
source /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/slurm/_common.sh

mkdir -p "${ROOT}/logs/exp"

# 2. Run the top-hat addition + re-eval
bash "${ROOT}/scripts/re_phase_b_tophat_only.sh"
