#!/bin/bash
#SBATCH --job-name=ice_eval_pos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/gt_positive_only/ice_eval_pos_%j.out
#SBATCH --error=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/logs/gt_positive_only/ice_eval_pos_%j.err

set -euo pipefail

bash /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/run_gt_positive_only_eval.sh
