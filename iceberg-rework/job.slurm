#!/bin/bash
#SBATCH --job-name=iceberg_unet
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # moosehead GPU partition max is 4 CPUs
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx3080:1           # options: gpu:rtx3080:1  gpu:a100:1 (faster)
#SBATCH --partition=gpu

# ---------------------------------------------------------------------------
# moosehead.bowdoin.edu config
# ---------------------------------------------------------------------------
REPO_DIR="/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"
CONDA_ENV="iceberg-unet"
MODE="s2"                               # s1 or s2 (binary segmentation: iceberg only)
DATA_DIR="${REPO_DIR}/data/v3_balanced"  # v3 dataset with shadow merged, IC masked, balanced
OUT_DIR="${REPO_DIR}/model/v3_balanced_aug"
# ---------------------------------------------------------------------------

mkdir -p logs
mkdir -p "${OUT_DIR}"

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "Mode        : ${MODE}"
echo "Data dir    : ${DATA_DIR}"
echo "Output dir  : ${OUT_DIR}"
echo "GPUs        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python "${REPO_DIR}/scripts/train.py" \
    --mode       "${MODE}"       \
    --data_dir   "${DATA_DIR}"   \
    --out_dir    "${OUT_DIR}"    \
    --encoder    resnet34        \
    --epochs     100             \
    --batch_size 16              \
    --lr         1e-4            \
    --workers    4

echo "Done. Results in ${OUT_DIR}"
