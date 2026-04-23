#!/bin/bash
#SBATCH --job-name=predict_all
#SBATCH --output=logs/predict_all_%j.out
#SBATCH --error=logs/predict_all_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=gpu

# ---------------------------------------------------------------------------
# predict_all.slurm
# Run predict_tifs.py (UNet++ v2_aug) on all 8 region/bin combinations.
# Overwrites any existing _pred.tif files from the old s2_20260227 checkpoint.
# Output: area_comparison/{region}/{bin}/unet/all_icebergs.gpkg
#         area_comparison/{region}/{bin}/unet/geotiffs/*_pred.tif
# ---------------------------------------------------------------------------

REPO_DIR="${HOME}/S2-iceberg-areas"
RESEARCH_ROOT="/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas"
CHECKPOINT="${RESEARCH_ROOT}/runs/s2_v2_aug/best_model.pth"

# 1. Environment setup
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate iceberg-unet

mkdir -p "${REPO_DIR}/logs"

echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "Checkpoint  : ${CHECKPOINT}"
echo "GPUs        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# 2. Run inference on all region/bin combinations
for region in KQ SK; do
    for bin in sza_lt65 sza_65_70 sza_70_75 sza_gt75; do

        imgs_dir="${RESEARCH_ROOT}/chips/${region}/${bin}/tifs"
        out_dir="${RESEARCH_ROOT}/area_comparison/${region}/${bin}/unet"

        # Skip if no chips exist
        n_chips=$(ls "${imgs_dir}"/*.tif 2>/dev/null | wc -l)
        if [ "$n_chips" -eq 0 ]; then
            echo "SKIP ${region}/${bin}: no .tif chips at ${imgs_dir}"
            continue
        fi

        echo "$(date '+%H:%M:%S') START ${region}/${bin}  (${n_chips} chips)"
        mkdir -p "${out_dir}"

        python "${REPO_DIR}/predict_tifs.py" \
            --checkpoint "${CHECKPOINT}" \
            --imgs_dir   "${imgs_dir}" \
            --out_dir    "${out_dir}"

        status=$?
        if [ $status -ne 0 ]; then
            echo "$(date '+%H:%M:%S') ERROR ${region}/${bin} — exit code ${status}"
        else
            n_preds=$(ls "${out_dir}/geotiffs"/*_pred.tif 2>/dev/null | wc -l)
            echo "$(date '+%H:%M:%S') DONE  ${region}/${bin}  (${n_preds} _pred.tif written)"
        fi
        echo ""

    done
done

echo "$(date '+%H:%M:%S') All prediction jobs complete."
