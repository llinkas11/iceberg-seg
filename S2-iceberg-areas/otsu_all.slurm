#!/bin/bash
#SBATCH --job-name=otsu_all
#SBATCH --output=logs/otsu_all_%j.out
#SBATCH --error=logs/otsu_all_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=main

# ---------------------------------------------------------------------------
# otsu_all.slurm
# Run otsu_threshold_tifs.py on all 8 region/bin combinations.
# No GPU needed. Can submit immediately alongside predict_all and threshold_all.
#
# Output: area_comparison/{region}/{bin}/otsu/all_icebergs_otsu.gpkg
# ---------------------------------------------------------------------------

REPO_DIR="${HOME}/S2-iceberg-areas"
RESEARCH_ROOT="/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas"

# 1. Environment setup
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate iceberg-unet

mkdir -p "${REPO_DIR}/logs"

echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "Otsu floor: 0.10  ceil: 0.50  sea_ice_frac: 0.15"
echo ""

# 2. Run Otsu on all region/bin combinations
for region in KQ SK; do
    for bin in sza_lt65 sza_65_70 sza_70_75 sza_gt75; do

        chips_dir="${RESEARCH_ROOT}/chips/${region}/${bin}/tifs"
        out_dir="${RESEARCH_ROOT}/area_comparison/${region}/${bin}/otsu"

        n_chips=$(ls "${chips_dir}"/*.tif 2>/dev/null | wc -l)
        if [ "$n_chips" -eq 0 ]; then
            echo "SKIP ${region}/${bin}: no .tif chips"
            continue
        fi

        echo "$(date '+%H:%M:%S') START ${region}/${bin}  (${n_chips} chips)"
        mkdir -p "${out_dir}"

        python "${REPO_DIR}/otsu_threshold_tifs.py" \
            --chips_dir    "${chips_dir}" \
            --out_dir      "${out_dir}" \
            --otsu_floor   0.10 \
            --otsu_ceil    0.50 \
            --sea_ice_frac 0.15 \
            --min_area_m2  100

        status=$?
        if [ $status -ne 0 ]; then
            echo "$(date '+%H:%M:%S') ERROR ${region}/${bin} — exit code ${status}"
        else
            echo "$(date '+%H:%M:%S') DONE  ${region}/${bin}"
        fi
        echo ""

    done
done

echo "$(date '+%H:%M:%S') All Otsu runs complete."
