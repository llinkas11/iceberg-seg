#!/bin/bash
#SBATCH --job-name=threshold_all
#SBATCH --output=logs/threshold_all_%j.out
#SBATCH --error=logs/threshold_all_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=main

# ---------------------------------------------------------------------------
# threshold_all.slurm
# Run threshold_tifs.py on all 8 region/bin combinations.
#
# Threshold: 0.22 (not the Fisser default of 0.12).
# Our chips carry the ESA N0500 +1000 DN offset (+0.10 reflectance), so
# Fisser's 0.12 threshold becomes 0.22 in our uncorrected reflectance space.
#
# Output: area_comparison/{region}/{bin}/threshold/all_icebergs_threshold.gpkg
# ---------------------------------------------------------------------------

REPO_DIR="${HOME}/S2-iceberg-areas"
RESEARCH_ROOT="/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas"
THRESHOLD="0.22"

# 1. Environment setup
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate iceberg-unet

mkdir -p "${REPO_DIR}/logs"

echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "Threshold   : B08 >= ${THRESHOLD} (= Fisser 0.12 in offset-corrected space)"
echo ""

# 2. Run threshold on all region/bin combinations
for region in KQ SK; do
    for bin in sza_lt65 sza_65_70 sza_70_75 sza_gt75; do

        chips_dir="${RESEARCH_ROOT}/chips/${region}/${bin}/tifs"
        out_dir="${RESEARCH_ROOT}/area_comparison/${region}/${bin}/threshold"

        # Skip if no chips exist
        n_chips=$(ls "${chips_dir}"/*.tif 2>/dev/null | wc -l)
        if [ "$n_chips" -eq 0 ]; then
            echo "SKIP ${region}/${bin}: no .tif chips at ${chips_dir}"
            continue
        fi

        echo "$(date '+%H:%M:%S') START ${region}/${bin}  (${n_chips} chips)"
        mkdir -p "${out_dir}"

        python "${REPO_DIR}/threshold_tifs.py" \
            --chips_dir "${chips_dir}" \
            --out_dir   "${out_dir}" \
            --threshold "${THRESHOLD}"

        status=$?
        if [ $status -ne 0 ]; then
            echo "$(date '+%H:%M:%S') ERROR ${region}/${bin} — exit code ${status}"
        else
            echo "$(date '+%H:%M:%S') DONE  ${region}/${bin}"
        fi
        echo ""

    done
done

echo "$(date '+%H:%M:%S') All threshold runs complete."
