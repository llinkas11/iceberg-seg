#!/usr/bin/env bash
# run_pipeline.sh — chip → UNet++ → threshold for one region across all SZA bins
#
# Usage:
#   bash run_pipeline.sh KQ
#   bash run_pipeline.sh SK
#
# Assumes iceberg-unet conda env is active.

set -euo pipefail

REGION="${1:-KQ}"
SCRIPTS=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts
DOWNLOADS="/mnt/research/v.gomezgilyaspik/students/smishra/sentinel2_downloads"
CKPT="runs/s2_20260227_231556/best_model.pth"
AOI="aois_greenland_area_distributions.gpkg"
BINS="sza_lt65 sza_65_70 sza_70_75 sza_gt75"

echo "========================================"
echo " Pipeline: $REGION"
echo " Checkpoint: $CKPT"
echo "========================================"

# ── Step 1: Chip all bins at once (chip_sentinel2.py walks region/sza_bin/*.zip itself)
echo ""
echo "========================================"
echo " Step 1/3: Chipping all $REGION bins"
echo "========================================"
python ${SCRIPTS}/chip_sentinel2.py \
    --safe_dir "$DOWNLOADS" \
    --out_dir  "chips" \
    --aoi      "$AOI"

# ── Steps 2 & 3: Per-bin inference and threshold
for BIN in $BINS; do
    CHIPS_TIF="chips/$REGION/$BIN/tifs"
    UNET_OUT="area_comparison/$REGION/$BIN/unet"
    THRESH_OUT="area_comparison/$REGION/$BIN/threshold"

    if [ ! -d "$CHIPS_TIF" ] || [ -z "$(ls "$CHIPS_TIF"/*.tif 2>/dev/null)" ]; then
        echo ""
        echo "--- [$BIN] No chips found — skipping"
        continue
    fi

    echo ""
    echo "========================================"
    echo " [$BIN] Step 2/3: UNet++ inference"
    echo "========================================"
    python ${SCRIPTS}/predict_tifs.py \
        --checkpoint "$CKPT" \
        --imgs_dir   "$CHIPS_TIF" \
        --out_dir    "$UNET_OUT"

    echo ""
    echo "========================================"
    echo " [$BIN] Step 3/3: Threshold (B08 >= 0.12)"
    echo "========================================"
    python ${SCRIPTS}/threshold_tifs.py \
        --chips_dir "$CHIPS_TIF" \
        --out_dir   "$THRESH_OUT"

done

echo ""
echo "========================================"
echo " Done. Results in area_comparison/$REGION/"
echo "========================================"
echo ""
echo " Next: python ${SCRIPTS}/compare_areas.py --region $REGION"