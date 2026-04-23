#!/bin/bash
# run_all_methods.sh — Run all 6 segmentation methods on the test-set chips for one SZA bin.
#
# All methods run on the SAME test chips (from prepare_test_chips_dir.py) so results
# are directly comparable against the same ground truth labels.
#
# Methods:
#   TR        fixed NIR threshold (0.22)
#   OT        per-chip Otsu on NIR
#   UNet      UNet++ argmax  (also saves softmax probs for post-processing)
#   UNet_TR   UNet++ + fixed threshold on P(iceberg)
#   UNet_OT   UNet++ + Otsu on P(iceberg)
#   UNet_CRF  UNet++ + DenseCRF
#   TODO: +TH (top-hat) variants — add once top-hat script is written
#
# Usage:
#   bash run_all_methods.sh <BIN> [CHECKPOINT]
#
# Examples:
#   bash run_all_methods.sh sza_70_75
#   bash run_all_methods.sh sza_gt75
#   bash run_all_methods.sh sza_lt65 /mnt/research/.../runs/s2_v3/best_model.pth
#
# Prerequisites:
#   python prepare_test_chips_dir.py   ← run once to build test_chips/ dir
#
# Outputs (all under area_comparison/test/<BIN>/<METHOD>/):
#   all_icebergs.gpkg   — merged iceberg polygons for test chips in this bin
#   gpkgs/              — per-chip .gpkg files
#   geotiffs/           — per-chip label GeoTIFFs      (UNet only)
#   probs/              — per-chip softmax prob .tifs   (UNet only, used by post-processing)

set -e

# DEPRECATED: run_all_methods.sh is superseded by run_methods.sh, which takes
# --manifest and --checkpoint explicitly and guards against cross-manifest
# inference drift. This wrapper continues to work but warns on every call; it
# will be removed once all slurm scripts are migrated.
echo "[deprecation] run_all_methods.sh: please migrate to run_methods.sh"
echo "[deprecation]   bash run_methods.sh --manifest <path> --checkpoint <path> --out_base <path> [--bin <BIN>]"
echo ""

BIN=${1:?     "Usage: bash run_all_methods.sh <BIN> [CHECKPOINT] [DEVICE]"}
CHECKPOINT=${2:-"/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/runs/s2_v2_aug/best_model.pth"}
DEVICE=${3:-""}

RESEARCH=/mnt/research/v.gomezgilyaspik/students/smishra
CHIPS=${RESEARCH}/S2-iceberg-areas/test_chips/${BIN}
OUT_BASE=${RESEARCH}/S2-iceberg-areas/area_comparison/test/${BIN}
SCRIPTS=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts

echo "============================================"
echo "SZA bin    : ${BIN}"
echo "Checkpoint : ${CHECKPOINT}"
echo "Chips dir  : ${CHIPS}  (test set only)"
echo "Output     : ${OUT_BASE}"
echo "============================================"
echo ""

# Verify test chips dir exists — must run prepare_test_chips_dir.py first
if [ ! -d "${CHIPS}" ]; then
    echo "ERROR: test chips dir not found: ${CHIPS}"
    echo "Run first: python /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/prepare_test_chips_dir.py"
    exit 1
fi

N_CHIPS=$(ls ${CHIPS}/*.tif 2>/dev/null | wc -l)
echo "Test chips in bin: ${N_CHIPS}"
echo ""

# ── 1. TR: fixed NIR threshold ────────────────────────────────────────────────
echo "[1/6] TR — fixed NIR threshold"
python ${SCRIPTS}/threshold_tifs.py \
    --chips_dir ${CHIPS} \
    --out_dir   ${OUT_BASE}/TR
echo ""

# ── 2. OT: per-chip Otsu ─────────────────────────────────────────────────────
echo "[2/6] OT — per-chip Otsu"
python ${SCRIPTS}/otsu_threshold_tifs.py \
    --chips_dir ${CHIPS} \
    --out_dir   ${OUT_BASE}/OT
echo ""

# ── 3. UNet: UNet++ argmax + save softmax probs ──────────────────────────────
echo "[3/6] UNet — UNet++ inference (also saves softmax probs)"
python ${SCRIPTS}/predict_tifs.py \
    --checkpoint ${CHECKPOINT} \
    --imgs_dir   ${CHIPS} \
    --out_dir    ${OUT_BASE}/UNet \
    --save_probs \
    ${DEVICE:+--device ${DEVICE}}
echo ""

PROBS_DIR=${OUT_BASE}/UNet/probs

# ── 4. UNet + TR ─────────────────────────────────────────────────────────────
echo "[4/6] UNet_TR — threshold on P(iceberg)"
python ${SCRIPTS}/threshold_probs.py \
    --probs_dir ${PROBS_DIR} \
    --out_dir   ${OUT_BASE}/UNet_TR
echo ""

# ── 5. UNet + OT ─────────────────────────────────────────────────────────────
echo "[5/6] UNet_OT — Otsu on P(iceberg)"
python ${SCRIPTS}/otsu_probs.py \
    --probs_dir ${PROBS_DIR} \
    --out_dir   ${OUT_BASE}/UNet_OT
echo ""

# ── 6. UNet + CRF ────────────────────────────────────────────────────────────
echo "[6/6] UNet_CRF — DenseCRF post-processing"
python ${SCRIPTS}/densecrf_tifs.py \
    --probs_dir ${PROBS_DIR} \
    --chips_dir ${CHIPS} \
    --out_dir   ${OUT_BASE}/UNet_CRF
echo ""

echo "============================================"
echo "Done. Outputs in: ${OUT_BASE}/"
echo ""
echo "Methods completed:"
for METHOD in TR OT UNet UNet_TR UNet_OT UNet_CRF; do
    GPKG=${OUT_BASE}/${METHOD}/all_icebergs.gpkg
    if [ -f "${GPKG}" ]; then
        echo "  ✓ ${METHOD}"
    else
        echo "  ✗ ${METHOD}  (all_icebergs.gpkg missing)"
    fi
done
echo "============================================"