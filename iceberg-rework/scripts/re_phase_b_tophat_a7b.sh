#!/bin/bash
# re_phase_b_tophat_a7b.sh: add the six top-hat companions (TR_TH, OT_TH,
# UNet_TH, UNet_TR_TH, UNet_OT_TH, UNet_CRF_TH) to the A7b Phase B re-run
# (Slurm 60323) and re-evaluate per-iceberg + chip-level metrics across 12
# methods. Mirrors re_phase_b_tophat_only.sh but restricted to A7b only;
# A0 and A1 already have their TH companions from Slurm 60300.
#
# Usage (on moosehead, inside iceberg-unet venv):
#   bash scripts/re_phase_b_tophat_a7b.sh
#
# rsync to moosehead from local:
#   rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/re_phase_b_tophat_a7b.sh \
#       llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/

set -euo pipefail

# 1. Repo + env paths
REPO_DIR="${ROOT:-/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework}"
PY="${PY:-/home/llinkas/.venvs/iceberg-unet312/bin/python}"
MANIFEST="${REPO_DIR}/data/v4_clean/manifest.json"
TEST_CHIPS_ROOT="${REPO_DIR}/data/v4_clean/test_chips"
BINS=(sza_lt65 sza_65_70 sza_70_75 sza_gt75)
BASE_METHODS=(TR OT UNet UNet_TR UNet_OT UNet_CRF)
ALL_METHODS="TR,OT,UNet,UNet_TR,UNet_OT,UNet_CRF,TR_TH,OT_TH,UNet_TH,UNet_TR_TH,UNet_OT_TH,UNet_CRF_TH"

# 2. A7b backbone
EXP_ID="exp_A7b_a1_size_aug"
TS="20260505_211329"
OUT_BASE="${REPO_DIR}/runs/${EXP_ID}/${TS}/re_phase_b_v4_clean"

# 3. Sanity check
[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest missing: $MANIFEST" >&2; exit 2; }
[[ -d "${OUT_BASE}/test" ]] || { echo "ERROR: missing base run dir: ${OUT_BASE}/test" >&2; exit 2; }

echo "==================================================="
echo ">>> Top-hat recovery on ${EXP_ID}"
echo "    Base run dir : ${OUT_BASE}/test"
echo "==================================================="

# 4. Per-(bin, method) top-hat recovery. tophat_recover.py reads gpkgs/ and
#    falls back when geotiffs/ is absent (only UNet writes geotiffs/), so
#    check the universal output instead.
for B in "${BINS[@]}"; do
    CHIPS="${TEST_CHIPS_ROOT}/${B}"
    for M in "${BASE_METHODS[@]}"; do
        BASE_OUT="${OUT_BASE}/test/${B}/${M}"
        TH_OUT="${OUT_BASE}/test/${B}/${M}_TH"
        if [[ ! -f "${BASE_OUT}/all_icebergs.gpkg" ]]; then
            echo "    SKIP [${B}/${M}]: missing base all_icebergs.gpkg"
            continue
        fi
        if [[ -f "${TH_OUT}/all_icebergs.gpkg" ]]; then
            echo "    [${B}/${M}_TH] already done, skip"
            continue
        fi
        mkdir -p "$TH_OUT"
        echo "    [${B}/${M}_TH] tophat_recover.py"
        "$PY" "${REPO_DIR}/scripts/tophat_recover.py" \
            --chips_dir       "$CHIPS" \
            --base_dir        "$BASE_OUT" \
            --out_dir         "$TH_OUT" \
            --base_method_id  "$M"
    done
done

# 5. Re-run per-iceberg eval covering all 12 methods (overwrites prior summary)
PER_ICE_DIR="${OUT_BASE}/per_iceberg"
mkdir -p "$PER_ICE_DIR"
echo "    eval_per_iceberg.py (12 methods) -> ${PER_ICE_DIR}"
"$PY" "${REPO_DIR}/scripts/eval_per_iceberg.py" \
    --manifest "$MANIFEST" \
    --test_dir "${OUT_BASE}/test" \
    --out_dir  "$PER_ICE_DIR" \
    --methods  "$ALL_METHODS"

# 6. Chip-level eval covering all 12 methods (overwrites prior summary)
EVAL_DIR="${OUT_BASE}/evaluation"
mkdir -p "$EVAL_DIR"
echo "    eval_methods.py (12 methods) -> ${EVAL_DIR}"
"$PY" "${REPO_DIR}/scripts/eval_methods.py" \
    --manifest "$MANIFEST" \
    --test_dir "${OUT_BASE}/test" \
    --out_dir  "$EVAL_DIR"

echo ""
echo "==================================================="
echo "Top-hat addition complete for A7b."
echo "Per-iceberg + chip-level CSVs now cover 12 methods."
echo "==================================================="
