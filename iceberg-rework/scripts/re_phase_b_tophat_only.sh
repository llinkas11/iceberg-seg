#!/bin/bash
# re_phase_b_tophat_only.sh: add the six top-hat companions (TR_TH, OT_TH,
# UNet_TH, UNet_TR_TH, UNet_OT_TH, UNet_CRF_TH) to the existing Phase B
# re-runs for both A0 and A1 backbones, then re-evaluate per-iceberg with
# the full 12-method list. Reads existing base predictions from
# runs/<exp>/<ts>/re_phase_b_v4_clean/test/<bin>/<METHOD>/, so the six
# base methods are not re-inferred.
#
# Usage (on moosehead, inside iceberg-unet venv):
#   bash scripts/re_phase_b_tophat_only.sh
#
# rsync to moosehead from local:
#   rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/re_phase_b_tophat_only.sh \
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

# 2. Backbone -> existing re_phase_b_v4_clean dir
declare -a BACKBONES=(
    "exp_A0_fisser_lt65_original|20260428_094028"
    "exp_A1_fisser_lt65_plus_nulls|20260429_234146"
)

# 3. Sanity check
[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest missing: $MANIFEST" >&2; exit 2; }

# 4. Per-backbone loop
for entry in "${BACKBONES[@]}"; do
    EXP_ID="${entry%|*}"
    TS="${entry#*|}"
    OUT_BASE="${REPO_DIR}/runs/${EXP_ID}/${TS}/re_phase_b_v4_clean"
    [[ -d "${OUT_BASE}/test" ]] || { echo "ERROR: missing base run dir: ${OUT_BASE}/test" >&2; exit 2; }

    echo "==================================================="
    echo ">>> Top-hat recovery on ${EXP_ID}"
    echo "    Base run dir : ${OUT_BASE}/test"
    echo "==================================================="

    # 5. Per-(bin, method) top-hat recovery
    for B in "${BINS[@]}"; do
        CHIPS="${TEST_CHIPS_ROOT}/${B}"
        for M in "${BASE_METHODS[@]}"; do
            BASE_OUT="${OUT_BASE}/test/${B}/${M}"
            TH_OUT="${OUT_BASE}/test/${B}/${M}_TH"
            # tophat_recover.py reads gpkgs/ and falls back when geotiffs/
            # is absent (only UNet writes geotiffs/), so check the universal
            # output instead.
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

    # 6. Re-run per-iceberg eval covering all 12 methods (overwrites prior summary)
    PER_ICE_DIR="${OUT_BASE}/per_iceberg"
    mkdir -p "$PER_ICE_DIR"
    echo "    eval_per_iceberg.py (12 methods) -> ${PER_ICE_DIR}"
    "$PY" "${REPO_DIR}/scripts/eval_per_iceberg.py" \
        --manifest "$MANIFEST" \
        --test_dir "${OUT_BASE}/test" \
        --out_dir  "$PER_ICE_DIR" \
        --methods  "$ALL_METHODS"

    # 7. Chip-level eval covering all 12 methods (overwrites prior summary)
    EVAL_DIR="${OUT_BASE}/evaluation"
    mkdir -p "$EVAL_DIR"
    echo "    eval_methods.py (12 methods) -> ${EVAL_DIR}"
    "$PY" "${REPO_DIR}/scripts/eval_methods.py" \
        --manifest "$MANIFEST" \
        --test_dir "${OUT_BASE}/test" \
        --out_dir  "$EVAL_DIR"
done

echo ""
echo "==================================================="
echo "Top-hat addition complete for A0 and A1 backbones."
echo "Per-iceberg + chip-level CSVs now cover 12 methods."
echo "==================================================="
