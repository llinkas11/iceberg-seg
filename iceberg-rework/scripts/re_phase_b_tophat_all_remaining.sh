#!/bin/bash
# re_phase_b_tophat_all_remaining.sh: add the six top-hat companions to the
# Phase B re-runs from re_phase_b_all_remaining.sh and re-evaluate per-iceberg
# + chip-level metrics across all 12 methods. Companion to that script;
# expects its outputs to exist on disk. CPU-only (no checkpoint reload).
# Backbone list lives in _exps_remaining.sh.
#
# Usage (on moosehead):
#   bash scripts/re_phase_b_tophat_all_remaining.sh
#
# rsync to moosehead from local:
#   rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/re_phase_b_tophat_all_remaining.sh \
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

# 2. (exp_id, ts) pairs sourced from _exps_remaining.sh
source "$(dirname "$0")/_exps_remaining.sh"
EXPS=("${EXPS_REMAINING[@]}")

# 3. Sanity check
[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest missing: $MANIFEST" >&2; exit 2; }

# 4. Per-backbone TH addition + 12-method eval re-run
for entry in "${EXPS[@]}"; do
    EXP_ID="${entry%|*}"
    TS="${entry#*|}"
    OUT_BASE="${REPO_DIR}/runs/${EXP_ID}/${TS}/re_phase_b_v4_clean"
    if [[ ! -d "${OUT_BASE}/test" ]]; then
        echo ">>> ${EXP_ID}: SKIP (no Phase B run dir at ${OUT_BASE})"
        continue
    fi

    echo "==================================================="
    echo ">>> ${EXP_ID} (ts=${TS}) top-hat addition"
    echo "==================================================="

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
    "$PY" "${REPO_DIR}/scripts/eval_per_iceberg.py" \
        --manifest "$MANIFEST" \
        --test_dir "${OUT_BASE}/test" \
        --out_dir  "${OUT_BASE}/per_iceberg" \
        --methods  "$ALL_METHODS"

    # 6. Re-run chip-level eval covering all 12 methods
    "$PY" "${REPO_DIR}/scripts/eval_methods.py" \
        --manifest "$MANIFEST" \
        --test_dir "${OUT_BASE}/test" \
        --out_dir  "${OUT_BASE}/evaluation"
done

echo ""
echo "==================================================="
echo "Top-hat addition complete for all remaining backbones."
echo "Per-iceberg + chip-level CSVs now cover 12 methods for ${#EXPS[@]} backbones."
echo "==================================================="
