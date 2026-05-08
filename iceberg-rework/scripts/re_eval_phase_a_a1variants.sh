#!/bin/bash
# re_eval_phase_a_a1variants.sh: re-evaluate the 8 A1-anchored Phase A
# variants (A5a..A9a aug=off, A7b..A9b aug=on) on the v4_clean test split
# for all four SZA bins. Sibling to re_eval_phase_a_all_sza.sh, which
# covered A0..A9. UNet method only.
#
# Timestamps are discovered dynamically per experiment because these
# trainings (Slurm 60309-60316) had not finished when this script was
# authored. The latest run dir under runs/exp_<id>_*/ is selected.
#
# Usage (on moosehead, inside iceberg-unet venv):
#   bash scripts/re_eval_phase_a_a1variants.sh
#
# rsync to moosehead from local:
#   rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/re_eval_phase_a_a1variants.sh \
#       llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/

set -euo pipefail

# 1. Repo + env paths
REPO_DIR="${ROOT:-/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework}"
PY="${PY:-/home/llinkas/.venvs/iceberg-unet312/bin/python}"
MANIFEST="${REPO_DIR}/data/v4_clean/manifest.json"
TEST_CHIPS_ROOT="${REPO_DIR}/data/v4_clean/test_chips"
BINS=(sza_lt65 sza_65_70 sza_70_75 sza_gt75)

# 2. The 8 new Phase A variants. exp_id only; timestamps resolved at runtime.
EXPS=(
    A5a_a1_2pos
    A6a_a1_adaptive
    A7a_a1_size
    A8a_a1_2pos_size
    A9a_a1_adaptive_size
    A7b_a1_size_aug
    A8b_a1_2pos_size_aug
    A9b_a1_adaptive_size_aug
)

# 3. Sanity check inputs once before the loop
[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest missing: $MANIFEST" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "ERROR: python missing: $PY" >&2; exit 2; }
for B in "${BINS[@]}"; do
    n=$(ls "$TEST_CHIPS_ROOT/$B"/*.tif 2>/dev/null | wc -l | tr -d ' ')
    [[ "$n" -gt 0 ]] || { echo "ERROR: no chips in $TEST_CHIPS_ROOT/$B" >&2; exit 2; }
done

echo "==================================================="
echo "Phase A A1-variants re-eval (UNet only, manifest=v4_clean)"
echo "Experiments  : ${#EXPS[@]}"
echo "Bins         : ${BINS[*]}"
echo "==================================================="

# 4. Outer loop: one experiment at a time
for EXP_SUFFIX in "${EXPS[@]}"; do
    # Match the on-disk dir name; YAML id is exp_<suffix> so dir is exp_<suffix>.
    EXP_ID="exp_${EXP_SUFFIX}"
    EXP_ROOT="${REPO_DIR}/runs/${EXP_ID}"
    if [[ ! -d "$EXP_ROOT" ]]; then
        echo ""
        echo ">>> ${EXP_ID}: SKIP (no run dir; training may not have started)"
        continue
    fi
    # 5. Discover the latest training timestamp under this experiment
    TS=$(ls -td "${EXP_ROOT}"/2* 2>/dev/null | head -1 | xargs -n1 basename)
    if [[ -z "${TS:-}" ]]; then
        echo ">>> ${EXP_ID}: SKIP (no timestamp dirs under ${EXP_ROOT})"
        continue
    fi
    CKPT="${EXP_ROOT}/${TS}/model/best_model.pth"
    OUT_BASE="${EXP_ROOT}/${TS}/re_eval_v4_clean"

    echo ""
    echo ">>> ${EXP_ID} (ts=${TS})"
    if [[ ! -f "$CKPT" ]]; then
        echo "    SKIP: checkpoint missing: $CKPT"
        continue
    fi

    # 6. Inference loop: one bin at a time, UNet only
    for B in "${BINS[@]}"; do
        OUT_UNET="${OUT_BASE}/test/${B}/UNet"
        if [[ -f "${OUT_UNET}/all_icebergs.gpkg" ]]; then
            echo "    [${B}] UNet already done, skip"
            continue
        fi
        mkdir -p "$OUT_UNET"
        echo "    [${B}] predict_tifs.py -> ${OUT_UNET}"
        "$PY" "${REPO_DIR}/scripts/predict_tifs.py" \
            --checkpoint "$CKPT" \
            --imgs_dir   "${TEST_CHIPS_ROOT}/${B}" \
            --out_dir    "$OUT_UNET" \
            --save_probs
    done

    # 7. Per-iceberg eval (Hungarian matching) against v4_clean GT
    PER_ICE_DIR="${OUT_BASE}/per_iceberg"
    mkdir -p "$PER_ICE_DIR"
    echo "    eval_per_iceberg.py -> ${PER_ICE_DIR}"
    "$PY" "${REPO_DIR}/scripts/eval_per_iceberg.py" \
        --manifest "$MANIFEST" \
        --test_dir "${OUT_BASE}/test" \
        --out_dir  "$PER_ICE_DIR" \
        --methods  UNet
done

echo ""
echo "==================================================="
echo "A1-variants re-eval complete. Per-experiment per_iceberg CSVs at:"
echo "  ${REPO_DIR}/runs/exp_A{5a..9a,7b..9b}_*/<ts>/re_eval_v4_clean/per_iceberg/eval_per_iceberg_summary.csv"
echo "==================================================="
