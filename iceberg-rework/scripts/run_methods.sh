#!/bin/bash
# run_methods.sh: run all six iceberg segmentation methods on the test split of
# one dataset manifest. Replacement for run_all_methods.sh; takes a manifest
# path instead of a BIN env var, so you cannot accidentally mix chips from a
# different dataset variant across methods.
#
# Methods:
#   TR        fixed NIR threshold (0.22)
#   OT        per-chip Otsu on NIR
#   UNet      UNet++ argmax (also saves softmax probs)
#   UNet_TR   UNet++ + fixed threshold on P(iceberg)
#   UNet_OT   UNet++ + Otsu on P(iceberg)
#   UNet_CRF  UNet++ + DenseCRF
#
# Usage:
#   bash run_methods.sh --manifest <path> --checkpoint <path> --out_base <path> [--bin <sza_bin>]
#
# Example:
#   bash run_methods.sh \
#       --manifest   data/v4_clean/manifest.json \
#       --checkpoint model/v3_balanced_sza_stage1_matched_seed42_aug_20260423/best_model.pth \
#       --out_base   results/baseline_v1/test \
#       --bin        sza_lt65
#
# If --bin is omitted, runs every SZA bin present in the manifest.
#
# Refusal checks:
#   * Refuses if the checkpoint has a sibling training_config.json whose
#     manifest_id does not match the one in --manifest (dataset drift guard).
#   * Refuses if the test chip dir for a bin has zero .tif files.

set -euo pipefail

# ----- Parse args -----------------------------------------------------------
MANIFEST=""
CHECKPOINT=""
OUT_BASE=""
BIN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest)   MANIFEST="$2";   shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --out_base)   OUT_BASE="$2";   shift 2 ;;
        --bin)        BIN="$2";        shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

for req in MANIFEST CHECKPOINT OUT_BASE; do
    if [[ -z "${!req}" ]]; then
        echo "ERROR: --${req,,} is required" >&2
        exit 2
    fi
done
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found: $MANIFEST" >&2
    exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi

# ----- Resolve paths --------------------------------------------------------
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$REPO_DIR/scripts"
PY="${PY:-/home/llinkas/.venvs/iceberg-unet312/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "ERROR: python interpreter not found at $PY" >&2
    echo "       set PY=... in the environment before calling run_methods.sh" >&2
    exit 4
fi

# Pull both fields in one subprocess; whitespace split is what read gives us.
read -r MANIFEST_ID CHIPS_SHA < <(
    "$PY" -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['manifest_id'], m['chips_sha'])" "$MANIFEST"
)

echo "==================================================="
echo "Manifest   : $MANIFEST  (id=$MANIFEST_ID)"
echo "chips_sha  : ${CHIPS_SHA:0:16}..."
echo "Checkpoint : $CHECKPOINT"
echo "Out base   : $OUT_BASE"
echo "SZA bin    : ${BIN:-<all>}"
echo "==================================================="

# ----- Dataset-drift guard --------------------------------------------------
CKPT_DIR="$(dirname "$CHECKPOINT")"
CKPT_CONFIG="$CKPT_DIR/training_config.json"
if [[ -f "$CKPT_CONFIG" ]]; then
    CKPT_MANIFEST=$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('manifest_id') or '')" "$CKPT_CONFIG")
    if [[ -n "$CKPT_MANIFEST" && "$CKPT_MANIFEST" != "$MANIFEST_ID" ]]; then
        echo "ERROR: checkpoint was trained on manifest '$CKPT_MANIFEST' but --manifest is '$MANIFEST_ID'." >&2
        echo "       Refusing to run: cross-manifest inference is a drift hazard." >&2
        echo "       Override with FORCE=1 if you really mean to do this." >&2
        if [[ "${FORCE:-0}" != "1" ]]; then
            exit 3
        fi
    fi
else
    echo "NOTE: checkpoint has no training_config.json sibling; cannot verify manifest match."
fi

# ----- Prepare test chips directory -----------------------------------------
# One-shot: builds a per-bin directory of symlinks under data/<manifest_id>/test_chips.
TEST_CHIPS_DIR="$REPO_DIR/data/$MANIFEST_ID/test_chips"
if [[ ! -d "$TEST_CHIPS_DIR" || -z "$(ls -A "$TEST_CHIPS_DIR" 2>/dev/null)" ]]; then
    echo ""
    echo "Building test chip dir: $TEST_CHIPS_DIR"
    "$PY" "$SCRIPTS/prepare_test_chips_dir.py" \
        --manifest "$MANIFEST" \
        --out_dir  "$TEST_CHIPS_DIR"
fi

# ----- Choose bins to run ---------------------------------------------------
if [[ -n "$BIN" ]]; then
    BINS=("$BIN")
else
    BINS=()
    for b in sza_lt65 sza_65_70 sza_70_75 sza_gt75; do
        if [[ -d "$TEST_CHIPS_DIR/$b" && -n "$(ls -A "$TEST_CHIPS_DIR/$b" 2>/dev/null)" ]]; then
            BINS+=("$b")
        fi
    done
fi

# ----- Run the six methods for each bin -------------------------------------
for B in "${BINS[@]}"; do
    CHIPS="$TEST_CHIPS_DIR/$B"
    OUT="$OUT_BASE/$B"
    N_CHIPS=$(ls "$CHIPS"/*.tif 2>/dev/null | wc -l | tr -d ' ')

    echo ""
    echo "--- $B  ($N_CHIPS chips)  out=$OUT ---"
    if [[ "$N_CHIPS" -eq 0 ]]; then
        echo "SKIP: no chips in $CHIPS"
        continue
    fi

    echo "[1/6] TR"
    "$PY" "$SCRIPTS/threshold_tifs.py" \
        --chips_dir "$CHIPS" \
        --out_dir   "$OUT/TR"

    echo "[2/6] OT"
    "$PY" "$SCRIPTS/otsu_threshold_tifs.py" \
        --chips_dir "$CHIPS" \
        --out_dir   "$OUT/OT"

    echo "[3/6] UNet (also saves softmax probs)"
    "$PY" "$SCRIPTS/predict_tifs.py" \
        --checkpoint "$CHECKPOINT" \
        --imgs_dir   "$CHIPS" \
        --out_dir    "$OUT/UNet" \
        --save_probs

    PROBS="$OUT/UNet/probs"
    echo "[4/6] UNet_TR"
    "$PY" "$SCRIPTS/threshold_probs.py" \
        --probs_dir "$PROBS" \
        --out_dir   "$OUT/UNet_TR"

    echo "[5/6] UNet_OT"
    "$PY" "$SCRIPTS/otsu_probs.py" \
        --probs_dir "$PROBS" \
        --out_dir   "$OUT/UNet_OT"

    echo "[6/6] UNet_CRF"
    "$PY" "$SCRIPTS/densecrf_tifs.py" \
        --probs_dir "$PROBS" \
        --chips_dir "$CHIPS" \
        --out_dir   "$OUT/UNet_CRF"
done

echo ""
echo "==================================================="
echo "Completed. Check $OUT_BASE/<bin>/<method>/ for gpkgs + method_config.json + skipped_chips.csv."
echo "==================================================="
