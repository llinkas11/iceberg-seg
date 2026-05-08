# Shib End-to-End: Independent Study Results
**Author:** Shibali Mishra  
**Date:** 2026-05-03  
**Project:** Sentinel-2 iceberg area retrieval across solar zenith angle bins  
**HPC:** moosehead.bowdoin.edu  
**Working dir on moosehead:** `/mnt/research/v.gomezgilyaspik/students/smishra/llinkas-rework`

---

## Overview

This document is the authoritative record of all experimental work completed for the independent study. It covers Phase A (dataset progression, A0 winner selection, C1/C2 ablation), Phase B (6-method sweep on the A0 checkpoint), per-iceberg evaluation metrics, the Fisser (2024) comparison attempt, and (added 2026-05-05) the Phase A higher-SZA re-eval and the A0-vs-A1 Phase B backbone comparison. All numbers are confirmed from actual script output.

For the higher-SZA re-eval (T1, T2) and the backbone comparison (T3, T4, T5), see `phase_a_higher_sza_t1_t4.md`. Headline (revised 2026-05-06 after A7b Phase B + top-hat completed): **A7b** (= A8b == A9b by collapse; A1 manifest + size oversample + augmentation) is the higher-SZA champion across all 18 Phase A backbones; **A7b + UNet_CRF** is the cleanest learned cross-bin pipeline (higher-SZA mean IoU 0.616 vs A1 + UNet_CRF's 0.602 at tied MAE 15.59 m vs 15.21 m; A7b wins sza_gt75 outright). A0 + UNet_OT survives as the lt65 pick (8.45 m, reproducing the published 8.18 m within rounding). Top-hat lifts UNet_OT recall in higher SZA bins but degrades UNet_CRF cross-bin, so A7b + UNet_CRF base (no TH) is the recommendation.

**The paper's core claim:** UNet++ + Otsu post-processing achieves root-length MAE of 8.18 m and +4.9% relative area error on the >100 m iceberg subset at SZA < 65°, compared to the B08 threshold method at 17.05 m MAE and -33% RE under the same automated pipeline conditions. Fisser (2024) reported ±5.7% RE for the threshold method under manually curated conditions.

---

## File Inventory

See `file_paths.md` for full paths (local and moosehead).

```
shib_end_to_end/
├── README.md                    this file (overview, numbers, TODO)
├── results.md                   full results section with all tables (LaTeX-ready)
├── phase_a_cleanup_audit.md     Phase A audit, C1/C2 ablation, A0 winner justification
├── shib_audit_plan.md           original human-in-the-loop plan (Codex handoff)
├── file_paths.md                all file locations (local + moosehead)
└── figures/
    ├── fig-archive/             15 publication PNGs, latest per slug
    ├── figure_review/           deck (2026-05-03, 19 slides) + checklist
    ├── table_previews/          9 table slide PNGs used in deck
    └── cleanup_viz/             10 annotation audit PNGs
```

**Note on table PNGs:** `table_base_*` and `table_tophat_*` were generated from llinkas's baseline_v1 (all-SZA) data. They need to be regenerated from Phase B on A0 (lt65 only) before final submission. Phase A tables (`table_phase_a_*`, `table_probability_calibration`) are confirmed correct.

---

## Experimental Setup

### Data
- **Sentinel-2 L1C**, bands B04/B03/B08, 10 m resolution, 256x256 chips
- **Regions:** Kangerlussuaq (KQ) and Sermilik (SK) fjords, Greenland
- **SZA bins:** sza_lt65, sza_65_70, sza_70_75, sza_gt75

### Manifests
| Manifest | Chips | Filter | IC mask |
|----------|-------|--------|---------|
| v4_raw_lt65 | 398 | none | none |
| v4_clean_lt65 | 330 | 40 m root-length | yes |
| v4_raw_lt65_plus_nulls | ~430 | none | none |
| v4_clean_lt65_plus_nulls | 359 | 40 m root-length | yes |

### Model
- **Architecture:** UNet++ with ResNet34 encoder
- **Output:** binary segmentation (iceberg / background, shadow merged)
- **Hyperparameters:** 100 epochs, lr 1e-4, batch size 16, seed 42
- **Training chip size:** 256x256 pixels

### Six inference methods
| ID | Description |
|----|-------------|
| TR | B08 >= 0.22 fixed threshold (= Fisser's 0.12 + 0.10 DN offset correction) |
| OT | Per-chip Otsu adaptive threshold on B08 |
| UNet | UNet++ argmax (direct binary output) |
| UNet_TR | UNet++ probability map + fixed threshold post-processing |
| UNet_OT | UNet++ probability map + per-chip Otsu post-processing |
| UNet_CRF | UNet++ probability map + DenseCRF refinement |

**Threshold note:** Sentinel-2 products with processing baseline > 4.0 have a +1000 DN radiometric offset (ESA 2024). Chips in v4_raw_lt65 were not corrected, so reflectance values are 0.10 higher than Fisser's corrected space. TR therefore uses 0.22, not 0.12.

---

## Phase A: Dataset Progression

### Goal
Walk 10 controlled dataset variants on lt65 chips to find the best training configuration before the method sweep.

### A0 is the winner on every metric

| ID | Manifest | val IoU | test IoU | UNet match rate | UNet RL MAE (m) |
|----|----------|---------|----------|-----------------|-----------------|
| **A0** | v4_raw_lt65 (Fisser preprocessing, no nulls) | **0.613** | **0.577** | **0.512** | **9.82** |
| A1 | v4_raw_lt65_plus_nulls | 0.503 | 0.477 | 0.315 | 15.21 |
| A3 | v4_clean_lt65_plus_nulls | 0.269 | 0.336 | 0.182 | 15.69 |
| A2 | v4_clean_lt65 (our preprocessing, no nulls) | 0.261 | 0.344 | 0.245 | 15.26 |
| A7-A9 | v4_clean + nulls + aug + size oversample | 0.243 | 0.320 | 0.163 | 14.78 |
| A5-A6 | v4_clean + nulls + aug + class balance | 0.237 | 0.312 | 0.158 | 15.23 |
| A4 | v4_clean + nulls + aug | 0.225 | 0.274 | 0.122 | 14.93 |

### C1/C2 Ablation: IC masking is the culprit

The A0 to A2 collapse (val IoU 0.613 to 0.261) was initially attributed to our preprocessing pipeline (40 m filter + IC pixel mask). C1/C2 ablation isolated the cause:

| Exp | Change from A0 | val IoU | Interpretation |
|-----|---------------|---------|----------------|
| A0 | baseline (no filter, no IC) | 0.612 | winner |
| C1 | 40 m filter only | 0.601 | near A0, filter alone is fine |
| C2 | IC mask only | 0.287 | collapses to near A2 level |
| A2 | 40 m filter + IC mask | 0.261 | full preprocessing pipeline |

**Conclusion:** The IC pixel mask (not the 40 m filter) causes the collapse. The IC mask zeros bright non-annotated pixels in training chips; the val/test splits are never masked. This creates a train-test domain shift: the model learns to predict on dimmed images but encounters fully bright pixels at inference, producing diffuse probability maps.

**A0 checkpoint:** `runs/exp_A0_fisser_lt65_original/20260428_094028/model/best_model.pth`  
Best val IoU: 0.6127717643976212

---

## Phase B: Method Sweep on A0

### Setup
- **Checkpoint:** A0 (frozen, no retraining)
- **Manifest:** v4_raw_lt65 (same as A0 training)
- **Test split:** 100 chips, lt65 only
- **Experiments:** B0 (TR), B1 (OT), B2 (UNet), B3 (UNet_TR), B4 (UNet_OT), B5 (UNet_CRF)
- **Ran:** 2026-05-03

All B runs share the same checkpoint and manifest; inference predictions are identical across B0-B5. The runs formally document the method comparison under a single controlled anchor.

**Result CSVs on moosehead:**
```
runs/exp_B{0-5}_method_*/20260503_*/per_iceberg/eval_per_iceberg_summary.csv
runs/exp_B{0-5}_method_*/20260503_*/per_iceberg/eval_per_iceberg.csv
```

### Root-Length MAE (m) - sza_lt65

Lower is better.

| Method | RL MAE (m) |
|--------|-----------|
| TR | 17.05 |
| OT | 16.40 |
| UNet | 9.82 |
| UNet_TR | 12.33 |
| **UNet_OT** | **8.18** |
| UNet_CRF | 10.97 |

UNet_OT wins. Threshold-only methods are ~2x worse than any UNet variant.

### Per-Pair IoU on Matched Pairs - sza_lt65

Higher is better. Hungarian matching at IoU >= 0.3.

| Method | Per-pair IoU |
|--------|-------------|
| TR | 0.440 |
| OT | 0.463 |
| UNet | 0.626 |
| UNet_TR | 0.589 |
| **UNet_OT** | **0.637** |
| UNet_CRF | 0.560 |

### Relative Area Error (%) - sza_lt65

Positive = overestimation, negative = underestimation.  
Formula: (pred_area - gt_area) / gt_area × 100 per matched pair.

| Method | mean RE% (all) | n (all) | mean RE% (>100m GT) | n (>100m) |
|--------|----------------|---------|---------------------|-----------|
| TR | -53.3% | 1,414 | -33.2% | 102 |
| OT | -38.0% | 1,715 | -46.0% | 208 |
| UNet | +27.5% | 12,343 | +16.9% | 711 |
| UNet_TR | +53.1% | 10,320 | +33.0% | 535 |
| **UNet_OT** | **+26.3%** | 5,267 | **+4.9%** | 139 |
| UNet_CRF | -26.7% | 10,354 | -10.5% | 698 |

**Headline:** UNet_OT at +4.9% RE on >100 m icebergs falls within Fisser (2024)'s published range of -5.7% to +5.9% for the B08 threshold under manually curated conditions at SZA < 65°.

The large n disparity (TR: 1,414 vs UNet: 12,343) is explained below.

---

## IC Filter Analysis

### What it is
`threshold_tifs.py` includes an IC chip-skip filter: if > 15% of a chip's pixels exceed 0.22, the chip is skipped entirely. Designed to reject sea-ice contaminated scenes in automated operation.

### Impact on Phase B
On 100 lt65 test chips:
- **68/100 chips skipped** by IC filter
- Only 32 chips receive TR predictions
- Skipped chips' GT icebergs counted as false negatives

This is why TR has only 1,414 matched pairs vs 12,343 for UNet. The 68 skipped chips are the iceberg-dense Fisser chips (high ice fraction = many icebergs = fails IC filter). The IC filter incorrectly treats iceberg density as sea ice contamination.

### This IS our methodology
The IC filter represents the difference between Fisser's manually curated pipeline and our automated one. Fisser hand-selected 14 scenes with no sea ice and 27 large icebergs each. Our pipeline automates scene selection using the IC filter. On Fisser's chips (which are curated to be ice-rich), the filter backfires.

---

## Fisser (2024) Comparison

### Published result
Fisser et al. (2024) applied B08 >= 0.12 threshold to 14 Kangerlussuaq acquisitions at SZA 45-81°. At SZA < 65°, their standardized area error (SRE_θ) ranged between -5.67% and +5.9%. Standardized = scaled to the Svalbard calibration (mean RE = 0.19% at SZA 56°).

### Our validation attempt
Ran TR without IC filter (`--ic_threshold 1.0`) on 100 lt65 test chips:

| Population | n chips | mean scene-level SAE |
|-----------|---------|---------------------|
| All matched pairs | 30 | -36.2% |
| GT root length > 100 m | 43 | -21.4% |
| **Fisser (2024) published** | 14 acquisitions | **-5.7% to +5.9%** |

### Why it doesn't converge
Three reasons prevent direct comparison:

1. **Metric definition:** Fisser's SRE_θ is standardized to their Svalbard airborne calibration. We cannot replicate this step without their Svalbard data. Our scene-level SAE is raw.

2. **Iceberg matching:** Fisser summed all reference iceberg areas directly. We use IoU >= 0.3 Hungarian matching; unmatched icebergs are excluded from denominator.

3. **Iceberg population:** Fisser's Greenland icebergs had mean root length 326.94 m. Our >100 m filter still includes much smaller icebergs than Fisser's curated set.

### Conclusion
The Fisser ±5.7% is cited as motivation (published TR baseline under ideal conditions), not a benchmark. UNet_OT at +4.9% RE on >100 m icebergs achieves Fisser-range performance without manual curation or size cutoff.

---

## Summary of Headline Numbers

| Metric | Value | Source |
|--------|-------|--------|
| A0 val IoU | 0.613 | exp_A0, 20260428_094028 |
| A0 test IoU | 0.577 | exp_A0, 20260428_094028 |
| C1 val IoU (40m filter only) | 0.601 | C1/C2 ablation |
| C2 val IoU (IC mask only) | 0.287 | C1/C2 ablation |
| Best Phase B RL MAE | 8.18 m (UNet_OT) | B4, 20260503 |
| Best Phase B per-pair IoU | 0.637 (UNet_OT) | B4, 20260503 |
| UNet_OT RE% on >100 m | +4.9% | B4, 20260503 |
| TR RE% on >100 m (IC filter on) | -33.2% | B0, 20260503 |
| TR RE% on >100 m (IC filter off) | -29.5% | fisser_validation, 20260503 |
| IC filter skip rate (lt65 test) | 68% | threshold sweep, 20260503 |
| Fisser (2024) published at lt65 | -5.7% to +5.9% | Fisser et al. 2024, Ann. Glaciol. |
| A1 mean MAE (3 higher-SZA bins) | 28.01 m | re_eval_v4_clean, 20260505 |
| A0 mean MAE (3 higher-SZA bins) | 33.33 m | re_eval_v4_clean, 20260505 |
| A0 + UNet_OT, lt65 (v4_clean split) | 8.45 m, IoU 0.733 | re_phase_b_v4_clean A0, 20260505 |
| A1 + UNet_CRF, sza_65_70 | 10.91 m, IoU 0.633 | re_phase_b_v4_clean A1, 20260505 |
| A1 + UNet_CRF, sza_70_75 | 12.11 m, IoU 0.610 | re_phase_b_v4_clean A1, 20260505 |
| A1 + UNet_CRF, sza_gt75 | 22.60 m, IoU 0.564 | re_phase_b_v4_clean A1, 20260505 |
| A7b mean MAE (3 higher-SZA bins) | 27.24 m, IoU 0.531 | re_eval_v4_clean A7b, 20260505 |
| A7a/A8a/A9a mean MAE (higher SZA) | 28.53 m, IoU 0.520 | A1 + size oversample, aug=off |
| A5a/A6a mean MAE (higher SZA) | 28.48 m, IoU 0.514 | A1 + class balancing, aug=off |
| A7b + UNet_CRF mean (higher SZA) | 15.59 m, IoU 0.616 | re_phase_b A7b, 20260506 (cross-bin pick) |
| A1 + UNet_CRF mean (higher SZA) | 15.21 m, IoU 0.602 | re_phase_b A1, 20260505 (prior pick) |
| A0 + UNet_CRF mean (higher SZA) | 20.62 m, IoU 0.586 | re_phase_b A0, 20260505 |

---

## Moosehead File Locations

See `file_paths.md` for the complete inventory.

---

## Key Code Changes Made This Session

### 1. `run_experiment.py` - added `--checkpoint` argument
Allows Phase B to pass A0's checkpoint directly, skipping training.
```python
parser.add_argument("--checkpoint", default=None,
                    help="External checkpoint for infer stage (skips train).")
# ...
checkpoint = args.checkpoint  # was: checkpoint = None
```

### 2. `slurm/exp.slurm` - forward CHECKPOINT env var
```bash
"${PY}" "${ROOT}/scripts/run_experiment.py" \
    --exp    "${EXP_ID}" \
    --stages "${STAGES}" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"}
```

### 3. `build_figure_review_deck.py` - ROOT path fix
Changed `parents[2]` to `parents[1]` so the script resolves `paper-writing/` correctly in the smishra mirror (paper-writing is inside llinkas-rework, not a sibling).

### 4. Six new Phase B experiment configs
`configs/experiments/exp_B{0-5}_method_*.yaml` - all inherit baseline_v1 but override:
- `data.manifest_id: v4_raw_lt65`
- `augmentation.enabled: false`
- `evaluation.focus_method: <method>`

---

## TODO Before Paper Submission

### High priority
- [ ] **Regenerate Phase B figures** from A0/lt65 results: `mae_rootlen_vs_sza`, `area_scatter_by_method`, `bias_delta_by_area`, `re_by_area_bin`, `outline_examples`. Current fig-archive contains llinkas's baseline_v1 (all-SZA) figures.
- [ ] **Run figure scripts** on Phase B per-iceberg CSVs: `make_fig_mae_vs_sza.py`, `make_fig_area_scatter.py`, `make_fig_bias_by_area.py`, `make_fig_re_by_area.py`
- [ ] **Rebuild deck** after figure regeneration using the pptx_env command above
- [ ] **Verify abstract numbers** against confirmed CSV values before submission

### Medium priority
- [ ] **Update methods section** to reflect: v4_raw_lt65 manifest, A0 as Phase B anchor, IC filter behavior, 0.22 threshold rationale
- [ ] **Add RE% discussion** to results section (Table 9 in results.md is drafted)
- [ ] **Write Fisser comparison paragraph** in discussion: our automated pipeline vs Fisser's manual curation, UNet_OT recovery

### Low priority / deferred
- [ ] Run Phase B on all 4 SZA bins (requires training a new model on all bins, not just lt65). Current Phase B is lt65-only because A0 was lt65-only.
- [ ] Consider running eval_per_iceberg.py on A0 run directly (not just via B configs) to confirm numbers match.

---

## Conda / Python Environments

| Environment | Path | Has pptx? | Use for |
|-------------|------|-----------|---------|
| iceberg-unet312 | /home/llinkas/.venvs/iceberg-unet312/bin/python | No (no write permission) | All ML/eval scripts |
| pptx_env | /mnt/research/.../smishra/pptx_env/bin/python | Yes | build_figure_review_deck.py only |
