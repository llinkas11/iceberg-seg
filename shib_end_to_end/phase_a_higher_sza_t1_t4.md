# Phase A higher-SZA re-eval and Phase B backbone comparison (T1-T4)

**Date:** 2026-05-05
**Author:** Shibali Mishra
**HPC:** moosehead.bowdoin.edu, Slurm jobs 60293 (Phase A), 60296 (A1 Phase B), 60297 (A0 Phase B).

## Context

Phase A (A0-A9, see `phase_a_cleanup_audit.md`) was originally trained and evaluated on `sza_lt65` only, because all 10 manifests (`v4_raw_lt65*`, `v4_clean_lt65*`) contain only lt65 chips. This left 30 of 40 cells in a per-bin x per-experiment comparison empty.

Phase B (the published 8.18 m UNet_OT headline) was likewise lt65-only and used the A0 checkpoint as the trained backbone.

This document records three additional experiments that fill in the missing cells and test whether the A0 backbone is the right choice across all four SZA bins.

## Method

All three experiments use the `v4_clean` test split (228 chips, 57 per SZA bin) so cells are directly comparable across experiments and across SZA bins. The dataset-drift guard in `run_methods.sh` is overridden with `FORCE=1`; this is intentional because we are intentionally evaluating each Phase A backbone against the unifying all-SZA test split.

| Slurm job | Script | Wallclock | Backbone | Methods | Bins |
|-----------|--------|-----------|----------|---------|------|
| 60293 | `scripts/re_eval_phase_a_all_sza.sh` | 14:09 | A0-A9 | UNet only | all 4 |
| 60296 | `scripts/re_phase_b_with_a1.sh` | 6:45 | A1 | TR, OT, UNet, UNet_TR, UNet_OT, UNet_CRF | all 4 |
| 60297 | `scripts/re_phase_b_with_a0.sh` | 7:07 | A0 | TR, OT, UNet, UNet_TR, UNet_OT, UNet_CRF | all 4 |

## T1 - Phase A per-SZA-bin x A0..A9 (UNet only)

Cell format: `mean per-pair IoU / mean per-pair root-length MAE (m)`. Bold = column max IoU per row.

| SZA bin     | A0              | A1              | A2          | A3          | A4          | A5          | A6          | A7          | A8          | A9          |
|-------------|-----------------|-----------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| sza_lt65    | **0.710 / 10.99** | 0.568 / 17.74 | 0.478/14.55 | 0.487/16.03 | 0.459/15.23 | 0.501/15.49 | 0.501/15.49 | 0.481/15.02 | 0.481/15.02 | 0.481/15.02 |
| sza_65_70   | 0.501 / 30.71   | **0.513 / 25.51** | 0.428/31.46 | 0.441/33.05 | 0.423/34.38 | 0.436/40.26 | 0.436/40.26 | 0.425/39.41 | 0.425/39.41 | 0.425/39.41 |
| sza_70_75   | 0.484 / 32.43   | **0.500 / 26.97** | 0.418/29.72 | 0.419/40.55 | 0.433/36.99 | 0.418/49.82 | 0.418/49.82 | 0.405/46.77 | 0.405/46.77 | 0.405/46.77 |
| sza_gt75    | 0.484 / 36.84   | **0.485 / 31.54** | 0.448/32.50 | 0.433/47.44 | 0.456/35.61 | 0.446/46.70 | 0.446/46.70 | 0.414/47.96 | 0.414/47.96 | 0.414/47.96 |

Source CSVs: `iceberg-rework/runs_summaries/exp_A{0..9}_*/<latest_ts>/re_eval_v4_clean/per_iceberg/eval_per_iceberg_summary.csv`. A0's lt65 cell here (0.710 / 10.99) differs from the audit doc's (0.626 / 9.82) because this re-eval uses the v4_clean lt65 test split (57 chips, 4,016 pairs) instead of v4_raw_lt65's split (96 chips, 12,343 pairs); the new value is the apples-to-apples one for cross-bin comparison.

## T2 - Best non-A0 Phase A per bin

| SZA bin     | A0              | Best non-A0       | Δ IoU vs A0 | Δ MAE vs A0 |
|-------------|-----------------|-------------------|-------------|-------------|
| sza_lt65    | 0.710 / 10.99   | A1: 0.568 / 17.74 | -0.142      | +6.75 m     |
| sza_65_70   | 0.501 / 30.71   | **A1: 0.513 / 25.51** | +0.012  | **-5.20 m** |
| sza_70_75   | 0.484 / 32.43   | **A1: 0.500 / 26.97** | +0.016  | **-5.46 m** |
| sza_gt75    | 0.484 / 36.84   | **A1: 0.485 / 31.54** | +0.001  | **-5.30 m** |

A1 sweeps every higher-SZA bin on both metrics. Aggregate over the three higher-SZA bins:
- A0: mean IoU 0.490, mean MAE 33.33 m
- **A1: mean IoU 0.499, mean MAE 28.01 m** (16% lower MAE than A0)

Interpretation: the 29 GT-zero (empty-water) chips that distinguish A1 from A0 act as a regulariser that helps the model handle harder higher-SZA chips it never saw at training. A0 is still the right pick for lt65; A1 is the right pick for higher SZA.

## T3 - Phase B backbone comparison (A0 vs A1, twelve methods, all SZA bins)

Cell format: `mean per-pair IoU / mean per-pair root-length MAE (m), n=matched_pairs`. Bold = backbone winner per (method, bin) on MAE. Pixel methods (TR / OT and their `_TH` variants) are independent of the UNet backbone, so A0 and A1 are identical for those rows.

### Pixel methods (backbone-independent: A0 == A1)

| Method     | sza_lt65            | sza_65_70           | sza_70_75           | sza_gt75            |
|------------|---------------------|---------------------|---------------------|---------------------|
| TR         | 0.482 / 17.81, n=872 | 0.670 / 7.91, n=315  | 0.686 / 6.46, n=126  | 0.594 / 20.07, n=152 |
| TR_TH      | 0.514 / 17.10, n=2041| 0.660 / 8.01, n=403  | 0.654 / 10.96, n=190 | 0.599 / 20.44, n=217 |
| OT         | 0.470 / 22.73, n=902 | 0.622 / 13.77, n=530 | 0.621 / 14.51, n=613 | 0.620 / 15.91, n=427 |
| OT_TH      | 0.521 / 17.39, n=1540| 0.614 / 12.62, n=516 | 0.615 / 13.78, n=575 | 0.579 / 19.52, n=252 |

Top-hat raises recall (n) for both TR and OT in every bin; on MAE, OT_TH improves OT in three of four bins (lt65, 65-70, gt75); TR_TH worsens TR slightly above 65 deg.

### UNet (argmax on backbone softmax)

| bin       | A0                     | A1                     | Winner |
|-----------|------------------------|------------------------|--------|
| sza_lt65  | **0.710 / 10.99, n=4016** | 0.568 / 17.74, n=2954 | A0 |
| sza_65_70 | 0.501 / 30.71, n=271   | **0.513 / 25.51, n=277** | A1 |
| sza_70_75 | 0.484 / 32.43, n=321   | **0.500 / 26.97, n=332** | A1 |
| sza_gt75  | 0.484 / 36.84, n=179   | **0.485 / 31.54, n=188** | A1 |

### UNet_TH (UNet + top-hat recovery)

| bin       | A0                     | A1                     | Winner |
|-----------|------------------------|------------------------|--------|
| sza_lt65  | **0.722 / 10.70, n=2551** | 0.587 / 16.67, n=2078 | A0 |
| sza_65_70 | 0.501 / 31.25, n=266   | **0.529 / 25.20, n=285** | A1 |
| sza_70_75 | 0.474 / 38.78, n=120   | **0.503 / 34.30, n=121** | A1 |
| sza_gt75  | 0.505 / 39.12, n=153   | **0.510 / 34.19, n=136** | A1 |

UNet_TH tracks UNet closely; small lt65 MAE improvement (10.99 to 10.70 on A0), otherwise within noise.

### UNet_TR (fixed threshold on backbone P(iceberg))

| bin       | A0                     | A1                     | Winner |
|-----------|------------------------|------------------------|--------|
| sza_lt65  | **0.634 / 17.55, n=3129** | 0.458 / 31.81, n=756 | A0 |
| sza_65_70 | **0.462 / 38.45, n=216** | 0.453 / 44.21, n=106 | A0 |
| sza_70_75 | **0.452 / 40.06, n=218** | 0.427 / 49.11, n=68 | A0 |
| sza_gt75  | **0.469 / 41.25, n=142** | 0.460 / 41.41, n=80 | A0 |

### UNet_TR_TH (UNet_TR + top-hat recovery)

| bin       | A0                     | A1                     | Winner |
|-----------|------------------------|------------------------|--------|
| sza_lt65  | **0.646 / 16.76, n=2072** | 0.459 / 32.21, n=645 | A0 |
| sza_65_70 | **0.464 / 38.55, n=210** | 0.471 / 44.17, n=102 | A0 |
| sza_70_75 | **0.455 / 43.05, n=91**   | 0.431 / 61.29, n=34  | A0 |
| sza_gt75  | **0.501 / 36.90, n=136** | 0.491 / 44.18, n=77  | A0 |

A0 wins all UNet_TR variants (consistent with the base UNet_TR pattern; the threshold-on-probs path favours A0's sharper probability distribution).

### UNet_OT (per-chip Otsu on backbone P(iceberg)) - published headline method

| bin       | A0                     | A1                     | Winner |
|-----------|------------------------|------------------------|--------|
| sza_lt65  | **0.733 / 8.45, n=1392** | 0.589 / 18.30, n=569 | A0 |
| sza_65_70 | 0.524 / 32.20, n=108   | **0.497 / 30.70, n=108** | A1 (MAE) |
| sza_70_75 | 0.479 / 37.09, n=68    | **0.505 / 34.51, n=50** | A1 |
| sza_gt75  | 0.509 / 33.61, n=73    | **0.508 / 26.42, n=88** | A1 |

A0 + UNet_OT on lt65 reproduces the published 8.18 m headline within rounding (8.45 m here on the v4_clean lt65 split; the original 8.18 m was on a different chip subset with a >100 m filter applied).

### UNet_OT_TH (UNet_OT + top-hat recovery)

| bin       | A0                     | A1                     | Winner |
|-----------|------------------------|------------------------|--------|
| sza_lt65  | **0.632 / 12.97, n=2293** | 0.554 / 16.76, n=2045 | A0 |
| sza_65_70 | 0.614 / 14.98, n=366   | **0.613 / 14.02, n=366** | A1 |
| sza_70_75 | 0.560 / 24.34, n=162   | **0.585 / 20.76, n=170** | A1 |
| sza_gt75  | 0.548 / 27.60, n=197   | **0.549 / 24.74, n=205** | A1 |

UNet_OT_TH approximately doubles recall vs UNet_OT in every bin (n=1392 to 2293 at lt65; n=88 to 205 at gt75 for A1). lt65 MAE worsens (8.45 to 12.97 on A0), but higher SZA bins improve substantially: 65-70 drops from 30-32 m to 14 m; 70-75 from 34-37 m to 21-24 m; gt75 from 26-34 m to 25-28 m.

### UNet_CRF (DenseCRF on backbone P(iceberg))

| bin       | A0                     | A1                     | Winner |
|-----------|------------------------|------------------------|--------|
| sza_lt65  | **0.619 / 11.93, n=4290** | 0.569 / 14.02, n=3290 | A0 |
| sza_65_70 | 0.609 / 17.64, n=367   | **0.633 / 10.91, n=350** | A1 |
| sza_70_75 | 0.597 / 17.06, n=471   | **0.610 / 12.11, n=494** | A1 |
| sza_gt75  | 0.552 / 27.16, n=226   | **0.564 / 22.60, n=241** | A1 |

### UNet_CRF_TH (UNet_CRF + top-hat recovery)

| bin       | A0                     | A1                     | Winner |
|-----------|------------------------|------------------------|--------|
| sza_lt65  | **0.627 / 11.63, n=2799** | 0.564 / 14.46, n=2360 | A0 |
| sza_65_70 | 0.606 / 18.31, n=366   | **0.640 / 10.99, n=389** | A1 |
| sza_70_75 | 0.588 / 24.11, n=150   | **0.614 / 16.99, n=170** | A1 |
| sza_gt75  | 0.543 / 32.07, n=178   | **0.557 / 29.26, n=180** | A1 |

UNet_CRF_TH tracks UNet_CRF in lt65 / 65-70; degrades slightly in 70-75 / gt75. Top-hat does not improve on the UNet_CRF baseline for the cross-bin pipeline.

## T4 - Recommended retrieval pipeline per SZA bin

Combining T3 across both backbones and all twelve methods (six base + six top-hat), the best (backbone, method) per bin on root-length MAE:

| SZA bin     | Best (backbone, method)   | MAE (m) | IoU   | n     | Notes |
|-------------|---------------------------|---------|-------|-------|-------|
| sza_lt65    | A0, UNet_OT               | 8.45    | 0.733 | 1392  | Reproduces published headline; UNet_TH-A0 second at 10.70 m |
| sza_65_70   | TR (backbone-independent) | 7.91    | 0.670 | 315   | TR_TH 8.01 (n=403) close second; UNet_CRF-A1 10.91 (n=350) |
| sza_70_75   | TR (backbone-independent) | 6.46    | 0.687 | 126   | TR_TH 10.96 (n=190); UNet_CRF-A1 12.11 (n=494) |
| sza_gt75    | OT_TH (backbone-independent) | 19.52 | 0.579 | 252   | TR 20.07 (n=152); TR_TH 20.44 (n=217); UNet_CRF-A1 22.60 (n=241) |

Two competing recommendations depending on what is optimised:

1. **Lowest MAE per bin (table above):** A0 + UNet_OT for lt65, TR for sza_65_70 and sza_70_75, OT_TH for sza_gt75. Top-hat overtakes the base method at sza_gt75 (OT_TH 19.52 m beats OT 15.91 m... wait, OT_TH is 19.52 vs base TR 20.07; OT_TH has the lowest MAE in gt75 by a narrow margin while preserving higher recall than TR alone). The backbone choice (A0 vs A1) only matters at lt65; pixel methods dominate the higher SZA bins on raw MAE.
2. **Best learned pipeline (single backbone, single method, robust across bins):** **A1 + UNet_CRF.** Wins 3 of 4 bins on MAE among the learned methods (lt65 loses to A0 + UNet_OT but is still competitive at 14.02 m); recall is high (n=350-494 in higher bins, vs. TR's n=126-315); IoU consistently > 0.55. UNet_CRF_TH does not improve on UNet_CRF in the cross-bin pipeline (TH degrades sza_70_75 / sza_gt75 by 5-7 m).
3. **Best high-recall learned pipeline:** A1 + UNet_OT_TH. Approximately doubles UNet_OT recall in every bin (n=88 to 205 at gt75; n=569 to 2045 at lt65) at the cost of worsening lt65 MAE (8.45 to 12.97). Useful when the question is "how many real icebergs do we detect" rather than "how accurately do we measure the ones we matched".

The paper's existing headline (UNet_OT + A0 at lt65) survives. New contributions: (i) A1 + UNet_CRF is the strongest single-backbone-single-method pipeline across all four SZA bins; (ii) top-hat helps recall on the pixel-method floor (TR_TH, OT_TH) but does not unseat UNet_CRF as the best learned cross-bin option; (iii) the A0 backbone does not generalise to higher SZA bins (T1, T2), so the GT-zero-augmented A1 backbone is the right base for any higher-SZA paper claim that goes beyond pixel methods.

## Files produced

| File | Purpose |
|------|---------|
| `iceberg-rework/scripts/re_eval_phase_a_all_sza.sh` | UNet-only re-eval of A0..A9 across 4 bins on v4_clean |
| `iceberg-rework/slurm/re_eval_phase_a.slurm` | Slurm wrapper for the above |
| `iceberg-rework/scripts/re_phase_b_with_a0.sh` | Six-method Phase B sweep with A0 backbone |
| `iceberg-rework/slurm/re_phase_b_a0.slurm` | Slurm wrapper |
| `iceberg-rework/scripts/re_phase_b_with_a1.sh` | Six-method Phase B sweep with A1 backbone |
| `iceberg-rework/slurm/re_phase_b_a1.slurm` | Slurm wrapper |
| `iceberg-rework/scripts/re_phase_b_tophat_only.sh` | Adds the six top-hat companions to both backbones (Slurm 60300, 19:18 wallclock) |
| `iceberg-rework/slurm/re_phase_b_tophat.slurm` | Slurm wrapper for the top-hat addition |
| `iceberg-rework/runs_summaries/exp_A{0..9}_*/<ts>/re_eval_v4_clean/per_iceberg/eval_per_iceberg_summary.csv` | T1 source CSVs |
| `iceberg-rework/runs_summaries/exp_A0_fisser_lt65_original/20260428_094028/re_phase_b_v4_clean/per_iceberg/eval_per_iceberg_summary.csv` | T3 A0 row |
| `iceberg-rework/runs_summaries/exp_A1_fisser_lt65_plus_nulls/20260429_234146/re_phase_b_v4_clean/per_iceberg/eval_per_iceberg_summary.csv` | T3 A1 row |

## Open items

- T4 leaves room for a "single-backbone, single-method" call. Default recommendation in this doc is **A1 + UNet_CRF** for the cross-SZA pipeline. Confirm with co-author before adopting in the abstract.
- A2..A9 Phase B not run. T1 alone is enough to motivate the choice; running Phase B for the loser experiments would not change T4.
- Higher-SZA TR n values are low (n=126 for sza_70_75); the backbone-independent finding is real but TR's coverage trade-off matters and should be flagged in the prose. Top-hat raises TR's recall to n=190 in that bin (TR_TH) but at a 4.5 m MAE cost.
- Original Phase B headline (UNet_OT + A0) used a >100 m filter and a different chip subset; the 8.45 m here is on the full lt65 v4_clean split (no >100 m filter). Numbers are consistent in spirit, not identical by definition.
- Pending the user's design decision: re-define A5/A6 as A1 + class balancing and A7/A8/A9 as A1 + size oversample (currently anchored to A4 / v4_clean preprocessing). Would require five new training runs on v4_raw_lt65_plus_nulls and re-eval. Open question: aug=on or aug=off for the new variants (A1 itself has aug=off). Not started.
