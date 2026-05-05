# Model progression

The paper's comparison rests on a progression, not a flat table. Every
experiment differs from its predecessor in exactly one controlled variable;
each step has a reader-facing motivation that survives the prose.

Two phases:

- **Phase A** walks the *dataset*: starts at Fisser's published recipe,
  progressively adds variables (null chips, our chip source, augmentations,
  balancing) to land on our best setup.
- **Phase B** walks the *method*: with the Phase A winner frozen as the
  training dataset, sweeps threshold, Otsu, UNet++, UNet+threshold,
  UNet+Otsu, UNet+CRF.

Metric ordering: **MAE first, IoU second.** Fisser does not report IoU or
MSE, so MAE on per-pair iceberg area and root length is the only number that
connects to their tables. IoU comes second for segmentation-community
reviewers. MSE is supplementary on request.

---

## Phase A: dataset progression (lt65-scoped)

Each step changes one thing from the step before. The progression has four
controlled-variable axes in order: (1) the preprocessing pipeline (A0 vs A2),
(2) null-chip injection (A2 vs A3), (3) augmentation (A3 vs A4), (4) class
and size balancing (A4 through A9, a 2x3 grid). Aug is off from A0 through
A3 so Fisser-equivalence is preserved through the first three steps; A4 is
the first step where we deviate from Fisser's recipe.

A0 anchors on `v4_raw_lt65` (398 lt65 Fisser chips, no 40 m filter, no IC
mask, no IC chip-drop). A2 anchors on `v4_clean_lt65` (the same 330 of those
chips that pass our IC chip-drop, with 40 m and IC mask applied). A0 -> A2
isolates the preprocessing-pipeline variable: same source, our cleaning vs
theirs.

| # | Dataset framing                                          | Manifest                       | Aug | Balancing           | Varies vs prior                     | Motivation                                                                                                                                                                 | YAML                                |
|---|----------------------------------------------------------|--------------------------------|-----|---------------------|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|
| A0 | Fisser lt65, Fisser preprocessing (no 40 m, no IC)       | `v4_raw_lt65` (398 chips)       | off | scheme_A            | anchor                              | Reproduce Fisser's published setup with their cleaning. If we do not match their numbers here, the rest of the comparison is unfounded.                                   | exp_A0_fisser_lt65_original         |
| A1 | A0 + GT-zero chips                                       | `v4_raw_lt65_plus_nulls`        | off | scheme_B            | +nulls (under raw preprocessing)    | Does showing the model empty water stop it hallucinating icebergs on null chips? Holds preprocessing fixed at Fisser's recipe.                                            | exp_A1_fisser_lt65_plus_nulls       |
| A2 | Fisser lt65, our preprocessing (40 m + IC mask)          | `v4_clean_lt65` (330 chips)     | off | scheme_A            | swap preprocessing pipeline         | Same Fisser chips, our 40 m root-length filter and IC pixel mask. Isolates the preprocessing-pipeline effect on the published Fisser baseline.                            | exp_A2_our_lt65                     |
| A3 | A2 + GT-zero chips                                       | `v4_clean_lt65_plus_nulls`      | off | scheme_C            | +nulls (under our preprocessing)    | Same question as A1 under our preprocessing. If A0->A1 and A2->A3 move the same direction, null-chip effect is preprocessing-independent.                                  | exp_A3_our_lt65_plus_nulls          |
| A4 | A3 + augmentations on                                    | `v4_clean_lt65_plus_nulls`      | on  | scheme_C            | +hflip / vflip / rot90              | First step where we leave Fisser's recipe. Does synthetic rotation / flipping recover from our smaller absolute training volume?                                          | exp_A4_our_lt65_plus_nulls_aug      |
| A5 | A4 + 2:1 positive-biased balancing                       | `v4_clean_lt65_plus_nulls`      | on  | scheme_D (2:1 fixed)| +class balance, fixed pos-majority  | Bias training loss toward the rarer positive signal. Tests "if we let the model see nulls but still emphasise positives, do we get the best of both?"                       | exp_A5_our_lt65_plus_nulls_aug_2pos |
| A6 | A4 + 2:1 adaptive balancing (fork from A4)               | `v4_clean_lt65_plus_nulls`      | on  | scheme_I (adaptive) | +class balance, majority:minority   | Swap the direction: 2:1 whichever way the natural per-bin distribution leans. Tests "does deferring to the data's own class signal beat forced pos-bias?"                  | exp_A6_our_lt65_plus_nulls_aug_adaptive |
| A7 | A4 + size oversample                                     | `v4_clean_lt65_plus_nulls`      | on  | scheme_J (size only)| +size balance via oversample (4x cap)| Replicate small / mid root-length bins up to the largest bin, never undersample. Tests the size axis on its own. With aug on, replicas get distinct geometric views per epoch. | exp_A7_our_lt65_plus_nulls_aug_size |
| A8 | A5 + size oversample                                     | `v4_clean_lt65_plus_nulls`      | on  | scheme_K (D + J)    | +size balance on top of fixed pos   | Class and size both balanced; class step uses fixed positive majority. Reads off the marginal lift of size balancing when class is already addressed.                       | exp_A8_our_lt65_plus_nulls_aug_2pos_size |
| A9 | A6 + size oversample                                     | `v4_clean_lt65_plus_nulls`      | on  | scheme_L (I + J)    | +size balance on top of adaptive    | Class and size both balanced; class step uses adaptive direction. Closes the 2x3 grid: class in {none, fixed-pos, adaptive} crossed with size in {none, oversample}.        | exp_A9_our_lt65_plus_nulls_aug_adaptive_size |

Notes:

- A5, A6, and A7 all compare against A4. A8 compares against A5; A9
  compares against A6.
- A0 vs A2 and A1 vs A3 form a 2x2: (preprocessing) x (nulls). A4 / A7 vs
  A5 / A8 vs A6 / A9 form a 2x3: (size balance: off, on) x (class balance:
  none, fixed-pos, adaptive).
- A4 is the natural "no-balancing" control for A5, A6, and A7.
- Oversampling pairs with augmentation: replicated chips get different
  random hflip / vflip / rot90 each pass, so the gradient sees more
  distinct instances of the rare bin without seeing the underlying pixels
  twice. The 4x cap on per-bin replication bounds memorisation risk on
  very small bins.
- The 40 m filter audit on Fisser lt65 (2026-04-27) found that 41,644 of
  70,818 components (58.8 %) are removed and 312 of 330 chips altered;
  IC masking touches 129 of 226 training chips (57.1 %). The A0 vs A2 cell
  is therefore not cosmetic: it isolates a substantial preprocessing
  intervention.
- The two `*_plus_nulls` manifests (A1, A3-A9) are now on disk
  (`31516dc09828007e...` for v4_clean_lt65_plus_nulls and
  `1e21d08fc96c3d53...` for v4_raw_lt65_plus_nulls), built by
  `scripts/build_lt65_nulls.py --merge_into_manifest`. Each appends the 29
  GT0 chips from `reference/lt65_nulls_selected.csv` to the base manifest's
  TRAIN split; val and test pkls pass through byte-stable.

---

## Phase B: method progression (Phase A winner)

Once Phase A selects a dataset, we freeze it and sweep the six methods. Every
row shares the same trained UNet checkpoint and the same test chips; only the
post-processing differs. One training run, six reports.

The original Phase B (Slurm 20260503) ran on the A0 checkpoint over the
`sza_lt65` test split only, because Phase A itself was lt65-scoped. The
2026-05-05 follow-up (Slurm 60296, 60297) re-runs the same six-method sweep
with both A0 and A1 backbones over all four SZA bins on the v4_clean test
split; see "Higher-SZA generalisation and backbone re-eval" below for the
backbone comparison.

| # | Method                      | What varies vs prior step       | Motivation                                                                                                                         | YAML                              |
|---|-----------------------------|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| B0 | Fixed NIR threshold (0.22)  | anchor                           | Fisser's baseline. Method-equivalent to running their script on our test set.                                                     | exp_B0_method_threshold           |
| B1 | Per-chip Otsu on B08        | adaptive threshold (no training) | Does chip-adaptive thresholding beat Fisser's fixed rule under varying SZA illumination? Thresholds cost nothing at inference.     | exp_B1_method_otsu                |
| B2 | UNet++ argmax               | learned segmentation             | The learned-vs-hand-engineered divide. Reviewer's question: "is the deep model actually necessary, or does Otsu get you there?"  | exp_B2_method_unet                |
| B3 | UNet++ + threshold on probs | threshold on continuous probs    | Does exposing softmax + a tunable threshold recover precision/recall trade-offs that argmax hides?                                 | exp_B3_method_unet_threshold      |
| B4 | UNet++ + Otsu on probs      | Otsu in model output space       | Same adaptive-threshold idea but after learning. Per-chip adaptive on the model's confidence.                                      | exp_B4_method_unet_otsu           |
| B5 | UNet++ + DenseCRF           | structured post-processing       | Does explicit spatial coherence clean up boundary artifacts the UNet alone produces at high SZA? Reviewer bait.                    | exp_B5_method_unet_crf            |

---

## Metrics reported at every step

Primary:

- **MAE on per-pair iceberg area (m2).** Matched via Hungarian on 1 - IoU,
  post-filter IoU >= 0.3. Reported per (method, sza_bin, experiment_id).
- **MAE on per-pair root length (m).** `sqrt(A_pred) - sqrt(A_ref)` in
  absolute value, same matching.
- **IoU** on matched pairs. Mean + median.

Secondary:

- **Detection stats.** `n_ref`, `n_pred`, `n_matched`, match rate, mean IoU
  on matched set. This is a selection-bias disclosure: MAE on 30%-matched
  pairs is not comparable to MAE on 90%-matched pairs without context.
- **MSE on area.** Supplementary, reported on request.
- **Relative error (Fisser Eq. 2).** For the Fisser-comparable rows (A0, A1)
  only. Over the observed SZA range, with Eq. 3 interpolation + Eq. 4 5-deg
  smoothing. Eq. 5 SRE explicitly not applied; no independent calibration
  set.

---

## Motivation figures (one per row)

The paper carries a compact grid of before-after panels, one per
progression step. Each panel is:

- One representative test chip.
- Column 1: ground-truth polygons.
- Column 2: prediction from the progression step's predecessor.
- Column 3: prediction from this step.
- Caption: the one-sentence motivation from the table above.

This collapses the progression story into one full-page figure a reader can
scan without reading prose.

---

## Higher-SZA generalisation and backbone re-eval (added 2026-05-05)

Both Phase A and the original Phase B were `sza_lt65`-scoped (manifests v4_raw_lt65* and v4_clean_lt65* contain only lt65 chips). Three follow-up runs on moosehead (Slurm 60293, 60296, 60297) re-evaluate every Phase A checkpoint and re-run Phase B with two different backbones across all four SZA bins on the unifying v4_clean test split. Full tables (T1 per-bin x A0..A9, T2 best non-A0 per bin, T3 A0 vs A1 Phase B, T4 recommended pipeline) live in `shib_end_to_end/phase_a_higher_sza_t1_t4.md`.

Headline findings:

- **A1 wins every higher-SZA bin** on both per-pair IoU and per-pair root-length MAE (T1, T2). A0 still wins lt65. The 29 GT-zero chips that distinguish A1 from A0 act as a regulariser on bins the model never saw at training: A1's mean MAE across the three higher-SZA bins is 28.01 m vs A0's 33.33 m, a 16% reduction.
- **The published lt65 headline (A0 + UNet_OT, 8.18 m MAE) reproduces** within rounding on the v4_clean lt65 split (8.45 m here); T3's A0 + UNet_OT lt65 cell is the canonical comparison.
- **A1 + UNet_CRF is the strongest single-backbone-single-method pipeline across all four bins** when the goal is one method that works everywhere; it wins 3 of 4 bins on MAE among learned methods (lt65 loses to A0 + UNet_OT but is competitive at 14.02 m MAE) and keeps n high (350-494 in higher bins) where TR's coverage drops to n=126 at sza_70_75.
- A0 + UNet_OT (lt65) and A1 + UNet_CRF (higher SZA) is the per-bin-best combination if the paper allows different methods per bin. T4 in the new artifact lays out both options.

These results do not change Phase A's lt65 winner (still A0); they extend the model-selection story to higher SZA bins that Phase A could not previously address.

---

## What is not in the progression

- Experiment `exp_ablation_no_augmentation` (the previous `exp_05`): aug-off
  ablation against the full baseline. Not a progression step; lives outside
  the table.
- Methods beyond the six in Phase B: top-hat variants, catboost, dynamic
  thresholding. Deferred per `plan.md`.

---

## Where the code lives

- `iceberg-rework/configs/baselines/baseline_v1.yaml`: canonical baseline.
  Phase A experiments are single-variable rollbacks from baseline; Phase B
  experiments re-run baseline inference with a different method focus.
- `iceberg-rework/configs/experiments/exp_A{0..6}_*.yaml`: Phase A.
- `iceberg-rework/configs/experiments/exp_B{0..5}_*.yaml`: Phase B.
- `iceberg-rework/configs/balancing/scheme_D_two_pos_per_null.yaml`: fixed
  2:1 pos-majority (A5).
- `iceberg-rework/configs/balancing/scheme_I_two_to_one_adaptive.yaml`:
  adaptive 2:1 majority:minority (A6).
- `iceberg-rework/configs/balancing/scheme_J_oversample_size_balanced.yaml`:
  oversample-only size balancing, 4x cap (A7).
- `iceberg-rework/configs/balancing/scheme_K_two_pos_per_null_size_balanced.yaml`:
  D + J composition (A8).
- `iceberg-rework/configs/balancing/scheme_L_adaptive_size_balanced.yaml`:
  I + J composition (A9).
- `iceberg-rework/scripts/eval_per_iceberg.py`: Hungarian matching + per-pair
  MAE + IoU + RE. All progression numbers flow through this one script.
