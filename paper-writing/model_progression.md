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

Each step changes one thing from the step before. Aug is OFF from A0 through
A3 so Fisser-equivalence is preserved; A4 is the first step where we deviate
from Fisser's recipe.

| # | Dataset                                   | Aug | Balancing           | Varies vs prior                     | Motivation                                                                                                                                                 | YAML                                |
|---|-------------------------------------------|-----|---------------------|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|
| A0 | Fisser lt65, positive-only                | off | scheme_A            | anchor                              | Reproduce Fisser's published setup. If we do not match their numbers here, the rest of the comparison is unfounded.                                        | exp_A0_fisser_lt65_original         |
| A1 | Fisser lt65 + GT-zero chips               | off | scheme_B            | +nulls                              | Does showing the model empty water stop it hallucinating icebergs on null chips? Fisser's positive-only design cannot answer this.                          | exp_A1_fisser_lt65_plus_nulls       |
| A2 | Our lt65, positive-only                   | off | scheme_A            | swap chip source (Fisser -> ours)   | Holding balancing constant, does the iceberg-labeler-annotated lt65 match or beat Fisser's chip set? Isolates data-source effect.                          | exp_A2_our_lt65                     |
| A3 | Our lt65 + GT-zero chips                  | off | scheme_C            | +nulls (on our source)              | Same question as A1 but on our source. If A0->A1 and A2->A3 move the same direction, the null-chip effect is source-independent.                           | exp_A3_our_lt65_plus_nulls          |
| A4 | Our lt65 + nulls + augmentations on       | on  | scheme_C            | +hflip / vflip / rot90              | First step where we leave Fisser's recipe. Does synthetic rotation/flipping recover from our smaller absolute training volume?                             | exp_A4_our_lt65_plus_nulls_aug      |
| A5 | A4 + 2:1 positive-biased balancing        | on  | scheme_D (2:1 fixed)| +class balance, fixed pos-majority  | Bias training loss toward the rarer positive signal. Tests "if we let the model see nulls but still emphasise positives, do we get the best of both?"      | exp_A5_our_lt65_plus_nulls_aug_2pos |
| A6 | A4 + 2:1 adaptive balancing (fork from A4)| on  | scheme_I (adaptive) | +class balance, majority:minority   | Swap the direction: 2:1 whichever way the natural per-bin distribution leans. Tests "does deferring to the data's own class signal beat forced pos-bias?" | exp_A6_our_lt65_plus_nulls_aug_adapt|

Notes:

- A5 and A6 are siblings. Both compare against A4, not against each other
  first.
- A2 vs A0 and A3 vs A1 form a 2x2: (source) x (nulls). The 2x2 is what
  supports a causal statement like "nulls help regardless of source".
- A4 is the natural "no-balancing" control for A5 and A6.

---

## Phase B: method progression (all SZA bins, Phase A winner)

Once Phase A selects a dataset (most likely A4, A5, or A6), we freeze it and
sweep the six methods over all four SZA bins. Every row shares the same
trained UNet checkpoint and the same test chips; only the post-processing
differs. One training run, six reports.

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
- `iceberg-rework/scripts/eval_per_iceberg.py`: Hungarian matching + per-pair
  MAE + IoU + RE. All progression numbers flow through this one script.
