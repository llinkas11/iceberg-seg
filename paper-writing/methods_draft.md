<!--
methods_draft.md: scaffolding for the paper's Methods section. Numeric claims
must be reconciled against `plan.md` and `reference/*.md` before going into
LaTeX. This file is the prose layer; the v4_clean pipeline, manifest schema,
and progression-driven experimental design live in the code at
`/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/`.

Last updated: 2026-04-30 (v4_clean dataset, binary segmentation, six-method
sweep, Hungarian per-pair evaluator, Fisser-comparable RE, oversample-only
size balancing for the Phase A 2x3 grid, plus explicit scope notes for
methodological branches introduced in planning but not executed).
-->

# Methods

## 2.0 Methodological Branches and Scope

The methodology contains one executed comparison path and several scoped branches. The executed path is the dataset and method progression reported in this paper: `v4_clean` data construction, UNet++ training, six inference methods (TR, OT, UNet++, UNet_TR, UNet_OT, UNet_CRF), and chip-level plus per-iceberg evaluation. All headline tables and figures come from this path unless explicitly labelled otherwise.

Several planning branches were investigated or specified but are not part of the executed comparison. They are documented here before the main pipeline to prevent them from being confused with reported results:

- **Tiny-iceberg annotation recovery**: fixed-threshold and Otsu candidate-generation workflows were developed for visual review of missed small icebergs, excluding overlap with existing annotations and suppressing broad bright regions. These candidates have not been merged into the training labels, and no retrained model in this paper uses them.
- **White top-hat post-processing**: a morphological white top-hat recovery step was specified as a possible `+TH` companion to each of the six base methods. The branch is treated as a sensitivity path for small-iceberg recovery, not as a headline method in the current six-method comparison.
- **Meteorological filtering**: wind speed and air temperature from CARRA were identified as possible confounders for scene selection. No wind-speed or temperature exclusion is applied in the reported dataset; these variables are retained as contextual metadata and possible future stratification axes.
- **CatBoost, contrast-based modelling, and dynamic thresholding**: these were listed as candidate method branches in the planning notes. They are deferred and have no trained model, inference outputs, or evaluation tables in the current paper.

## 2.1 Study Regions and Imagery

The study covers two marine-terminating glacier fjords on the east coast of Greenland: Kangerlussuaq (KQ) and Sermilik (SK). Both fjords drain outlet glaciers of the Greenland Ice Sheet and are documented sources of significant iceberg flux (Moyer and others, 2019; Enderlin and others, 2014). Kangerlussuaq lies near 68.5 N; Sermilik near 65.7 N.

Imagery is drawn from the Sentinel-2 Level-1C (L1C) archive (Drusch and others, 2012). L1C products provide top-of-atmosphere reflectance at 10 m resolution. Three bands are used: B04 (red, 665 nm), B03 (green, 560 nm), and B08 (near-infrared, 842 nm). All three feed the model; B08 is the primary discriminant for iceberg detection at the SZA range of interest.

Scenes are stratified by solar zenith angle (SZA) into four bins: SZA < 65 (`sza_lt65`), 65-70 (`sza_65_70`), 70-75 (`sza_70_75`), and > 75 (`sza_gt75`). The boundaries follow Fisser and others (2024), who documented a threshold near SZA 65 above which threshold-based retrieval error increases substantially. Acquisitions span September through November across multiple years to sample illumination variation while excluding winter darkness and summer melt-pond ambiguity.

## 2.2 Radiometric Calibration Note

All scenes use ESA Sentinel-2 baseline N0500 or later. Under this baseline, ESA applies a +1000 DN offset to all band digital numbers before distribution. When converted to TOA reflectance using the standard 10$^{-4}$ scaling factor, this offset adds +0.10 to all reflectance values relative to the offset-corrected space used by Fisser and others (2024). Our processing pipeline applies the 10$^{-4}$ scaling without subtracting the offset. Consequently, Fisser's calibrated threshold of B08 >= 0.12 corresponds to B08 >= 0.22 in our reflectance space, and all threshold values in this paper are reported in offset-uncorrected space. Because every scene shares the same processing baseline, the +0.10 offset is uniform across all chips and does not affect relative comparisons between methods or SZA bins.

## 2.3 Data Acquisition and Chipping

Scenes were downloaded from the Copernicus Data Space using the OData API, filtered to scene-level cloud cover below 1 % and constrained to per-fjord AOI polygons. Each scene was tiled into 256 x 256 pixel chips (2.56 x 2.56 km at 10 m resolution) using a sliding window with no overlap. Chips are stored as three-band GeoTIFFs (B04 / B03 / B08) with spatial reference metadata preserved from the source scene.

## 2.4 Chip Filtering

Chips were filtered before annotation and training. First, scene-level QA60 cloud + cirrus fraction within chip bounds < 1 % (stricter than the 10 % default in the Roboflow pipeline). Second, an annotation-aware ice-coverage (IC) filter: IC is computed at the chip level as the fraction of non-annotated pixels with B08 >= 0.22, and chips with IC >= 15 % have their bright non-annotated pixels masked to zero (training only; validation and test are never masked). The IC threshold and the annotation-aware refinement are documented in `reference/b08_analysis_results_discussion.md` sections 3.1 through 3.6.

A 40 m root-length cutoff is applied per individual iceberg. Connected components smaller than 16 pixels (1,600 m$^2$) are removed both from the COCO annotations (by area) and from the Fisser pickled masks (by connected-component size after shadow merge). The cutoff matches the Fisser (2025) dataset minimum and removes rasterisation artefacts (median 3 pixels) without losing any iceberg above one resolution element.

## 2.5 Annotation and Dataset Composition

The combined dataset is `v4_clean`, materialised at `data/v4_clean/manifest.json` on the working HPC. Composition before filtering was 984 chips; after the 40 m and IC filters, 916 chips survive into v4_clean.

Two annotation sources contribute:

1. Fisser and others (2024) pickled masks: 398 chips at SZA < 65 from Kangerlussuaq Fjord, with three-class polygon annotations (ocean, iceberg, shadow). Shadow is merged into iceberg before any analysis (see Section 2.6). 330 of these chips survive the 40 m + IC filters.
2. Roboflow-annotated chips: 586 chips at SZA > 65 across both fjords, with single-class iceberg annotations using the SAM3 smart-select tool followed by manual correction. All 586 contribute to v4_clean.

Splits in v4_clean target 65 / 15 / 25 train/val/test. The effective split is 551 / 137 / 228 because the test set is capped at 57 chips per SZA bin (228 = 57 x 4) for cross-bin metric balance. Within v4_clean training, 193 chips received annotation-aware IC masking. The class distribution is binary: ocean 94.4 % / iceberg 5.6 % at the pixel level.

The manifest is content-addressed: every chip row carries `chip_stem`, `tif_path`, `tif_sha`, `sza_bin`, `source`, `n_icebergs`, `has_iceberg`, `ic_aware`, `split`, and `pkl_position`. A `chips_sha` is computed over the sorted (chip_stem, tif_sha, split) tuples and stamped into every downstream output, so cross-experiment comparison can be grounded in identical chip membership.

For evaluation across SZA bins, 330 Fisser pickled chips have synthetic GeoTIFFs at `data/raw_chips/fisser/<chip_stem>.tif`. These carry a 10 m identity transform and no CRS; they exist solely so the polygonisation + rasterisation evaluation pipeline (Section 2.11) can read every chip through one code path.

## 2.6 UNet++ Segmentation Model

The segmentation model is UNet++ (Zhou and others, 2018) with a ResNet-34 encoder pretrained on ImageNet (He and others, 2016). UNet++ extends the standard encoder-decoder with nested dense skip connections that aggregate features across encoder depths, improving boundary localisation for objects at varying scales. The model accepts three-channel (B04 / B03 / B08) input chips at 256 x 256 pixels and produces a single-channel sigmoid output (binary segmentation: iceberg vs not-iceberg). Shadow is merged into iceberg before training (Fisser class 2 -> class 1) so the model never sees shadow as a separate class.

Training uses a composite Dice + BCE loss, with the BCE positive weight set to the inverse of the iceberg-pixel fraction in the training set. Optimiser: AdamW (learning rate 10$^{-4}$, weight decay 10$^{-4}$). Schedule: cosine annealing over 100 epochs, batch size 16. The checkpoint with highest validation IoU across all epochs is retained.

Augmentation is on by default in baseline_v1: random horizontal flip, random vertical flip, and random 90 degree rotation. Augmentation is the single variable that flips between the two adjacent rows of the dataset progression (see Section 2.13); aug-on and aug-off variants run on byte-identical chip sets so the lift can be isolated.

Training is launched via `slurm/baseline_v1.slurm` under the environment variable `ICEBERG_EXPERIMENT=1`. Under that flag the training script refuses to run without an explicit `--seed`, so every published checkpoint is reproducible. The seed (default 42) propagates to Python, NumPy, Torch CPU and CUDA, and the cuDNN deterministic flag.

## 2.7 Fixed NIR Threshold (TR)

The fixed-threshold method applies a chip-wide cutoff of B08 >= 0.22 to each test chip, classifying pixels above the threshold as iceberg. This corresponds to Fisser and others (2024)'s calibrated value of 0.12 in offset-corrected reflectance space, adjusted for the +0.10 uniform offset in our pipeline (Section 2.2). The threshold is applied independently per chip with no spatial context. Connected components smaller than 100 m$^2$ are discarded.

A chip-level IC block filter is also applied at evaluation: chips where the fraction of pixels with B08 >= 0.22 exceeds 15 % are skipped (likely sea-ice contamination). Skipped chips are recorded in `skipped_chips.csv` and counted in the per-method skip total reported alongside accuracy metrics; they do not silently drop out of the comparison.

## 2.8 Per-Chip Otsu Thresholding (OT)

The Otsu method computes an independent threshold per chip from the B08 histogram (Otsu, 1979), adapting to local illumination. The threshold is computed on non-zero B08 pixels. Three guards are applied: chips with Otsu threshold < 0.10 are skipped as radiometrically flat; chips with > 15 % of pixels above the computed threshold are skipped as likely sea-ice dominated; thresholds above 0.50 (rare, sparse-histogram-driven) are clipped. Polygons smaller than 100 m$^2$ are discarded, matching the fixed-threshold baseline.

## 2.9 UNet++ + Threshold (UNet_TR), UNet++ + Otsu (UNet_OT), UNet++ + DenseCRF (UNet_CRF)

Three additional methods consume the UNet++ softmax probability map P(iceberg) instead of the raw B08 reflectance:

- **UNet_TR** binarises P(iceberg) at a fixed cutoff (default 0.5).
- **UNet_OT** runs per-chip Otsu on P(iceberg). The same flat-prob and IC-block guards as Section 2.8 apply.
- **UNet_CRF** applies DenseCRF (Krahenbuhl and Koltun, 2011) post-processing to the chip's softmax probability stack, using bilateral and gaussian pairwise terms. The chip's reflectance image provides the bilateral term so DenseCRF can pull boundaries onto reflectance edges. Default parameters: `sxy_bilateral = 80, srgb = 13, sxy_gaussian = 3, compat = 10, iterations = 5`.

All three methods share the same UNet++ checkpoint and the same softmax probabilities, so any difference between them is attributable to the post-processing rule alone.

## 2.10 White Top-Hat Filtering for Small-Iceberg Recovery (deferred)

A morphological white top-hat post-processing step was scoped for small-iceberg recovery but is not in the current six-method comparison. The intended branch applies a disk-structured white top-hat filter to the B08 band, thresholds the response, removes components below the 40 m root-length cutoff, subtracts pixels already covered by a base method, and writes a merged base-plus-recovered `+TH` output for each base method. See `tiny_icebergs_methods_addendum.md` for the annotation-review context; any `+TH` results must be reported separately from the six headline methods.

## 2.11 Evaluation

Predictions for every method are stored as GeoPackages with one polygon per detected iceberg, carrying `area_m2`, `class_name`, and `source_file`. Evaluation rasterises these polygons back to pixel masks using the chip's own affine transform and runs two parallel pipelines.

**Pixel-level metrics (`scripts/eval_methods.py`)**: per chip, computes IoU, precision, recall, F1, predicted area in m$^2$, ground-truth area in m$^2$, absolute area error, and squared area error against the iceberg-class mask. Aggregations by (method, sza_bin) report mean IoU / precision / recall / F1, area MAE (mean absolute area error in m$^2$), area MSE, chip count, and skip count. Aggregations also break out a GT-positive-only view, which excludes chips with zero ground-truth iceberg pixels.

For a predicted binary mask $P$ and reference mask $R$, chip-level IoU is:

$$\mathrm{IoU} = \frac{|P \cap R|}{|P \cup R|} = \frac{TP}{TP + FP + FN}$$

For chip $i$, predicted and reference iceberg area are:

$$A_{\mathrm{pred},i} = n_{\mathrm{pred},i} \cdot r^2,\quad A_{\mathrm{ref},i} = n_{\mathrm{ref},i} \cdot r^2$$

where $r = 10$ m is the Sentinel-2 pixel size. Area error for chip $i$ is:

$$e_i = A_{\mathrm{pred},i} - A_{\mathrm{ref},i}$$

For a reporting group with $N$ evaluated chips:

$$\mathrm{MAE}_{area} = \frac{1}{N}\sum_{i=1}^{N}|e_i|$$

$$\mathrm{MSE}_{area} = \frac{1}{N}\sum_{i=1}^{N}e_i^2$$

**Per-pair metrics (`scripts/eval_per_iceberg.py`)**: per chip, ground-truth and predicted masks are decomposed into connected components. Pairwise IoU between every GT and predicted component is computed, with a bounding-box prefilter that skips disjoint pairs without touching pixel data. Pairs are matched by Hungarian assignment on cost `1 - IoU` (`scipy.optimize.linear_sum_assignment`); pairs with IoU < 0.3 are dropped as unmatched. This is described in detail in Section 2.12.

Both pipelines stamp the experiment id, chips_sha, manifest_id, and method_config_sha into their output CSVs, so any downstream comparison can verify the inputs were identical.

## 2.12 Per-Pair Area Error Metrics (Fisser-Comparable)

Beyond the area-distribution comparison, per-iceberg error is quantified against a visually delineated reference set and reported as error statistics comparable to Fisser and others (2024). This places the six methods (TR, OT, UNet++, UNet_TR, UNet_OT, UNet_CRF) on the same accuracy axis as Fisser's published Sentinel-2 results.

### Reference set

Reference polygons are the iceberg masks of the test split of v4_clean. Per-chip ground-truth is the corresponding row of `y_test.pkl` (binary uint8). Connected components of the iceberg class are extracted with `scipy.ndimage.label` and decomposed into individual reference icebergs. The same 40 m root-length filter applied during dataset construction (Section 2.4) is implicit, since smaller components were already removed from `y_test.pkl`. Each component carries `area_m2`, `area_px`, and a bounding-box slice for downstream use.

A second reference layer (the iceberg-labeler hand-validated polygons) is wired in as a future drop-in via `--reference labeler` once the labeler GPKG export pipeline lands. The per-pair pipeline accepts both reference sources without code change.

### Metrics

Let $A_\mathrm{S2}$ be a predicted iceberg area and $A_\mathrm{ref}$ the matched reference area. For each matched pair:

$$\mathrm{RE} = 100 \cdot \frac{A_\mathrm{S2} - A_\mathrm{ref}}{A_\mathrm{ref}} \quad (\%)$$

following Fisser and others (2024, eqn 2). Alongside RE, we report two absolute error metrics and the per-pair IoU used in matching:

$$\mathrm{AE}_\mathrm{area} = |A_\mathrm{S2} - A_\mathrm{ref}| \quad (\text{m}^2)$$

$$\mathrm{AE}_\mathrm{root} = |\sqrt{A_\mathrm{S2}} - \sqrt{A_\mathrm{ref}}| \quad (\text{m})$$

$$\mathrm{IoU} = \frac{|P \cap R|}{|P \cup R|}$$

Positive RE indicates Sentinel-2 overestimation. Per-bin Mean Absolute Error (MAE) is the mean of $\mathrm{AE}_\mathrm{area}$ or $\mathrm{AE}_\mathrm{root}$ over matched pairs. For $M$ matched iceberg pairs:

$$\mathrm{MAE}_{area} = \frac{1}{M}\sum_{j=1}^{M}|A_{\mathrm{S2},j} - A_{\mathrm{ref},j}|$$

$$\mathrm{MAE}_{root} = \frac{1}{M}\sum_{j=1}^{M}|\sqrt{A_{\mathrm{S2},j}} - \sqrt{A_{\mathrm{ref},j}}|$$

$$\mathrm{MSE}_{area} = \frac{1}{M}\sum_{j=1}^{M}(A_{\mathrm{S2},j} - A_{\mathrm{ref},j})^2$$

Root-length MAE is preferred for cross-size comparison because it does not amplify with the square of iceberg linear scale.

### Matching

Per chip, the pairwise IoU matrix between reference and predicted components is computed, with a bounding-box overlap prefilter that skips disjoint-bbox pairs in O(1). For overlapping pairs, intersection is computed only over the bbox-intersection window, and union is derived as `area_ref + area_pred - inter` rather than a second full-chip OR. Matching is by Hungarian assignment on `1 - IoU` (`scipy.optimize.linear_sum_assignment`). Pairs with IoU < 0.3 are dropped as unmatched. Unmatched references count as false negatives; unmatched predictions count as false positives. For each (method, sza_bin) combination the match rate `n_matched / n_ref` is reported alongside the metric averages as a selection-bias disclosure: a method that matches 30 % of ground-truth icebergs is not directly comparable to one that matches 90 % without that context.

### SZA-binned reporting

For each method, per-pair MAE on area, per-pair MAE on root length, and per-pair IoU are reported per SZA bin (`sza_lt65`, `sza_65_70`, `sza_70_75`, `sza_gt75`) so the SZA dependence of method accuracy can be read directly off one row per method. The full per-pair CSV (`eval_per_iceberg.csv`) carries every match individually, so finer stratifications (per ground-truth size bucket, per region, per chip source) can be computed downstream.

### Non-comparability of Fisser equation 5

Fisser and others (2024, eqn 5) standardise the SZA-dependent error to a 56 reference anchored in an independent Dornier aerial survey. Our dataset carries no Dornier-equivalent calibration, so applying equation 5 with our own visually delineated reference would be circular: the anchor and the thing being standardised would share the same source. We therefore report raw, interpolated, and smoothed RE per Fisser equations 3 and 4 over the observed SZA range, but do not compute the standardised SRE. We perform no extrapolation beyond the sampled range.

### Expected qualitative signal

UNet++ is trained to separate iceberg from ocean only, with shadow merged before analysis. As SZA increases the shadow that accompanies each iceberg grows, but UNet++ continues to exclude it, so we expect UNet++ RE to trend negative with rising SZA. The B08 fixed threshold and per-chip Otsu both respond to reflectance alone and include the high-reflectance iceberg edges that bleed into shadow pixels when the shadow darkens; we expect their RE to trend positive with rising SZA. DenseCRF sharpens UNet++ boundaries and is expected to track UNet++ closely. Quantifying the sign and magnitude of this SZA-dependent split is the primary contribution of this paper.

No ground-truth iceberg area measurements independent of photointerpretation are available. Using visually delineated labels as the reference inherits the interpreter's identification of the iceberg-water edge in B08, which is itself SZA-dependent at the highest angles. This limitation is shared with Fisser and others (2024, Greenland leg), who use the same class of reference. The per-bin match-rate and IQR statistics disclose the resulting uncertainty.

## 2.13 Experimental Design and Reproducibility

Experiments follow a controlled progression. Phase A walks the dataset axis on lt65-only chips, with four controlled variables in order: (1) preprocessing pipeline (Fisser cleaning vs ours), (2) null-chip injection, (3) augmentation, (4) class and size balancing (a 2x3 grid across A4-A9). A0 anchors on `v4_raw_lt65` (the same Fisser lt65 chips with no 40 m component filter and no IC mask, faithful to Fisser's published recipe); A2 anchors on `v4_clean_lt65` (the 330 chips that pass our IC chip-drop, with 40 m and IC mask applied). The A0 to A2 contrast isolates the preprocessing pipeline as a single controlled variable. A3 to A9 inherit `v4_clean_lt65_plus_nulls` and vary one balancing axis at a time. Phase B walks the method axis on the Phase A winner: B0 (fixed B08 threshold) -> B5 (UNet++ + DenseCRF). Each row changes exactly one controlled variable from the row before it; multi-family changes (Fisser reproduction changing both the chip source and the augmentation flag) are explicit via a `controlled_variable:` declaration in the experiment YAML.

The configuration system has four layers:

1. `configs/baselines/baseline_v1.yaml`: canonical baseline. Every experiment inherits this.
2. `configs/balancing/scheme_*.yaml`: 12 declarative balancing schemes (A through L).
3. `configs/datasets/`: dataset recipes; one per source variant.
4. `configs/experiments/exp_*.yaml`: one file per experiment (19 files at the time of writing). Top-level `change:` block declares the deltas from baseline.

A validator (`scripts/validate_experiment.py`) refuses any experiment whose `change:` block touches more than one controlled family unless `controlled_variable:` is set explicitly. The runner (`scripts/run_experiment.py`) drives an experiment through five stages: manifest, train, infer, evaluate, figures. Each stage stamps the experiment id, manifest chips_sha, resolved config_sha, and git_sha into its outputs. A figure registry (`scripts/_fig_registry.py`) routes every `savefig` through an append-only `fig-archive/` and a live `figures.md` index, and exposes a `write_table` helper for tables-as-PNGs.

Source code, configuration, and run logs are version-controlled at `github.com/llinkas11/iceberg-seg`. Materialised manifests, model checkpoints, and inference outputs live on the HPC working tree at `/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/` and are not in version control because of size.

### 2.13.1 Class balancing

Stage 1 of `balance_training.py` operates on the GT-positive vs GT-zero ratio per SZA bin. Three direction choices are encoded as separate schemes:

- **scheme_E (natural)**: no resampling. The training distribution is left as-is, and is the implicit default for `baseline_v1` (`balancing_scheme: none`).
- **scheme_D (fixed positive majority)**: resample each bin to a 2:1 GT+ : GT0 ratio in the same direction regardless of the natural distribution. The intent is to bias the loss toward the rarer positive signal.
- **scheme_I (adaptive majority:minority)**: resample each bin to 2:1 majority : minority where "majority" is the class with the larger natural per-bin count. Lets the natural class-imbalance signal lead.

A4, A5, A6 isolate the effect of these three settings on a fixed dataset (our lt65 + nulls + augmentations).

### 2.13.2 Size balancing (oversample-only)

Stage 2 of `balance_training.py` operates on the per-iceberg root-length bin within the GT-positive pool. Three bins: `rl_40_100`, `rl_100_300`, `rl_300_plus`. The default 2:1 Fisser-style implementation undersamples large bins down to a target ratio, which limits training data. To avoid that, the size balancer offers an oversample-only mode (added 2026-04-27): replicate the small and mid bins up to the count of the largest bin, never undersample. A `max_oversample_ratio` cap (default 4.0) bounds how aggressively a small bin can grow, so a 5-chip bin never trains 12x harder than a 60-chip bin.

Oversample-only pairs deliberately with augmentation. A replicated chip is shown to the model multiple times per epoch, but each time receives a different random hflip / vflip / rot90 (8 possible variants total). The gradient sees more *distinct geometric instances* of the rare bin without seeing identical pixels twice. This is fundamentally different from in-line augmentation alone, which diversifies each chip's geometry but does nothing to rebalance the per-class gradient frequency the optimiser sees over the course of an epoch.

scheme_J implements size balancing alone (no class change). schemes K and L compose stage-1 (D or I) with stage-2 (J) so the dataset progression can isolate each axis.

### 2.13.3 Preprocessing-pipeline isolation (A0 vs A2) and the v4_raw companion

Two preprocessing operations distinguish our pipeline from Fisser's published recipe: a 40 m root-length filter applied to connected components within each chip's mask, and an annotation-aware IC pixel mask that zeros bright non-annotated pixels in training chips with IC >= 15 % of non-annotated area. Auditing the lt65 split of `v4_clean` against the raw Fisser pkls quantifies the footprint of each operation. The 40 m filter removes 41,644 of 70,818 mask components on Fisser lt65 chips (58.8 %), and 312 of 330 chips have at least one component dropped. The IC pixel mask zeros pixels in 129 of 226 training Fisser lt65 chips (57.1 %). Two chips become effectively negative after the 40 m filter; no chip is removed from the manifest by either step.

Because both operations substantially edit Fisser's data, the experimental design treats preprocessing as a controlled variable rather than a fixed setting. A0 and A1 anchor on `v4_raw_lt65` (398 lt65 Fisser chips, no filter, no mask, no IC chip-drop), preserving Fisser's published cleaning. A2 through A9 anchor on `v4_clean_lt65` (330 chips with our 40 m + IC pipeline applied). The A0 to A2 contrast isolates the preprocessing pipeline. To support a robustness check across all SZA bins, a parallel `v4_raw` manifest (984 chips, no filters, all bins) backs a companion baseline run that reports the same per-pair MAE / IoU / detection tables as `v4_clean`. The paper presents both tables; the delta is read directly as the preprocessing-pipeline impact.

This contrast also defines the broader dataset-philosophy axis in the study: an idealized iceberg-only retrieval setting versus a realistic fjord-scene setting that retains sea ice, melange, and ambiguous bright background.

### 2.13.4 Phase A 2x3 grid

A4-A9 form a 2x3 grid:

|                  | Size balance: off | Size balance: oversample (J) |
|------------------|-------------------|------------------------------|
| Class: none      | A4                | A7                            |
| Class: fixed pos | A5                | A8                            |
| Class: adaptive  | A6                | A9                            |

Each row reads "same as the column-1 cell, but with size oversampling added." The grid is designed to read off the marginal lift of each balancing axis when the other axis is held fixed. Pairs to look at are A4-A5-A6 (class on a fixed dataset), A4-A7 (size on natural class distribution), A5-A8, A6-A9 (size on top of class).

This grid is a design structure, not a guarantee that all six cells remain empirically distinct. On `v4_clean_lt65_plus_nulls`, GT-positive chips are the natural majority, so scheme_D and scheme_I produce identical class-balanced training sets. Under the 4x oversampling cap, schemes J, K, and L also converge to the same size-balanced training set. The resulting Phase A comparison therefore reads as three realized training conditions: A4, A5/A6, and A7/A8/A9.

### 2.13.5 Reporting metrics on the run

Every experiment run produces three CSV families:

- `evaluation/eval_summary.csv`: chip-level mean IoU / precision / recall / F1 / area MAE / area MSE per (method, sza_bin), plus GT-positive-only aggregations.
- `per_iceberg/eval_per_iceberg_summary.csv`: per-pair mean IoU / area MAE / root-length MAE / RE per (method, sza_bin) using Hungarian matching.
- `per_iceberg/eval_per_iceberg_detection.csv`: per-method `n_ref`, `n_pred`, `n_matched`, match rate, precision. Selection-bias disclosure for cross-method MAE comparison.

The per-pair table is the headline for Fisser comparability. The chip-level table is the segmentation-community-standard companion. Detection stats are required context for any cross-method MAE claim.

The headline tables are reported on `v4_clean` (canonical baseline). A parallel run on `v4_raw` (no 40 m filter, no IC mask, all SZA bins) produces the same three CSV families. The delta between the two table sets, reported under matched (method, sza_bin) pairs, attributes performance differences to the preprocessing pipeline alone.

## Planned Methodology Figures

### Fig. 4. Evaluation schematic

Create a reader-friendly schematic explaining how binary predictions become the reported per-iceberg metrics. The figure should use a left-to-right pipeline:

`ground-truth mask + predicted mask -> connected components -> IoU matrix -> Hungarian matching -> matched/unmatched objects -> MAE / IoU / MSE reporting table`

The schematic should include:

- **Ground-truth components**: show a simple mask or real test chip with ground-truth iceberg components outlined in blue and labelled `GT1`, `GT2`, `GT3`.
- **Predicted components**: show predicted iceberg components outlined in orange and labelled `P1`, `P2`, `P3`.
- **IoU-based matching**: show a compact IoU matrix with ground-truth components as rows and predicted components as columns. Highlight the selected Hungarian matches and note that matches with IoU < 0.3 are dropped.
- **Matched and unmatched outcomes**: explicitly mark matched pairs as evaluated pairs, unmatched ground-truth objects as false negatives, and unmatched predictions as false positives.
- **Metric table**: include a compact reporting table with columns `Method`, `SZA bin`, `Area MAE`, `Root-length MAE`, `Mean IoU`, `MSE`, and `Match rate`. Use real baseline values if available; otherwise use clearly schematic placeholder values.

Implementation target:

- Add `iceberg-rework/scripts/make_fig4_evaluation_schematic.py`.
- Prefer a real `v4_clean` test chip with multiple ground-truth and predicted components.
- Load baseline outputs from `runs/exp_baseline_v1/20260424_185158/inference/` and `runs/exp_baseline_v1/20260424_185158/per_iceberg/` if available locally.
- If real outputs are unavailable, generate simple synthetic masks that exactly illustrate the same connected-component, IoU, Hungarian-match, false-negative, and false-positive logic.
- Save through the figure registry to `iceberg-rework/viz/paper_figures/fig4_evaluation_schematic.png`.
- Optionally copy the final PNG to `paper-writing/figures/fig4_evaluation_schematic.png`.

Draft caption:

Fig. 4. Evaluation workflow for per-iceberg metrics. Ground-truth and predicted masks are decomposed into connected components, pairwise IoU is computed, and Hungarian assignment selects non-overlapping matches. Matched pairs contribute area MAE, root-length MAE, IoU, and MSE; unmatched ground-truth objects are counted as false negatives and unmatched predictions as false positives. Match rate is reported alongside error metrics to disclose selection bias.

Validation checklist:

- Confirm the figure explicitly shows ground-truth components, predicted components, the IoU matrix, Hungarian matching, unmatched false negatives, unmatched false positives, and the MAE / IoU / MSE reporting table.
- Confirm the schematic matches the evaluator: Hungarian assignment on `1 - IoU`, matches below IoU < 0.3 dropped, MAE primary, IoU secondary, and MSE supplementary.
- Confirm the figure makes no new model-method claims; it explains evaluation only.
