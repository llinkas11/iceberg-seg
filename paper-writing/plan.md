# Iceberg Segmentation Pipeline: Project State

**Status:** Refactor complete (Phases 1-6). v4_clean dataset built and frozen. baseline_v1 trained and evaluated. Phase A 2x3 grid (size balance x class balance) registered as 10 YAMLs, runner ready. Phase B method sweep is a single inference dispatch on the trained checkpoint, runner ready. Per-pair MAE + IoU tables published for the canonical baseline. **2026-05-05 addendum:** Phase A re-eval on all four SZA bins and Phase B backbone comparison (A0 vs A1) complete; A1 wins higher-SZA bins, A0 still wins lt65, A1 + UNet_CRF is the strongest single-pipeline option across all bins. See `shib_end_to_end/phase_a_higher_sza_t1_t4.md` for T1-T4.

**Last verified:** 2026-05-05.
**Latest git commit:** `7f8b100` on `main`, github.com/llinkas11/iceberg-seg.
**Authoritative state:** this file.
**Methodology:** `methods_draft.md`.
**Experimental progression:** `model_progression.md`.
**Repository design + audit:** `refactor_plan.md`.

This file is the single handoff document. Read top to bottom for full context.

---

## Context

Reworking smishra's Sentinel-2 iceberg segmentation pipeline to match Fisser 2025 standards and produce a defensible paper. All editable work in `/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/`, mirrored to `github.com/llinkas11/iceberg-seg`. Source chip data in `smishra/rework/` (read-only).

The current work is structured by progression:

- **Phase A** walks the *dataset* in a 2x3 grid: A0 (Fisser lt65 reproduction) -> A6/A9 (our lt65 + nulls + augmentations + class balancing + size balancing).
- **Phase B** walks the *method*: B0 (fixed B08 threshold) -> B5 (UNet++ + DenseCRF), all six on the Phase A winner. Single training run, six reports.

See `model_progression.md` for the full table with motivations.

---

## Resolved Prerequisites

| ID | Issue | Resolution |
|----|-------|------------|
| PR-1 | CARRA vs ERA5 | ERA5 via Open-Meteo. Wind max 8.4 m s$^{-1}$ (all pass). 324 chips temp <= 0 C (mostly sza_gt75, documented confound, not filtered). |
| PR-2 | Fisser SAFE files | Not in downloads. Fisser chips accepted as pre-filtered by Fisser et al. IC computed from tif B08 directly using annotation-aware method. |
| PR-3 | Root length definition | Per-individual-iceberg (connected component >= 16 px), not per-chip aggregate. |
| PR-4 | Validation set | 65/15/25 train/val/test; effective split is 551/137/228 (test capped at 57 per SZA bin for cross-bin metric balance). Validation used for checkpoint selection, never masked. |
| PR-5 | Re-annotation | 1,756 missed candidates found across 129 chips. Decision pending review of viz/missed_icebergs/. |
| PR-6 | CatBoost / dynamic threshold | Deferred. Dynamic IC threshold rejected because 15-25 % of iceberg pixels fall below 0.22 at every SZA bin (b08_analysis_results_discussion.md §3.6). |
| PR-7 | Shadow class | Merged into iceberg (class 2 -> 1). Model is binary. Shadow merge bridges fragmented icebergs, nearly doubling survivors after 40 m filter. |
| PR-8 | Fisser test chip evaluability | All 330 Fisser pkl chips have synthetic GeoTIFFs at `data/raw_chips/fisser/<chip_stem>.tif`. eval_methods + eval_per_iceberg load Fisser test chips through the same code path as Roboflow chips. |
| PR-9 | Augmentation vs class imbalance distinction | Augmentation diversifies *each chip's geometric views*, not *class frequency*. Oversample-only size balancing (scheme_J) addresses gradient-frequency imbalance separately, paired with augmentation. |
| PR-10 | Preprocessing-pipeline isolation | Audit on Fisser lt65 (2026-04-27): 41,644 of 70,818 components removed by 40 m filter (58.8 %); 312 of 330 chips altered; 129 of 226 training chips IC-masked (57.1 %). Both filters substantially edit Fisser data. Decision: build a `v4_raw` companion (no 40 m, no IC mask, no IC chip-drop) and report a parallel table. A0/A1 anchor on `v4_raw_lt65`; A2-A9 anchor on `v4_clean_lt65`. A0 -> A2 isolates the preprocessing pipeline as a single controlled variable. |
| PR-11 | UNet_TR threshold config drift | Investigation triggered by A2's UNet_TR collapse (4 / 8,468 matches). Root cause: `baseline_v1.yaml` declared `methods.UNet_TR.prob_threshold: 0.5`, but `run_methods.sh` never forwarded it to `threshold_probs.py`, so the script's hardcoded `THRESHOLD = 0.22` always won. Every UNet_TR number in the project, including the canonical baseline_v1 headline table, was produced at 0.22 not 0.5. Fixed 2026-04-29: `run_experiment.py.stage_infer` now reads `prob_threshold` from the merged config and passes `--prob_threshold` to `run_methods.sh`, which forwards `--threshold` to `threshold_probs.py`. `baseline_v1.yaml` updated to declare `prob_threshold: 0.22` (matches operational value, preserves reproducibility of published baseline numbers). UNet_OT and UNet_CRF have similar config drift (vestigial YAML keys not consumed by their scripts); audited but not fixed in this round, since they use script-default behaviour and changing it would invalidate baseline numbers. |
| PR-12 | A2 UNet calibration shift is the experimental finding, not a bug | Probability-distribution audit (2026-04-29) on lt65 UNet outputs: median P(iceberg) on test pixels is 0.001 for baseline_v1, 0.013 for A0 (v4_raw_lt65), but 0.278 for A2 (v4_clean_lt65). 59 % of A2's pixels fall in 0.20-0.35 vs 0.4 % for baseline. UNet_TR predicted-polygon median area is 200 m^2 for A2 vs 2,200 m^2 for baseline (50 % of A2 polygons below 200 m^2 = speckle). Root cause: A2 trains on 198 chips with 112 IC-masked (57 %) and is evaluated on unmasked test chips, a train-test domain shift baked into the IC-mask design. All three runs use byte-identical hyperparameters (resnet34, 100 epochs, lr 1e-4, seed 42, batch 16, no_augment). Calibration drift is therefore a property of the data choice not the code. A0 vs A2 quantifies the cost of applying Fisser's 40 m + IC preprocessing in a low-data regime: best val IoU drops 0.61 -> 0.26. Reported in the paper as the headline preprocessing-impact finding rather than as a bug to fix. |

---

## Key Methodological Decisions

### 1. Shadow Merge
Fisser class 2 (shadow) merged into class 1 (iceberg) before all analysis. Model is binary. Documented in `reference/descriptive_stats_results_discussion.md` §2.

### 2. 40 m Root-Length Cutoff
Connected components < 16 pixels (1,600 m$^2$, 40 m root length) removed. Matches Fisser 2025. Most removed components are rasterisation artefacts (median 3 pixels). Documented in `reference/descriptive_stats_results_discussion.md` §2.

### 3. IC Filtering (Annotation-Aware)
IC = fraction of non-annotated pixels with B08 >= 0.22 (Fisser's 0.12 + 0.10 DN offset). Training chips with IC >= 15 % have bright non-annotated pixels masked to zero. Validation and test never masked. 193 training chips masked. Justification: `reference/b08_analysis_results_discussion.md` §3.1-3.6.

### 4. DN Offset
All reflectances +0.10 high (processing baseline >= 4.0 adds 1000 DN; chip_sentinel2.py does DN * 10$^{-4}$ without subtracting). Fisser's 0.12 = our 0.22. Documented in `project_radiometric_offset.md` (smishra).

### 5. Temperature Confound
98.9 % of sza_gt75 chips have temp <= 0 C. Filtering would destroy the bin. Documented as confound, not filtered. Wind not a confound (max 8.4 m s$^{-1}$, all below 15 m s$^{-1}$). See `reference/descriptive_stats_results_discussion.md` §4.

### 6. Oversample-only Size Balancing (new)
Size balancing within positive chips uses oversample-only with a 4x replication cap. Replicate small / mid root-length bins up to the largest bin count; never undersample. Pairs with augmentation: each replicated chip gets a different random geometric variant per epoch, so the gradient sees more *distinct* instances of the rare bin without seeing identical pixels twice. Implemented in `balance_training.py.rebalance_area_bins(oversample_only=True, max_oversample_ratio=4.0)`.

---

## Pipeline Stages

The refactored pipeline is five named stages, all driven by `scripts/run_experiment.py`:

```
1. manifest    Build or verify the data manifest (chip list + split + chips_sha)
2. train       UNet++ training under ICEBERG_EXPERIMENT=1 (seed required)
3. infer       All six methods on the trained checkpoint, one manifest
4. evaluate    Chip-level (eval_methods) + per-pair (eval_per_iceberg) metrics
5. figures     Registry-archived plots (Phase 6 figure registry)
```

Every stage stamps `experiment_id`, `chips_sha`, `git_sha`, `seed`, and final metrics into provenance JSON next to its outputs.

---

## Completed Steps

### Step 0: Fisser Provenance Audit (2026-04-15)
`audit_fisser_provenance.py` -> `reference/fisser_provenance_audit.csv`. 398 / 398 tifs accessible, all dates parseable. 80 unique scenes, 0 matching SAFE zips in downloads.

### Step 1: 40 m Root-Length Filter + Shadow Merge (2026-04-15)
`filter_small_icebergs.py` -> `data/annotations_filtered.coco.json`, `data/fisser_filtered/*.pkl`. COCO 18,312 -> 7,947 annotations. Fisser 96,648 -> 39,534 components.

### Step 2: IC Quality Filtering (2026-04-15)
`filter_quality.py` -> `reference/ic_filter_10km.csv`. 356 of 984 chips fail IC >= 15 %. Masking applied only to training chips.

### Step 3: Meteorological Data (2026-04-15)
`fetch_met_data.py` -> `reference/met_data.csv`. 0 chips > 15 m s$^{-1}$ wind. 324 chips temp <= 0 C.

### Step 4: Missed Icebergs (2026-04-15)
`visualize_missed_icebergs.py` -> `viz/missed_icebergs/` + `reference/missed_icebergs_summary.csv`. 1,756 missed candidates across 129 chips. Median RL 60.8 m.

### Step 5: Descriptive Statistics (2026-04-15)
`descriptive_stats.py` -> `viz/descriptive_stats/` + `reference/descriptive_stats.csv` + `reference/descriptive_stats_results_discussion.md` + `reference/b08_analysis_results_discussion.md`.

### Step 6: Clean Dataset Build (2026-04-24)
`build_clean_dataset.py` -> `data/v4_clean/`. Replaces and supersedes the earlier `v3_clean` dir.

- 916 chips total. 65/15/25 nominal split; effective 551/137/228 (test capped at 57 per SZA bin).
- 193 training chips IC-masked (5.9 M pixels zeroed; 1.4 M iceberg pixels preserved).
- 330 Fisser chips have synthetic GeoTIFFs at `data/raw_chips/fisser/`. tif_path is populated in every chip row of the manifest.
- `manifest.json` is the single dataset identity. `chips_sha = fc4b3b16334f2916...`.

### Step 7: Test pool + lt65 nulls (2026-04-23)
- `build_v4_test_pools.py` -> `data/v4_test_pools/<bin>/{pos,null}/<chip_stem>.tif`. Per-bin test pools for 2:1 sampling. Manifest at `reference/v4_test_pools.csv`. Test chips pass through raw (no IC mask, no balancing).
- `build_lt65_nulls.py` -> 29 lt65 GT0 chips (6 KQ + 23 SK), ranked by ascending B08 p95 and stratified to match the Fisser GT-positive regional distribution. Manifest at `reference/lt65_nulls_selected.csv`. QC contact sheet at `viz/lt65_nulls_qc/contact_sheet.png`.

### Refactor Phase 1 (2026-04-23): provenance + seed + Fisser tifs
- `_method_common.py`: shared `write_method_config`, `write_skipped_chips`, `load_manifest`, `get_git_sha`, `sha256_of_*`, `SKIP_*` constants.
- `train.py` refuses to run without `--seed` under `ICEBERG_EXPERIMENT=1`. Writes `training_config.json` next to `best_model.pth`.
- `build_clean_dataset.py` emits `manifest.json` with `chips_sha`. Synthesises GeoTIFFs for Fisser chips. Defaults to 65/15/25.

### Refactor Phase 2 (2026-04-23): unified runner
- `run_methods.sh` takes `--manifest <path>` and `--checkpoint <path>`. Refuses on `chips_sha` mismatch (cross-manifest drift guard).
- `prepare_test_chips_dir.py` reads `--manifest` directly.

### Refactor Phase 3 (2026-04-23): balancing schemes (now 12 schemes)
Twelve declarative YAMLs under `configs/balancing/`:

| Scheme | Stage 1 (class) | Stage 2 (size) |
|---|---|---|
| A `fisser_original` | drop GT0 | none |
| B `fisser_plus_nulls` | 1:1 GT+/GT0 (lt65) | Fisser-style undersample |
| C `our_lt65_plus_nulls` | 1:1 GT+/GT0 (lt65, our source) | none by default |
| D `two_pos_per_null` | 2:1 GT+/GT0 fixed pos-bias | none |
| E `natural` | none | none |
| F `fixed_total_114` | total cap | none |
| G `equalized_across_sza` | per-bin count | none |
| H `custom` | per-bin ratio table | none |
| I `two_to_one_adaptive` | 2:1 majority:minority | none |
| **J `oversample_size_balanced`** | none | oversample-only, max 4x |
| **K `two_pos_per_null_size_balanced`** | D | J |
| **L `adaptive_size_balanced`** | I | J |

J/K/L (added 2026-04-27) extend the size axis with oversample-only logic in `balance_training.rebalance_area_bins(oversample_only=True, max_oversample_ratio=4.0)`. No chip is dropped at the size step.

### Refactor Phase 4 (2026-04-23): experiment runner
- `validate_experiment.py` enforces the single-controlled-variable rule via `controlled_variable:` declaration.
- `run_experiment.py` drives manifest -> train -> infer -> evaluate -> figures.
- 19 experiment YAMLs under `configs/experiments/` (see Experiment Inventory below).

### Refactor Phase 5 (2026-04-23): evaluation parity
- `eval_methods.py`: per-chip pixel metrics + chip-level area MAE/MSE; aggregations by (method, sza_bin) including GT-positive-only and skip count. Configurable skip policy.
- `eval_per_iceberg.py`: Hungarian matching on `1 - IoU` (default `iou_threshold = 0.3`); per-pair MAE on area and root length; per-pair IoU; relative error (RE, Fisser eq. 2). bbox-prefilter + area-derived union for speed. Detection stats CSV (`n_ref`, `n_pred`, `n_matched`, match rate, precision) for selection-bias disclosure.

### Refactor Phase 6 (2026-04-24, completed 2026-04-27): figure registry
- `_fig_registry.py`: `write(fig, slug, caption, out_dir)` and `write_table(headers, rows, title, slug, caption, out_dir)`. Saves to `<out_dir>/fig-archive/<YYYYMMDD_HHMMSS>__<slug>.png` (append-only) and updates `<out_dir>/figures.md` row for the slug. Slug-format-enforced via regex.
- All paper-bound figure scripts migrated: `compare_model_eval.py`, `make_figure21_iou_gt_positive_comparison.py`, `compare_areas.py`, `eval_methods.py`, `descriptive_stats.py`. Per-chip dumpers (visualize_*, predict.py, build_lt65_nulls QC PNGs) intentionally still call `fig.savefig` directly.

### Code Conversion to 2-Class
All inference scripts run as `num_classes=1` (binary). Shadow class removed throughout.

---

## Experiment Inventory (19 experiments)

### Baseline + ablations
| YAML | Purpose |
|---|---|
| `exp_baseline_v1` | No-op anchor. Drives the canonical baseline run (`runs/exp_baseline_v1/<ts>/`). |
| `exp_ablation_no_aug` | baseline_v1 with `augmentation.enabled: false`. |
| `exp_ablation_no_nulls` | baseline_v1 with `data.balancing_scheme: scheme_A_fisser_original` (drops GT0 training chips across all bins). |

### Phase A (lt65-scoped dataset progression)
| ID | Manifest | Dataset framing | Aug | Class balance | Size balance | YAML |
|---|---|---|---|---|---|---|
| A0 | `v4_raw_lt65` | Fisser lt65, Fisser preprocessing (no 40 m, no IC) | off | drop-GT0 | none | `exp_A0_fisser_lt65_original` |
| A1 | `v4_raw_lt65_plus_nulls` | A0 + nulls | off | 1:1 GT+/GT0 | undersample | `exp_A1_fisser_lt65_plus_nulls` |
| A2 | `v4_clean_lt65` | Fisser lt65, our preprocessing (40 m + IC) | off | drop-GT0 | none | `exp_A2_our_lt65` |
| A3 | `v4_clean_lt65_plus_nulls` | A2 + nulls | off | 1:1 GT+/GT0 | none | `exp_A3_our_lt65_plus_nulls` |
| A4 | `v4_clean_lt65_plus_nulls` | A3 + augmentations | on | 1:1 GT+/GT0 | none | `exp_A4_our_lt65_plus_nulls_aug` |
| A5 | `v4_clean_lt65_plus_nulls` | A4 + 2:1 fixed pos-bias | on | 2:1 fixed | none | `exp_A5_our_lt65_plus_nulls_aug_2pos` |
| A6 | `v4_clean_lt65_plus_nulls` | A4 + adaptive 2:1 | on | 2:1 adaptive | none | `exp_A6_our_lt65_plus_nulls_aug_adaptive` |
| A7 | `v4_clean_lt65_plus_nulls` | A4 + size oversample | on | 1:1 GT+/GT0 | oversample (4x cap) | `exp_A7_our_lt65_plus_nulls_aug_size` |
| A8 | `v4_clean_lt65_plus_nulls` | A5 + size oversample | on | 2:1 fixed | oversample (4x cap) | `exp_A8_our_lt65_plus_nulls_aug_2pos_size` |
| A9 | `v4_clean_lt65_plus_nulls` | A6 + size oversample | on | 2:1 adaptive | oversample (4x cap) | `exp_A9_our_lt65_plus_nulls_aug_adaptive_size` |

Phase A reads as a chain: **A0 -> A2** isolates the preprocessing pipeline (raw Fisser vs our 40 m + IC), **A2 -> A3** isolates null-chip injection, **A3 -> A4** isolates augmentation, and **A4-A9** form a 2x3 grid: {no class balance, fixed pos-bias, adaptive} crossed with {no size balance, oversample}.

The two `*_plus_nulls` manifests (A1, A3-A9) are not yet on disk; they require the C1 follow-up (refactor `build_lt65_nulls.py` to merge nulls from `reference/lt65_nulls_selected.csv` into a base manifest).

### Phase B (method sweep on baseline)
All B experiments share the baseline_v1 trained checkpoint; one training run produces all six method outputs. Each B YAML pins `evaluation.focus_method` so reporting tables read the headline method off the row.

| ID | Method focus | YAML |
|---|---|---|
| B0 | Fixed B08 threshold | `exp_B0_method_threshold` |
| B1 | Per-chip Otsu on B08 | `exp_B1_method_otsu` |
| B2 | UNet++ argmax | `exp_B2_method_unet` |
| B3 | UNet++ + threshold on probs | `exp_B3_method_unet_threshold` |
| B4 | UNet++ + Otsu on probs | `exp_B4_method_unet_otsu` |
| B5 | UNet++ + DenseCRF | `exp_B5_method_unet_crf` |

---

## Phase A Leaderboard (2026-04-30)

All ten Phase A experiments trained and evaluated. Byte-identical hyperparameters across all runs (resnet34, 100 epochs, lr=1e-4, batch=16, seed=42). Sorted by best val IoU.

| ID | manifest | val IoU | test IoU | UNet match rate | UNet RL MAE (m) | comment |
|---|---|---|---|---|---|---|
| **A0** | `v4_raw_lt65` (Fisser preproc, no nulls) | **0.613** | **0.577** | **0.512** | **9.82** | Phase A winner. |
| A1 | `v4_raw_lt65_plus_nulls` | 0.503 | 0.477 | 0.315 | 15.21 | -0.11 IoU vs A0 from null injection. |
| A3 | `v4_clean_lt65_plus_nulls` | 0.269 | 0.336 | 0.182 | 15.69 | |
| A2 | `v4_clean_lt65` | 0.261 | 0.344 | 0.245 | 15.26 | -0.35 IoU vs A0 from preprocessing. |
| A7=A8=A9 | size oversample (J/K/L) | 0.243 | 0.320 | 0.163 | 14.78 | Identical training set on this manifest. |
| A5=A6 | class balance (D/I) | 0.237 | 0.312 | 0.158 | 15.23 | Identical training set on this manifest. |
| A4 | `v4_clean_lt65_plus_nulls` (no balance) | 0.225 | 0.274 | 0.122 | 14.93 | Lowest val IoU. |

Empirical 2x3 collapse: A5 == A6 (D and I equivalent on GT+-majority data) and A7 == A8 == A9 (size oversample saturates at the same equilibrium under 4x cap). Phase A's 2x3 grid is empirically a 1x3 progression on `v4_clean_lt65_plus_nulls`.

A0 wins, A0 vs A2 isolates the preprocessing pipeline as the dominant Phase A axis, and A2-A9 plateau at ~0.24-0.27 IoU. Phase A balancing-grid sweep confirms PR-12: once preprocessing has degraded calibration, no balancing scheme recovers it.

Phase B uses the **canonical baseline_v1** checkpoint (all four SZA bins) rather than A0; A0 is best on lt65 alone but does not generalise.

---

## Baseline_v1 Trained Run

Job 56554 completed 2026-04-24 at `runs/exp_baseline_v1/20260424_185158/`.

### Training (`runs/.../model/training_config.json`)
- best_val_iou = 0.323, test_iou = 0.314 (pixel-level), test_loss = 1.025
- seed = 42, manifest_id = v4_clean, experiment_mode = true, reproducible = true
- 100 epochs on 551 training chips. ~10 min total.

### Per-pair MAE on root length, in metres (eval_per_iceberg, the Fisser-comparable headline)

| Method | < 65 | 65-70 | 70-75 | > 75 |
|---|---|---|---|---|
| TR | 17.8 | 7.9 | 6.5 | 20.1 |
| OT | 22.7 | 13.8 | 14.5 | 15.9 |
| UNet | 10.5 | 11.5 | 13.9 | 15.6 |
| UNet_TR | 14.2 | 15.5 | 18.6 | 19.6 |
| UNet_OT | **8.0** | 12.0 | 13.9 | 15.3 |
| UNet_CRF | 10.1 | **7.4** | **9.0** | **12.6** |

UNet_CRF wins three of four bins. UNet_OT wins lt65. Threshold-only methods worsen with rising SZA.

### Per-pair IoU on matched pairs

| Method | < 65 | 65-70 | 70-75 | > 75 |
|---|---|---|---|---|
| TR | 0.48 | 0.67 | 0.69 | 0.59 |
| OT | 0.47 | 0.62 | 0.62 | 0.62 |
| UNet | 0.70 | 0.67 | 0.64 | 0.65 |
| UNet_TR | 0.66 | 0.64 | 0.60 | 0.61 |
| UNet_OT | **0.73** | 0.67 | 0.64 | 0.64 |
| UNet_CRF | 0.65 | **0.69** | 0.67 | 0.66 |

### Detection stats (selection-bias disclosure)

| Method | n_pred_total | n_matched | match_rate | precision |
|---|---|---|---|---|
| TR | 16,975 | 1,465 | 20.3% | 8.6% |
| OT | 42,429 | 2,472 | 34.3% | 5.8% |
| UNet | 10,878 | 3,927 | **54.5%** | 36.1% |
| UNet_TR | 11,196 | 3,503 | 48.7% | 31.3% |
| UNet_OT | 5,702 | 1,906 | 26.5% | 33.4% |
| UNet_CRF | 13,031 | 4,163 | **57.8%** | 32.0% |

### One known interpretation note
Chip-level pixel IoU (`eval_summary.csv`) is 0.005-0.013 for every method. That happens because the chips are 94 % ocean by pixel; any false-positive pixel blows up the union and crashes the metric. The per-pair IoU table above is the meaningful one for method comparison. Mention this in the paper.

---

## Open Questions

### Variant manifests for Phase A
Six canonical manifests now exist on disk (see Critical File Paths > Data),
covering A0 through A9. The two `*_plus_nulls` variants (chips_sha `31516dc0…`
and `1e21d08f…`) were materialised on 2026-04-27 by `build_lt65_nulls.py
--merge_into_manifest`, which appends 29 lt65 GT0 chips to a base manifest's
TRAIN split with val and test pkls copied byte-stable.

### Missed icebergs (PR-5)
1,756 missed candidates. Decide whether to re-annotate.

### Oversized annotations
92 icebergs > 400,000 m$^2$. Likely multi-iceberg clumps from Otsu pre-annotation. Review and possibly split in Roboflow.

### CatBoost / dynamic thresholding
Listed in `new-plan.txt` as TBD. Deferred.

---

## Critical File Paths

### Configuration (`iceberg-rework/configs/`)
| File | Purpose |
|---|---|
| `baselines/baseline_v1.yaml` | Canonical baseline. Every experiment inherits this. |
| `experiments/exp_*.yaml` | 19 experiments (see Experiment Inventory above). |
| `balancing/scheme_*.yaml` | 12 balancing schemes (A through L). |

### Data (`iceberg-rework/data/`)
| File | chips_sha | n | Filters | Purpose |
|---|---|---|---|---|
| `v4_clean/manifest.json` | `fc4b3b16…` | 916 | 40 m + IC | Canonical baseline dataset (all bins). |
| `v4_clean_lt65/manifest.json` | `b26077e1…` | 330 | 40 m + IC | Phase A2 base (lt65 only, our preprocessing). |
| `v4_clean_lt65_plus_nulls/manifest.json` | `31516dc0…` | 359 | 40 m + IC + 29 nulls | Phase A3-A9 base (training-time GT0 injection). |
| `v4_raw_lt65/manifest.json` | `2f923c35…` | 398 | none | Phase A0 base (lt65 only, Fisser preprocessing). |
| `v4_raw_lt65_plus_nulls/manifest.json` | `1e21d08f…` | 427 | none + 29 nulls | Phase A1 base (Fisser preprocessing + nulls). |
| `v4_raw/manifest.json` | `149b2476…` | 984 | none | Companion baseline, all bins, no filters (parallel paper table). |
| `v4_clean/train_validate_test/` | | | | Materialised pkls. |
| `v4_clean/split_log.csv` | | | | Per-chip metadata. |
| `raw_chips/fisser/<chip_stem>.tif` | | | | Synthetic GeoTIFFs for Fisser chips (398 written; 330 referenced by v4_clean). |
| `v4_test_pools/<bin>/{pos,null}/` | | | | Per-bin test pools for 2:1 sampling. |
| `v4_clean_lt65_balanced/` | | | | Additive legacy variant: 28 lt65 pos + 29 lt65 null. Superseded by v4_*_lt65 manifests + the C1 null-merge step. |

### Reference (`iceberg-rework/reference/`)
| File | Purpose |
|---|---|
| `v4_test_pools.csv` | Test pool manifest. |
| `lt65_nulls_selected.csv` | 29 lt65 GT0 chips picked for the v4 lt65 test pool. |
| `b08_analysis_results_discussion.md` | IC + B08 narrative + tables. |
| `descriptive_stats_results_discussion.md` | Dataset characterisation. |
| `met_data.csv` | ERA5 wind + temperature per chip. |
| `fisser_provenance_audit.csv` | Fisser chip tif paths, dates, regions. |

### Scripts (`iceberg-rework/scripts/`)
| File | Purpose |
|---|---|
| `build_clean_dataset.py` | Build manifest from Fisser pkls + Roboflow COCO + IC filter. |
| `train.py` | UNet++ training; seed required under ICEBERG_EXPERIMENT=1. |
| `run_methods.sh` | Six methods on one manifest + checkpoint. Cross-manifest drift guard. |
| `run_experiment.py` | Drive one experiment through all five stages. |
| `validate_experiment.py` | Single-controlled-variable rule. |
| `eval_methods.py` | Chip-level IoU + MAE + MSE; figure registry-routed. |
| `eval_per_iceberg.py` | Per-pair MAE + IoU (Hungarian matching, bbox prefilter). |
| `compare_areas.py` | Method comparison plots; figure registry-routed. |
| `compare_model_eval.py` | Baseline-vs-variant heatmaps; figure registry-routed. |
| `descriptive_stats.py` | Dataset characterisation figures + tables; figure registry-routed. |
| `balance_training.py` | Stage-1 (class) and stage-2 (size) balancing. Oversample-only mode. |
| `_method_common.py` | Shared provenance helpers + skip-reason constants. |
| `_fig_registry.py` | `write` and `write_table`; append-only fig-archive + figures.md. |

### Slurm (`iceberg-rework/slurm/`)
| File | Purpose |
|---|---|
| `_common.sh` | Shared bash preamble (sourced by absolute path because slurm copies the script to /var/spool). |
| `baseline_v1.slurm` | Train + infer + evaluate the canonical baseline from scratch. |
| `baseline_v1_resume.slurm` | Resume infer + evaluate from an existing trained checkpoint. RUN_TS configurable. |
| `exp.slurm` | Generic per-experiment runner. `EXP_ID=exp_<id> sbatch slurm/exp.slurm`. Optional STAGES env var. |

### Documentation (in `paper-writing/`)
| File | Purpose |
|---|---|
| `plan.md` | This file. Project state. |
| `methods_draft.md` | Methods-section draft. |
| `model_progression.md` | Phase A 2x3 grid + Phase B method sweep. |
| `results.md` | Results-section draft (Phase A leaderboard, preprocessing impact, Phase B per-method per-bin headlines). Source-of-truth tables for `main.tex`. |
| `iceberg-rework-README.md` | Project README with folder layout + tables. |
| `refactor_plan.md` | 12-section repository design + audit. |
| `reference/descriptive_stats_results_discussion.md` | Dataset stats narrative. |
| `reference/b08_analysis_results_discussion.md` | IC / B08 analysis narrative. |

---

## How to Submit Anything

```bash
# Baseline (canonical run from scratch)
ssh moosehead 'cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework && \
  sbatch slurm/baseline_v1.slurm'

# Resume baseline infer + evaluate from an existing trained checkpoint
ssh moosehead 'cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework && \
  RUN_TS=20260424_185158 sbatch slurm/baseline_v1_resume.slurm'

# Any experiment (A0-A9, B0-B5, ablations, baseline_v1)
ssh moosehead 'cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework && \
  EXP_ID=exp_A7_our_lt65_plus_nulls_aug_size sbatch slurm/exp.slurm'

# Partial pipeline (e.g. infer + evaluate only)
ssh moosehead 'cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework && \
  EXP_ID=exp_A7_our_lt65_plus_nulls_aug_size STAGES=infer,evaluate sbatch slurm/exp.slurm'

# Validate any experiment without running it
python scripts/validate_experiment.py --exp <exp_id>

# Tail logs
ssh moosehead 'tail -f /mnt/research/.../iceberg-rework/logs/exp/ice_exp_<job_id>.out'
```

---

## Verified Pipeline State (2026-04-27)

- Repo on github.com/llinkas11/iceberg-seg, latest commit `7f8b100`.
- Six canonical manifests on disk (chips_sha shown to 16 chars):
  - `v4_clean`                 : `fc4b3b16334f2916...`, 916 chips, 40 m + IC, all bins.
  - `v4_clean_lt65`            : `b26077e13fe536e2...`, 330 chips, 40 m + IC, lt65 only.
  - `v4_clean_lt65_plus_nulls` : `31516dc09828007e...`, 359 chips, 40 m + IC + 29 nulls in train.
  - `v4_raw_lt65`              : `2f923c35d858ba06...`, 398 chips, no filters, lt65 only.
  - `v4_raw_lt65_plus_nulls`   : `1e21d08fc96c3d53...`, 427 chips, no filters + 29 nulls in train.
  - `v4_raw`                   : `149b247671b70880...`, 984 chips, no filters, all bins.
- `build_clean_dataset.py` regression-tested 2026-04-28: rebuilding with no flags reproduces v4_clean's `chips_sha = fc4b3b16334f2916...` exactly. Same 551/137/228 split, 193 IC-masked chips. Confirms the new flag-gated branches (IC-skip, SZA-early-filter, idempotent shadow merge) leave the canonical build byte-stable.
- baseline_v1 trained: `runs/exp_baseline_v1/20260424_185158/model/best_model.pth` (104 MB, 100 epochs, val IoU 0.323, test IoU 0.314 pixel-level). Trained on v4_clean.
- All 10 Phase A YAMLs validate on HPC after the manifest_id repointing (A0/A1 -> v4_raw_lt65 / v4_raw_lt65_plus_nulls; A2-A9 -> v4_clean_lt65 / v4_clean_lt65_plus_nulls).
- All 19 experiment YAMLs validate locally and on HPC.
- All 12 balancing scheme YAMLs parse.
- `_fig_registry` exposes `write` and `write_table`. Every paper-bound figure script routes through it.
- `eval_per_iceberg.compute_iou_matrix` verified on L-pair degenerate case (area-derived union matches naive full-mask OR to float precision).
- `balance_training.rebalance_area_bins(oversample_only=True)` unit-tested: (5, 30, 60) input -> (60, 60, 60) without cap; -> (20, 60, 60) with cap=4x. No chip dropped.
- Per-pair MAE + IoU + detection-stats tables exist for baseline_v1 at `runs/exp_baseline_v1/20260424_185158/per_iceberg/`.
- Dependencies (scikit-image, scipy, PyYAML, pydensecrf2) installed in `~/.venvs/iceberg-unet312` and listed in `requirements.txt`.

---

## Handoff Checklist for Next Context Window

1. Read this file (`plan.md`) end-to-end.
2. Optional: skim `methods_draft.md` for paper prose state.
3. Optional: skim `model_progression.md` for the experimental narrative + 2x3 grid logic.
4. Read the latest `git log --oneline | head -30` to see most recent commits.
5. Check job queue: `ssh moosehead 'squeue -u llinkas'`.
6. Open questions to resolve with user before scaling experiments:
   - Phase A "our lt65" scoping (a, b, or c above).
   - Whether to materialise variant manifests (`fisser_lt65_original`, `our_lt65`, `our_lt65_plus_nulls`).
   - Whether to start Phase B reporting on baseline_v1 results.
7. The figure registry is the canonical place to add any new plot. Use `_fig_registry.write` for plots and `_fig_registry.write_table` for tables-as-PNGs.

---

## Pending: trim the script-check pack after Farias review

The script-check pack at `iceberg-rework/script-check-README.md` was originally written to send all 23 per-script questions to the external reviewer (Farias). Five of those questions (Q1, Q7, Q15, Q16, Q17) have since been answered empirically in this project, with figures and CSVs under `paper-writing/figure_review/script_check_answers/<slug>/` and a written summary in `methods_draft.md` Section 2.14.

Once Farias has finished reviewing the pack, do the following so future readers see only the open questions:

1. In `iceberg-rework/script-check-README.md`, delete the bullet for each answered question (Q1, Q7, Q15, Q16, Q17) and its `*Pre-checked* ...` sub-bullet. Leave a one-line pointer at the section level (or in the README header) noting that the answered set is documented in `paper-writing/methods_draft.md` Section 2.14 and `paper-writing/figure_review/script_check_answers/`.
2. In each affected production script, add a brief inline comment at the parameter that was checked, summarising the empirical justification. The five spots are:
   - `iceberg-rework/scripts/threshold_tifs.py`, at `ic_threshold = 0.15`: `# Q1 sweep (n=23,981) confirmed 15% sits in the slow middle of the ic_frac ECDF; moving to 0.20 / 0.30 buys only a few percent. See paper-writing/figure_review/script_check_answers/q01_ic_cutoff_sweep/.`
   - `iceberg-rework/scripts/otsu_threshold_tifs.py`, at `min_otsu_thresh = 0.10`: `# Q7 confirmed 0.10 floor: 6.1% of chips skip; raising to 0.15 jumps to 41% (would discard half the population). See .../q07_otsu_floor_distribution/.`
   - `iceberg-rework/scripts/threshold_probs.py`, at `threshold = 0.22`: `# Q15 test-split sensitivity (n=171): F1 climbs to 0.528 at tau=0.90 vs 0.464 at 0.22 (delta +0.064). Calibration deferred until val probs exist. See .../q15_unet_threshold_f1/.`
   - `iceberg-rework/scripts/otsu_probs.py`, at the `range_p < 0.01` flat-prob skip: `# Q16 confirmed: 0% would-skip at every cutoff in {0.005, 0.01, 0.02, 0.05}; guard is non-binding on this population. See .../q16_flat_prob_distribution/.`
   - `iceberg-rework/scripts/otsu_probs.py`, at the no-floor branch: `# Q17 ruled out a 0.5 floor: 100% of chips have Otsu < 0.5; floor would activate everywhere and remove 22.5% of iceberg pixels. See .../q17_otsu_on_prob_floor/.`
3. Keep the answer scripts under `iceberg-rework/scripts/script_check_answers/` and the artifact folders under `paper-writing/figure_review/script_check_answers/` so the reasoning is reproducible. The checklist row in `paper-writing/figure_review/figure_review_checklist.csv` for each answered slug stays at status `draft` with the headline finding in `needed_edits` until the figure is incorporated into the deck.

Do NOT remove the answered questions from the README before the reviewer has read it; they are the asked-and-answered evidence the reviewer needs to see to know we did our own due diligence.
