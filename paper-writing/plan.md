# Iceberg Segmentation Pipeline: Project State

**Status:** Refactor complete (Phases 1-6). Configuration system, manifest schema, experiment runner, method runner, per-iceberg evaluator, and figure registry all in place. baseline_v1 trained successfully on `v4_clean`; method inference + evaluation re-running after dependency fix.
**Last verified:** 2026-04-24.
**Authoritative state:** this file. Methodology details: `methods_draft.md`. Experimental progression: `model_progression.md`. Repository design + audit: `refactor_plan.md`.

## Context

Reworking smishra's Sentinel-2 iceberg segmentation pipeline to match Fisser 2025 standards and produce a defensible paper. All editable work in `/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/`, mirrored to `github.com/llinkas11/iceberg-seg`. Source chip data in `smishra/rework/` (read-only).

The current work is structured by progression:

- **Phase A** walks the *dataset*: A0 (Fisser lt65 reproduction) -> A6 (our lt65 + nulls + augmentations + adaptive balancing).
- **Phase B** walks the *method*: B0 (fixed B08 threshold) -> B5 (UNet++ + DenseCRF), all six on the Phase A winner.

See `model_progression.md` for the full table with motivations.

---

## Resolved Prerequisites

| ID | Issue | Resolution |
|----|-------|------------|
| PR-1 | CARRA vs ERA5 | ERA5 via Open-Meteo. Wind max 8.4 m s$^{-1}$ (all pass), 324 chips temp <= 0 C (mostly sza_gt75, documented as confound, not filtered). |
| PR-2 | Fisser SAFE files | Not in downloads. Fisser chips accepted as pre-filtered by Fisser et al. IC computed from tif B08 directly using annotation-aware method. |
| PR-3 | Root length definition | Per-individual-iceberg (connected component >= 16 px), not per-chip aggregate. |
| PR-4 | Validation set | 65/15/25 train/val/test; effective split is 551/137/228 due to per-SZA-bin test caps (57 per bin). Validation used for checkpoint selection, never masked. |
| PR-5 | Re-annotation | 1,756 missed candidates found across 129 chips. Decision pending review of viz/missed_icebergs/. |
| PR-6 | CatBoost / dynamic threshold | Deferred. Dynamic IC threshold rejected because 15-25 % of iceberg pixels fall below 0.22 at every SZA bin (b08_analysis_results_discussion.md §3.6). |
| PR-7 | Shadow class | Merged into iceberg (class 2 -> 1). Model is binary. Shadow merge bridges fragmented icebergs, nearly doubling survivors after 40 m filter. |
| PR-8 | Fisser test chip evaluability | All Fisser pkl chips now have synthetic GeoTIFFs at `data/raw_chips/fisser/<chip_stem>.tif` (3-band float32, 10 m identity transform, no CRS). eval_methods + eval_per_iceberg now load Fisser test chips through the same code path as Roboflow chips. |

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

---

## Pipeline Stages

The refactored pipeline is now five named stages, all driven by `scripts/run_experiment.py`:

```
1. manifest    Build or verify the data manifest (chip list + split + chips_sha)
2. train       UNet++ training under ICEBERG_EXPERIMENT=1 (seed required)
3. infer       All six methods on the trained checkpoint, one manifest
4. evaluate    Chip-level (eval_methods) + per-pair (eval_per_iceberg) metrics
5. figures     Registry-archived plots
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
`visualize_missed_icebergs.py` -> `viz/missed_icebergs/`, `reference/missed_icebergs_summary.csv`. 1,756 missed candidates across 129 chips. Median RL 60.8 m.

### Step 5: Descriptive Statistics (2026-04-15)
`descriptive_stats.py` -> `viz/descriptive_stats/` + `reference/descriptive_stats.csv` + `reference/descriptive_stats_results_discussion.md` + `reference/b08_analysis_results_discussion.md`.

### Step 6: Clean Dataset Build (2026-04-16, rebuilt as v4_clean 2026-04-24)
`build_clean_dataset.py` -> `data/v4_clean/`. Replaces and supersedes the earlier `v3_clean` dir.

- 916 chips total, 65/15/25 nominal split (effective 551/137/228 due to per-bin test caps).
- 193 training chips IC-masked (5.9 M pixels zeroed; 1.4 M iceberg pixels preserved).
- Test stratified to 57 chips per SZA bin.
- 330 Fisser chips now have synthetic GeoTIFFs at `data/raw_chips/fisser/`. tif_path is populated in every chip row of the manifest, so evaluation no longer silently drops Fisser chips.
- `manifest.json` is the single dataset identity. `chips_sha` (sha256 over chip_stem + tif_sha + split tuples) is the single hash that must match across runs.

### Step 7: Training-Set Variants and Balancing Schemes
`balance_training.py` (rewritten 2026-04-23) consumes a clean dataset and applies a balancing scheme. Nine schemes are declared as YAML under `configs/balancing/`:

| Scheme | Rule |
|---|---|
| A `fisser_original` | Drop GT0 training chips. |
| B `fisser_plus_nulls` | Keep Fisser positives; inject GT0 chips at 1:1, lt65 only. |
| C `our_lt65_plus_nulls` | As B but on our chip source. |
| D `two_pos_per_null` | 2:1 GT+ : GT0 fixed pos-bias, all bins. |
| E `natural` | No resampling. Explicit no-op for A/B partner runs. |
| F `fixed_total_114` | Cap total to 114 chips, stratified by SZA + class. |
| G `equalized_across_sza` | Equal per-SZA-bin count. |
| H `custom` | Per-bin ratio table in YAML. |
| I `two_to_one_adaptive` | 2:1 majority : minority, direction picked from natural per-bin distribution. |

Strategy classes are deferred until the first scheme that has no existing implementation needs to run; A through D are covered by `balance_training.py` today.

### Step 7c: Per-Bin Test Pools (2026-04-23)
`build_v4_test_pools.py` -> `data/v4_test_pools/<bin>/{pos,null}/<chip_stem>.tif`. Manifest at `reference/v4_test_pools.csv`. Test chips pass through raw (no IC mask, no balancing). Used for cross-bin metric comparison at evaluation time.

- Bin counts: lt65 56 pos / 29 null; sza_65_70 21 / 36; sza_70_75 20 / 37; sza_gt75 23 / 34.
- 2:1 sampling cap per bin: lt65 56/28, sza_65_70 21/10, sza_70_75 20/10, sza_gt75 23/11. Cross-bin equalised cap is 20 pos / 10 null (bottlenecked by sza_70_75).

### Step 7b: lt65 Null Chips (2026-04-23)
`build_lt65_nulls.py`. 4,444 lt65 chips scanned; 278 accepted; 29 selected (6 KQ + 23 SK), ranked by ascending B08 p95 and descending dark-pixel fraction. Stratified to match the Fisser GT-positive regional distribution (19.5 % KQ / 80.5 % SK). Manifest at `reference/lt65_nulls_selected.csv`. QC contact sheet at `viz/lt65_nulls_qc/contact_sheet.png`.

### Refactor Phase 1 (2026-04-23): provenance + seed + Fisser tifs
- `_method_common.py`: shared `write_method_config`, `write_skipped_chips`, `load_manifest`, `get_git_sha`, `sha256_of_*`, `SKIP_*` constants. Single source of truth for method-output provenance.
- `train.py`: refuses to run without `--seed` under `ICEBERG_EXPERIMENT=1`. Writes `training_config.json` next to `best_model.pth` (args, seed, manifest_id, git_sha, final metrics, experiment_mode flag).
- `build_clean_dataset.py`: emits `manifest.json` with `chips_sha`. Synthesises GeoTIFFs for Fisser chips. Defaults to 65/15/25.

### Refactor Phase 2 (2026-04-23): unified runner
- `run_methods.sh`: takes `--manifest <path>` and `--checkpoint <path>`. Refuses if the checkpoint's `training_config.json.manifest_id` does not match (cross-manifest drift guard).
- `prepare_test_chips_dir.py`: reads `--manifest` directly to populate per-bin test chip dirs.

### Refactor Phase 3 (2026-04-23): balancing schemes
Nine YAMLs landed under `configs/balancing/`. Strategy class layer deferred.

### Refactor Phase 4 (2026-04-23): experiment runner
- `validate_experiment.py`: enforces single-controlled-variable rule via `controlled_variable:` declaration.
- `run_experiment.py`: drives manifest -> train -> infer -> evaluate -> figures.
- 14 experiment YAMLs under `configs/experiments/`: A0-A6, B0-B5, baseline_v1, ablation_no_aug.

### Refactor Phase 5 (2026-04-23): evaluation parity
- `eval_methods.py` now emits per-chip `pred_area_m2`, `gt_area_m2`, `abs_area_err_m2`, `sq_area_err_m2`. Summary aggregates to `mae_area_m2`, `mse_area_m2`, `n_chips`, `n_skipped`. Skip policy configurable.
- `eval_per_iceberg.py` rewritten with Hungarian matching on `1 - IoU` (default `iou_threshold = 0.3`); per-pair MAE on area and root length; per-pair IoU; relative error (RE, Fisser eq. 2). Detection stats CSV (`n_ref`, `n_pred`, `n_matched`, match rate, precision) for selection-bias disclosure.

### Refactor Phase 6 (2026-04-24): figure registry
- `_fig_registry.py`: `write(fig, slug, caption, out_dir)`. Saves to `<out_dir>/fig-archive/<YYYYMMDD_HHMMSS>__<slug>.png` (append-only) and updates `<out_dir>/figures.md` row for the slug.
- `compare_model_eval.py` and `make_figure21_iou_gt_positive_comparison.py` migrated. Remaining migrations (compare_areas, eval_methods, descriptive_stats) deferred until paper rev cycle.

### Code Conversion to 2-Class
All inference scripts run as `num_classes=1` (binary). Shadow class removed throughout.

---

## Current Work

### Job 56554 (baseline_v1 resume)
Resume of the `baseline_v1` training run. Inference + evaluation against the trained checkpoint at `runs/exp_baseline_v1/20260424_185158/model/best_model.pth`. PD on the gpu partition.

When complete, headline tables will land at:
- `runs/exp_baseline_v1/20260424_185158/evaluation/eval_summary.csv` (chip-level IoU/MAE/MSE per method × SZA bin).
- `runs/exp_baseline_v1/20260424_185158/per_iceberg/eval_per_iceberg_summary.csv` (per-pair MAE on area + root length, per-pair IoU per method × SZA bin).
- `runs/exp_baseline_v1/20260424_185158/per_iceberg/eval_per_iceberg_detection.csv` (match rate per method).

---

## Remaining Work

### Phase A Materialisation
A0-A6 each reference a manifest that does not yet exist (`fisser_lt65_original`, `fisser_lt65_plus_nulls`, `our_lt65`, `our_lt65_plus_nulls`). To run any A experiment, the corresponding manifest must be built first via `build_clean_dataset.py` with the appropriate balancing scheme.

**Open scoping question** (raised but not resolved): every lt65 chip in v4_clean is Fisser-sourced. There is no Roboflow-annotated lt65 in the v4_clean training split, so "our lt65" as a separate chip source needs definition. Three possible interpretations:
- (a) A2 / A3 are redundant with A0 / A1; drop them.
- (b) "Our lt65" means same chips, our preprocessing.
- (c) "Our lt65" is a separate iceberg-labeler chip set not yet integrated.

Pending direction.

### Phase B Materialisation
B0-B5 share the baseline_v1 checkpoint; one training run produces all six. Once 56554 succeeds, each of B0-B5 is a reporting filter on the same outputs.

### Phase 7 Cleanup (deferred)
Retire `IDS2026/S2-iceberg-areas/` to `_archive/`. Delete `scripts/*.bak` and `scripts/__pycache__/` on HPC. Update `paper-writing/methods_draft.md` to reflect the new manifest + experiment system.

### Open Questions
1. Phase A "our lt65" interpretation (above).
2. **Missed icebergs (PR-5):** 1,756 missed candidates. Decide on re-annotation.
3. **92 oversized annotations:** > 400,000 m$^2$. Likely multi-iceberg clumps. Review and possibly split in Roboflow.
4. **CatBoost / dynamic thresholding:** deferred.

---

## Critical File Paths

### Configuration
| File | Purpose |
|---|---|
| `configs/baselines/baseline_v1.yaml` | Canonical baseline; every experiment inherits this. |
| `configs/experiments/exp_*.yaml` | 14 experiments: Phase A (A0-A6), Phase B (B0-B5), baseline_v1, ablation_no_aug. |
| `configs/balancing/scheme_*.yaml` | 9 balancing schemes (A through I). |

### Data
| File | Purpose |
|---|---|
| `data/v4_clean/manifest.json` | Single dataset identity. chips_sha = `fc4b3b16334f2916...`. |
| `data/v4_clean/train_validate_test/` | Materialised pkls for training (consumed by `train.py --data_dir data/v4_clean`). |
| `data/v4_clean/split_log.csv` | Per-chip metadata: split, pkl_position, chip_stem, tif_path, sza_bin, source, n_icebergs, ic_aware, ic_masked, wind_ms, temp_c. |
| `data/raw_chips/fisser/<chip_stem>.tif` | Synthetic GeoTIFFs for Fisser chips so evaluation finds them. |
| `data/v4_test_pools/<bin>/{pos,null}/` | Test chip pools for 2:1 sampling at evaluation time. |

### Reference
| File | Purpose |
|---|---|
| `reference/v4_test_pools.csv` | Test pool manifest (bin, gt_label, chip_stem, source, n_icebergs, ic_frac, tif_src, tif_pool). |
| `reference/lt65_nulls_selected.csv` | 29 lt65 GT0 chips selected for the v4 lt65 test pool. |
| `reference/b08_analysis_results_discussion.md` | IC / B08 analysis with results, discussion, methods. |
| `reference/descriptive_stats_results_discussion.md` | Dataset characterisation. |
| `reference/met_data.csv` | ERA5 wind + temperature per chip. |
| `reference/fisser_provenance_audit.csv` | Fisser chip tif paths, dates, regions. |

### Scripts
| File | Purpose |
|---|---|
| `scripts/build_clean_dataset.py` | Build a clean manifest from Fisser pkls + Roboflow COCO + IC filter. |
| `scripts/train.py` | UNet++ training with seed enforcement and training_config.json emission. |
| `scripts/run_methods.sh` | Run all six methods for one manifest + checkpoint. |
| `scripts/run_experiment.py` | Drive one experiment through manifest -> train -> infer -> evaluate -> figures. |
| `scripts/validate_experiment.py` | Single-controlled-variable rule; refuses multi-family change without `controlled_variable:`. |
| `scripts/eval_methods.py` | Chip-level IoU + MAE + MSE per method × SZA bin. |
| `scripts/eval_per_iceberg.py` | Per-pair MAE + IoU (Hungarian matching). |
| `scripts/_method_common.py` | Shared provenance helpers + skip-reason constants. |
| `scripts/_fig_registry.py` | Append-only figure archive + figures.md index. |

### Slurm
| File | Purpose |
|---|---|
| `slurm/_common.sh` | Sourced by every slurm script. Sets ROOT, PY, ICEBERG_EXPERIMENT=1. |
| `slurm/baseline_v1.slurm` | Train + infer + evaluate baseline_v1 from scratch. |
| `slurm/baseline_v1_resume.slurm` | Resume infer + evaluate against an existing trained checkpoint. RUN_TS configurable. |

### Documentation (all in `paper-writing/`)
| File | Purpose |
|---|---|
| `plan.md` | This file. Project state. |
| `methods_draft.md` | Methods-section draft for the paper. |
| `model_progression.md` | Phase A / Phase B experimental progression. |
| `iceberg-rework-README.md` | Project-level README with folder layout + tables. |
| `refactor_plan.md` | Repository design + audit (12-section response to the original refactor brief). |
| `reference/descriptive_stats_results_discussion.md` | Dataset stats narrative. |
| `reference/b08_analysis_results_discussion.md` | IC / B08 analysis narrative. |

---

## Verified Pipeline State (2026-04-24)

- Repository on github.com/llinkas11/iceberg-seg, commit `bf151dd`.
- v4_clean manifest built; chips_sha = `fc4b3b16334f2916...`.
- baseline_v1 trained: `runs/exp_baseline_v1/20260424_185158/model/best_model.pth` (104 MB, 100 epochs, val IoU recorded in training_config.json).
- 14 experiment YAMLs validate; all 9 balancing schemes parse.
- `_fig_registry` unit tests pass: append-only archiving, slug-prefix safety, regenerate-replaces-row behaviour.
- `eval_per_iceberg.compute_iou_matrix` verified on L-pair degenerate case (area-derived union matches naive full-mask OR to float precision).
- skimage, scipy, PyYAML, pydensecrf2 installed in `~/.venvs/iceberg-unet312` and listed in `requirements.txt`.
