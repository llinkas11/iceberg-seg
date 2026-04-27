# Iceberg Segmentation Pipeline: Project State

**Status:** Refactor complete (Phases 1-6). v4_clean dataset built and frozen. baseline_v1 trained and evaluated. Phase A 2x3 grid (size balance x class balance) registered as 10 YAMLs, runner ready. Phase B method sweep is a single inference dispatch on the trained checkpoint, runner ready. Per-pair MAE + IoU tables published for the canonical baseline.

**Last verified:** 2026-04-27.
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
| ID | Dataset | Aug | Class balance | Size balance | YAML |
|---|---|---|---|---|---|
| A0 | Fisser lt65, positive-only | off | drop-GT0 | none | `exp_A0_fisser_lt65_original` |
| A1 | Fisser lt65 + nulls | off | 1:1 GT+/GT0 | undersample | `exp_A1_fisser_lt65_plus_nulls` |
| A2 | Our lt65, positive-only | off | drop-GT0 | none | `exp_A2_our_lt65` |
| A3 | Our lt65 + nulls | off | 1:1 GT+/GT0 | none | `exp_A3_our_lt65_plus_nulls` |
| A4 | A3 + augmentations | on | 1:1 GT+/GT0 | none | `exp_A4_our_lt65_plus_nulls_aug` |
| A5 | A4 + 2:1 fixed pos-bias | on | 2:1 fixed | none | `exp_A5_our_lt65_plus_nulls_aug_2pos` |
| A6 | A4 + adaptive 2:1 | on | 2:1 adaptive | none | `exp_A6_our_lt65_plus_nulls_aug_adaptive` |
| A7 | A4 + size oversample | on | 1:1 GT+/GT0 | oversample (4x cap) | `exp_A7_our_lt65_plus_nulls_aug_size` |
| A8 | A5 + size oversample | on | 2:1 fixed | oversample (4x cap) | `exp_A8_our_lt65_plus_nulls_aug_2pos_size` |
| A9 | A6 + size oversample | on | 2:1 adaptive | oversample (4x cap) | `exp_A9_our_lt65_plus_nulls_aug_adaptive_size` |

A4-A9 form a 2x3 grid: {no class balance, fixed pos-bias, adaptive} crossed with {no size balance, oversample}.

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

### Phase A "our lt65" scoping (high priority)
v4_clean has zero Roboflow-annotated lt65 chips in the training split (every lt65 training chip is Fisser-sourced). A2/A3 as written currently swap nothing meaningful from A0/A1. Three interpretations are open:

- (a) A2/A3 redundant with A0/A1; drop them. Phase A collapses to 8 rows.
- (b) "Our lt65" means same chips, our preprocessing (40 m + IC filter applied; Fisser's wasn't). Weaker claim but valid.
- (c) "Our lt65" is a separate iceberg-labeler chip set not yet integrated. Needs new manifest source first.

User direction is required before A2-A9 can produce meaningful results.

### Variant manifests (`fisser_lt65_original`, `fisser_lt65_plus_nulls`, `our_lt65`, `our_lt65_plus_nulls`)
A0-A9 reference manifests that do not yet exist on disk. Each must be built via `build_clean_dataset.py` with the appropriate balancing scheme, OR via a new manifest-recipe layer. Not blocking Phase B.

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
| File | Purpose |
|---|---|
| `v4_clean/manifest.json` | Single dataset identity. `chips_sha = fc4b3b16334f2916...`. |
| `v4_clean/train_validate_test/` | Materialised pkls. |
| `v4_clean/split_log.csv` | Per-chip metadata. |
| `raw_chips/fisser/<chip_stem>.tif` | 330 synthetic GeoTIFFs for Fisser chips. |
| `v4_test_pools/<bin>/{pos,null}/` | Per-bin test pools for 2:1 sampling. |
| `v4_clean_lt65_balanced/` | Additive variant: 28 lt65 pos + 29 lt65 null. |

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
- v4_clean manifest built. chips_sha = `fc4b3b16334f2916...`.
- baseline_v1 trained: `runs/exp_baseline_v1/20260424_185158/model/best_model.pth` (104 MB, 100 epochs, val IoU 0.323, test IoU 0.314 pixel-level).
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
