# Sentinel-2 Iceberg Segmentation: project README

**Working tree (HPC):** `/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/`
**Source code (versioned):** `https://github.com/llinkas11/iceberg-seg`
**Source chip data (read-only):** `/mnt/research/v.gomezgilyaspik/students/smishra/rework/`
**Last updated:** 2026-04-27

This file is the project-level overview. For methodology prose, see `methods_draft.md`. For the live state of work and stage-by-stage progress, see `plan.md`. For the experimental progression narrative (Phase A and Phase B), see `model_progression.md`. For the deep audit and target architecture, see `refactor_plan.md`.

---

## What this project is

Sentinel-2 L1C iceberg segmentation across four solar zenith angle (SZA) bins in Kangerlussuaq (KQ) and Sermilik (SK) fjords on the east coast of Greenland. The paper compares six retrieval methods on a single shared dataset:

- **TR**: fixed B08 NIR threshold (Fisser-equivalent, 0.22 in offset-uncorrected space)
- **OT**: per-chip Otsu thresholding on B08
- **UNet**: UNet++ (ResNet-34, ImageNet-init) binary segmentation
- **UNet_TR**: fixed threshold on the UNet++ softmax probability
- **UNet_OT**: per-chip Otsu on the UNet++ softmax probability
- **UNet_CRF**: UNet++ followed by DenseCRF post-processing

Every method runs on the same test chip set, the same trained checkpoint (where applicable), and the same minimum-area filter, so cross-method differences attribute to the method.

The paper's headline metric is **mean absolute error (MAE) on iceberg area**, per matched pair, per SZA bin. This is the only number that plugs directly into Fisser and others (2024)'s reported tables. Per-pair IoU is the segmentation-community-standard companion. Match rate is reported alongside both as a selection-bias disclosure.

---

## Sensor and grid

- Sentinel-2 L1C, bands B04 (red), B03 (green), B08 (NIR), 10 m resolution.
- Chip size: 256 x 256 pixels (2.56 x 2.56 km).
- SZA bins: `sza_lt65` (Jul-Sep), `sza_65_70` (Sep-Oct), `sza_70_75` (Oct), `sza_gt75` (Nov).

---

## Repository layout

The HPC working tree mirrors `github.com/llinkas11/iceberg-seg`. Materialised pkls, model checkpoints, and inference outputs are HPC-only (too large for git).

```
iceberg-rework/
|-- README.md                              quick-start; sibling of this file
|-- requirements.txt                       (PyYAML, scikit-image, scipy, pydensecrf2, etc.)
|-- plan.md                                project state
|-- new-plan.txt                           original advisor brief
|-- job.slurm                              legacy training launcher (superseded by slurm/)
|
|-- configs/                               configuration system (Phase 4 of refactor)
|   |-- baselines/baseline_v1.yaml         canonical baseline; every experiment inherits
|   |-- experiments/exp_*.yaml             19 experiments: A0-A9, B0-B5, baseline_v1, ablation_no_aug, ablation_no_nulls
|   |-- balancing/scheme_*.yaml            12 schemes (A-L). J=oversample-only size balance; K=D+J; L=I+J
|   |-- datasets/                          (reserved for dataset recipe YAMLs)
|   `-- methods/                           (reserved; methods configured in baseline_v1.yaml today)
|
|-- scripts/                               all pipeline code
|   |-- _method_common.py                  shared provenance helpers + SKIP_* constants
|   |-- _fig_registry.py                   append-only figure archive + figures.md index
|   |-- audit_fisser_provenance.py         map Fisser chips to tifs, parse dates
|   |-- filter_small_icebergs.py           40 m RL filter, shadow merge
|   |-- filter_quality.py                  IC filter (annotation-aware, B08 >= 0.22)
|   |-- fetch_met_data.py                  ERA5 wind + temperature
|   |-- visualize_missed_icebergs.py       unannotated bright-object scan
|   |-- descriptive_stats.py               per-iceberg histograms + tables
|   |-- build_clean_dataset.py             v4_clean manifest + pkls + synthetic Fisser tifs
|   |-- build_lt65_nulls.py                lt65 GT0 chip selection
|   |-- build_v4_test_pools.py             per-bin test pools for 2:1 sampling at eval time
|   |-- build_gt_positive_training.py      drop-GT0 training variant builder (scheme A)
|   |-- balance_training.py                staged balancing (schemes A through D today)
|   |-- prepare_test_chips_dir.py          symlink test tifs by manifest into per-bin dirs
|   |-- train.py                           UNet++ training (binary, seed-required under ICEBERG_EXPERIMENT=1)
|   |-- predict_tifs.py                    UNet++ inference + softmax probs + polygons
|   |-- threshold_tifs.py                  TR method
|   |-- otsu_threshold_tifs.py             OT method
|   |-- threshold_probs.py                 UNet_TR method
|   |-- otsu_probs.py                      UNet_OT method
|   |-- densecrf_tifs.py                   UNet_CRF method
|   |-- crf_utils.py                       pydensecrf2 wrapper
|   |-- run_methods.sh                     one manifest + one checkpoint -> 6 methods x 4 bins
|   |-- run_experiment.py                  manifest -> train -> infer -> evaluate -> figures
|   |-- validate_experiment.py             single-controlled-variable rule
|   |-- eval_methods.py                    chip-level IoU + MAE + MSE table
|   |-- eval_per_iceberg.py                per-pair MAE + IoU (Hungarian matching)
|   |-- compare_model_eval.py              baseline vs variant comparison tables and heatmaps
|   `-- make_figure21_*.py                 GT-positive IoU heatmap generator
|
|-- slurm/                                 sbatch wrappers
|   |-- _common.sh                         shared bash preamble (sourced by absolute path)
|   |-- baseline_v1.slurm                  full pipeline from scratch
|   |-- baseline_v1_resume.slurm           resume from an existing trained checkpoint
|   `-- exp.slurm                          generic per-experiment runner, EXP_ID env var
|
|-- data/
|   |-- annotations_filtered.coco.json     COCO with < 40 m RL icebergs removed
|   |-- fisser_filtered/                   Fisser pkls with shadow merged + 40 m filter
|   |-- v3_clean/                          legacy 60/15/25 build (kept for backward comparison)
|   |-- v3_balanced/                       legacy 2:1 area-binned balanced training set
|   |-- v4_clean/                          current canonical dataset
|   |   |-- manifest.json                  single source of truth, chips_sha = fc4b3b16334f2916...
|   |   |-- split_log.csv                  per-chip metadata
|   |   `-- train_validate_test/           materialised pkls
|   |-- v4_clean_lt65_balanced/            additive: 28 lt65 pos + 29 lt65 null
|   |-- v4_test_pools/<bin>/{pos,null}/    per-bin test chip pools for 2:1 sampling
|   `-- raw_chips/fisser/<chip_stem>.tif   330 synthetic GeoTIFFs for Fisser pkls
|
|-- reference/
|   |-- fisser_provenance_audit.csv
|   |-- met_data.csv                       ERA5 wind + temperature
|   |-- descriptive_stats.csv              per-bin iceberg + ocean + contrast statistics
|   |-- lt65_nulls_selected.csv            29 lt65 GT0 chips picked for the test pool
|   |-- v4_test_pools.csv                  per-bin test pool manifest
|   |-- b08_analysis_results_discussion.md IC + B08 narrative + tables
|   `-- descriptive_stats_results_discussion.md
|
|-- viz/
|   |-- filter_40m/{coco,fisser}/          before-and-after for the 40 m cutoff
|   |-- missed_icebergs/                   side-by-side originals + missed-candidate overlays
|   |-- descriptive_stats/                 histograms + tables
|   `-- lt65_nulls_qc/contact_sheet.png    QC on the 29 selected nulls
|
|-- model/                                 trained checkpoints (legacy v3 runs preserved)
|
|-- runs/                                  experiment outputs
|   `-- exp_baseline_v1/<timestamp>/
|       |-- model/best_model.pth
|       |-- model/training_config.json
|       |-- model/training_log.csv
|       |-- inference/<sza_bin>/<METHOD>/  per-method gpkgs + method_config.json + skipped_chips.csv
|       |-- evaluation/                    chip-level CSVs from eval_methods.py
|       `-- per_iceberg/                   per-pair CSVs from eval_per_iceberg.py
|
|-- logs/baseline/                         slurm stdout + stderr
|-- fig-archive/                           append-only figure archive (per Phase 6)
`-- figures.md                             live index of latest figures, with captions
```

---

## v4_clean dataset

The canonical dataset is `data/v4_clean/`. Build / verify with:

```
python scripts/build_clean_dataset.py
```

That script produces `manifest.json`, `split_log.csv`, and the `train_validate_test/*.pkl` pyramids. It also writes 330 synthetic GeoTIFFs under `data/raw_chips/fisser/` so evaluation can find Fisser chips through the same code path as Roboflow chips.

### Composition

| | Total | sza_lt65 | sza_65_70 | sza_70_75 | sza_gt75 |
|---|---|---|---|---|---|
| Train | 551 | 226 (Fisser) | 56 | 83 | 186 |
| Val   | 137 | 47 (Fisser) | 14 | 26 | 50 |
| Test  | 228 | 57 (Fisser) | 57 | 57 | 57 |

Source breakdown: every lt65 chip is Fisser-sourced; every other-bin chip is Roboflow-annotated.

Class distribution (training pixels): ocean 94.4 %, iceberg 5.6 %.

### Identity

Every chip row in `manifest.json` carries `chip_stem`, `tif_path`, `tif_sha`, `sza_bin`, `source`, `n_icebergs`, `has_iceberg`, `ic_aware`, `split`, `pkl_position`. A single `chips_sha` over the sorted (chip_stem, tif_sha, split) tuples is the dataset identity. Every downstream output stamps that hash so cross-experiment reproducibility can be verified.

Current chips_sha: `fc4b3b16334f2916...`.

---

## How experiments run

Three layers, top to bottom:

1. **Slurm script** (`slurm/baseline_v1.slurm`, `slurm/baseline_v1_resume.slurm`): SBATCH directives + source `slurm/_common.sh` + invoke the runner.
2. **Experiment runner** (`scripts/run_experiment.py`): given an experiment id, walks the five stages (manifest, train, infer, evaluate, figures), stamps provenance into every output.
3. **Stage scripts** (`train.py`, `run_methods.sh`, `eval_methods.py`, `eval_per_iceberg.py`): the actual work.

The experiment runner refuses to start unless `validate_experiment.py` accepts the experiment YAML. The validator enforces the single-controlled-variable rule: an experiment may touch only one of `data | methods | augmentation | training | inference | evaluation` unless it explicitly declares `controlled_variable:`.

`run_methods.sh` refuses to run if the checkpoint's `training_config.json.manifest_id` does not match the manifest you pass in (cross-manifest drift guard).

`train.py` refuses to run without `--seed` under `ICEBERG_EXPERIMENT=1` (set by every slurm wrapper). Every published checkpoint is reproducible; the seed propagates to Python, NumPy, Torch CPU and CUDA, and the cuDNN deterministic flag.

---

## Running baseline_v1 from a clean slate

```
ssh moosehead
cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework
sbatch slurm/baseline_v1.slurm
```

If the run fails between stages (most commonly inference, because of an unfamiliar dependency), the slurm script prints the exact resume command. Resume with:

```
RUN_TS=<timestamp_dir> sbatch slurm/baseline_v1_resume.slurm
```

---

## Methodological decisions (capsule version)

### Shadow merge
Fisser's three-class masks (ocean, iceberg, shadow) are reduced to binary by remapping shadow into iceberg before any analysis. Aligns Fisser annotations with Roboflow annotations (which do not distinguish shadow). Documented in `reference/descriptive_stats_results_discussion.md` Section 2.

### 40 m root-length cutoff
Connected components smaller than 16 pixels (1,600 m$^2$, 40 m root length) are removed. Matches Fisser (2025) dataset minimum.

### Annotation-aware IC filtering
IC = fraction of non-annotated pixels with B08 >= 0.22. Training chips with IC >= 15 % have bright non-annotated pixels masked to zero. Validation and test never masked. 193 training chips were masked in the v4_clean build. Justification: `reference/b08_analysis_results_discussion.md` sections 3.1-3.6. Sensitivity sweep across the full 23,981-chip pool at IC cutoffs in {10, 15, 20, 25, 30}% is summarised in `methods_draft.md` Section 2.14 and stored under `figure_review/script_check_answers/q01_ic_cutoff_sweep/`.

### DN offset
Reflectances are +0.10 high relative to Fisser's space because chip_sentinel2.py applies the 10$^{-4}$ scaling without subtracting the 1000 DN N0500 offset. Fisser's 0.12 = our 0.22. Internal consistency holds because every chip shares the offset.

---

## Critical numbers (verified 2026-04-27)

- v4_clean: 916 chips total. chips_sha = `fc4b3b16334f2916...`.
- Splits: 551 / 137 / 228. Test cap: 57 chips per SZA bin.
- baseline_v1 trained checkpoint at `runs/exp_baseline_v1/20260424_185158/model/best_model.pth`: 100 epochs, val IoU 0.323, test IoU 0.314 (pixel-level), seed 42.
- Per-pair MAE on root length (m), best per bin: UNet_OT 8.0 at lt65; UNet_CRF 7.4 / 9.0 / 12.6 at 65-70 / 70-75 / >75. Threshold-only methods (TR, OT) over-detect with low precision (5-9%) and worsen with rising SZA.
- 19 experiment YAMLs: baseline_v1 + A0-A9 + B0-B5 + ablation_no_aug + ablation_no_nulls. All validate locally and on HPC.
- 12 balancing schemes: A-I (single axis) + J/K/L (size + composed).
