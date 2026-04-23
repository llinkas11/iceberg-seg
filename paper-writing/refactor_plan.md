# Iceberg Pipeline Refactor Plan

**Target codebase:** `/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/` (live, canonical)
**Mirror on HPC/local:** `~/Desktop/IDS2026/S2-iceberg-areas/` is older, partially divergent, and should be retired after migration.
**Split policy (new baseline):** 65 / 15 / 25 train/val/test.
**Figure policy (project-wide):** every generated figure is written both to `fig-archive/<YYYYMMDD_HHMM>__<slug>.png` (append-only, never overwritten) and to `figures.md` (live table of current figures with captions). When a figure is regenerated, `figures.md` points at the new `fig-archive/` entry; the old entry stays in the archive.

---

## 0. HPC sync check (answered)

`llinkas/iceberg-rework/` is a **strict superset** of `smishra/llinkas-rework/`. The 3 divergent files are all llinkas-ahead:

| File | What changed in llinkas copy |
|---|---|
| `scripts/balance_training.py` | Rewritten to SZA-aware two-stage balancer (stage 1 GT0/GT+, stage 2 area bins). Adds `--out_dir` guard so v3_balanced baseline is not overwritten. |
| `scripts/predict_tifs.py` | Handles missing `val_iou` key in older checkpoints; fixes `prob_meta` count for variable band outputs. |
| `scripts/train.py` | Adds `--seed` (Python / NumPy / Torch / cuDNN) + seeded worker init + generator for deterministic reruns; adds `persistent_workers`; records `ckpt_metric` in checkpoint. |

llinkas is also ahead by **many extra scripts, five new dataset variants, four model variants, four results directories**. smishra has nothing unique to merge back. **No sync action required.** smishra's `llinkas-rework/` can be treated as the pre-fork snapshot.

---

## 1. Current Repository Audit

### 1.1 Two parallel codebases (this is itself a risk)

| Location | Role today | Recommendation |
|---|---|---|
| `IDS2026/S2-iceberg-areas/` (local OneDrive) | Original pre-rework repo. `train.py` still says "s2 3-class", `predict_tifs.py` still polygonises shadow class, no `--seed`, no probs output, no CRF, no `eval_methods.py`. Used by no current experiment. | Archive. Keep as `_archive/s2-iceberg-areas-prerework/`. Do not edit. |
| `iceberg-rework/scripts/` (HPC) | Live. All six methods, binary segmentation, IC masking, seeded training, eval\_methods, per-iceberg eval, comparison tooling. | **Canonical.** All refactor work lands here. |

### 1.2 Entrypoints in `iceberg-rework/scripts/`

**Dataset construction:**
- `build_fisser_index.py`: index Fisser source chips.
- `build_test_index.py`: index test-chip dir.
- `filter_small_icebergs.py`: 40 m root-length filter, shadow-into-iceberg merge.
- `filter_quality.py`: IC filter (annotation-aware B08 ≥ 0.22, ≥ 15 % threshold).
- `build_clean_dataset.py`: produces `data/v3_clean/` (split + IC-masked training pkls + `split_log.csv`). **This is the de-facto manifest source.** Default split is 60/15/25 (needs change to 65/15/25).
- `balance_training.py`: consumes `v3_clean`, writes a balanced-variant dir. Two modes (stage 1 only / stage 1 + stage 2). Default out is `--out_dir` specified per-call.
- `build_gt_positive_training.py`: drops GT0 training chips, copies val/test unchanged. Clean isolation of training-only mutation.
- `add_small_icebergs.py`, `filter_small_icebergs.py`, `prepare_new_training_data.py`, `prepare_test_chips_dir.py`: variant builders / assemblers.

**Training:** `train.py` (accepts `--data_dir` pointing at any variant; `--seed` optional; augmentation flag is `--no_augment`).

**Inference (6 methods on a chip dir):** `threshold_tifs.py`, `otsu_threshold_tifs.py`, `predict_tifs.py`, `threshold_probs.py`, `otsu_probs.py`, `densecrf_tifs.py`, + `threshold_masked_tifs.py`.

**Orchestrators:**
- `run_all_methods.sh`: runs 6 methods for one SZA bin against one chip dir with one checkpoint.
- `run_pipeline.sh`: older full pipeline (chip → unet → threshold), KQ/SK loop.
- `run_model_comparison_eval.sh`, `run_matched_comparison_eval.sh`, `run_gt_positive_only_eval.sh`: driver shells for the experiments run on 2026-04-23.

**SLURM wrappers:** `job.slurm`, `train_matched.slurm`, `gt_positive_only_eval.slurm`, `model_comparison_eval.slurm`, `matched_comparison_eval.slurm`, `train_gt_positive_only.slurm`.

**Evaluation:**
- `eval_methods.py`: loads `y_test.pkl` + `split_log.csv`, rasterises per-method GPKGs back to masks, computes IoU / precision / recall / F1 per chip, per (method, sza_bin).
- `eval_per_iceberg.py`: per-iceberg MAE, RERL, contrast via CC-IoU matching.
- `compare_model_eval.py`: pairwise (baseline vs variant) summary tables, heatmaps.
- `make_figure21_iou_gt_positive_comparison.py`: figure.

**Analysis / QA:** `audit_fisser_provenance.py`, `dataset_analysis.py`, `descriptive_stats.py`, `summarize_training_run.py`, `visualize_missed_icebergs.py`, `visualize_predictions.py`.

**Utilities / one-offs:** `annotate_roboflow_otsu.py`, `cloud_filter_roboflow.py`, `export_*.py`, `upload_*.py`, `fetch_met_data.py`, `rebin_downloads.py`.

**Notebooks:** none in `iceberg-rework/` (good). `IDS2026/notebooks/` has a legacy notebook dir, not used in current pipeline.

**Config files:** none. Every experiment is parameterised by CLI flags and env vars in shell scripts. No central config system.

**Data variants on HPC (each is a full pickled 65/15/25-like snapshot, not a pointer list):**
`data/v3_clean`, `data/v3_balanced`, `data/v3_balanced_sza_stage1`, `data/v3_balanced_sza_stage1_stage2`, `data/v3_balanced_stage1_test`, `data/v3_balanced_stage2_test`, `data/v3_train_gt_positive_only`, `data/fisser_filtered/`, `data/annotations_filtered.coco.json`.

**Model variants:** `model/v3_balanced_sza_stage1_aug_20260423`, `model/v3_balanced_sza_stage1_matched_seed42_aug_20260423`, `model/v3_clean_matched_seed42_aug_20260423`, `model/v3_train_gt_positive_only_aug_20260423`.

**Results variants:** `results/model_comparison_20260423_*`.

### 1.3 Dead code / duplicated code

- `scripts/train.py.bak_matched_20260423`, `scripts/balance_training.py.bak`, `scripts/__pycache__/`: delete after migration tag.
- `IDS2026/S2-iceberg-areas/predict.py` vs `predict_tifs.py` vs rework `predict_tifs.py`: three near-identical model-loading paths.
- `threshold_tifs.py` vs `threshold_probs.py` vs `threshold_masked_tifs.py`: three near-identical polygonisation loops; only input array and threshold source differ.
- `otsu_threshold_tifs.py` vs `otsu_probs.py`: duplicated loop + identical filter guards.
- `compare_areas.py` (local repo) vs `compare_model_eval.py` (rework): overlap.
- Loss/DiceLoss/IoU scoring defined in `train.py`, then separately inlined in `eval_methods.py` and `eval_per_iceberg.py`.

### 1.4 Scripts that mutate or silently depend on state

- `balance_training.py` default `CLEAN_DIR = data/v3_clean`: safe (reads, writes to a different out\_dir).
- `build_clean_dataset.py` reads from `SMISHRA = .../smishra/rework/chips` → **external dependency on smishra's chips tree.** If smishra ever deletes those chips, the pipeline breaks. Mitigate by either copying chips into `iceberg-rework/data/raw_chips/` or resolving the dependency via a content-addressed manifest.
- `_build_tif_index()` in `build_clean_dataset.py` is a module-level global, safe but implicit.
- `chip_sentinel2.py` writes into `chips/{region}/{sza_bin}/tifs` with no versioning; a rerun silently overwrites.
- `run_all_methods.sh` hardcodes default `CHECKPOINT=smishra/S2-iceberg-areas/runs/s2_v2_aug/best_model.pth` and `CHIPS=smishra/S2-iceberg-areas/test_chips/{BIN}`. **Critical hidden coupling:** whoever calls the script can silently swap model or test chips without it being recorded in the output dir.
- `predict_tifs.py --save_probs` writes a `probs/` dir that `threshold_probs.py`, `otsu_probs.py`, and `densecrf_tifs.py` implicitly consume. If UNet is rerun with different softmax handling, the three downstream methods give different answers without any signal that the upstream changed.
- Augmentation is a **boolean flag in `train.py`** (`--no_augment` flips the `IcebergDataset(augment=True)` constructor). It is not stored in the checkpoint args in a way that downstream tooling reads. Checkpoints do record `vars(args)` in the `.pth`, but nothing verifies it at inference.
- No seed is stored in data-variant dirs. `build_clean_dataset.py` uses `random.Random(seed)` internally; if the seed CLI flag changes, a rebuilt v3\_clean would silently shift chip-to-split membership.

### 1.5 Hidden dependencies summary

- `eval_methods.py` requires `tif_path` in `split_log.csv` → requires `build_clean_dataset.py` to have written non-blank paths (currently blank for `source=fisser` rows; see split\_log row 0). This breaks evaluation on any test chip whose source is Fisser because `get_chip_transform` returns `None, None`.
- `run_all_methods.sh` requires `prepare_test_chips_dir.py` to have been run, nothing enforces that.
- `densecrf_tifs.py` requires a chip tif AND a probs tif; any mismatch in extent is silently handled.

---

## 2. Risks to Scientific Validity

Grouped by failure mode.

### 2.1 Split can drift between variant rebuilds
- `build_clean_dataset.py` has a `--train_frac/--val_frac/--test_frac` CLI. Running it again with the same flags and same seed reproduces membership, but there is no written-to-disk assertion. A changed chip inventory in `smishra/rework/chips/` (new chips added, or any filter threshold modified) silently reshuffles splits.
- **Fix:** freeze the split as a content-addressed `manifest.json` listing `chip_stem → split` and a SHA over the sorted chip list; refuse to run if the hash on disk differs from the manifest's recorded hash.

### 2.2 Variant construction changes multiple things at once
- Current practice: every variant is a new pickled directory. `build_gt_positive_training.py` does the right thing (training-only filter, val/test copied by `shutil.copy2`). But:
- `balance_training.py` does NOT copy val/test into the balanced out\_dir as a naming-stable alias; it writes new X\_validation.pkl / x\_test.pkl with potentially different pkl orderings. If `pkl_position` ordering is not stable, `eval_methods.py` joins on `pkl_position` and can misalign.
- **Fix:** enforce that test + val pkl byte-for-byte match `v3_clean`'s test + val for every non-test-modifying variant. Add a post-build check.

### 2.3 Augmentation is coupled to "the training script"
- Augmentation toggle is in `train.py` only. Two variants that differ ONLY in augmentation end up in different `model/*` dirs with different names, but the augmentation state is not persisted as a first-class field anywhere downstream reads. If a stranger looks at `results/model_comparison_*`, they must trace the run via model dir name convention to recover this fact.
- **Fix:** write `training_config.json` next to `best_model.pth` containing every hyperparameter + data manifest id + seed + aug flags. Eval tooling reads it, stamps it into output CSVs.

### 2.4 Test set can differ silently between methods
- `run_all_methods.sh` is passed a `BIN` and resolves `CHIPS=${RESEARCH}/S2-iceberg-areas/test_chips/${BIN}`. That path is built by `prepare_test_chips_dir.py` which is itself parameterised. Two runs on different dates can compute IoU over **different chip sets** while both being called "sza\_lt65 test".
- **Fix:** `run_all_methods.sh` must take a `--manifest_id` and resolve chip paths from the manifest's test records, not from a directory listing.

### 2.5 Metrics computed differently across methods
- `eval_methods.py` rasterises GPKG polygons back to pixel masks at each chip's *tif* transform / height / width. That works uniformly for all methods. Good.
- BUT: `predict_tifs.py` applies its own `--min_area_m2` filter before polygonisation; `threshold_tifs.py` uses `MIN_AREA_M2 = 100`; `otsu_threshold_tifs.py` uses `MIN_AREA_M2 = 100`; `threshold_probs.py` / `otsu_probs.py` / `densecrf_tifs.py` must be checked (read them). Any mismatch means UNet vs Otsu IoU difference is partly due to different minimum-area cuts.
- **Fix:** centralise `MIN_AREA_M2` in a single method-config section; log it per run; assert equality when comparing.

### 2.6 Implicit filtering in methods
- `otsu_threshold_tifs.py` silently skips chips with `thresh < otsu_floor=0.10`, `bright_frac > 0.15` (sea ice), < 100 valid pixels, or any threshold\_otsu exception. A "skipped" chip produces no polygons and is thus evaluated as "all zeros prediction" by `eval_methods.py`, which affects recall. This is a modelling choice disguised as an I/O filter.
- **Fix:** have each method also write a `skipped_chips.csv` so the evaluation joins on exact chip membership and treats skip either as (a) false negative or (b) excluded from per-method metric, chosen explicitly via a flag, not silently.

### 2.7 Silent seed surfaces
- `train.py --seed` exists but defaults to `None` (non-deterministic). Any result produced without `--seed` is fundamentally unreproducible; comparing an unseeded run to a seeded run entangles training noise with the variable of interest.
- `build_clean_dataset.py` uses its own seed, not propagated from a top-level config.
- `balance_training.py` uses its own seed.
- **Fix:** enforce seed propagation at every stochastic boundary (dataset build, balancing, training, any augmentation) through a single `config.seed` field. `train.py` should refuse to run without a seed under experiment mode.

### 2.8 Checkpoint identity is fragile
- Model dir name encodes intent (`v3_balanced_sza_stage1_matched_seed42_aug_20260423`), but the naming convention is not enforced by code. A rename breaks nothing visibly but loses provenance.
- **Fix:** a `training_config.json` side-by-side with `best_model.pth` + a pointer at the consumed data manifest.

### 2.9 Fisser rows have blank `tif_path`
- `split_log.csv` row 0 shows `tif_path=""` for a Fisser chip. `eval_methods.get_chip_transform` returns `None` and the chip is skipped. This means Fisser test chips are **invisible to `eval_methods.py`** without it being obvious.
- **Fix:** during `build_clean_dataset.py`, write a synthetic tif for Fisser chips (CRS + transform recoverable from Fisser provenance audit), OR store pixel-space ground truth and evaluate without a georeferenced round-trip for Fisser rows.

### 2.10 Chip source directory not content-addressed
- Everything ultimately points at `smishra/rework/chips/`. If that tree changes, intentionally or not, every dataset variant, every trained model, every evaluation result silently becomes incompatible with any newly rebuilt variant.
- **Fix:** copy-on-write into `iceberg-rework/data/raw_chips/` and record a SHA index.

---

## 3. Canonical Baseline Definition, `baseline_v1`

### 3.1 Baseline spec (`configs/baselines/baseline_v1.yaml`)

```yaml
id: baseline_v1
created: 2026-04-23
rationale: |
  First canonical baseline after the 2026-Q1 rework. Captures the intended
  Fisser-aligned binary segmentation pipeline with 65/15/25 split, IC-masked
  training, shadow merged into iceberg, 40 m root-length filter, and standard
  augmentations.

data:
  manifest_id: v4_clean_65_15_25
  source: fisser_roboflow_merged
  split:
    train_frac: 0.65
    val_frac:   0.15
    test_frac:  0.25
    stratify_by: sza_bin
    seed: 42
  filters:
    shadow_merge:       true          # Fisser class 2 -> class 1
    root_length_min_m:  40
    ic_threshold:       0.15
    ic_mask_scope:      train_only    # val + test never masked
    b08_threshold_ic:   0.22          # Fisser 0.12 + 0.10 DN offset
  balancing_scheme: none              # v4_clean is the unbalanced baseline;
                                      # balancing is a separate named variant

preprocessing:
  chip_size_px: 256
  pixel_area_m2: 100.0
  bands: [B04, B03, B08]
  dn_scale: 1.0e-4
  dn_offset_applied: false            # acknowledged: raw TOA +0.10 relative to Fisser

augmentation:
  enabled: true
  ops: [hflip, vflip, rot90]

model:
  arch: UnetPlusPlus
  encoder: resnet34
  encoder_weights: imagenet
  num_classes: 1                      # binary
  in_channels: 3

training:
  epochs: 100
  batch_size: 16
  lr: 1.0e-4
  weight_decay: 1.0e-4
  optimizer: AdamW
  scheduler: CosineAnnealingLR
  scheduler_eta_min: 1.0e-6
  loss: dice_plus_bce
  seed: 42
  workers: 4
  ckpt_metric: val_iou

inference:
  min_area_m2: 100          # applied uniformly across ALL methods
  bands_expected: 3
  probs_output: true        # UNet saves P(iceberg) so TR/OT/CRF on probs are reproducible

methods:
  TR:       { b08_threshold: 0.22 }
  OT:       { otsu_floor: 0.10, otsu_ceil: 0.50, sea_ice_frac: 0.15, b08_idx: 2 }
  UNet:     { threshold: 0.5 }
  UNet_TR:  { prob_threshold: 0.5 }
  UNet_OT:  { otsu_floor_prob: 0.10, otsu_ceil_prob: 0.90 }
  UNet_CRF: { sxy_bilateral: 80, srgb: 13, sxy_gaussian: 3, compat: 10, iters: 5 }

evaluation:
  metrics:
    pixel:
      - iou
      - dice
      - precision
      - recall
      - f1
      - mae_area
      - mse_area
    per_iceberg:
      - matched_iou
      - mae_area
      - rerl
      - contrast_b08
  aggregations:
    - all_chips                # mean over every test chip
    - gt_positive_only         # only chips with any GT iceberg
    - per_sza_bin              # broken out by sza_lt65 / _65_70 / _70_75 / _gt75
    - per_region               # KQ / SK
  skipped_chip_policy: count_as_false_negative   # explicit, not implicit

reporting:
  fig_archive: fig-archive/
  fig_index:   figures.md
```

### 3.2 What a baseline run produces

```
runs/baseline_v1/<timestamp>/
  training_config.json           # verbatim copy of the config used
  best_model.pth
  training_log.csv
  inference/<method>/
    gpkgs/
    probs/           # UNet only
    skipped_chips.csv
    method_config.json
  evaluation/
    eval_summary.csv
    eval_summary_gt_positive_only.csv
    eval_per_chip.csv
    eval_per_iceberg.csv
    figures.md        # per-run, links to fig-archive/
```

### 3.3 Test-locking requirement

Once `baseline_v1` is frozen:
- Its `manifest.json` test-chip list is immutable.
- Any new variant using the same `manifest_id` MUST produce the same SHA for test+val chip lists (enforced by `tools/check_manifest.py`).
- Any new variant changing test membership gets a new `manifest_id` and cannot be compared head-to-head against `baseline_v1` without an explicit flag.

---

## 4. Proposed New Repository Structure

Adapted to what exists. Everything below is under `iceberg-rework/`.

```
iceberg-rework/
  configs/
    baselines/
      baseline_v1.yaml
    experiments/                      # one YAML per experiment, each inherits a baseline
      exp_01_fisser_lt65_source.yaml
      exp_02_fisser_lt65_plus_nulls.yaml
      exp_03_our_lt65.yaml
      exp_04_our_lt65_plus_nulls.yaml
      exp_05_no_augmentation.yaml
      exp_06_balancing_natural.yaml
      exp_07_method_otsu.yaml
      exp_08_method_unet.yaml
      exp_09_method_crf.yaml
    datasets/                         # named dataset recipes (source + filters)
      v4_clean_65_15_25.yaml
      fisser_lt65_original.yaml
      fisser_lt65_plus_nulls.yaml
      our_lt65.yaml
      our_lt65_plus_nulls.yaml
    balancing/                        # one YAML per named scheme (A-H below)
      scheme_A_fisser_original.yaml
      scheme_B_fisser_plus_nulls.yaml
      scheme_C_our_lt65_plus_nulls.yaml
      scheme_D_equal_pos_null.yaml
      scheme_E_natural.yaml
      scheme_F_fixed_total_114.yaml
      scheme_G_equalized_across_sza.yaml
      scheme_H_custom.yaml
    methods/                          # per-method defaults, versioned
      TR.yaml
      OT.yaml
      UNet.yaml
      UNet_TR.yaml
      UNet_OT.yaml
      UNet_CRF.yaml

  data/
    raw_chips/                        # canonical content-addressed chip store
      index.sha.csv                   # chip_stem -> SHA of .tif
    manifests/                        # ALL dataset manifests live here
      v4_clean_65_15_25/manifest.json
      fisser_lt65_original/manifest.json
      fisser_lt65_plus_nulls/manifest.json
      ...
    splits/                           # only the split files; chip bytes come from raw_chips/
      v4_clean_65_15_25/split_log.csv
    processed/                        # pkl pyramids if needed for fast training
      v4_clean_65_15_25/train_validate_test/
    _archive/                         # frozen snapshots of prior data variants

  src/
    iceberg/
      __init__.py
      config.py                       # load + validate YAML, apply inheritance
      manifest.py                     # build / read / hash manifests
      seed.py                         # central seed helper
      datasets/
        build_clean.py                # replaces build_clean_dataset.py
        variants.py                   # fisser_lt65_original, +nulls, our_lt65, etc.
      balancing/
        __init__.py
        strategies.py                 # pluggable schemes A..H
      preprocessing/
        chip_sentinel2.py
        ic_filter.py
        root_length.py
      augmentation/
        transforms.py                 # single source of truth for augs
      methods/
        base.py                       # IcebergMethod ABC: predict_chip / write_gpkg
        threshold.py                  # TR
        otsu.py                       # OT
        unet.py                       # UNet (wraps smp)
        threshold_probs.py            # UNet_TR
        otsu_probs.py                 # UNet_OT
        densecrf.py                   # UNet_CRF
      training/
        train.py                      # thin wrapper around iceberg.models
        losses.py
      inference/
        run_method.py                 # dispatches by method name
      evaluation/
        pixel_metrics.py              # iou/dice/prec/rec/f1/MAE/MSE
        per_iceberg.py                # MAE/RERL/contrast/matched IoU
        aggregate.py                  # all_chips / gt_positive_only / per_sza / per_region
      figures/
        registry.py                   # write(fig, slug, caption) -> fig-archive + figures.md

  scripts/                            # thin CLI wrappers over src/
    run_experiment.py                 # primary entrypoint; see §7
    build_manifest.py
    validate_experiment.py
    freeze_baseline.py

  slurm/
    train.slurm                       # takes EXP_ID, nothing else
    infer.slurm
    evaluate.slurm

  runs/
    baseline_v1/<timestamp>/
    exp_02/<timestamp>/
    ...

  fig-archive/                        # append-only
  figures.md                          # live index

  reports/                            # narrative writeups
  notebooks_archive/
  _archive/                           # current scripts/*.bak, __pycache__, legacy
```

Migration philosophy: **keep current `scripts/` running during transition**; add `src/iceberg/` alongside; new work uses `run_experiment.py`; legacy scripts are deleted only once each has a `src/` equivalent with passing byte-for-byte regression.

---

## 5. Dataset Manifest System

### 5.1 Manifest schema (`data/manifests/<manifest_id>/manifest.json`)

```json
{
  "manifest_id": "fisser_lt65_plus_nulls_v2",
  "created": "2026-04-23T18:00:00Z",
  "source_recipe": "configs/datasets/fisser_lt65_plus_nulls.yaml",
  "source_recipe_sha": "sha256:ab12...",
  "chip_source": "data/raw_chips/",
  "chip_source_sha": "sha256:9f03...",

  "bin_scope": ["sza_lt65"],
  "region_scope": ["KQ", "SK"],

  "split_policy": {
    "train_frac": 0.65,
    "val_frac":   0.15,
    "test_frac":  0.25,
    "stratify_by": ["sza_bin", "has_iceberg"],
    "seed": 42
  },

  "balancing_scheme": "scheme_B_fisser_plus_nulls",
  "balancing_scheme_sha": "sha256:c00c...",
  "balancing_report": "data/manifests/fisser_lt65_plus_nulls_v2/balance_report.csv",

  "gt_positive_count": 57,
  "gt_zero_count":    29,
  "total_chips":      86,

  "counts_by_split": {
    "train": { "gt_positive": 37, "gt_zero": 19 },
    "val":   { "gt_positive":  9, "gt_zero":  4 },
    "test":  { "gt_positive": 11, "gt_zero":  6 }
  },

  "chips": [
    {
      "chip_stem": "fisser_0371",
      "tif_path":  "data/raw_chips/fisser_0371.tif",
      "tif_sha":   "sha256:...",
      "sza_bin":   "sza_lt65",
      "region":    "KQ",
      "source":    "fisser",
      "has_iceberg": true,
      "n_icebergs": 119,
      "max_rl_m":   412.3,
      "ic":         0.79,
      "ic_masked":  true,
      "split":      "train",
      "pkl_position": 0
    },
    "..."
  ],

  "chips_sha": "sha256:deadbeef..."
}
```

`chips_sha` is computed over the sorted `[chip_stem, tif_sha, split]` triples. It is the identity that downstream methods and evaluation stamp into every output CSV. Two manifests with identical `chips_sha` are the same dataset; any disagreement explains itself by field.

### 5.2 Manifest operations

- `iceberg.manifest.build(config_path)`: resolve recipe + balancing scheme → write manifest.json + balance\_report.csv + pkls.
- `iceberg.manifest.verify(manifest_path)`: recompute SHAs, assert match.
- `iceberg.manifest.diff(a, b)`: print one-line-per-changed-field (chip set, split, balancing, seed); refuses to operate if more than one family of field differs, so you cannot accidentally publish a two-variable diff.

### 5.3 The three "dataset" levels

| Level | Artifact | When produced |
|---|---|---|
| Recipe | `configs/datasets/*.yaml` | Hand-authored |
| Manifest | `data/manifests/<id>/manifest.json` | `build_manifest.py --dataset <recipe_id>` |
| Materialised pkls | `data/processed/<id>/train_validate_test/` | Emitted alongside manifest for training convenience. The manifest remains the source of truth; pkls are a cache. |

---

## 6. Balancing Module Refactor

### 6.1 Strategy interface (`src/iceberg/balancing/strategies.py`)

```python
class BalancingStrategy(ABC):
    scheme_id: str

    @abstractmethod
    def apply(self, manifest: Manifest, seed: int) -> Manifest:
        """Return a new Manifest with only training-level reshuffling.
        val + test must be pointer-identical to input."""
```

Concrete implementations, one class per scheme, all registered in
`BALANCING_SCHEMES = {"scheme_A_fisser_original": FisserOriginal, ...}`.

### 6.2 The eight schemes

| Scheme | What it changes | Scope |
|---|---|---|
| A `fisser_original` | Keep lt65 positive-only; drop GT0 chips. | Train, lt65 only |
| B `fisser_plus_nulls` | Keep all Fisser positives; inject GT0 chips up to target ratio. | Train, lt65 only |
| C `our_lt65_plus_nulls` | Same as B but on our lt65 chips. | Train, lt65 only |
| D `equal_pos_null_ratio` | Enforce 1:1 GT+ / GT0 per sza\_bin. | Train, all bins |
| E `natural` | Leave natural distribution untouched. | Train |
| F `fixed_total_114` | Cap training to 114 chips total, stratified. | Train |
| G `equalized_across_sza` | Equal training count across all SZA bins. | Train |
| H `custom` | User-supplied ratio table in YAML. | Train |

### 6.3 Scope separation

Each scheme instance declares its scope at import:

- `stage`: `pre_split` (runs before train/val/test split; NOT used in v1, too risky), or `train_only` (runs on the training subset only, which is what every scheme above does).
- `class_balance`: GT+ vs GT0 ratio target.
- `size_balance`: root-length bin target (stage-2 style).
- `sza_balance`: across-sza target.

The scheme YAML makes all four dimensions explicit. If any is unspecified, the default is "no change".

### 6.4 Invariants enforced at apply time

1. `val` split pkl bytes unchanged.
2. `test` split pkl bytes unchanged.
3. `manifest.chips[i].pkl_position for split='test'` has the same value before and after.
4. Balance report CSV written: per-(sza, group) input count → action → output count.

Apply fails with a hard error if any invariant breaks.

### 6.5 Current code → new code mapping

| Current | New |
|---|---|
| `scripts/balance_training.py --balance_positive_area_bins` | scheme D (pos/null ratio) + stage 2 as separate scheme composed in YAML |
| staged balancing in `balance_training.py` | the staged logic becomes two composable strategies applied in order |
| `scripts/build_gt_positive_training.py` | scheme A (`fisser_original` specialised to all-bins) |

---

## 7. Single Config Experiment Runs

### 7.1 Experiment YAML (`configs/experiments/exp_02_fisser_lt65_plus_nulls.yaml`)

```yaml
id: exp_02
inherits: baseline_v1
change:
  data:
    manifest_id: fisser_lt65_plus_nulls_v2
    source: fisser_lt65_plus_nulls
    balancing_scheme: scheme_B_fisser_plus_nulls
notes: |
  Same as baseline_v1 except we use Fisser's original lt65 positives and
  add GT0 chips at a 1:1 ratio.
```

### 7.2 Validator rules (`validate_experiment.py`)

- **Exactly one top-level change path** in `change`. More than one = refuse. `data.*` counts as one family; `methods.UNet.threshold` counts as another.
- **All inherited fields preserved** unless explicitly overridden.
- **Manifest compatibility:** if `change` touches data but not evaluation, the new manifest must share `val` and `test` SHAs with the baseline's manifest. Otherwise the experiment is marked `incomparable_to_baseline: true` and can only be compared against another non-baseline manifest.
- **Seed preserved** unless the experiment explicitly changes it (in which case it must also mark itself as `requires_reseeded_baseline: true`, forcing a baseline re-run at the new seed for head-to-head comparison).
- **Augmentation propagation:** if `change.augmentation.enabled` flips, validator refuses unless the experiment also declares `controlled_variable: augmentation`.

### 7.3 Runner (`scripts/run_experiment.py`)

```
python scripts/run_experiment.py --exp exp_02 [--stages manifest,train,infer,evaluate,figures]
```

Stages:
1. `manifest`: resolve recipe → write manifest → assert val/test SHAs per validator rules.
2. `train`: read experiment training config → call `src/iceberg/training/train.py` with full provenance → write `training_config.json` next to checkpoint.
3. `infer`: for each method in `methods:`, run `src/iceberg/inference/run_method.py --method <M> --manifest <id> --checkpoint <path>` writing per-method `method_config.json`.
4. `evaluate`: `src/iceberg/evaluation/*` producing summary + per-chip + per-iceberg CSVs.
5. `figures`: generate per-experiment figures via `iceberg.figures.registry.write()`.

Every stage stamps the experiment id, manifest id, chips\_sha, and a SHA of the resolved merged config into its outputs.

---

## 8. Unified Method Runner

### 8.1 Common interface

```python
class IcebergMethod(ABC):
    name: str
    requires: set[str]    # e.g. {"chip_tif"}, or {"chip_tif","probs_tif"}

    @abstractmethod
    def predict_chip(self,
                     chip: ChipInputs,
                     cfg: MethodConfig) -> PredictionOutputs: ...
```

`ChipInputs` holds `tif_path`, loaded `bands (C,H,W) float32`, `transform`, `crs`, and optionally `probs (K,H,W)`.
`PredictionOutputs` holds `polygons: GeoDataFrame`, `binary_mask: (H,W) uint8`, `skipped: bool`, `skip_reason: str|None`.

### 8.2 Runner dispatcher

```python
def run_method(method_name: str, manifest: Manifest, method_cfg: dict, out_dir: Path) -> None:
    method = METHOD_REGISTRY[method_name](method_cfg)
    for chip in manifest.iter_chips(split="test"):
        chip_inputs = load_chip_inputs(chip, needs=method.requires)
        out = method.predict_chip(chip_inputs, method_cfg)
        write_outputs(out, out_dir, chip.chip_stem)
    write_merged_gpkg(out_dir)
```

### 8.3 Why this replaces six scripts

- `threshold_tifs.py`, `otsu_threshold_tifs.py`, `predict_tifs.py`, `threshold_probs.py`, `otsu_probs.py`, `densecrf_tifs.py` → six `IcebergMethod` classes.
- Minimum-area filter applied once, in `run_method`, not inside each method.
- Skip policy applied once, recorded in `skipped_chips.csv`.
- `predict_chip` is the only place method logic lives. This makes apples-to-apples comparison structural.

### 8.4 Same-input guarantee

`run_method` reads chips from `manifest.iter_chips(split="test")`. The chip dir is not a parameter. A different manifest is the only way to change what chips run. This closes the hole in `run_all_methods.sh` where `CHIPS` was an ambient env var.

---

## 9. Evaluation Standardisation

### 9.1 Pixel metrics (`src/iceberg/evaluation/pixel_metrics.py`)

Computed per chip, stored in `eval_per_chip.csv`:
`iou`, `dice`, `precision`, `recall`, `f1`, `tp_px`, `fp_px`, `fn_px`, `tn_px`, `pred_area_m2`, `gt_area_m2`, `area_mae_m2`, `area_mse_m2`.

Aggregated in `eval_summary.csv`:
mean, median, 25/75 percentiles, count. For each of:
- all chips
- GT-positive-only chips
- per SZA bin
- per region
- per (SZA bin × region)

### 9.2 Per-iceberg metrics (`src/iceberg/evaluation/per_iceberg.py`)

Via greedy CC-IoU matching (from existing `eval_per_iceberg.py`):
`matched_iou`, `area_mae_m2`, `area_mse_m2`, `rerl_m`, `contrast_b08`.

Aggregated in `eval_per_iceberg_summary.csv` with the same aggregation dimensions.

### 9.3 MAE / MSE parity with older papers

`area_mae_m2` and `area_mse_m2` come from both (a) pixel-mask area difference per chip and (b) per-iceberg matched area difference, reported separately and clearly labeled. Fisser-style reports use chip-area MAE; earlier threshold-only papers use per-scene MAE; the system produces both so external comparisons are apples-to-apples.

### 9.4 Skip policy

Configurable in `baseline.evaluation.skipped_chip_policy`:
- `count_as_false_negative`: predicted mask is all zeros, metrics computed normally. **Default.**
- `exclude`: chip is dropped from aggregation; count appears in the summary.

Never silent. Always logged in `eval_summary.csv` as `n_skipped_{method}`.

### 9.5 Fisser-row handling

For any chip whose `tif_path` is empty or missing (historically true for Fisser chips), evaluation either:
- synthesises a CRS-less identity transform and evaluates in pixel space (acceptable because we do not need real-world area for Fisser per-chip metrics; area in m² is derived from `pixel_area_m2` constant), or
- the manifest build stage writes a synthetic georeferenced tif so all chips use the same code path.

Pick once. Record in baseline YAML.

---

## 10. Experiment Matrix

| Experiment | Inherits | `change:` | Comparable to baseline? |
|---|---|---|---|
| `baseline_v1`       |, | none |, |
| `exp_01`            | baseline_v1 | `data.source: fisser_lt65_original` | yes (same val+test SHA if enforced) |
| `exp_02`            | baseline_v1 | `data.source: fisser_lt65_plus_nulls`, `balancing_scheme: B` | yes |
| `exp_03`            | baseline_v1 | `data.source: our_lt65` | yes |
| `exp_04`            | baseline_v1 | `data.source: our_lt65_plus_nulls`, `balancing_scheme: C` | yes |
| `exp_05`            | baseline_v1 | `augmentation.enabled: false`, `controlled_variable: augmentation` | yes |
| `exp_06`            | baseline_v1 | `data.balancing_scheme: scheme_E_natural` | yes |
| `exp_07`            | baseline_v1 | `methods: [OT]`  (only reports OT) | yes (same manifest, same test) |
| `exp_08`            | baseline_v1 | `methods: [UNet]` | yes |
| `exp_09`            | baseline_v1 | `methods: [UNet_CRF]` | yes |
| `exp_10` (suggested) | baseline_v1 | `training.seed: 137` | requires `requires_reseeded_baseline` flag |
| `exp_11` (suggested) | baseline_v1 | `inference.min_area_m2: 400` | yes, data-independent |

All of exp\_07 / 08 / 09 produce results over the same manifest and the same trained UNet. They differ only in which method section is enabled at inference.

---

## 11. Immediate Refactor Priorities

### 11.1 Must fix first (blocks valid comparison today)

1. **Produce `v4_clean_65_15_25` manifest.** Rerun `build_clean_dataset.py` with `--train_frac 0.65 --val_frac 0.15 --test_frac 0.25` and write a proper `manifest.json` with chips\_sha. Every future experiment references this ID.
2. **Fix Fisser-row `tif_path` blanks in `split_log.csv`.** Either (a) write synthetic tifs at build time, or (b) route Fisser chips through a pixel-space evaluation branch. Without this, Fisser test chips are invisible to `eval_methods.py`.
3. **Write `training_config.json` next to every checkpoint.** Minimal viable: dump `vars(args)` + git SHA + manifest id + data SHA into `<out_dir>/training_config.json` from `train.py`. Four-line patch.
4. **Make `run_all_methods.sh` refuse to run without `--manifest_id`.** Remove hardcoded `CHIPS=` / `CHECKPOINT=` defaults. Derive both from the experiment YAML.
5. **Centralise `min_area_m2` across all six methods.** Today each has its own constant. Until unified, every UNet-vs-OT comparison is contaminated.
6. **Enforce seeds.** `train.py`: default `--seed 42`; refuse `--seed None` when invoked by `run_experiment.py`. `build_clean_dataset.py` + `balance_training.py` read seed from the same config.

### 11.2 Freeze next (lock before running more experiments)

7. **Freeze `baseline_v1.yaml`.** Write to `configs/baselines/baseline_v1.yaml` with `v4_clean_65_15_25` as its manifest, tag the git commit, and produce one baseline run to tag as the reference.
8. **Write `validate_experiment.py`.** Single-variable-change guard is the whole point of this refactor; until it exists, accidental multi-variable diffs will keep happening.
9. **Stand up `data/raw_chips/` content store.** Copy (or symlink with recorded SHAs) the smishra chips used by each manifest so future deletions don't silently break the chain.
10. **Stand up `fig-archive/` + `figures.md` registry.** Implement `iceberg.figures.registry.write(fig, slug, caption)`. Wire into `compare_model_eval.py` and `make_figure21_iou_gt_positive_comparison.py` first.

### 11.3 Can wait

11. **Port scripts into `src/iceberg/`.** Mechanical refactor, do it after the four-plus-six steps above are in. Keep current scripts running in the meantime.
12. **Write per-method YAMLs and pluggable methods.** Once `src/iceberg/` exists, migrate six scripts into six `IcebergMethod` classes.
13. **Unified augmentation module.** Move the in-line `IcebergDataset.__getitem__` augmentation logic to `src/iceberg/augmentation/transforms.py`.
14. **Delete `_archive/` and backup files** (`*.bak`, `__pycache__`) once migration is tagged.
15. **Retire `IDS2026/S2-iceberg-areas/`** from active use. Move to `_archive/s2-iceberg-areas-prerework/`.

---

## 12. Deliverables

### 12.1 Refactor roadmap (order, owner-agnostic)

1. Write `baseline_v1.yaml` (1 day).
2. Rewrite `build_clean_dataset.py` to emit a manifest + propagate seed (2 days).
3. Patch `train.py` to write `training_config.json` (0.5 day).
4. Patch each of the 6 method scripts to take `--min_area_m2` from a shared config + emit `skipped_chips.csv` + write `method_config.json` (1.5 days).
5. Rewrite `run_all_methods.sh` to accept `--exp` and resolve every path from YAML (1 day).
6. Write `scripts/run_experiment.py` + `scripts/validate_experiment.py` (2 days).
7. Write `scripts/build_manifest.py` + content-addressed `data/raw_chips/` (1 day).
8. Stand up `iceberg.figures.registry` + migrate 2 existing figure scripts (1 day).
9. Re-run `baseline_v1` end-to-end. Record chips\_sha, manifest sha, eval csvs. Tag commit.
10. Port to `src/iceberg/`, delete dead scripts, retire old repo (3-5 days, incremental).

Total: ~2 weeks part-time for a single working dev, assuming no re-training bottleneck.

### 12.2 Config schema

See §3.1 (`baseline_v1`), §5.1 (manifest), §7.1 (experiment). All YAML validated via a single JSONSchema file at `configs/_schema/config.schema.json`.

### 12.3 Manifest schema

See §5.1.

### 12.4 Experiment inheritance system

See §7. Implementation note: inheritance is deep-merge of YAML dicts with an explicit `change:` block at the top level. `validate_experiment.py` enforces that `change:` touches exactly one family (one of: `data`, `methods`, `augmentation`, `training`, `inference`, `evaluation`).

### 12.5 File-by-file migration recommendations

| Current path | Action |
|---|---|
| `scripts/build_clean_dataset.py` | Rewrite in-place to emit manifest.json + split\_log.csv, default 65/15/25, propagate seed from config, resolve chip source from `data/raw_chips/`. |
| `scripts/balance_training.py` | Extract staged logic into `src/iceberg/balancing/strategies.py` as two composable strategies; keep script as thin CLI shim during migration. |
| `scripts/build_gt_positive_training.py` | Reexpress as `scheme_A_fisser_original` with scope=all-bins; delete script after migration. |
| `scripts/train.py` | Add `training_config.json` emission + enforce seed. Later move loss/IoU helpers into `src/iceberg/training/`. |
| `scripts/predict_tifs.py` | Keep working during migration. Later becomes `src/iceberg/methods/unet.py`. Probs output stays. |
| `scripts/threshold_tifs.py`, `threshold_probs.py`, `threshold_masked_tifs.py` | Consolidate into one `ThresholdMethod` class parameterised by `input_source` ∈ {`b08`, `probs`, `masked_b08`}. |
| `scripts/otsu_threshold_tifs.py`, `otsu_probs.py` | Consolidate into one `OtsuMethod` class parameterised by `input_source`. |
| `scripts/densecrf_tifs.py` | Becomes `CRFMethod`. |
| `scripts/eval_methods.py` | Split into `pixel_metrics.py` + `aggregate.py`. Current file becomes thin CLI. |
| `scripts/eval_per_iceberg.py` | Move to `src/iceberg/evaluation/per_iceberg.py`. |
| `scripts/compare_model_eval.py`, `make_figure21_*.py` | Use `iceberg.figures.registry` instead of direct `savefig`. |
| `scripts/run_all_methods.sh` | Replace with `scripts/run_experiment.py --exp <id> --stages infer`. Old shell kept during migration with a deprecation banner. |
| `scripts/run_model_comparison_eval.sh`, `run_matched_comparison_eval.sh`, `run_gt_positive_only_eval.sh` | Each becomes an experiment YAML. |
| `job.slurm`, `train_matched.slurm`, etc. | Replace with a single `slurm/train.slurm` that only takes `EXP_ID`. |
| `scripts/__pycache__`, `scripts/*.bak` | Delete after tagging migration commit. |
| `IDS2026/S2-iceberg-areas/*` | Move into `_archive/s2-iceberg-areas-prerework/`. Stop editing. |
| `IDS2026/paper-writing/methods_draft.md` | Update to reference the new config + manifest system once it is live. |

### 12.6 Risks if we do nothing

- Results for `exp_02` vs `exp_03` vs `exp_04` cannot be trusted to isolate a single variable, because balancing + split + chip membership + augmentation can all silently drift between runs.
- Fisser test chips are being skipped by `eval_methods.py` today because of the blank `tif_path` issue, the current IoU table under-represents lt65 performance.
- The six methods use three different `min_area_m2` defaults today; UNet-vs-Otsu deltas partly reflect this, not the method.
- `run_all_methods.sh` will happily run Otsu on test chips from smishra's dir and UNet on test chips from llinkas's dir if the env vars drift, cross-method comparisons are not guaranteed to be over the same images.
- Any repeat of an unseeded training run produces a different checkpoint, so re-running `baseline_v1` does not reproduce `baseline_v1`.
- Adding a new dataset variant currently requires writing a new `build_*.py` script each time. This encourages copy-paste → divergent balancing → invalid comparisons.
- No content addressing of raw chips means a single `rm` in smishra's tree silently invalidates every past and future experiment.

---

## Appendix A: Split policy change

Current `v3_clean` is 551 / 137 / 228 (60 / 15 / 25 of 916). New baseline is 65 / 15 / 25. Of 916:
- 65 % → 595 train (+44 vs today)
- 15 % → 137 val (unchanged)
- 25 % → 229 test (+1 vs today)

Practically: build a new `v4_clean_65_15_25` manifest. Do **not** edit `v3_clean` in place. All past results stamped with `v3_clean` remain reproducible; new results stamped with `v4_clean_65_15_25`.

## Appendix B: Figure policy (project-wide)

Every `iceberg.figures.registry.write(fig, slug, caption)` call:
1. Saves `fig-archive/<YYYYMMDD_HHMM>__<slug>.png` (append-only).
2. Updates `figures.md` so the entry for `slug` points at the latest filename with the caption.
3. If `slug` already exists in `figures.md`, replaces the pointer. The old archive file stays on disk.

Scripts that currently call `fig.savefig(...)` directly are migrated one at a time; until migrated, they are ignored by the registry and produce figures only in ad-hoc run dirs.
