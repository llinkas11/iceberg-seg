# iceberg-rework: HPC quick-start

**Working tree:** `/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/`
**Source code:** `https://github.com/llinkas11/iceberg-seg`
**Last updated:** 2026-04-24

For the project-level README with folder layout and methodology, see
`paper-writing/iceberg-rework-README.md` (sibling tree under
`/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/paper-writing/`,
or in the same git repo at `paper-writing/iceberg-rework-README.md`).

For project state, see `paper-writing/plan.md`. For methodology, see
`paper-writing/methods_draft.md`. For experimental progression, see
`paper-writing/model_progression.md`.

---

## Setup

The HPC venv is already in place at `~/.venvs/iceberg-unet312/` with all
dependencies installed (PyTorch + CUDA 12.1, segmentation-models-pytorch,
rasterio, geopandas, shapely, pandas, matplotlib, scikit-image, scipy,
PyYAML, pydensecrf2). If a fresh setup is needed, install:

```
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Run the canonical baseline

From a moosehead shell:

```
cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework
sbatch slurm/baseline_v1.slurm
```

The slurm wrapper sets `ICEBERG_EXPERIMENT=1` and calls
`scripts/run_experiment.py --exp exp_baseline_v1 --stages manifest,train,infer,evaluate`.
Outputs land at `runs/exp_baseline_v1/<timestamp>/`.

If the run fails between stages, the wrapper prints the timestamp and the
exact resume command. To resume from after a successful train:

```
RUN_TS=<timestamp_dir> sbatch slurm/baseline_v1_resume.slurm
```

The resume script wipes any partial inference output, runs all six methods
on the existing trained checkpoint, then runs the chip-level and per-pair
evaluators in parallel.

---

## Run any other experiment

```
python scripts/validate_experiment.py --exp <experiment_id>     # confirm
python scripts/run_experiment.py    --exp <experiment_id>       # execute
```

Available experiments under `configs/experiments/`:

- `exp_baseline_v1`: no-op anchor, used for the headline baseline run.
- `exp_A0_fisser_lt65_original` through `exp_A6_our_lt65_plus_nulls_aug_adaptive`: Phase A dataset progression (lt65-scoped).
- `exp_B0_method_threshold` through `exp_B5_method_unet_crf`: Phase B method sweep on the Phase A winner.
- `exp_ablation_no_aug`: baseline with augmentation disabled.

See `paper-writing/model_progression.md` for the full progression and
motivations row by row.

---

## Single-stage runs

For each stage independently:

**Build / verify the dataset manifest:**
```
python scripts/build_clean_dataset.py    # writes data/v4_clean/manifest.json + pkls
```

**Train UNet++ (binary, seed-required under ICEBERG_EXPERIMENT=1):**
```
ICEBERG_EXPERIMENT=1 python scripts/train.py \
    --mode s2 \
    --data_dir data/v4_clean \
    --out_dir model/<run_id> \
    --encoder resnet34 \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --seed 42
```

**Run all six methods on a manifest + checkpoint:**
```
bash scripts/run_methods.sh \
    --manifest   data/v4_clean/manifest.json \
    --checkpoint model/<run_id>/best_model.pth \
    --out_base   results/<run_id>/inference \
    [--bin sza_lt65]                    # optional; default: all bins
```

**Chip-level evaluation:**
```
python scripts/eval_methods.py \
    --manifest data/v4_clean/manifest.json \
    --test_dir results/<run_id>/inference \
    --out_dir  results/<run_id>/evaluation
```

**Per-pair evaluation (Hungarian + per-pair MAE + IoU):**
```
python scripts/eval_per_iceberg.py \
    --manifest      data/v4_clean/manifest.json \
    --test_dir      results/<run_id>/inference \
    --out_dir       results/<run_id>/per_iceberg \
    --matcher       hungarian \
    --iou_threshold 0.3
```

---

## Output schema

Every method run writes three provenance files into its output dir:

- `all_icebergs.gpkg`: merged polygons across all chips for that method + bin.
- `method_config.json`: every parameter the method used + git SHA + timestamp + chips_sha + run_kind.
- `skipped_chips.csv`: one row per chip the method refused to score, with a reason from a small enumerated set (`too_few_bands`, `ic_block_filter`, `otsu_floor`, `flat_prob`, `chip_tif_not_found`, `too_few_prob_bands`).

Every training run writes `training_config.json` next to `best_model.pth` with all hyperparameters, the seed, the manifest_id, the git SHA, the experiment_mode flag, and the final metrics.

Every experiment run writes `run_stamp.json` at the root of its run dir with the experiment id, baseline id, config_sha, and the list of stages executed.

---

## Where the figures live

`scripts/_fig_registry.write(fig, slug, caption, out_dir)` saves to
`<out_dir>/fig-archive/<YYYYMMDD_HHMMSS>__<slug>.png` (append-only) and
updates `<out_dir>/figures.md` so the slug points at the latest archive
entry. Old entries stay on disk.

Every paper-bound figure script in the repo now routes through the
registry: `compare_areas.py`, `compare_model_eval.py`, `eval_methods.py`,
`descriptive_stats.py`, and `make_figure21_iou_gt_positive_comparison.py`.
Tables published as PNGs use `_fig_registry.write_table(headers, rows,
title, slug, caption, out_dir)`.

Per-chip dumpers (`predict.py`, `visualize_*`, `build_lt65_nulls.py`,
`add_small_icebergs.py`, `filter_small_icebergs.py`,
`otsu_threshold_tifs.py`'s diagnostic PNGs) intentionally still call
`fig.savefig` directly because they emit hundreds of files per run; routing
those through `figures.md` would bloat the index.

---

## Troubleshooting

**Job fails with `ModuleNotFoundError`**: a Python dep is missing in the venv.
Install with `~/.venvs/iceberg-unet312/bin/pip install <name>` and add to
`requirements.txt`. As of 2026-04-24, the venv has every dep this code
imports.

**Job fails with `checkpoint not found at ...`**: the slurm wrapper's
`RUN_TS` does not match an actual `runs/exp_baseline_v1/<RUN_TS>/model/`
dir. Override with `RUN_TS=<correct_timestamp> sbatch slurm/...`.

**`validate_experiment.py` complains about multi-family change**: your
experiment YAML touches more than one of `data | methods | augmentation |
training | inference | evaluation`. Either split into two experiments, or
add `controlled_variable: <family>` to the YAML if the multi-family change
is intentional.

**`run_methods.sh` refuses with "checkpoint was trained on manifest X but
--manifest is Y"**: the dataset-drift guard fires when the trained model
is not on the same dataset you are evaluating on. Either retrain on the
right manifest, or set `FORCE=1` to override (only do this if you really
know what you are doing).

---

## Logs

Every slurm job writes `logs/baseline/<job_name>_<job_id>.{out,err}`.
Tail with:

```
tail -f logs/baseline/ice_baseline_v1_<id>.out
```

`squeue -u llinkas` shows queued and running jobs.

`sacct -j <id>` shows historical job state, exit codes, and elapsed time.
