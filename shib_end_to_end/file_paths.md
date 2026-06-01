# File Paths: Moosehead + Local
**Last updated:** 2026-05-05 (added 2026-05-05 follow-up runs)

---

## Moosehead Working Root

```
/mnt/research/v.gomezgilyaspik/students/smishra/llinkas-rework/
```

All paths below are relative to this root unless noted.

---

## Checkpoints

```
# Phase A winner (A0) - use this for all Phase B inference
runs/exp_A0_fisser_lt65_original/20260428_094028/model/best_model.pth
  best_val_iou: 0.6127717643976212
  manifest_id:  v4_raw_lt65
  trained on:   398 lt65 chips, Fisser preprocessing, no 40m filter, no IC mask
```

---

## Manifests

```
data/v4_raw_lt65/manifest.json                 398 chips, lt65, no filter, no IC
data/v4_clean_lt65/manifest.json               330 chips, lt65, 40m filter + IC
data/v4_raw_lt65/test_chips/sza_lt65/          100 test chip .tif files
data/v4_raw_lt65/train_validate_test/          pkl files (y_train, y_val, y_test)
data/v4_raw_lt65/split_log.csv
```

---

## Phase B Experiment Runs (2026-05-03)

All 6 experiments use A0 checkpoint + v4_raw_lt65 manifest.

```
runs/exp_B0_method_threshold/20260503_012136/
runs/exp_B1_method_otsu/20260503_012343/
runs/exp_B2_method_unet/20260503_012543/
runs/exp_B3_method_unet_threshold/20260503_012741/
runs/exp_B4_method_unet_otsu/20260503_012937/
runs/exp_B5_method_unet_crf/20260503_013135/
```

Each run directory contains:
```
inference/          method prediction gpkgs by SZA bin
evaluation/         chip-level IoU CSVs
per_iceberg/        per-pair CSVs (added 2026-05-03)
  eval_per_iceberg.csv           per-pair rows (method, chip, gt_area, pred_area, re_pct, ...)
  eval_per_iceberg_summary.csv   aggregated by method+sza_bin
  eval_per_iceberg_chips.csv     aggregated by chip
  eval_per_iceberg_detection.csv match rate, precision, n_gt, n_pred
manifest_stamp.json
run_stamp.json
```

---

## 2026-05-05 follow-up: Phase A higher-SZA re-eval + backbone-comparison Phase B

Three Slurm jobs evaluate the Phase A backbones and re-run Phase B against the v4_clean test split (228 chips, 57 per SZA bin). Full T1-T4 tables in [phase_a_higher_sza_t1_t4.md](phase_a_higher_sza_t1_t4.md).

**Phase A re-eval** (Slurm 60293, UNet only, A0..A9 x 4 SZA bins):
```
runs/exp_A{0..9}_*/<canonical_ts>/re_eval_v4_clean/
  test/<sza_bin>/UNet/{geotiffs/, gpkgs/, all_icebergs.gpkg, method_config.json, skipped_chips.csv}
  per_iceberg/eval_per_iceberg_summary.csv
```

**A1 Phase B re-run** (Slurm 60296, six methods x 4 SZA bins):
```
runs/exp_A1_fisser_lt65_plus_nulls/20260429_234146/re_phase_b_v4_clean/
  test/<sza_bin>/{TR, OT, UNet, UNet_TR, UNet_OT, UNet_CRF}/
  evaluation/eval_summary.csv
  per_iceberg/eval_per_iceberg_summary.csv
```

**A0 Phase B re-run** (Slurm 60297, same layout):
```
runs/exp_A0_fisser_lt65_original/20260428_094028/re_phase_b_v4_clean/
  ... (same as A1)
```

Driver scripts: `iceberg-rework/scripts/re_eval_phase_a_all_sza.sh`, `re_phase_b_with_a0.sh`, `re_phase_b_with_a1.sh`. Slurm wrappers: `iceberg-rework/slurm/re_eval_phase_a.slurm`, `re_phase_b_a0.slurm`, `re_phase_b_a1.slurm`. All three Phase A backbones evaluated with `FORCE=1` to override `run_methods.sh`'s dataset-drift guard (intentional cross-manifest inference against the unifying v4_clean test split). Top-hat variants (`--with_tophat`) NOT included; would add 6 more `<METHOD>_TH` outputs per bin.

Local mirrors of the per_iceberg summary CSVs are in `iceberg-rework/runs_summaries/exp_A*/<ts>/{re_eval_v4_clean,re_phase_b_v4_clean}/per_iceberg/eval_per_iceberg_summary.csv`.

---

## Fisser Validation Run (2026-05-03)

TR without IC filter on 100 lt65 test chips.

```
runs/fisser_validation/inference/sza_lt65/TR/all_icebergs.gpkg
runs/fisser_validation/per_iceberg/eval_per_iceberg.csv
runs/fisser_validation/per_iceberg/eval_per_iceberg_summary.csv
```

---

## Phase B RE% Aggregated Summary

```
runs/phase_b_re_pct_summary.csv
  columns: method, n_pairs, mean_re_pct, median_re_pct
  population: sza_lt65 matched pairs only (other bins empty)
```

---

## Configs

```
configs/experiments/exp_B0_method_threshold.yaml
configs/experiments/exp_B1_method_otsu.yaml
configs/experiments/exp_B2_method_unet.yaml
configs/experiments/exp_B3_method_unet_threshold.yaml
configs/experiments/exp_B4_method_unet_otsu.yaml
configs/experiments/exp_B5_method_unet_crf.yaml
configs/experiments/exp_A0_fisser_lt65_original.yaml
configs/baseline_v1.yaml
```

---

## SLURM Scripts

```
slurm/exp.slurm                   main experiment runner (patched: supports --checkpoint)
slurm/eval_per_iceberg_b.slurm    per-iceberg eval batch job for B0-B5
slurm/fisser_validation.slurm     TR no-IC-filter validation run
```

---

## Key Scripts

```
scripts/run_experiment.py          pipeline driver (patched: --checkpoint arg)
scripts/run_methods.sh             6-method inference runner
scripts/eval_methods.py            chip-level IoU evaluation
scripts/eval_per_iceberg.py        per-pair Hungarian matching + MAE/IoU/RE%
scripts/threshold_tifs.py          TR method (0.22 threshold, IC filter at ic_frac > 0.15)
scripts/build_figure_review_deck.py deck builder (ROOT patched: parents[1])
```

---

## Figure Archive (moosehead)

```
paper-writing/figures/fig-archive/    publication PNGs (latest per slug)
paper-writing/figure_review/
  figure_review_deck.pptx            latest deck (19 slides, built 2026-05-03)
  figure_review_checklist.csv
```

**To rebuild deck:**
```bash
cd /mnt/research/v.gomezgilyaspik/students/smishra/llinkas-rework
/mnt/research/v.gomezgilyaspik/students/smishra/pptx_env/bin/python \
  scripts/build_figure_review_deck.py \
  --out-dir /mnt/research/v.gomezgilyaspik/students/smishra/llinkas-rework/paper-writing/figure_review
```

---

## Python Environments

| Env | Path | Use for |
|-----|------|---------|
| iceberg-unet312 | `/home/llinkas/.venvs/iceberg-unet312/bin/python` | All ML/eval/figure scripts |
| pptx_env | `/mnt/research/v.gomezgilyaspik/students/smishra/pptx_env/bin/python` | `build_figure_review_deck.py` only |

---

## Local (Mac)

```
/Users/smishra/iceberg-seg/shib_end_to_end/   this directory
  README.md                                   overview + headline numbers + TODO
  results.md                                  LaTeX-ready results tables (all phases)
  shib_audit_plan.md                          original Codex handoff plan
  file_paths.md                               this file
  figures/
    fig-archive/                              15 files, latest per slug (publication figures)
    figure_review/
      figure_review_deck_2026-05-03.pptx     19 slides (slides 1-10 = figures, 11-19 = tables)
      figure_review_checklist.csv
    table_previews/                           9 PNGs rendered as table slides in the deck
      table_phase_a_leaderboard.png           Phase A leaderboard (our numbers, confirmed)
      table_phase_a_preprocessing_nulls.png   2x2 preprocessing x nulls (confirmed)
      table_probability_calibration.png       P(iceberg) calibration audit (confirmed)
      table_base_mae_rootlen.png              RL-MAE by method/SZA (baseline_v1 - needs regen from Phase B)
      table_base_iou.png                      IoU by method/SZA (baseline_v1 - needs regen)
      table_base_detection.png               detection stats (baseline_v1 - needs regen)
      table_tophat_mae_rootlen.png            top-hat post-processing results (llinkas only)
      table_tophat_iou.png                    top-hat IoU (llinkas only)
      table_tophat_detection.png             top-hat detection (llinkas only)
    cleanup_viz/                              10 annotation audit PNGs
```

**rsync to pull updated deck:**
```bash
rsync -av smishra@moosehead.bowdoin.edu:\
/mnt/research/v.gomezgilyaspik/students/smishra/llinkas-rework/paper-writing/figure_review/figure_review_deck.pptx \
/Users/smishra/iceberg-seg/shib_end_to_end/figures/figure_review/figure_review_deck_$(date +%Y-%m-%d).pptx
```
