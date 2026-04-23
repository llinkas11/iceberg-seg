# Iceberg Segmentation Pipeline Rework Plan

**Status:** Steps 0-7 + 6b complete. All scripts converted to binary. eval_per_iceberg.py written. Ready for GPU training.
**Last verified:** 2026-04-16. build_clean_dataset.py and balance_training.py rerun successfully after /simplify code review fixes.

## Context

Reworking smishra's Sentinel-2 iceberg segmentation pipeline to match Fisser 2025 standards and produce a defensible paper. All editable work in `llinkas/iceberg-rework/`, source data in `smishra/rework/` (read-only).

---

## Resolved Prerequisites

| ID | Issue | Resolution |
|----|-------|------------|
| PR-1 | CARRA vs ERA5 | ERA5 via Open-Meteo used. Wind max 8.4 m/s (all pass), 324 chips temp <= 0C (mostly sza_gt75, documented as confound not filtered) |
| PR-2 | Fisser SAFE files | Not available in downloads. Fisser chips accepted as pre-filtered by Fisser et al. IC computed from tif B08 directly using annotation-aware method |
| PR-3 | Root length definition | Per-individual-iceberg (connected component >= 16 px), not per-chip aggregate |
| PR-4 | Validation set | 60/15/25 train/val/test adopted. Validation used for checkpoint selection, never masked |
| PR-5 | Re-annotation | 1,756 missed candidates found across 129 chips. Decision pending review of viz/missed_icebergs/ |
| PR-6 | CatBoost/dynamic threshold | Deferred. Dynamic IC threshold rejected because 15-25% of iceberg pixels fall below 0.22 at every SZA bin (see b08_analysis_results_discussion.md Section 3.6) |
| PR-7 | Shadow class | **Merged into iceberg (class 2 -> 1).** Applied throughout all scripts and data. Model is binary segmentation (single class: iceberg). Shadow merge bridges fragmented icebergs, nearly doubling survivors after 40m filter |

---

## Key Methodological Decisions (with justification references)

### 1. Shadow Merge
Fisser class 2 (shadow) merged into class 1 (iceberg) before all analysis. Aligns Fisser 3-class with Roboflow annotations. Model is binary (single class: iceberg). Documented in `reference/descriptive_stats_results_discussion.md` Section 2.

### 2. 40m Root-Length Cutoff
Individual icebergs (connected components) < 16 pixels (1,600 m2, 40m RL) removed. Matches Fisser 2025 dataset minimum. Most removed components are rasterization artifacts (median 3 pixels), not real icebergs. Documented in `reference/descriptive_stats_results_discussion.md` Section 2.

### 3. IC Filtering (Annotation-Aware)
Adapts Fisser et al. (2024) IC method to chip level. IC = fraction of non-annotated pixels with B08 >= 0.22 (Fisser's 0.12 + DN offset). Training chips with IC >= 15% have bright non-annotated pixels masked to zero. Validation and test never masked. 193 training chips masked. Full justification in `reference/b08_analysis_results_discussion.md` Sections 3.1-3.6.

### 4. DN Offset
All reflectances +0.10 high (processing baseline >= 4.0 adds 1000 DN, chip_sentinel2.py does DN*1e-4 without subtracting). Fisser's 0.12 = our 0.22. Documented in smishra's `project_radiometric_offset.md`.

### 5. Temperature Confound
98.9% of sza_gt75 chips have temp <= 0C. Filtering would destroy the bin. Documented as confound, not filtered. Wind is not a confound (max 8.4 m/s, all below 15 m/s threshold). See `reference/descriptive_stats_results_discussion.md` Section 4.

---

## Completed Steps

### Step 0: Fisser Provenance Audit ✓
- Script: `audit_fisser_provenance.py`
- Output: `reference/fisser_provenance_audit.csv`
- Result: 398/398 tifs accessible, all dates parseable. 80 unique scenes, 0 matching SAFE zips in downloads.

### Step 1: 40m Root-Length Filter + Shadow Merge ✓
- Script: `filter_small_icebergs.py` (merges shadow before CC analysis, uses np.bincount)
- Output: `data/annotations_filtered.coco.json`, `data/fisser_filtered/*.pkl`
- Result: COCO 18,312 -> 7,947 annotations. Fisser 96,648 -> 39,534 components (shadow merge bridges fragments). Viz in `viz/filter_40m/`

### Step 2: IC Quality Filtering ✓
- Script: `filter_quality.py` (annotation-aware chip-level, B08 >= 0.22)
- Output: `reference/ic_filter_10km.csv`, analysis in `reference/b08_analysis_results_discussion.md`
- Result: 356 of 984 chips fail IC >= 15%. Masking applied only to training chips in build_clean_dataset.py.

### Step 3: Meteorological Data ✓
- Script: `fetch_met_data.py`
- Output: `reference/met_data.csv`
- Result: 0 chips > 15 m/s wind. 324 chips temp <= 0C. No filtering applied.

### Step 4: Missed Icebergs ✓
- Script: `visualize_missed_icebergs.py`
- Output: `viz/missed_icebergs/` (30 side-by-side images), `reference/missed_icebergs_summary.csv`
- Result: 1,756 missed candidates across 129 annotated chips. Median RL 60.8m.

### Step 5: Descriptive Statistics ✓
- Script: `descriptive_stats.py`
- Output: `viz/descriptive_stats/` (histograms + table PNGs), `reference/descriptive_stats.csv`
- Analysis docs: `reference/descriptive_stats_results_discussion.md`, `reference/b08_analysis_results_discussion.md`

### Step 6: Clean Dataset Build ✓
- Script: `build_clean_dataset.py` (includes annotation-aware IC masking for training chips)
- Output: `data/v3_clean/` (551 train, 137 val, 228 test, binary masks)
- Result: 916 chips. 193 training chips IC-masked (5.9M pixels zeroed, 1.4M iceberg pixels preserved). Test stratified 57 per SZA bin.

### Step 7: Training Set Balancing ✓
- Script: `balance_training.py`
- Output: `data/v3_balanced/` (364 train, 137 val, 228 test)
- Result: null and rl_100_300 undersampled. 2:1 max ratio. Iceberg pixels 7.0%.

### Code Conversion to 2-Class ✓
Scripts updated to binary segmentation (num_classes=1):
- `train.py` (line 192: `else 2`)
- `predict_tifs.py` (line 56: `else 2`, shadow polygonization removed)
- `predict.py` (line 100: `else 2`, shadow color removed)
- `export_onnx.py` (line 22: `else 2`)
- `visualize_predictions.py` (line 66: `classes=2`)
- `export_roboflow.py` (shadow removed from CLASS_NAMES)
- `export_manual_annotations_roboflow.py` (same)
- `eval_methods.py` (comment updated)
- `threshold_probs.py`, `otsu_probs.py`, `densecrf_tifs.py` (docstrings updated to 2-band probs)
- `job.slurm` (DATA_DIR -> data/v3_balanced)

### eval_per_iceberg.py Written ✓
- Per-iceberg MAE, RERL, contrast via greedy connected-component IoU matching
- Designed for binary from the start
- Cannot run until model is retrained and predictions generated

---

## Remaining Steps

### Step 8: Retrain UNet++ (binary)
**Status: READY TO RUN. Requires GPU.**

```bash
# From moosehead SSH session:
cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework
sbatch job.slurm
```

Or interactively:
```bash
python scripts/train.py --mode s2 \
    --data_dir data/v3_balanced \
    --out_dir model/v3_balanced_aug \
    --encoder resnet34 --epochs 100 --batch_size 16 --lr 1e-4
```

- Augmentation: hflip + vflip + rot (on by default, --no_augment to disable)
- Output: `model/v3_balanced_aug/best_model.pth`, `training_log.csv`
- Expected: training on 364 chips, binary (BCE + DiceLoss, sigmoid), ~100 epochs

### Step 9: Run Inference (All 6 Methods)
**Status: BLOCKED on Step 8.**

1. Build new test_chips dir from v3 test set:
   ```bash
   python scripts/prepare_test_chips_dir.py  # may need path updates for v3 split_log
   ```

2. Run all methods per SZA bin:
   ```bash
   bash scripts/run_all_methods.sh sza_lt65 model/v3_balanced_aug/best_model.pth
   bash scripts/run_all_methods.sh sza_65_70 model/v3_balanced_aug/best_model.pth
   bash scripts/run_all_methods.sh sza_70_75 model/v3_balanced_aug/best_model.pth
   bash scripts/run_all_methods.sh sza_gt75 model/v3_balanced_aug/best_model.pth
   ```

3. Evaluate:
   ```bash
   python scripts/eval_methods.py \
       --test_pkl data/v3_clean/train_validate_test/y_test.pkl \
       --test_index data/v3_clean/split_log.csv \
       --results_dir results/v3/area_comparison/test \
       --out_dir results/v3/test_outputs
   ```

**Note:** `run_all_methods.sh` currently points CHIPS and OUT_BASE to smishra's directories. These paths must be updated for v3 test chips before running.

### Step 10: Per-Iceberg Evaluation
**Status: SCRIPT WRITTEN. BLOCKED on Step 9.**

```bash
python scripts/eval_per_iceberg.py \
    --pred_dir results/v3/area_comparison/test/{sza_bin}/UNet \
    --test_pkl data/v3_clean/train_validate_test/y_test.pkl \
    --test_x_pkl data/v3_clean/train_validate_test/x_test.pkl \
    --test_index data/v3_clean/split_log.csv \
    --out_dir results/v3
```

Metrics: MAE (area), RERL (root length), contrast (B08 iceberg - ocean), per-iceberg IoU.

---

## Open Questions (for advisor/team)

1. **Missed icebergs (PR-5):** 1,756 missed candidates found. Review `viz/missed_icebergs/` to decide if re-annotation is needed.
2. **92 oversized annotations:** Icebergs > 400,000 m2 (exceeding Fisser max). Likely multi-iceberg clumps from Otsu pre-annotation. Review and potentially split in Roboflow.
3. **CatBoost / dynamic thresholding:** Listed in new-plan.txt as TBD. Deferred for now.

---

## Critical File Paths

| File | Purpose |
|------|---------|
| `data/v3_balanced/train_validate_test/` | **Training data** (364 train, 137 val, 228 test, binary masks, IC-masked training) |
| `data/v3_clean/split_log.csv` | Definitive chip-to-split mapping with ic_aware, ic_masked, met data |
| `reference/b08_analysis_results_discussion.md` | IC/B08 analysis: results, discussion, exhaustive methods |
| `reference/descriptive_stats_results_discussion.md` | Dataset characterization: size, temporal, meteorological |
| `reference/met_data.csv` | ERA5 wind + temperature per chip |
| `reference/fisser_provenance_audit.csv` | Fisser chip tif paths, dates, regions |
| `viz/descriptive_stats/` | All histogram and table figures |
| `viz/missed_icebergs/` | Side-by-side missed iceberg visualizations |
| `viz/filter_40m/` | Before/after 40m cutoff visualizations |
| `job.slurm` | SLURM job template (DATA_DIR = data/v3_balanced) |

---

## Verified Pipeline State (2026-04-16)

- `build_clean_dataset.py` runs cleanly: 916 chips, 193 training IC-masked, val/test untouched
- `balance_training.py` runs cleanly: 364 balanced training chips
- Data verified: unique classes {0, 1} only, no class 2
- UNet++ binary model initializes: 26,078,609 parameters (1 output channel)
- Split log has all columns: split, pkl_position, stem, chip_stem, tif_path, sza_bin, source, n_icebergs, ic_aware, ic_masked, wind_ms, temp_c
- IC masking confirmed: 193 training chips masked, 0 val, 0 test
