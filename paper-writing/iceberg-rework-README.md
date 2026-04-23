# Sentinel-2 Iceberg Segmentation — Rework (llinkas)

**Working directory:** `/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/`
**Source data:** `/mnt/research/v.gomezgilyaspik/students/smishra/rework/` (read-only)
**Last updated:** 2026-04-16

---

## Project Overview

Rework of smishra's Sentinel-2 iceberg segmentation pipeline. Compares UNet++ against classical threshold methods across four solar zenith angle (SZA) bins in Kangerlussuaq (KQ) and Sermilik (SK) fjords, East Greenland.

**Key methodological changes from smishra's v2 pipeline:**
- Shadow (class 2) merged into iceberg (class 1). Model is binary (single class: iceberg).
- 40 m root-length cutoff applied per individual iceberg (connected component >= 16 pixels).
- IC filtering uses annotation-aware chip-level method with fixed B08 >= 0.22 threshold (Fisser's 0.12 + DN offset). Sea ice masked in training chips with IC >= 15%. Validation/test never masked.
- Dataset resplit to 60/15/25 train/val/test with SZA-stratified test set.
- Training set balanced to 2:1 ratio per iceberg size bin.
- Fisser chips filtered for IC >= 15% (68 removed by provenance audit, further masking via annotation-aware IC).

**Sensor:** Sentinel-2 L1C, bands B04/B03/B08, 10 m resolution
**Chip size:** 256 x 256 pixels = 2.56 x 2.56 km
**SZA bins:** sza_lt65 (Jul-Sep), sza_65_70 (Sep-Oct), sza_70_75 (Oct), sza_gt75 (Nov)

---

## Folder Layout

```
iceberg-rework/
├── README.md
├── requirements.txt
├── job.slurm
├── new-plan.txt                         ← original rework plan from advisor
│
├── scripts/                             ← all pipeline scripts (editable)
│   ├── audit_fisser_provenance.py       ← map Fisser chips to tifs, parse dates, compute IC
│   ├── filter_small_icebergs.py         ← remove icebergs < 40m RL, merge shadow into iceberg
│   ├── filter_quality.py                ← annotation-aware IC filtering (Fisser 10km method adapted)
│   ├── fetch_met_data.py                ← ERA5 wind speed + temperature for all chips
│   ├── visualize_missed_icebergs.py     ← find unannotated bright objects 40-500m RL
│   ├── descriptive_stats.py             ← per-iceberg histograms, tables, Fisser comparison
│   ├── build_clean_dataset.py           ← merge filtered data, 60/15/25 split, binary masks
│   ├── balance_training.py              ← 2:1 oversampling/undersampling per area bin
│   ├── train.py                         ← UNet++ training (from smishra, unchanged)
│   ├── predict_tifs.py                  ← inference on .tif chips (from smishra, unchanged)
│   ├── eval_methods.py                  ← pixel-wise IoU/precision/recall (from smishra)
│   ├── compare_areas.py                 ← area statistics + plots (from smishra)
│   ├── run_all_methods.sh               ← run all 6 methods on one SZA bin
│   ├── run_pipeline.sh                  ← chip -> UNet++ -> threshold pipeline
│   └── ... (other scripts from smishra)
│
├── data/
│   ├── annotations_filtered.coco.json   ← COCO with icebergs < 40m RL removed
│   ├── fisser_filtered/                 ← Fisser masks: shadow merged, < 40m RL removed
│   ├── v3_clean/                        ← 60/15/25 split (916 chips, IC-filtered, binary masks)
│   │   ├── train_validate_test/         ← X_train.pkl, Y_train.pkl, etc.
│   │   └── split_log.csv               ← extended metadata per chip
│   └── v3_balanced/                     ← balanced training set (340 train chips)
│       ├── train_validate_test/         ← balanced X_train.pkl, unchanged val/test
│       └── balance_report.csv
│
├── reference/
│   ├── fisser_provenance_audit.csv      ← Fisser chip tif accessibility + IC + dates
│   ├── fisser_quality_filter.csv        ← Fisser IC filter results (old Otsu method, superseded)
│   ├── ic_filter_10km.csv               ← 10km block IC results for Roboflow chips
│   ├── met_data.csv                     ← ERA5 wind + temperature per chip
│   ├── filter_40m_summary.csv           ← before/after counts for 40m RL filter
│   ├── missed_icebergs_summary.csv      ← unannotated bright object candidates
│   ├── descriptive_stats.csv            ← per-bin iceberg/ocean/contrast statistics
│   └── b08_analysis_results_discussion.md ← results, discussion, methods for IC/B08 analysis
│
├── viz/
│   ├── filter_40m/                      ← before/after visualizations for 40m cutoff
│   │   ├── coco/                        ← Roboflow chips
│   │   └── fisser/                      ← Fisser chips
│   ├── missed_icebergs/                 ← side-by-side original + annotated/missed
│   └── descriptive_stats/               ← all histograms and table figures
│       ├── hist_root_length.png         ← per-bin subplots
│       ├── hist_month.png
│       ├── hist_wind.png
│       ├── hist_temp.png
│       ├── hist_area.png                ← per-bin + log-log power law
│       ├── hist_iceberg_vs_neighborhood_b08.png  ← cf. Fisser 2024 Fig. 9
│       ├── table_iceberg_b08_by_sza.png
│       ├── table_iceberg_b08_pixels_by_sza.png
│       ├── table_sza_characterization.png
│       ├── table_annotation_aware_ic.png
│       ├── table_relative_abundance.png
│       ├── table_fisser_comparison.png
│       └── table_b08_per_sza.png
│
└── model/                               ← (empty until retraining)
```

---

## Data Sources (read-only, in smishra/rework/)

| Item | Path (under smishra/rework/) | Description |
|------|------------------------------|-------------|
| Fisser chips | `data/fisser_original/train_validate_test/` | 398 pkl chips (sza_lt65) |
| Roboflow COCO | `data/roboflow_export/train/train/` | 586 images + 18,322 annotations |
| Combined v2 pkls | `data/train_validate_test/` | smishra's 984-chip dataset (not used directly) |
| Split log | `data/split_log.csv` | smishra's original split mapping |
| Chip tifs | `chips/{KQ,SK}/{sza_bin}/tifs/` | 23,981 GeoTIFF chips (symlinks) |
| Test chips | `test_chips/{sza_bin}/` | smishra's 96 held-out chips |
| Fisser tif index | `reference/fisser_index.csv` | Fisser pkl -> tif path mapping |
| Scene catalogue | `reference/scene_catalogue.csv` | 319 S2 scenes with metadata |
| Model checkpoint | `model/s2_v2_aug/best_model.pth` | smishra's best model (test IoU 0.3617) |

---

## v3 Dataset (this rework)

### Processing Pipeline

```
Fisser pkls (398)  ──┐
                     ├─ merge shadow (class 2 -> 1)
Roboflow COCO (586) ─┤  remove icebergs < 40m RL
                     ├─ IC quality filter (68 Fisser chips removed)
                     ├─ 60/15/25 stratified split -> 916 chips
                     ├─ annotation-aware IC masking (training only, IC >= 15%)
                     └─ 2:1 area-bin balancing (training only) -> 364 train chips
```

### v3_clean Split (before balancing)

| Split | Total | sza_lt65 | sza_65_70 | sza_70_75 | sza_gt75 | Null |
|-------|-------|----------|-----------|-----------|----------|------|
| Train | 551 | 214 | 61 | 95 | 181 | 202 |
| Val | 137 | 59 | 20 | 14 | 44 | 50 |
| Test | 228 | 57 | 57 | 57 | 57 | 109 |

### v3_balanced Training Set

| Bin | Original | Balanced | Action |
|-----|----------|----------|--------|
| null | 199 | 110 | undersampled |
| rl_40_100 | 55 | 55 | unchanged |
| rl_100_300 | 208 | 110 | undersampled |
| rl_300_plus | 89 | 89 | unchanged |
| **Total** | **551** | **364** | |

Class distribution (training): ocean 93.0%, iceberg 7.0%

---

## Key Methodological Decisions

### Shadow Merge
Fisser's 3-class masks (ocean/iceberg/shadow) are reduced to binary (iceberg only) by remapping shadow (class 2) into iceberg (class 1) before connected component analysis. This aligns with Roboflow annotations which do not distinguish shadow. The model performs binary segmentation (single class: iceberg). Merging bridges fragmented iceberg components, nearly doubling the icebergs surviving the 40m filter (16,343 to 32,536 in Fisser Y_train).

### 40m Root-Length Cutoff
Individual icebergs (connected components) smaller than 16 pixels (1,600 m2, 40 m root length) are removed to match the Fisser (2025) dataset cutoff. Applied to both COCO annotations (by area field) and Fisser masks (by connected component size after shadow merge).

### IC Filtering
Annotation-aware chip-level IC using the Fisser B08 >= 0.22 threshold (= 0.12 corrected). Annotated iceberg pixels are excluded from IC calculation. Training chips with IC >= 15% have bright non-annotated pixels masked to zero. Validation and test chips are never masked. Full justification in `reference/b08_analysis_results_discussion.md`.

### DN Offset
All reflectances are +0.10 high due to processing baseline offset. Fisser's 0.12 = our 0.22. chip_sentinel2.py does DN x 1e-4 without subtracting 1000. UNet++ trained on offset-shifted chips, internally consistent.

---

## Remaining Steps

- [x] Implement annotation-aware IC masking in build_clean_dataset.py (2026-04-16)
- [x] Rebuild v3_clean and v3_balanced with IC masking applied to training chips (2026-04-16: 193 train chips masked, val/test untouched)
- [x] Implement new evaluation metrics (MAE, RERL, contrast) in eval_per_iceberg.py (script written; runs after Step below)
- [ ] Retrain UNet++ (binary) on v3_balanced: `sbatch job.slurm` from moosehead (or `python scripts/train.py --mode s2 --data_dir data/v3_balanced --out_dir model/v3_balanced_aug`)
- [ ] Run all 6 methods on new test set (run_all_methods.sh paths must be updated for v3 test chips)
- [ ] Run eval_per_iceberg.py and compare results with Fisser 2025

---

## Notes

- All scripts point to llinkas paths for script execution and smishra paths for source data.
- The `filter_quality.py` script was rewritten from Otsu-based to Fisser's 10km block method, then further refined to annotation-aware chip-level. The ic_filter_10km.csv and fisser_quality_filter.csv are intermediate outputs from earlier iterations.
- train.py uses num_classes=1 (binary segmentation) for all modes. Inference scripts expand the single-channel sigmoid output to 2-band probs [P(ocean), P(iceberg)] for downstream compatibility with threshold and CRF scripts.
- Validation and test sets in v3_balanced are identical to v3_clean (only training is balanced).
