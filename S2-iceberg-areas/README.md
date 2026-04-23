# Iceberg area retrieval from Sentinel-2 data at varying solar angles

## About the repository
This repository contains the Python code underlying the peer-reviewed paper titled "Impact of varying solar angles on Arctic iceberg area retrieval from Sentinel-2 near-infrared data". In the paper we calibrated a near-infrared reflectance threshold for the iceberg area retrieval from Sentinel-2 data. Further, we quantified the error variation of the iceberg area retrieval with the solar angle. We recommend reading the paper for details. In this repository we share the code that was written to conduct the analysis. This repository contains code that was written to conduct the experiments. The purpose is to publish the calculations implemented in the code along with the paper. This repository is no standalone software and it is not an installable Python package. Furthermore, it does not contain the data needed to run the analysis. Please reach out if you seek the data used in the study.

## Summary
The separation between ice and ocean pixels in near-infrared data appears straight-forward initially. Many have used constant reflectance thresholds for this purpose. In our paper we propose to use the following rule for detecting and delineating icebergs from top-of-atmosphere Sentinel-2 near-infrared (B08) reflectance (scaled to 0-1 range) data:

$$iceberg = B08 >= 0.12$$

Connected pixels are considered one object. The individual iceberg area is then the sum over the area covered by the connected pixels. In the paper we examined the effects of varying solar zenith angles on the error in the area retrieval, using the reflectance threshold.

We recommend limiting the iceberg area retrieval from these data to solar zenith angles below 65°, which is 5° lower than the broad recommendation by the [European Space Agency](https://scihub.copernicus.eu/news/News00610). The error in the iceberg area retrieval is consistent up to 65°, staying approximately between an overestimated +6% and an underestimated -6%. The specific error margins apply to the threshold used here, but we expect the varying solar illuminations to challenge the iceberg area retrieval also when using other algorithms.

## Background
Iceberg areas are important for studies on freshwater and nutrient fluxes in glacial fjords, and for iceberg drift and deterioration modelling. Satellite remote sensing encompasses valuable tools and datasets to calculate iceberg areas. Synthetic aperture radar (SAR) and optical satellite data are of particular relevance. We generally aim to use iceberg area retrievals from SAR data in conjunction with optical data to study Arctic iceberg populations. The snapshot character of optical satellite acquisitions on cloud-free days is suitable for verifying retrievals from SAR data, but cannot facilitate a consistent time series. To use optical data for verification we have to understand errors in the iceberg area retrieval from optical data first, which was the objective of the paper.

## Contact
Use the contact details of the first author shared in the publication referenced above.

---

# UNet++ Iceberg Segmentation

This repository also contains pre-processed datasets and training code for a UNet++ deep learning model that segments icebergs directly from satellite imagery. Two datasets are provided: one for Sentinel-1 SAR data and one for Sentinel-2 optical data.

## Repository structure

```
S2-iceberg-areas/
│
├── GreenlandExperiment/
│   └── GreenlandExperiment.py      # SZA analysis — 14 S2 scenes at SZA 45°–81°
│
├── SvalbardExperiment/
│   ├── SvalbardExperiment.py       # Threshold calibration vs. Dornier aircraft survey
│   ├── process_s2.py               # Detect icebergs in S2 at 17 thresholds
│   └── process_dornier_data.py     # Extract reference polygons from aircraft rasters
│
├── S1UnetPlusPlus/
│   ├── imgs/                       # 383 SAR chips (256×256, 3 bands: HH, HV, HH/HV)
│   ├── masks/                      # 383 binary masks  (0=ocean, 1=iceberg)
│   └── train_validate_test/        # Pre-split numpy arrays as .pkl files
│
├── S2UnetPlusPlus/
│   ├── imgs/                       # 398 optical chips (256×256, 3 spectral bands)
│   ├── masks/                      # 398 three-class masks (0=ocean, 1=iceberg, 2=shadow)
│   └── train_validate_test/        # Pre-split numpy arrays as .pkl files
│
├── train.py                        # UNet++ training script
├── predict.py                      # Inference on test set — PNG visualisations + CSV
├── predict_tifs.py                 # Georeferenced inference on raw .tif chips → GeoPackage
└── job.slurm                       # HPC SLURM submission script
```

---

## Datasets

### S1UnetPlusPlus — Sentinel-1 SAR

| | |
|---|---|
| **Sensor** | Sentinel-1 SAR, EW mode, GRDM product |
| **Bands** | 3 — HH backscatter, HV backscatter, HH/HV ratio |
| **Chip size** | 256 × 256 pixels |
| **Masks** | Binary: `0 = ocean`, `1 = iceberg` |
| **Split** | 310 train / 38 validation / 35 test |
| **Coverage** | Arctic, 2016–2024 |
| **Class balance** | ~98.5% ocean, ~1.5% iceberg |

### S2UnetPlusPlus — Sentinel-2 Optical

| | |
|---|---|
| **Sensor** | Sentinel-2A/B, Level-1C (top-of-atmosphere reflectance) |
| **Bands** | 3 spectral bands (includes NIR B08) |
| **Chip size** | 256 × 256 pixels |
| **Masks** | 3-class: `0 = ocean`, `1 = iceberg`, `2 = shadow` |
| **Split** | 323 train / 39 validation / 36 test |
| **Coverage** | Arctic, 2016–2024 |
| **Class balance** | ~84% ocean, ~8.5% iceberg, ~7% shadow |

The shadow class captures the effect described in the paper: at high solar zenith angles, icebergs cast shadows that appear as dark patches similar to open water in NIR imagery. Labelling these separately prevents them from being misclassified as ocean during training.

All images are pre-normalized to `[0, 1]` float32 and stored in PyTorch `(N, C, H, W)` format.

### Pickle file layout

Each `train_validate_test/` folder contains six files:

```
X_train.pkl       — (N, 3, 256, 256) float32  training images
Y_train.pkl       — (N, 1, 256, 256) int64    training masks
X_validation.pkl  — (M, 3, 256, 256) float32  validation images
Y_validation.pkl  — (M, 1, 256, 256) int64    validation masks
x_test.pkl        — (K, 3, 256, 256) float32  test images
y_test.pkl        — (K, 1, 256, 256) int64    test masks
```

Load in Python:
```python
import pickle
with open("S2UnetPlusPlus/train_validate_test/X_train.pkl", "rb") as f:
    X_train = pickle.load(f)   # numpy array, shape (323, 3, 256, 256)
```

---

## Installation

```bash
pip install torch torchvision segmentation-models-pytorch
pip install rasterio geopandas shapely pandas
```

---

## Training

`train.py` trains a UNet++ model with a ResNet34 encoder on either the S1 or S2 dataset.

```bash
# Sentinel-2 optical (3-class)
python train.py \
    --mode       s2 \
    --data_dir   S2UnetPlusPlus \
    --out_dir    runs/s2_exp1 \
    --encoder    resnet34 \
    --epochs     100 \
    --batch_size 16 \
    --lr         1e-4

# Sentinel-1 SAR (binary)
python train.py \
    --mode       s1 \
    --data_dir   S1UnetPlusPlus \
    --out_dir    runs/s1_exp1
```

### What the script does

- Loads train/validation/test splits from the `.pkl` files
- Builds `smp.UnetPlusPlus` with ImageNet pretrained encoder weights
- Handles class imbalance automatically:
  - S1: applies a `pos_weight ≈ 70` to BCEWithLogitsLoss (iceberg is only 1.5% of pixels)
  - S2: computes inverse-frequency class weights from the training masks
- Loss = DiceLoss + BCE/CrossEntropyLoss
- Optimiser: AdamW with cosine annealing LR schedule
- Augmentation: random horizontal flip, vertical flip, 90° rotation
- Saves the best checkpoint by validation IoU

### Outputs

```
runs/s2_exp1/
├── best_model.pth      — best checkpoint (by val IoU)
└── training_log.csv    — epoch, train_loss, val_loss, train_iou, val_iou, lr
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | required | `s1` (binary) or `s2` (3-class) |
| `--data_dir` | required | Path to `S1UnetPlusPlus` or `S2UnetPlusPlus` |
| `--out_dir` | required | Where to save checkpoint and log |
| `--encoder` | `resnet34` | Any encoder supported by `segmentation-models-pytorch` |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `16` | Batch size |
| `--lr` | `1e-4` | Initial learning rate |
| `--workers` | `4` | DataLoader worker processes |
| `--no_pretrain` | off | Disable ImageNet pretrained encoder weights |

---

## Inference on test set

`predict.py` runs the trained model on the test split and produces visual outputs for every chip.

```bash
python predict.py \
    --checkpoint runs/s2_exp1/best_model.pth \
    --data_dir   S2UnetPlusPlus \
    --out_dir    predictions/s2_exp1
```

### Outputs

```
predictions/s2_exp1/
├── visualizations/
│   ├── chip_000.png    ─┐
│   ├── chip_001.png     │  one panel per test chip:
│   └── ...             ─┘  [input image | ground truth | prediction]
├── predicted_masks/
│   ├── predicted_masks.npy     — (N, 256, 256) uint8, class labels per pixel
│   └── ground_truth_masks.npy  — (N, 256, 256) int64, corresponding ground truth
└── summary.csv         — per-chip IoU, Dice, iceberg % predicted vs ground truth
```

Each PNG shows the satellite chip with coloured overlays: **cyan = iceberg**, **orange = shadow** (S2 only), side-by-side with the ground truth for direct comparison.

---

## Georeferenced inference on raw .tif chips

`predict_tifs.py` runs inference on the raw `.tif` files in `imgs/`, preserving each chip's spatial coordinate reference system. The output can be opened directly in QGIS or any GIS software.

```bash
python predict_tifs.py \
    --checkpoint  runs/s2_exp1/best_model.pth \
    --imgs_dir    S2UnetPlusPlus/imgs \
    --out_dir     georef_predictions/s2_exp1
```

### Outputs

```
georef_predictions/s2_exp1/
├── geotiffs/
│   └── <chip_name>_pred.tif      — single-band GeoTIFF, pixel value = class label
├── gpkgs/
│   └── <chip_name>_icebergs.gpkg — iceberg polygons for that chip
└── all_icebergs.gpkg             — all chips merged into one GeoPackage
```

`all_icebergs.gpkg` is the primary result. Each row is one delineated iceberg polygon:

| Column | Description |
|---|---|
| `geometry` | Polygon in the chip's native UTM CRS |
| `class_name` | `iceberg` or `shadow` |
| `area_m2` | Polygon area in square metres |
| `source_file` | Source `.tif` filename (encodes satellite, date, tile, position) |
| `iceberg_id` | Unique integer ID across all chips |

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `best_model.pth` |
| `--imgs_dir` | required | Directory of `.tif` chips |
| `--out_dir` | required | Output directory |
| `--min_area_m2` | `100.0` | Minimum polygon area to keep (filters noise) |

---

## Running on an HPC cluster

Edit the top section of `job.slurm` to match your cluster:

```bash
REPO_DIR="/path/to/S2-iceberg-areas"   # path to this repo on the cluster
CONDA_ENV="iceberg"                     # your conda environment name
MODE="s2"                               # s1 or s2
#SBATCH --partition=gpu                 # your cluster's GPU partition name
```

Then submit:

```bash
mkdir -p logs
sbatch job.slurm
```

Stdout and stderr are written to `logs/iceberg_unet_<jobid>.out/.err`.

---

## Full pipeline summary

```
1. PREPARE (already done — data is in this repo)
   S1/S2 .tif chips → normalized, split → .pkl files

2. TRAIN
   python train.py --mode s2 --data_dir S2UnetPlusPlus --out_dir runs/s2_exp1
   → runs/s2_exp1/best_model.pth

3a. VISUALISE TEST RESULTS
   python predict.py --checkpoint runs/s2_exp1/best_model.pth \
                     --data_dir S2UnetPlusPlus --out_dir predictions/s2_exp1
   → PNG panels + summary.csv

3b. GEOREFERENCED OUTPUT (open in QGIS)
   python predict_tifs.py --checkpoint runs/s2_exp1/best_model.pth \
                          --imgs_dir S2UnetPlusPlus/imgs \
                          --out_dir georef_predictions/s2_exp1
   → all_icebergs.gpkg
```

---

## Current extension work

The repository is currently being extended for an independent-study workflow that goes beyond the original paper code.

### DenseCRF sandbox

A lightweight two-phase CRF sandbox now lives in `test-crf/`. It separates:

1. one-time UNet++ probability caching
2. repeated CPU-side DenseCRF experiments on cached probabilities

The current CRF implementation follows a bilateral-only formulation using network softmax probabilities as unary terms.

### Tiny-iceberg label review

A separate `tiny-icebergs/` workflow is being used to review likely missed small iceberg annotations in the high-SZA Roboflow labels.

Key points:

- the new Roboflow export `final-labeling-1` was kept as a raw snapshot
- a corrected working copy `final-labeling-1_fixednulls` was created for known null-scene fixes
- review outputs are three-panel images:
  1. NIR chip
  2. existing annotations in blue
  3. existing annotations in blue plus new threshold candidates in red
- the preferred current review rule is a fixed NIR threshold of `0.30`
- a large-bright-region guard is applied to reduce false positives on broad bright land-like features
- an Otsu-based variant exists for comparison, but fixed `0.30` currently performs better in manual review

These workflows are still part of dataset review and method development, not yet the final published pipeline.
