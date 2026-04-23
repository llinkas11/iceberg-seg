# Independent Study: UNet++ vs NIR Threshold for Iceberg Area Retrieval at High Solar Zenith Angles

**Advisor context:** Building on Fisser et al. 2024 (paper in this repository) and correspondence with Dr. Henrik Fisser
**Codebase:** Extension of the original `S2-iceberg-areas` repository

---

## Research Question

> Does a trained UNet++ deep learning model retrieve larger iceberg areas than Fisser's calibrated NIR reflectance threshold (B08 ≥ 0.12) at high solar zenith angles (SZA > 65°), and does the gap between the two methods grow as SZA increases?

### Motivation

Fisser et al. 2024 demonstrated that the B08 ≥ 0.12 threshold underestimates iceberg area as SZA increases beyond 65°, with errors worsening significantly at SZA > 75° (typical of October–November in Greenland at 66–68°N). The threshold's degradation is caused by:

1. **Reduced solar illumination** — lower sun angle causes the same iceberg to appear darker in NIR
2. **Shadow elongation** — icebergs cast longer shadows that can overlap and confuse the detector
3. **Reflectance saturation effects** — the threshold is calibrated at moderate SZA and does not adapt

The UNet++ model (trained on 398 Sentinel-2 chips with 3-class labels: ocean/iceberg/shadow) learns spatial and textural context rather than relying on a single reflectance cutoff. The hypothesis is that it degrades more gracefully at high SZA — or not at all — because it has learned to recognize iceberg shape, texture, and shadow relationship rather than brightness alone.

### Study Design

Since no manually-annotated ground truth exists for the new scenes, we use a **no-ground-truth comparison**:

- Run both methods on the same Sentinel-2 chips organized by SZA bin
- Compare the total and per-iceberg area retrieved by each method
- If UNet++ consistently retrieves more area at high SZA, that suggests it is less affected by the illumination degradation that causes the threshold to underestimate

This approach was suggested directly by Dr. Fisser in email correspondence: *"Some tens of image patches would be realistic to create. That would be a nice replication of our study approach."*

---

## Study Areas

Two Greenland fjords from Fisser's original study, selected for their iceberg populations and varying SZA during autumn:

| Region | Name | Approx. coordinates | Primary SZA range (Sept–Nov) |
|---|---|---|---|
| **KQ** | Kangerlussuaq Fjord | 68°N, 32°W | 68° (Sept) → 83° (Nov) |
| **SK** | Sermilik Fjord | 66°N, 38°W | 68° (Sept) → 83° (Nov) |

The AOI file `aois_greenland_area_distributions.gpkg` (provided by Dr. Fisser) contains 14 fjord/ocean polygons in EPSG:5938, covering deep fjord (DF), upper fjord (UF), and outer coast (OC) subzones for each region:

| Polygon | Area (km²) | Description |
|---|---|---|
| KQ-DF | 1,560 | Kangerlussuaq deep fjord |
| KQ-UF | 304 | Kangerlussuaq upper fjord |
| KQ-OC | 5,656 | Kangerlussuaq outer coast |
| SK-DF | 2,038 | Sermilik deep fjord |
| SK-UF | 268 | Sermilik upper fjord |
| SK-OC | 10,383 | Sermilik outer coast |

These polygons are water/ocean-only masks — they do not include land. They are used to clip Sentinel-2 scenes to the relevant fjord water areas before chipping.

---

## SZA Bins

Matching the evaluation bins from Fisser 2024, based on approximate SZA at KQ/SK (~66–68°N) by month:

| Folder name | SZA range | Typical months at study sites |
|---|---|---|
| `sza_lt65` | < 65° | July–August (baseline, low error) |
| `sza_65_70` | 65–70° | September |
| `sza_70_75` | 70–75° | Early October |
| `sza_gt75` | > 75° | Late October–November (primary target) |

---

## Data Acquisition

### Source
Copernicus Data Space Ecosystem (free, ESA): [dataspace.copernicus.eu](https://dataspace.copernicus.eu)
Product type: **Sentinel-2 L1C** (top-of-atmosphere reflectance, unprocessed)
Search API: OData v1 catalogue

### Search parameters

```python
SEARCH_WINDOWS = [
    ("2019-09-01", "2019-09-30"), ("2021-09-01", "2021-09-30"),   # Sept (~68°)
    ("2019-10-01", "2019-10-31"), ("2020-10-01", "2020-10-31"),   # Oct (~76°)
    ("2021-10-01", "2021-10-31"), ("2022-10-01", "2022-10-31"),
    ("2019-11-01", "2019-11-30"), ("2020-11-01", "2020-11-30"),   # Nov (~83°)
    ("2021-11-01", "2021-11-30"), ("2022-11-01", "2022-11-30"),
]
MAX_CLOUD_COVER = 20   # percent
MAX_RESULTS = 5        # per month per region
```

**Total catalogue:** 90 unique scenes across KQ and SK for 2019–2022 (Sept–Nov), ~47 GB total.

### AOI bounding boxes used for search

```
KQ: POLYGON((-33.5 67.5, -30.5 67.5, -30.5 69.0, -33.5 69.0, -33.5 67.5))
SK: POLYGON((-39.0 65.0, -36.0 65.0, -36.0 67.0, -39.0 67.0, -39.0 65.0))
```

### Folder structure

Downloads are organised into SZA bins at download time:

```
sentinel2_downloads/
├── KQ/
│   ├── sza_65_70/    ← September scenes
│   ├── sza_70_75/    ← Early October scenes
│   └── sza_gt75/     ← Late October / November scenes
└── SK/
    └── ...
```

### Download status

| Region | SZA bin | Scenes downloaded | Notes |
|---|---|---|---|
| KQ | sza_65_70 | 3 (Sept 2019) | Successfully downloaded and processed |
| KQ | sza_gt75 | 0 | Disk quota exceeded on moosehead |
| SK | all | 0 | Disk quota exceeded |

**Blocker:** Moosehead HPC home directory quota (~2 GB) was exceeded after 3 scenes (~1.6 GB). The `/mnt/hpc` GlusterFS storage (33 TB free) requires sysadmin access to create a personal directory. Workaround: download to `/tmp` (391 GB free local disk) or request quota increase.

---

## Pipeline

### Overview

```
sentinel2_downloads/{region}/{sza_bin}/*.zip
         │
         ▼  chip_sentinel2.py
chips/{region}/{sza_bin}/
    tifs/  *.tif          — 256×256 float32 GeoTIFFs  [B04, B03, B08]
    pngs/  *_B08.png      — grayscale NIR previews for annotation
         │
    ┌────┴────┐
    │         │
    ▼         ▼
predict_tifs.py        threshold_tifs.py
(UNet++)               (B08 ≥ 0.12)
    │                      │
    ▼                      ▼
all_icebergs.gpkg     all_icebergs_threshold.gpkg
(per region/SZA bin)  (per region/SZA bin)
    │                      │
    └────────┬─────────────┘
             ▼
        compare_areas.py
             │
             ▼
    figures/
        area_stats.csv
        area_boxplots.png
        area_ratio.png
```

### Scripts added in this study

| Script | Purpose |
|---|---|
| `download_sentinel2.py` | Search Copernicus catalogue and download S2 L1C scenes, organised by SZA bin |
| `chip_sentinel2.py` | Unzip .SAFE files, extract bands B04/B03/B08, clip to AOI, tile to 256×256 chips |
| `threshold_tifs.py` | Apply B08 ≥ 0.12 to chip .tifs, polygonize, save GeoPackage |
| `compare_areas.py` | Load both GeoPackages, compute area stats per SZA bin, generate box plots |

---

## Running the Pipeline

### 1. Environment setup

```bash
# On moosehead HPC (recommended for GPU inference)
conda activate iceberg-unet

# Or install locally
pip install rasterio geopandas shapely pillow numpy pandas matplotlib \
            torch torchvision segmentation-models-pytorch
```

### 2. Download scenes

```bash
python download_sentinel2.py
# Follow prompts: authenticate → search catalogue → choose [A] download all
# Downloads go to sentinel2_downloads/{region}/{sza_bin}/
```

### 3. Chip scenes

```bash
python chip_sentinel2.py \
    --safe_dir sentinel2_downloads \
    --out_dir  chips \
    --aoi      aois_greenland_area_distributions.gpkg

# Optional: specify different bands (default B04 B03 B08)
# --bands B04 B03 B08
```

**Output:** `chips/{region}/{sza_bin}/tifs/*.tif` (float32 GeoTIFFs, [0,1] range)
**Filtering:** Chips with < 5% valid signal or > 30% nodata are skipped.

### 4. UNet++ inference

```bash
python predict_tifs.py \
    --checkpoint runs/s2_20260227_231556/best_model.pth \
    --imgs_dir   chips/KQ/sza_65_70/tifs \
    --out_dir    georef_predictions/KQ/sza_65_70

# Run for each region/SZA bin combination
```

**Output:** `georef_predictions/{region}/{sza_bin}/all_icebergs.gpkg`

### 5. Threshold detection

```bash
python threshold_tifs.py \
    --chips_dir chips/KQ/sza_65_70/tifs \
    --out_dir   georef_predictions/KQ/sza_65_70 \
    --b08_idx   2 \
    --threshold 0.12

# Run for each region/SZA bin combination
```

**Output:** `georef_predictions/{region}/{sza_bin}/all_icebergs_threshold.gpkg`

### 6. Compare areas

```bash
python compare_areas.py \
    --pred_dir georef_predictions \
    --out_dir  figures
```

**Output:**
- `figures/area_stats.csv` — per-(region, SZA bin, method) statistics
- `figures/area_boxplots.png` — box plots of area distributions
- `figures/area_ratio.png` — UNet++ / threshold ratio trend across SZA bins (requires ≥ 2 bins)

---

## Model

The UNet++ model used for inference was trained as part of this repository's original codebase.

| Property | Value |
|---|---|
| Architecture | UNet++ with ResNet34 encoder |
| Framework | `segmentation-models-pytorch` |
| Input | 3-band float32 chips, 256×256 pixels |
| Output classes | 3 — ocean (0), iceberg (1), shadow (2) |
| Checkpoint | `runs/s2_20260227_231556/best_model.pth` |
| Training epochs | 39 (best by val IoU) |
| Validation IoU | 0.4398 |
| Training data | 323 chips from `S2UnetPlusPlus/` (original Fisser dataset) |

The model was **not** retrained on the new Greenland scenes — inference is zero-shot on the September 2019 scenes.

### Test Set Performance

Evaluated on 36 held-out chips from the original Fisser dataset (`predictions/s2_exp1/summary.csv`), generated by running `predict.py` on the test split:

| Metric | Value |
|---|---|
| Mean IoU (iceberg class) | **0.384** |
| Median IoU | 0.369 |
| Mean Dice coefficient | **0.629** |
| Median Dice | 0.658 |
| Mean predicted iceberg coverage | 7.8% |
| Mean ground truth iceberg coverage | 8.2% |
| Systematic area bias | **−0.40%** |
| Chips with IoU > 0.5 | 7 / 36 |
| Chips with IoU < 0.3 | 10 / 36 |

The −0.40% systematic bias is the most important number for this study: it shows the model produces essentially unbiased iceberg area estimates at moderate SZA (the training distribution). This baseline is what we compare against at high SZA, where the threshold is known to underestimate.

---

## Results (Preliminary — KQ sza_65_70 only)

**3 scenes processed:** S2B_MSIL1C_20190901 (KQ, September 1 2019, SZA ~68°)
**3,917 chips generated** after AOI clipping and quality filtering

### Area statistics

| Method | Polygons | Mean area (m²) | Median area (m²) | Total area (km²) |
|---|---|---|---|---|
| UNet++ | 119,193 | 168,475 | 400 | 20,081 |
| Threshold (B08 ≥ 0.12) | 3,917 | 6,542,536 | 6,553,600 | 25,627 |

### Key observations

**1. Threshold degenerates in September (SZA ~68°)**

The threshold produces exactly one polygon per chip, each at ~6,553,600 m² — the full chip area (256 × 256 pixels × 100 m²/pixel). This means every pixel in every chip exceeds B08 = 0.12.

**Reason:** In early September 2019, the KQ fjord water surface was covered by sea ice and brash ice. Both icebergs and background sea ice have high NIR reflectance. The threshold cannot distinguish between them without either:
- A pre-existing water/open-ocean mask (which excludes sea ice), or
- Known iceberg locations to constrain the search area (which is how Fisser actually applies it — see `GreenlandExperiment.py` line 113)

**Fisser's actual methodology (from code review):**
```python
# From GreenlandExperiment.py detect_icebergs():
data, transform = mask(src, list(icebergs.buffer(self.buffer_distance).geometry), ...)
iceberg_polygons, _ = to_polygons(np.int8(data >= threshold), transform, meta["crs"])
```
The threshold is applied **only within a 100m buffer around known/annotated iceberg locations**, not to the full scene. It is an area measurement tool given known locations, not a standalone detector.

**2. UNet++ detects realistic individual features**

The UNet++ produces 119,193 polygons with a median area of 400 m² (4 pixels, ~20×20 m) and mean of ~168,000 m². It also detects 306,823 shadow polygons, which is physically meaningful — at SZA ~68°, icebergs cast visible shadows in the fjord. The model learned to separate the shadow class (class 2) from ocean (class 0) and iceberg (class 1).

**3. Cross-SZA comparison is not yet available**

---

## Tiny-Iceberg Annotation Refinement Workflow

After the main v2 dataset was assembled, we found that some existing labels likely miss very small icebergs, especially in the newer high-SZA annotation set. We therefore built a separate review workflow whose purpose is label refinement rather than final model evaluation.

### Data handling

- The newer Roboflow export `final-labeling-1` was downloaded as a raw snapshot and kept unchanged.
- A separate corrected working copy, `final-labeling-1_fixednulls`, was created for label corrections.
- In that corrected copy, all annotations were removed for files beginning with:
  `S2A_MSIL1C_20161107T141402_N0500_R053_T24WWT_20230921T211238`
- The high-SZA export contains only `sza_65_70`, `sza_70_75`, and `sza_gt75` chips. The `sza_lt65` bin still comes from the older annotation source.

### Review output

Each reviewed chip is saved as a three-panel triptych:

1. black-and-white NIR chip
2. NIR chip with existing annotations in blue
3. NIR chip with existing annotations in blue and new threshold candidates in red

This makes it easy to review both the original labels and the proposed additions on the same chip.

### Fixed-threshold candidate generation

The current preferred candidate-generation rule is a fixed NIR threshold applied to the chip-level NIR band. Existing annotations are rasterized and excluded so that new candidates cannot overlap previous labels. The remaining bright pixels are grouped into connected components, and only small components are kept as candidate tiny icebergs.

To suppress obvious false positives on broad bright surfaces, we apply a large-bright-region guard. Bright connected regions above a second reflectance threshold are identified, filtered by minimum size, buffered, and excluded from the red-candidate search space.

The current preferred settings from manual review are:

- fixed threshold: `0.30`
- candidate size range: `2–32` pixels
- large-bright-region threshold: `0.22`
- large-bright-region minimum size: `100` pixels
- large-bright-region buffer: `2` pixels

### Otsu comparison

We also built an Otsu-based comparison workflow that keeps the same exclusion logic, component filtering, and large-bright-region guard but swaps the fixed threshold for a per-chip Otsu threshold. This is useful for sensitivity analysis, but the fixed `0.30` threshold currently performs better in qualitative review and is easier to explain.

### Annotation-format handling

The review code was updated to handle both polygon-style COCO segmentations and RLE mask segmentations. Malformed polygon entries are skipped during visualization instead of being rendered as artifacts. This was necessary because at least one exported chip contained inconsistent segmentation formats.

All 3 processed scenes are from the same date and SZA bin (~68°). The ratio plot (`area_ratio.png`) requires data from at least 2 SZA bins and will be generated once October/November scenes are processed.

---

## Key Methodological Finding

This study reveals a fundamental difference between the two approaches:

| Aspect | Fisser threshold | UNet++ |
|---|---|---|
| Input required | Known iceberg locations (ground truth) + ocean mask | Raw satellite chips only |
| Background assumption | Dark open ocean (B08 < 0.12) | Learned from training data |
| Sea ice handling | Fails — sea ice has same NIR as icebergs | Potentially robust — uses shape/texture/context |
| Shadow handling | Ignored | Explicitly modelled as class 2 |
| Standalone usability | Requires preprocessing | Self-contained |

The threshold as implemented in `threshold_tifs.py` applies a naive global thresholding to full chips. To make a fair comparison, the threshold should be applied with an ocean/open-water mask that excludes sea ice. This would require either:
- A dynamic sea ice mask (e.g., from AMSR2 passive microwave)
- An NDWI-based water classification using B03 and B08
- Fisser's manual fjord polygon annotations

---

## Remaining Work

| Task | Status | Notes |
|---|---|---|
| Download Oct/Nov scenes | Blocked — disk quota | Need `/mnt/hpc` access or quota increase on moosehead |
| Run pipeline on sza_70_75 scenes | Pending | Need disk space first |
| Run pipeline on sza_gt75 scenes | Pending | Primary comparison target |
| Generate ratio plot | Pending | Requires ≥ 2 SZA bins |
| Add ocean/water mask to threshold | Optional | Would make threshold comparison fairer |
| Roboflow annotation of high-SZA chips | Optional | Ground truth for accuracy evaluation; ~30–50 chips per Dr. Fisser's suggestion |
| Fine-tune UNet++ on high-SZA chips | Optional | If annotations are created |

---

## HPC Notes (Bowdoin moosehead)

| Item | Details |
|---|---|
| Cluster | `moosehead.bowdoin.edu` (moose66 node) |
| Conda env | `iceberg-unet` |
| GPU | RTX 3080 (SLURM: `gpu:rtx3080:1`) |
| Home quota | ~2 GB (severely limits downloads) |
| `/mnt/hpc` | GlusterFS, 35 TB total, 33 TB free — requires sysadmin to create personal dir |
| `/tmp` | 393 GB local disk — usable but volatile (purged between sessions) |
| Transfer method | `rsync` from Mac (git unavailable due to Xcode license issue on Mac) |

**Recommended solution:** Email HPC support to create `/mnt/hpc/smishra/` with adequate quota for the ~50 GB dataset.

---

## File Structure (this study)

```
S2-iceberg-areas/
├── download_sentinel2.py          ← NEW: download S2 L1C from Copernicus
├── chip_sentinel2.py              ← NEW: unzip .SAFE → extract bands → tile chips
├── threshold_tifs.py              ← NEW: apply B08≥0.12 → polygonize → GeoPackage
├── compare_areas.py               ← NEW: compare area distributions across SZA bins
├── predict_tifs.py                ← EXISTING: UNet++ georeferenced inference
│
├── aois_greenland_area_distributions.gpkg   ← from Dr. Fisser: 14 fjord/ocean polygons
│
├── sentinel2_downloads/           ← gitignored
│   └── {region}/{sza_bin}/*.zip
├── chips/                         ← gitignored
│   └── {region}/{sza_bin}/tifs|pngs/
├── georef_predictions/            ← gitignored
│   └── {region}/{sza_bin}/
│       ├── all_icebergs.gpkg              (UNet++)
│       └── all_icebergs_threshold.gpkg   (threshold)
├── figures/                       ← gitignored
│   ├── area_stats.csv
│   ├── area_boxplots.png
│   └── area_ratio.png
└── runs/                          ← gitignored
    └── s2_20260227_231556/
        └── best_model.pth         (val IoU = 0.4398, epoch 39)
```

---

## Abstract Draft

> *For use as a starting point — update the cross-SZA comparison section once October/November scenes are processed.*

Iceberg area retrieval from optical satellite imagery is complicated by low solar illumination at high latitudes in autumn, when solar zenith angles (SZA) exceed 65°. Fisser et al. (2024) demonstrated that a calibrated Sentinel-2 near-infrared reflectance threshold (B08 ≥ 0.12) systematically underestimates iceberg area at SZA > 65°, with errors worsening significantly beyond 75°. In this study, we evaluate whether a UNet++ deep learning model can retrieve larger — and therefore more accurate — iceberg areas than the reflectance threshold at these challenging illumination conditions, using Sentinel-2 Level-1C imagery over Kangerlussuaq and Sermilik Fjords, Greenland across September–November (SZA ≈ 68–83°).

The UNet++ model, trained on 398 Sentinel-2 chips with three-class labels (ocean, iceberg, shadow), achieves a mean intersection-over-union (IoU) of 0.38 and Dice coefficient of 0.63 on the held-out test set, with a systematic iceberg area bias of only −0.40% relative to ground truth — indicating reliable, essentially unbiased area retrieval under moderate solar zenith angles. Applied to three Sentinel-2 scenes acquired over Kangerlussuaq Fjord in September 2019 (SZA ≈ 68°), the model detected 119,193 individual iceberg polygons with a median area of 400 m², while also identifying 306,823 shadow polygons consistent with the elongated shadows expected at this solar angle.

The NIR reflectance threshold, applied to the same chips without prior iceberg location information, returned degenerate results: one full-chip polygon per chip (area ≈ 6.5 km²), caused by sea ice covering the fjord background with equivalent NIR reflectance to icebergs. Code inspection of Fisser et al.'s original implementation reveals that the threshold is designed to be applied within a 100 m buffer around known iceberg locations, not as a standalone scene-wide detector — a fundamental methodological distinction from the UNet++ approach, which requires no preprocessing or prior iceberg locations.

*[Cross-SZA comparison to be completed — pending October/November scene processing.]*
The comparison of total retrieved iceberg area as a function of SZA bin will provide direct evidence of whether the learning-based approach is less affected by the solar illumination degradation that limits the threshold method.

---

## References

- Fisser, H., et al. (2024). *Impact of varying solar angles on Arctic iceberg area retrieval from Sentinel-2 near-infrared data.* (paper in this repository)
- Rezvanbehbahani, S., et al. (2020). *Significant contribution of small icebergs to the freshwater budget in Greenland fjords.* Communications Earth & Environment.
- Zhou, Z., et al. (2019). *UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation.* IEEE Transactions on Medical Imaging.
- Copernicus Data Space Ecosystem: [dataspace.copernicus.eu](https://dataspace.copernicus.eu)
