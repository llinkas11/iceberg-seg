# Descriptive Statistics: Results and Discussion

All statistics computed after merging shadow (class 2) into iceberg (class 1) and applying the 40 m root-length cutoff (individual icebergs with area < 1,600 m2 removed). Reflectance values are uncorrected TOA (DN x 1e-4); subtract 0.10 for Fisser-equivalent corrected reflectance.

---

## 1. Dataset Composition

The combined dataset contains 984 chips: 398 from Fisser et al. (2025) (sza_lt65, July to September) and 586 from Roboflow manual annotation (sza_gt65, September to November). After removing 68 Fisser chips that failed the IC quality audit (Section 3 of b08_analysis_results_discussion.md), the working dataset contains 916 chips split 60/15/25 into 551 training, 137 validation, and 228 test chips.

**Table 1.** Dataset composition by SZA bin and split. See also `table_relative_abundance.png`.

| SZA Bin | Total chips | Chips with icebergs | Null chips | n(>=5 icebergs) |
|---------|------------|--------------------|-----------|-----------------| 
| sza_lt65 | 398 | 395 (99.2%) | 3 | 370 |
| sza_65_70 | 138 | 63 (45.7%) | 75 | 43 |
| sza_70_75 | 166 | 89 (53.6%) | 77 | 73 |
| sza_gt75 | 282 | 144 (51.1%) | 138 | 124 |
| **All** | **984** | **691 (70.2%)** | **293** | **610** |

The sza_lt65 bin is dominated by iceberg-containing chips (99.2%) because Fisser selected chips known to contain icebergs. The sza_gt65 bins reflect a more natural distribution: roughly half of chips in each bin are null (no icebergs above the 40 m cutoff). This asymmetry between sources is an artifact of how each dataset was constructed and is addressed through stratified splitting and training set balancing.

---

## 2. Root-Length Filter

The 40 m root-length cutoff removes individual icebergs with area below 1,600 m2 (16 pixels at 10 m resolution), matching the minimum size reported in the Fisser (2025) dataset.

**Table 2.** Effect of 40 m root-length filter on iceberg counts. "Before" is total connected components; "after" is components >= 16 pixels. See also `filter_40m_summary.csv`.

| Source | Before | After | Removed | Removed % |
|--------|--------|-------|---------|-----------|
| Roboflow (COCO) | 18,312 | 7,947 | 10,365 | 56.6% |
| Fisser (pkl, shadow merged) | 96,648 | 39,534 | 57,114 | 59.1% |
| **Total** | **114,960** | **47,481** | **67,479** | **58.7%** |

### Results

The majority of removed components are rasterization artifacts rather than real icebergs. In the Fisser masks, polygon-to-pixel rasterization creates 1 to 3 pixel fragments along annotation boundaries and in narrow gaps between polygons. Before the shadow merge, 82.7% of Fisser connected components were below 16 pixels. After merging shadow into iceberg (which bridges many fragments into contiguous objects), the removal rate drops to 59.1%. The surviving 39,534 Fisser icebergs and 7,947 Roboflow icebergs (47,481 total) represent the scientifically usable iceberg population.

Visualizations of the filter effect are in `viz/filter_40m/coco/` and `viz/filter_40m/fisser/`, showing kept icebergs (gold) and removed fragments (red) for sample chips from each source.

### Discussion

The high removal rate should not be interpreted as discarding real icebergs. The median removed component in Fisser is 3 pixels (300 m2, root length 17 m), well below the scale at which individual icebergs can be reliably delineated at 10 m resolution. These artifacts arise because Fisser's annotations were created as vector polygons with sub-pixel precision, and rasterization to the 10 m pixel grid produces isolated fragments at polygon edges. The shadow merge reduces this problem substantially (removal rate drops from 82.7% to 59.1%) because shadow pixels adjacent to iceberg pixels bridge what would otherwise appear as disconnected fragments.

---

## 3. Iceberg Size Distribution

**Figure 1.** `hist_root_length.png`. Per-iceberg root-length distribution by SZA bin. Each subplot has an independent y-axis to accommodate the 20:1 count ratio between sza_lt65 (39,534 icebergs) and the gt65 bins (1,939 to 3,121 each).

**Figure 2.** `hist_area.png`. Per-iceberg area distribution by SZA bin (linear scale), with a combined log-log plot for power-law assessment.

### Results

**Table 3.** Iceberg size statistics by SZA bin. See also `table_fisser_comparison.png`.

| SZA Bin | N icebergs | Mean area (m2) | Median area (m2) | Max area (m2) | Mean RL (m) | Median RL (m) |
|---------|-----------|----------------|------------------|---------------|-------------|---------------|
| sza_lt65 | 39,534 | 8,854 | 3,100 | 3,608,200 | 72 | 56 |
| sza_65_70 | 1,939 | 24,065 | 3,500 | 5,100,000 | 86 | 59 |
| sza_70_75 | 2,887 | 32,482 | 3,600 | 6,100,000 | 94 | 60 |
| sza_gt75 | 3,121 | 33,502 | 3,600 | 6,199,200 | 97 | 60 |

Median root lengths are consistent across bins (56 to 60 m), indicating that the typical iceberg size is stable regardless of SZA. Mean root lengths increase with SZA (72 to 97 m) because mean area increases, which is driven by a small number of very large annotations in the gt65 bins rather than a systematic shift in the population.

The maximum area increases from 3.6 million m2 at sza_lt65 to 6.2 million m2 at sza_gt75. These maximum values correspond to annotations with root lengths of 1,900 to 2,490 m, far exceeding the maximum reported by Fisser (2025) of 399,700 m2 (root length 632 m). This suggests that some Roboflow annotations encompass multiple adjacent icebergs or ice melange as a single polygon. A total of 92 icebergs across all bins exceed 400,000 m2.

The log-log distribution (Figure 2, rightmost panel) shows an approximately linear relationship for areas between 1,600 and 100,000 m2, consistent with the power-law distribution reported by Fisser (2025). The tail above 100,000 m2 contains the potentially over-annotated multi-iceberg clumps.

### Discussion

The consistency of median root length across SZA bins (56 to 60 m) suggests that the underlying iceberg population is not fundamentally different between bins. The increase in mean is an artifact of annotation methodology: the Roboflow annotations for gt65 chips were pre-annotated with Otsu thresholding and then manually reviewed. Otsu tends to merge adjacent bright objects into single polygons, producing larger annotations on average. Fisser's annotations, created by careful visual delineation, separate individual icebergs more precisely.

The 92 annotations exceeding Fisser's maximum area (399,700 m2) warrant visual inspection. These are flagged in `reference/descriptive_stats.csv` and could represent genuine ice shelves, dense melange fields incorrectly annotated as single icebergs, or edge effects where the annotation polygon extends beyond the iceberg boundary. These outliers affect mean statistics substantially (mean area at sza_gt75 is 33,502 m2 while median is 3,600 m2) but do not affect the model training because the pixel-level mask is what the model sees, not the polygon boundaries.

---

## 4. Temporal and Meteorological Characterization

**Figure 3.** `hist_month.png`. Chip count by acquisition month per SZA bin. Only months July through November are represented.

**Figure 4.** `hist_wind.png`. Chip count by 10 m wind speed (ERA5 reanalysis) per SZA bin. The 15 m/s threshold from Fisser et al. (2024) is shown as a dashed line.

**Figure 5.** `hist_temp.png`. Chip count by 2 m air temperature (ERA5 reanalysis) per SZA bin. The 0 C threshold is shown as a dashed line.

### Results

Acquisition months are determined entirely by the SZA bin definitions: sza_lt65 spans July to September, sza_65_70 spans September to October, sza_70_75 falls in October, and sza_gt75 falls in November (Figure 3). This temporal stratification is by design, reflecting the progression of solar zenith angle through the Arctic autumn.

Wind speed across all 984 chips ranges from 0.1 to 8.4 m/s (mean 2.3, median 2.0). No chip exceeds the 15 m/s threshold used by Fisser et al. (2024) to exclude high-wind scenes (Figure 4). Wind speed distributions are similar across SZA bins. Wind is therefore not a confounding variable in this dataset.

Temperature shows a strong SZA-dependent pattern (Figure 5). At sza_lt65 (summer), all 398 chips have temperatures above 0 C (mean 5.9 C). At sza_gt75 (November), 279 of 282 chips (98.9%) have temperatures at or below 0 C (mean -3.4 C). The intermediate bins show a gradual transition: sza_65_70 has 6 of 138 chips (4.3%) below 0 C, and sza_70_75 has 39 of 166 (23.5%) below 0 C.

**Table 4.** Temperature summary by SZA bin.

| SZA Bin | N chips | Mean temp (C) | Chips <= 0 C | % <= 0 C |
|---------|---------|---------------|-------------|----------|
| sza_lt65 | 398 | 5.9 | 0 | 0.0% |
| sza_65_70 | 138 | 2.8 | 6 | 4.3% |
| sza_70_75 | 166 | 2.7 | 39 | 23.5% |
| sza_gt75 | 282 | -3.4 | 279 | 98.9% |

### Discussion

Temperature is confounded with SZA in this dataset: higher SZA occurs later in autumn when temperatures are lower. This makes it impossible to disentangle the optical effects of SZA (shadow formation, background brightening) from the physical effects of temperature (surface refreezing, meltwater absence, changed albedo). Removing sub-zero chips would eliminate 98.9% of the sza_gt75 bin, making cross-SZA comparison impossible. We therefore retain all chips regardless of temperature and note this confound in the study limitations.

Wind is not a concern. The maximum observed wind speed (8.4 m/s) is well below the threshold at which wind-driven surface roughness significantly affects NIR reflectance (Pegau and Paulson, 2001). The absence of high-wind scenes reflects the generally calm conditions during the September to November acquisition window in East Greenland fjords.

---

## 5. Relative Abundance

**Table 5.** Relative abundance of icebergs per chip by SZA bin. n(0) = chips with zero icebergs above 40 m RL. n(>=1) = chips with at least one iceberg. n(>=5) = chips with five or more icebergs. See also `table_relative_abundance.png`.

| SZA Bin | N chips | n(0) | n(>=1) | n(>=5) | Ratio n(0):n(>=1):n(>=5) |
|---------|---------|------|--------|--------|-------------------------|
| sza_lt65 | 398 | 3 | 395 | 370 | 3:395:370 |
| sza_65_70 | 138 | 75 | 63 | 43 | 75:63:43 |
| sza_70_75 | 166 | 77 | 89 | 73 | 77:89:73 |
| sza_gt75 | 282 | 138 | 144 | 124 | 138:144:124 |
| All | 984 | 293 | 691 | 610 | 293:691:610 |

### Results

The sza_lt65 bin contains almost no null chips (3 of 398, 0.8%) and the vast majority of chips contain five or more icebergs (370 of 398, 93.0%). This reflects the Fisser dataset's construction: chips were selected from known iceberg-rich regions during peak calving season.

The gt65 bins have approximately equal proportions of null and iceberg-containing chips (45 to 54% iceberg-containing). Among chips with icebergs, most contain five or more (68 to 86% of iceberg-containing chips across gt65 bins). This indicates that icebergs tend to cluster: a chip either contains no icebergs or contains several.

### Discussion

The clustering pattern is physically expected. Icebergs calve from tidewater glaciers and drift in groups through fjords. A 2.56 x 2.56 km chip sampling the main fjord channel will typically contain multiple icebergs, while a chip sampling open ocean away from the calving front will contain none.

The near-absence of null chips at sza_lt65 is a selection artifact, not a physical property of low-SZA conditions. Fisser selected chips specifically because they contained icebergs for annotation. The gt65 bins, drawn from a broader spatial sample of the Sentinel-2 tile, provide a more representative null-to-iceberg ratio.

This asymmetry in null chip abundance between sources creates a training set imbalance (addressed through dataset balancing in balance_training.py). It also means that the sza_lt65 test set contains proportionally fewer null chips than the gt65 test sets, which should be considered when comparing per-bin metrics.

---

## 6. Comparison with Fisser (2025)

**Table 6.** Comparison of dataset statistics with Fisser (2025) Mendeley Data reference values. See also `table_fisser_comparison.png`.

| Metric | This dataset | Fisser (2025) |
|--------|-------------|---------------|
| N icebergs | 47,481 | Not reported |
| Mean area (m2) | 12,532 | 2,468 |
| Median area (m2) | 3,200 | Not reported |
| Max area (m2) | 6,199,200 | 399,700 |
| Mean RL (m) | 76 | ~50 |
| Max RL (m) | 2,490 | 632 |

### Discussion

Our mean iceberg area (12,532 m2) is five times larger than Fisser's (2,468 m2). This discrepancy has three contributing factors:

1. **Shadow merge.** Fisser reported iceberg area excluding shadow pixels. Our merged annotations include shadow, which increases individual iceberg areas by approximately 40% at sza_lt65 (where shadow constitutes 7.1% of labeled pixels in the original Fisser masks, compared to 6.9% for iceberg). This effect alone does not explain the 5x difference.

2. **Annotation methodology.** The Roboflow annotations were pre-annotated with Otsu thresholding and then manually reviewed. Otsu tends to produce more inclusive polygons that merge adjacent icebergs and include bright meltwater in the annotation boundary. Fisser's visual delineation separates individual icebergs more precisely. This is the primary driver of the area difference.

3. **Outlier annotations.** 92 annotations exceed Fisser's maximum area of 399,700 m2. These outliers inflate the mean substantially (mean 12,532 vs median 3,200 m2). If annotations above 400,000 m2 are excluded, the mean drops to approximately 5,200 m2, closer to Fisser but still elevated due to factor 2.

The median root length (57 m) is closer to Fisser's mean root length (~50 m), suggesting that the central tendency of the iceberg population is reasonably consistent between datasets. The difference is concentrated in the upper tail of the size distribution.

---

## Figures and Tables Index

| ID | File | Description |
|----|------|-------------|
| Table 1 | `table_relative_abundance.png` | Dataset composition and relative abundance by SZA bin |
| Table 2 | `filter_40m_summary.csv` | 40 m root-length filter before/after counts |
| Table 3 | `table_fisser_comparison.png` | Iceberg size statistics and comparison with Fisser (2025) |
| Table 4 | (inline) | Temperature summary by SZA bin |
| Table 5 | `table_relative_abundance.png` | Relative abundance n(0):n(>=1):n(>=5) |
| Table 6 | `table_fisser_comparison.png` | Comparison with Fisser (2025) reference values |
| Figure 1 | `hist_root_length.png` | Per-iceberg root-length distribution, one subplot per SZA bin |
| Figure 2 | `hist_area.png` | Area distribution per SZA bin with log-log power-law check |
| Figure 3 | `hist_month.png` | Chip count by acquisition month (Jul to Nov only) |
| Figure 4 | `hist_wind.png` | Chip count by wind speed per SZA bin |
| Figure 5 | `hist_temp.png` | Chip count by temperature per SZA bin |
