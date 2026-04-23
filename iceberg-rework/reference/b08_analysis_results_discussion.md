# B08 Reflectance Analysis: Results, Discussion, and Methodological Implications

All reflectance values are uncorrected TOA reflectance (DN x 1e-4). To convert to Fisser's offset-corrected space, subtract 0.10. The threshold of 0.22 used throughout is equivalent to Fisser et al. (2024)'s 0.12 after accounting for the +1000 DN processing baseline offset that chip_sentinel2.py does not subtract. Shadow pixels (Fisser class 2) have been merged into the iceberg class (class 1) prior to all analysis, producing binary masks (iceberg only). This aligns the Fisser 3-class annotation schema with the Roboflow schema and the binary segmentation model.

---

## 1. Per-Iceberg B08 Reflectance by SZA Bin

### Results

Mean B08 reflectance inside annotated icebergs varies non-monotonically with solar zenith angle (Table 1). At sza_lt65 (July to September), pixel-area-weighted iceberg B08 averages 0.515. At sza_65_70 (September to October), iceberg reflectance drops to 0.336, the lowest of any bin. It recovers partially at sza_70_75 (October) to 0.413 and rises to 0.677 at sza_gt75 (November).

**Table 1.** Per-iceberg mean B08 reflectance by SZA bin. N is the number of individual icebergs (connected components >= 16 pixels after merging shadow into iceberg). "% below 0.22" is the fraction of individual icebergs whose mean B08 falls below the Fisser detection threshold. See also `table_iceberg_b08_by_sza.png`.

| SZA Bin | N icebergs | Mean B08 | Median B08 | % icebergs below 0.22 |
|---------|-----------|----------|-----------|----------------------|
| sza_lt65 | 39,534 | 0.4108 | 0.3922 | 13.2% |
| sza_65_70 | 1,710 | 0.2897 | 0.2876 | 20.2% |
| sza_70_75 | 2,375 | 0.3808 | 0.3775 | 6.3% |
| sza_gt75 | 2,773 | 0.5400 | 0.5208 | 21.2% |

The pixel-level view (Table 2) weights by total iceberg area rather than treating each iceberg equally. This captures the contribution of large icebergs more faithfully.

**Table 2.** Pixel-level B08 reflectance inside icebergs by SZA bin. "% px < 0.22" is the fraction of all iceberg pixels (not icebergs) falling below the threshold. See also `table_iceberg_b08_pixels_by_sza.png`.

| SZA Bin | Total pixels | Px mean B08 | Px median B08 | % px < 0.22 |
|---------|-------------|-------------|--------------|-------------|
| sza_lt65 | 3,500,163 | 0.5153 | 0.5107 | 15.2% |
| sza_65_70 | 104,990 | 0.3362 | 0.3332 | 24.5% |
| sza_70_75 | 166,135 | 0.4134 | 0.4102 | 15.0% |
| sza_gt75 | 219,731 | 0.6768 | 0.6544 | 16.5% |

Within-iceberg standard deviation is lowest at sza_65_70 (0.103) and highest at sza_gt75 (0.277), indicating that icebergs at sza_65_70 are uniformly dim while icebergs at sza_gt75 contain a wide spread of pixel values within individual objects (Table 3).

### Discussion

The sza_65_70 bin represents a transition zone where iceberg reflectance is compressed. At moderate SZA (65 to 70 degrees), solar illumination has decreased enough that iceberg surfaces no longer reflect strongly, but shadow formation has not yet progressed to the point of creating distinct bright and dark faces. The result is uniformly dim icebergs (low std of 0.103) that are close in reflectance to the surrounding water. This is consistent with Fisser et al. (2024), who identified SZA of approximately 65 degrees as the onset of significant retrieval error.

The high reflectance and high variability at sza_gt75 reflect the shadow segregation effect described by Fisser et al. (2024). At SZA above 75 degrees, sun-facing slopes remain very bright while shadow-facing slopes drop well below the detection threshold. The mean is pulled upward by the bright faces, but 16.5% of pixels are below 0.22 (Table 2). This bimodal within-iceberg distribution makes sza_gt75 the most challenging bin for both threshold and learned methods.

The partial recovery at sza_70_75 suggests that shadow formation at intermediate angles creates enough geometric contrast (bright sunlit face adjacent to dark shadow) that the average reflectance increases relative to the uniformly dim sza_65_70 condition.

---

## 2. Iceberg, Ocean, and Contrast Characterization by SZA Bin

### Results

**Table 3.** Full SZA bin characterization. Iceberg B08 and ocean B08 are pixel-area-weighted means. Contrast is iceberg B08 minus ocean B08. Within-ice std is the mean of per-iceberg standard deviations weighted by iceberg area. "% px < 0.22" is the pixel-level fraction of iceberg pixels below threshold. See also `table_sza_characterization.png`.

| SZA Bin | Months | Iceberg B08 | Ocean B08 | Contrast | Within-ice std | % px < 0.22 |
|---------|--------|-------------|-----------|----------|---------------|-------------|
| sza_lt65 | Jul, Aug, Sep | 0.5153 | 0.2850 | 0.2303 | 0.1332 | 15.2% |
| sza_65_70 | Sep, Oct | 0.3362 | 0.1597 | 0.1764 | 0.1025 | 24.5% |
| sza_70_75 | Oct | 0.4134 | 0.1890 | 0.2244 | 0.1351 | 15.0% |
| sza_gt75 | Nov | 0.6768 | 0.1734 | 0.5034 | 0.2766 | 16.5% |

Background (non-iceberg) B08 reflectance is highest at sza_lt65 (0.285) and lower at the other bins (0.160 to 0.189).

Iceberg-to-ocean contrast is lowest at sza_65_70 (0.176), moderate at sza_lt65 (0.230) and sza_70_75 (0.224), and highest at sza_gt75 (0.503).

The iceberg and 100 m neighborhood B08 distributions are shown per SZA bin in Figure 1.

**Figure 1.** `hist_iceberg_vs_neighborhood_b08.png`. Normalized histograms of B08 reflectance inside annotated icebergs (red) and within the 100 m neighborhood belt around icebergs (blue) at each SZA bin. The fixed threshold of 0.22 is shown as a dashed line. Shadow has been merged into the iceberg class. Comparable to Fisser et al. (2024) Figure 9.

### Discussion

Contrast collapse at sza_65_70 arises from simultaneous compression on both sides of the reflectance distribution. Iceberg reflectance drops from 0.515 to 0.336 as reduced illumination dims surfaces. Ocean reflectance is slightly elevated (0.160) compared to the theoretical dark water baseline. The net contrast of 0.176 is 23% lower than the next closest bin (sza_70_75 at 0.224). This narrow margin explains why any fixed threshold will perform poorly at SZA 65 to 70 degrees: the iceberg and background distributions overlap substantially (Figure 1, sza_65_70 panel).

The high contrast at sza_gt75 (0.503) is initially counterintuitive given that this is the highest SZA bin. It arises because the shadow segregation effect concentrates reflected energy onto sun-facing slopes, producing very high pixel values on those faces (mean 0.677), while ocean remains dark (0.173). However, the high within-iceberg variance (0.277) means this contrast applies only to the bright fraction of each iceberg. The shadow fraction falls below detection thresholds, leading to systematic underestimation of iceberg area despite high apparent contrast. This is visible in Figure 1 (sza_gt75 panel) as the long left tail of the iceberg distribution extending below 0.22.

The sza_lt65 background of 0.285 reflects the environmental context of the Fisser et al. (2025) training chips. These were extracted from Kangerlussuaq and Sermilik fjords during July to September, when calving activity produces abundant brash ice surrounding larger icebergs. This bright background material is not sea ice in the climatological sense (pack ice or fast ice) but rather a natural feature of the calving environment. The 100 m neighborhood analysis confirms this: 49.3% of neighborhood pixels around sza_lt65 icebergs exceed the 0.22 threshold (Figure 1, sza_lt65 panel).

---

## 3. Implications for IC Filtering Methodology

### Results

The Fisser et al. (2024) ice coverage filter uses a fixed B08 threshold of 0.12 (0.22 uncorrected) in 10 km blocks, excluding blocks where more than 15% of pixels exceed the threshold. Applying an annotation-aware adaptation of this method at chip level (2.56 km), where annotated iceberg pixels are excluded from the IC calculation, produces the following counts of chips exceeding IC >= 15%:

**Table 4.** Annotation-aware IC at chip level. IC is computed as the fraction of non-annotated pixels with B08 >= 0.22. "IC >= 15%" is the number of chips where the non-iceberg background exceeds the Fisser threshold.

| SZA Bin | N chips | IC mean | IC median | Chips with IC >= 15% |
|---------|---------|---------|-----------|---------------------|
| sza_lt65 | 398 | 0.380 | 0.288 | 239 (60%) |
| sza_65_70 | 138 | 0.071 | 0.004 | 20 (14%) |
| sza_70_75 | 166 | 0.070 | 0.006 | 28 (17%) |
| sza_gt75 | 282 | 0.111 | 0.022 | 69 (24%) |

At sza_lt65, the majority of chips (60%) exceed IC 15% even after excluding annotated icebergs, reflecting the pervasive brash ice environment described in Section 2.

### Discussion

A single fixed threshold cannot serve as both an iceberg detection tool and a sea ice identification tool across SZA bins. At sza_65_70, the iceberg and non-iceberg distributions overlap to the point where 24.5% of iceberg pixels are indistinguishable from background using B08 alone (Table 2, Figure 1). At sza_lt65, the surrounding brash ice environment means that bright non-iceberg pixels are pervasive regardless of actual sea ice conditions (Table 4).

These findings rule out a universal pixel-level masking strategy applied to all chips. Masking every bright non-annotated pixel in every chip would create an artificially clean training environment that does not represent real satellite imagery. Chips where 2% of the background is brash ice are realistic scenes. Chips where 40% is pack ice are contaminated. The IC threshold should separate these two cases, not sterilize the entire dataset.

### Methods: Sea Ice Filtering Protocol

#### 3.1 Annotation-Aware IC Computation

Fisser et al. (2024) computed ice coverage (IC) by applying a fixed B08 reflectance threshold of 0.12 (corrected TOA reflectance) to 10 km squared blocks derived from the full Sentinel-2 tile (10,980 x 10,980 pixels at 10 m resolution). Blocks where IC exceeded 15% were excluded. We adapt this method to operate at the chip level (256 x 256 pixels, 2.56 x 2.56 km) because the training data consists of pre-tiled chips, and the parent Sentinel-2 tiles are not available for all chips (the Fisser et al. (2025) Mendeley dataset provides only the tiled chips, not the source tiles).

A direct application of Fisser's threshold at chip scale conflates iceberg signal with sea ice signal. At 10 km scale, even a large iceberg occupies less than 0.1% of the block, so icebergs do not inflate the IC metric. At chip scale, annotated icebergs can occupy 5 to 25% of the chip area (Table 3), which would cause the chip to fail IC regardless of actual sea ice presence. To address this, we exclude annotated iceberg pixels from the IC calculation:

    IC = count(B08 >= 0.22 AND pixel not in iceberg annotation) / count(pixels not in iceberg annotation)

The threshold of 0.22 (uncorrected TOA reflectance) is equivalent to Fisser's 0.12 after accounting for the +1000 DN offset introduced by the Sentinel-2 processing baseline >= 4.0 (N0500/N0510). The chip_sentinel2.py tiling script converts DN to reflectance as DN x 1e-4 without subtracting this offset, so all reflectance values in the dataset are shifted by +0.10 relative to corrected reflectance. The 15% IC threshold is retained from Fisser et al. (2024).

This formulation is applied identically to both data sources. For the Fisser et al. (2025) chips (sza_lt65, N=398), the iceberg annotation mask is derived from the published pixel-level labels with shadow (class 2) merged into iceberg (class 1). For the Roboflow chips (sza_gt65, N=586), the annotation mask is derived from the COCO polygon annotations rasterized at 256 x 256 pixels. Merging shadow into iceberg ensures that Fisser's 3-class labels (ocean, iceberg, shadow) and Roboflow's 2-class labels (ocean, iceberg) define the annotation mask consistently across both sources.

#### 3.2 IC Results Across SZA Bins

The annotation-aware IC metric produces the following distribution (Table 4, `table_annotation_aware_ic.png`):

| SZA Bin | N chips | IC mean | IC median | IC p90 | Chips with IC >= 15% |
|---------|---------|---------|-----------|--------|---------------------|
| sza_lt65 | 398 | 0.380 | 0.288 | 0.975 | 239 (60.1%) |
| sza_65_70 | 138 | 0.071 | 0.004 | 0.287 | 20 (14.5%) |
| sza_70_75 | 166 | 0.070 | 0.006 | 0.247 | 28 (16.9%) |
| sza_gt75 | 282 | 0.111 | 0.022 | 0.342 | 69 (24.5%) |

In total, 356 of 984 chips (36.2%) exceed IC 15%. Of these, 275 contain annotated icebergs and 81 are null (no icebergs). The sza_lt65 bin dominates the failures (239 of 356, 67.1%), reflecting the brash ice environment of summer calving fjords described in Section 2.

For the 628 chips that pass IC < 15%, the bright non-annotated background constitutes only 1.5% to 2.5% of chip area across bins. This represents natural background variability (scattered brash ice, meltwater, slight reflectance noise) that is characteristic of realistic Sentinel-2 fjord scenes.

#### 3.3 Masking Protocol

We adopt a selective masking approach that modifies only training chips exceeding the IC threshold, preserving the natural background in passing chips and leaving evaluation data unmodified.

**Training chips with IC >= 15%:** For each failing chip, pixels satisfying (B08 >= 0.22) AND (not in iceberg annotation) are set to zero across all three bands (B04, B03, B08). This removes sea ice and other bright contaminants while preserving annotated iceberg pixels and the dark ocean surrounding them. Across the 356 failing chips, this operation masks 10,654,099 pixels (37.1% to 50.3% of failing chip area, depending on SZA bin) and preserves 2,751,589 annotated iceberg pixels. The 81 null failing chips (no icebergs) are retained with their bright pixels masked to zero, contributing ocean-only training examples.

**Training chips with IC < 15%:** These 628 chips are left unmodified. Their backgrounds contain 1.5% to 2.5% bright non-annotated pixels, which represent natural environmental conditions (brash ice, meltwater, reflectance variation) that the model should learn to distinguish from icebergs. Masking these chips would create a synthetic training environment where the non-iceberg background is artificially dark, which does not reflect the conditions the model will encounter at inference.

**Validation and test chips:** These are never masked, regardless of their IC values. The validation set is used for model checkpoint selection, and the test set is used for final evaluation. Both must reflect real-world conditions to produce unbiased performance estimates. If masking were applied to evaluation data, performance metrics would be optimistically biased because the model would be evaluated on artificially clean scenes that do not occur in unprocessed satellite imagery.

#### 3.4 Justification for 15% IC Threshold

The 15% threshold is inherited directly from Fisser et al. (2024) Section 3.2 to maintain methodological comparability. It distinguishes chips where the non-iceberg background is dominated by open water (IC < 15%, median IC = 0.004 to 0.029 across the three gt65 bins) from chips where a substantial fraction of the background consists of sea ice or other bright contaminants (IC >= 15%, median IC = 0.29 to 0.39). The separation between these two populations is visible in the IC distribution: p25 values are near zero (0.0001 to 0.012) across all bins, while p90 values range from 0.247 to 0.975, indicating a long right tail of heavily contaminated chips rather than a gradual continuum.

#### 3.5 Justification for Masking Rather Than Discarding

Discarding all 356 chips that exceed IC 15% would remove 36.2% of the dataset, including 275 chips with annotated icebergs. At sza_lt65, discarding would eliminate 239 of 398 Fisser chips (60.1%), reducing the low-SZA training pool from 398 to 159 chips and severely imbalancing the dataset across SZA bins. The icebergs in these chips are themselves valid training examples: they are correctly annotated, they span the full range of iceberg sizes and morphologies in the dataset, and the sea ice surrounding them does not alter the spectral properties of the iceberg pixels. Masking preserves these 2,751,589 iceberg pixels (representing the majority of all Fisser iceberg pixels) while removing only the confounding background.

#### 3.6 Justification Against Per-Bin Dynamic Threshold

The overlap between iceberg and non-iceberg B08 distributions is present at all SZA bins. Table 2 shows that 15.0% to 24.5% of iceberg pixels fall below the 0.22 threshold in every bin. A dynamic threshold set higher than 0.22 would reduce false identification of sea ice but would also mask a larger fraction of real iceberg pixels. A dynamic threshold set lower than 0.22 would preserve more iceberg signal but would fail to identify dim sea ice. No fixed or adaptive threshold can cleanly separate the two distributions because they overlap in reflectance space at every SZA (Figure 1).

The annotation-aware formulation avoids this problem entirely. By excluding annotated icebergs from the IC calculation, the threshold of 0.22 is used only to characterize the brightness of the non-iceberg background. It does not serve as an iceberg detector. The iceberg/non-iceberg distinction is made by the human-verified annotations, which are not subject to the spectral ambiguity that limits any reflectance-based threshold.

---

## Figures and Tables Index

| ID | File | Description |
|----|------|-------------|
| Table 1 | `table_iceberg_b08_by_sza.png` | Per-iceberg mean B08 by SZA bin |
| Table 2 | `table_iceberg_b08_pixels_by_sza.png` | Pixel-level B08 inside icebergs by SZA bin |
| Table 3 | `table_sza_characterization.png` | Full SZA bin characterization: iceberg, ocean, contrast, std |
| Table 4 | (inline above) | Annotation-aware IC at chip level |
| Figure 1 | `hist_iceberg_vs_neighborhood_b08.png` | Iceberg vs 100 m neighborhood B08 distributions per SZA bin |
| Figure 2 | `hist_root_length.png` | Per-iceberg root length distribution per SZA bin |
| Figure 3 | `hist_month.png` | Chip count by acquisition month per SZA bin |
| Figure 4 | `hist_area.png` | Iceberg area distribution per SZA bin with log-log power law check |
| Figure 5 | `hist_wind.png` | Chip count by wind speed per SZA bin |
| Figure 6 | `hist_temp.png` | Chip count by temperature per SZA bin |
| Table 5 | `table_relative_abundance.png` | Relative abundance n(0):n(>=1):n(>=5) per SZA bin |
| Table 6 | `table_fisser_comparison.png` | Comparison with Fisser 2025 reference values |
