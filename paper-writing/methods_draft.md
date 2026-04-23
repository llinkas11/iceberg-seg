<!--
RECONCILIATION NOTE (2026-04-17)

Multiple items in this draft predate the 2026-04-16 pivot to the v3_balanced / binary
pipeline. Before lifting prose into the new Overleaf template, override against plan.md
and reference/*.md:

- §2.4 Chip Filtering: the per-chip Otsu 15% sea-ice chip exclusion is superseded by
  the annotation-aware IC filter (IC = fraction of *non-annotated* pixels with B08 >= 0.22;
  15% threshold; 193 training chips had bright non-annotated pixels masked to zero; val/test
  never masked). Full justification: reference/b08_analysis_results_discussion.md §3.1-3.6.

- §2.5 Dataset composition: "984 annotated chips ... 790 training, 98 validation, 96 test,
  24 per SZA bin in test" is stale. Current: 984 pre-filter -> 916 after 40 m RL filter ->
  v3_balanced training 364 / val 137 / test 228 (57 per SZA bin in test). Class distribution
  is binary: ocean 93.0% / iceberg 7.0%. Source: plan.md, iceberg-rework-README.md v3 tables.

- §2.6 UNet++: "three-class output (ocean=0, iceberg=1, shadow=2)" is stale.
  Binary segmentation (single class: iceberg); shadow merged before analysis (plan.md §PR-7).
  "Test IoU 0.3617" is smishra's v2_aug checkpoint on 984 chips, NOT the current
  v3_balanced training; the new checkpoint is pending plan.md Step 8.

- §2.9 Hybrid pipelines / DenseCRF: tested on 4-chip sandbox; IoU dropped 0.011 to 0.013;
  NOT applied to full dataset. Retain only as a negative-comparison point in prose.

- §2.10 White top-hat: primary source is tiny_icebergs_methods_addendum.md.

- §2.11 Evaluation: missing per-iceberg MAE / RERL / contrast metrics (plan.md Step 10,
  implemented in eval_per_iceberg.py).

Keep as scaffolding. Do NOT copy numeric claims or class-structure language without
reconciling against plan.md and reference/*.md.
-->

# Methods

## 2.1 Study Regions and Imagery

The study covers two marine-terminating glacier fjords in Greenland: Kangerlussuaq (KQ) and Sermilik (SK). Both fjords are sites of active calving from outlet glaciers of the Greenland Ice Sheet and have been previously documented as sources of significant iceberg flux (Moyer et al., 2019; Enderlin et al., 2014). Kangerlussuaq Fjord is located on the east coast at approximately 68.5°N, and Sermilik Fjord at approximately 65.7°N.

Imagery is drawn from the Sentinel-2 Level-1C (L1C) archive (Drusch et al., 2012). L1C products provide top-of-atmosphere (TOA) reflectance at 10 m resolution for the visible and near-infrared bands used here: B04 (red, 665 nm), B03 (green, 560 nm), and B08 (near-infrared, 842 nm). All three bands are used as model inputs; B08 is the primary discriminant for iceberg detection.

Scenes are stratified by solar zenith angle (SZA) into four bins: SZA < 65° (sza_lt65), 65-70° (sza_65_70), 70-75° (sza_70_75), and > 75° (sza_gt75). These boundaries follow Fisser et al. (2024), who documented a threshold near SZA 65° above which threshold-based retrieval error increases substantially. The dataset comprises 175 scenes in total: 86 from Kangerlussuaq (25, 15, 30, and 16 scenes per bin) and 89 from Sermilik (33, 18, 10, and 28 scenes per bin). Scenes were acquired between September and November across multiple years to sample a range of illumination conditions while excluding winter darkness and summer melt-pond ambiguity.

## 2.2 Radiometric Calibration Note

All scenes used in this study were processed under ESA Sentinel-2 baseline N0500 or later. Under this baseline, ESA applies a +1000 DN offset to all band digital numbers before distribution. When converted to TOA reflectance using the standard scaling factor of 10^-4, this offset adds +0.10 to all reflectance values relative to the offset-corrected space used by Fisser et al. (2024). Our processing pipeline applies the 10^-4 scaling without subtracting this offset. Consequently, Fisser's calibrated threshold of B08 >= 0.12 corresponds to B08 >= 0.22 in our reflectance space, and all threshold values in this paper are reported in offset-uncorrected space. Because all scenes in the dataset share the same processing baseline, the +0.10 offset is uniform across all chips and does not affect relative comparisons between methods or between SZA bins.

## 2.3 Data Acquisition and Chipping

Scenes were downloaded from the Copernicus Data Space using the OData API, filtered to cloud cover below a scene-level threshold and constrained to the fjord AOI polygon for each study region. Each scene was chipped into 256x256 pixel tiles (2.56 x 2.56 km at 10 m resolution) using a sliding window with no overlap. Chips are stored as three-band GeoTIFF files (B04, B03, B08) with spatial reference metadata preserved from the source scene. A total of 23,981 chips were produced across both fjords and all SZA bins.

## 2.4 Chip Filtering

Before annotation and model training, chips were filtered on two criteria. First, chips with cloud cover exceeding 1% were excluded using the Sentinel-2 scene classification layer (MSK_CLASSI_B00.jp2), which provides a per-pixel cloud/cloud-shadow classification at 20 m resolution, resampled to the chip grid. Second, chips where more than 15% of pixels exceeded the per-chip Otsu threshold on B08 were excluded as likely sea-ice dominated. This combination of cloud and sea-ice filtering removed chips where iceberg detection would be unreliable or where bright non-target surfaces would corrupt annotations.

## 2.5 Annotation and Dataset Composition

The training dataset comprises 984 annotated chips from two sources. The first source is 398 chips from Kangerlussuaq Fjord at SZA < 65° provided by Fisser et al. (2025). These chips carry three-class polygon annotations (ocean, iceberg, shadow) and represent well-illuminated conditions where iceberg boundaries are distinct. The second source is 586 chips at SZA > 65° annotated on the Roboflow platform by 33 annotators. High-SZA chips were annotated with a single class (iceberg); ocean and shadow pixels are background. Annotation used the SAM3 (Segment Anything Model) smart-select tool in Roboflow for initial boundary proposals, followed by manual correction. Iceberg boundaries were delineated at the visible ice-water boundary in the NIR band. Shadow regions were not explicitly labeled.

The 984-chip dataset was split into training (790 chips), validation (98 chips), and test (96 chips) sets. The test set contains exactly 24 chips per SZA bin to ensure balanced evaluation across illumination conditions. The training set class distribution reflects the strong ocean dominance of fjord scenes: 92.5% ocean pixels, 4.6% iceberg pixels, and 2.9% shadow pixels.

## 2.6 UNet++ Segmentation Model

The segmentation model is UNet++ (Zhou et al., 2018) with a ResNet34 encoder pretrained on ImageNet (He et al., 2016). UNet++ extends the standard encoder-decoder architecture with nested dense skip connections that aggregate features from multiple encoder depths, improving boundary localization for objects at varying scales. The model accepts three-channel (B04, B03, B08) input chips at 256x256 pixels and produces a three-class output (ocean=0, iceberg=1, shadow=2). For area retrieval, only the iceberg channel is extracted from the argmax prediction; ocean and shadow are treated as background in all reported results.

The model was trained with a composite loss function combining Dice loss and cross-entropy loss with inverse-frequency class weights to compensate for ocean-class dominance. The optimizer was AdamW (learning rate 1e-4, weight decay 1e-3) with a cosine annealing learning rate schedule over 100 epochs and batch size 16. Training used a single NVIDIA RTX 3080 GPU on the Bowdoin College high-performance computing cluster. The checkpoint with highest validation IoU across all epochs was retained.

Data augmentation was selected from a 16-combination sweep covering horizontal flip, vertical flip, random 90° rotation, and color jitter, evaluated by test-set IoU. The winning configuration (horizontal flip + vertical flip + random 90° rotation, hereafter v2_aug) achieved a test IoU of 0.3617, compared to 0.3464 for the no-augmentation baseline. Color jitter augmentation degraded performance in all combinations tested, consistent with the near-grayscale nature of NIR imagery and the dominance of reflectance magnitude over color in iceberg discrimination.

## 2.7 Fixed NIR Threshold

The fixed NIR threshold applies a scene-wide cutoff of B08 >= 0.22 to each chip, classifying pixels above the threshold as iceberg. This threshold corresponds to Fisser et al.'s (2024) calibrated value of 0.12 in offset-corrected reflectance space, adjusted for the +0.10 uniform offset in our pipeline (Section 2.2). The threshold is applied independently to each chip with no spatial context. Pixels forming connected components smaller than 100 m2 (1 pixel at 10 m resolution) are discarded. This method provides the simplest possible baseline and directly replicates the approach whose SZA-dependent limitations were characterized by Fisser et al. (2024).

## 2.8 Per-Chip Otsu Thresholding

Per-chip Otsu thresholding computes an independent threshold for each chip from the B08 histogram using Otsu's method (Otsu, 1979), adapting to local illumination conditions rather than applying a fixed scene-wide cutoff. The threshold is computed on non-zero B08 pixels within the chip. Three guards are applied: chips where the Otsu threshold falls below 0.10 are skipped as radiometrically flat (likely open ocean or cloud); thresholds above 0.50 are clipped to 0.50 to prevent unstable values in sparse histograms; and chips where more than 15% of pixels exceed the computed threshold are skipped as likely sea-ice dominated. Polygons smaller than 100 m2 are discarded, consistent with the fixed threshold baseline.

## 2.9 Hybrid Pipelines

Three additional pipelines combine UNet++ probability outputs with threshold-based post-processing. In UNet++/Threshold, the UNet++ iceberg probability map is binarized at a fixed probability cutoff to produce a refined segmentation. In UNet++/Otsu, the per-chip Otsu method is applied to the UNet++ iceberg probability map rather than to the raw B08 reflectance. These hybrid pipelines test whether applying adaptive thresholding to a learned probability surface outperforms applying it to raw reflectance.

DenseCRF (Krahenbuhl and Koltun, 2011) post-processing was evaluated as an additional refinement step in a four-chip sandbox using bilateral-only parameters (sxy=40, srgb=3, compat=4, iterations=5). CRF refinement reduced mean IoU by 0.011 to 0.013 relative to the UNet++ baseline (baseline mean IoU: 0.345; best CRF run: 0.332). Given this consistent degradation and the limited probability surface confidence at high SZA conditions, CRF post-processing was not applied to the full dataset and is not included in the main results comparison.

## 2.10 White Top-Hat Filtering for Small-Iceberg Recovery

Each segmentation method produces a binary iceberg mask that may miss icebergs smaller than the effective receptive field of the method. To recover these objects, a morphological white top-hat transform is applied as an optional post-processing step to every pipeline output. The white top-hat extracts bright features smaller than a structuring element from the residual between the original B08 image and its morphological opening. Candidate pixels are those that (a) exceed a fixed NIR threshold of 0.30 on B08, (b) form connected components of 2 to 32 pixels (200 to 3,200 m2 at 10 m resolution), and (c) do not overlap any existing segmentation mask by even one pixel. A large-bright-region guard suppresses false positives near shorelines and sea-ice edges: connected regions above B08 >= 0.22 that span at least 100 pixels are identified, buffered outward by 2 pixels, and excluded from the candidate search space. The top-hat step is applied independently to each method's output, producing a "+TH" variant for every pipeline (e.g., Threshold+TH, Otsu+TH, UNet+++TH). This allows direct comparison of each method with and without small-iceberg recovery across all SZA bins.

## 2.11 Evaluation

All methods produce iceberg polygons stored as GeoPackage files with area in square meters. Comparison is performed at the SZA-bin level rather than the chip level, aggregating total detected area, polygon count, mean polygon area, and median polygon area across all chips in each region-bin combination. UNet++ serves as the reference method; the ratio of each threshold-based method's total area to UNet++ total area quantifies overestimation or underestimation as a function of SZA. Results are reported separately for Kangerlussuaq and Sermilik to assess whether retrieval behavior generalizes across fjords.

## 2.12 Per-Pair Area Error Metrics (Fisser-Comparable)

Beyond the area-distribution comparison of Section 2.11, we quantify per-iceberg error against a visually delineated reference set and report error statistics comparable to Fisser and others (2024). This allows the four methods (UNet++, fixed NIR threshold, per-chip Otsu, and DenseCRF) to be placed on the same accuracy axis as the Fisser (2024) Sentinel-2 results.

### Reference set

Reference polygons are the set of labeler-validated iceberg outlines produced in `iceberg-labeler`. For every chip whose assignment status is complete, we retain polygons with `PolygonDecision.action` in {accepted, modified, added} from results with `chip_verdict` in {accepted, edited}. Chips carrying any of the tags {cloud, ambiguous, land-edge, melange} are dropped, because those tags flag invalid ground truth. The `sea-ice` and `dark-water` tags are retained and recorded, since those are physical study conditions, not labels of poor annotation. Each polygon is converted from chip-pixel coordinates back to the chip's UTM reference frame via the inverse of the chip's rasterio affine, and polygons shorter than 40 m in root length (sqrt of area) are discarded to match the S2 10 m ground sample distance and the size-filter convention of plan.md.

### Metrics

Let $A_\mathrm{S2}$ be a predicted iceberg area and $A_\mathrm{ref}$ the matched reference area. For each matched pair we compute the relative error (after Fisser and others, 2024, eqn 2):

$$\mathrm{RE} = 100 \cdot \frac{A_\mathrm{S2} - A_\mathrm{ref}}{A_\mathrm{ref}} \quad (\%) \tag{2}$$

alongside two absolute error metrics and the segmentation Intersection over Union (IoU) already used during matching:

$$\mathrm{AE}_\mathrm{area} = |A_\mathrm{S2} - A_\mathrm{ref}| \quad (\text{m}^2)$$

$$\mathrm{AE}_\mathrm{root} = |\sqrt{A_\mathrm{S2}} - \sqrt{A_\mathrm{ref}}| \quad (\text{m})$$

$$\mathrm{IoU} = \frac{|P \cap R|}{|P \cup R|}.$$

Positive RE implies Sentinel-2 overestimation. Per-bin Mean Absolute Error (MAE) is the mean of $\mathrm{AE}_\mathrm{area}$ or $\mathrm{AE}_\mathrm{root}$; root-length MAE is preferred for cross-size comparison because it does not amplify with the square of iceberg linear scale.

### Matching

Within each chip the reference and predicted polygons are paired by Hungarian assignment on the cost $1 - \mathrm{IoU}$, computed with `scipy.optimize.linear_sum_assignment`. Pairs with IoU below 0.3 are discarded as unmatched. If any reference polygon overlaps two or more predicted polygons with IoU $\geq$ 0.1, the group is flagged ambiguous and excluded from RE to avoid biasing the per-pair statistic with split or merged detections (Fisser and others, 2024, report the analogous exclusion for hand-curated pairs). Unmatched references enter detection statistics as false negatives; unmatched predictions enter as false positives. For each (method, region, SZA bin) combination we report the match rate n_matched / n_ref as a selection-bias disclosure, so that readers can judge how much of each method's polygon inventory enters the RE estimate.

### Size and SZA aggregation

Per-pair metrics are stratified two ways.

1. Size (Fisser and others, 2024, Fig 3a-c analog): pairs are grouped by reference root length into log$_2$ buckets [40, 80), [80, 160), [160, 320), [320, 640), [640, $\infty$) m. For each method we report n_pairs, mean and median RE, the 25th and 75th percentiles, area MAE, root-length MAE, and mean IoU per bucket.

2. Solar zenith angle (Fisser and others, 2024, Fig 3d analog): per-chip SZA is taken from the `Sun_Angles_Grid` in the Sentinel-2 tile metadata (`MTD_TL.xml`), bilinearly interpolated at the chip centroid, with pysolar from the acquisition time and chip centroid latitude and longitude as fallback and cross-check. Per-degree median RE is computed over integer degree bins across the observed range (approximately 60 to 78$^\circ$), linearly interpolated to fill intermediate integer degrees within the sampled range (Fisser and others, 2024, eqn 3), then smoothed with a 5$^\circ$ centered running mean whose window collapses to 3 or 4 samples at the edges (Fisser and others, 2024, eqn 4). The interquartile range (25th and 75th percentile RE) is smoothed with the same filter and plotted as a shaded band.

### Non-comparability of Fisser equation 5

Fisser and others (2024, eqn 5) further standardize the smooth SZA-dependent error to a 56$^\circ$ reference anchored in an independent Dornier aerial survey. Our dataset carries no Dornier-equivalent calibration set, so applying equation 5 with our own visually delineated reference would be circular (the anchor and the thing standardized share the same source). We therefore report raw, interpolated, and smoothed RE, but do not compute the standardized SRE. Our accessible SZA range (approximately 60 to 78$^\circ$) is a subset of Fisser's 45 to 81$^\circ$; we perform no extrapolation outside the sampled range.

### Expected qualitative signal

UNet++ is trained to separate iceberg from ocean only, with shadow merged before analysis (plan.md §PR-7). As SZA increases the shadow that accompanies each iceberg grows, but UNet++ continues to exclude it, so we expect UNet++ RE to trend negative with rising SZA. The B08 fixed threshold and per-chip Otsu both respond to reflectance alone and include the high-reflectance iceberg edges that bleed into shadow pixels when the shadow darkens; we expect their RE to trend positive with rising SZA. DenseCRF, when available, sharpens UNet++ boundaries and is expected to track UNet++ closely. Quantifying the sign and magnitude of this SZA-dependent split is the primary contribution of Section 2.12.

No ground-truth iceberg area measurements independent of photointerpretation are available for direct accuracy validation. Using visually delineated labels as the reference inherits the interpreter's identification of the iceberg/water edge in B08, which is itself SZA-dependent at the highest angles. This limitation is shared with Fisser and others (2024, Greenland leg), who use the same class of reference. The match-rate and IQR statistics disclose the resulting uncertainty.
