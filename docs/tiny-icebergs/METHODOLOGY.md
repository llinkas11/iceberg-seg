# Tiny-Iceberg Review Methodology

## Goal

The tiny-iceberg workflow is a review and label-refinement process for chips where existing annotations may miss very small icebergs. The workflow is designed to be simple, inspectable, and easy to rerun while keeping the raw annotation export unchanged.

## Data sources

Two annotation sources are used together:

1. Older manually annotated COCO export for `sza_lt65`
2. New Roboflow export `final-labeling-1` for the higher-SZA bins

The newer Roboflow export was downloaded as a raw snapshot and kept unchanged in a separate folder. A metadata lookup table was built from `split_log.csv` so each chip could be linked to its solar-angle bin without moving files inside the raw export.

### High-SZA Roboflow subset

The downloaded `final-labeling-1` export contains `588` chips:

- `138` chips in `sza_65_70`
- `166` chips in `sza_70_75`
- `284` chips in `sza_gt75`

It contains no `sza_lt65` chips, so the low-angle bin still comes from the older annotation source.

## Dataset corrections

### Null-scene correction

A corrected working copy called `final-labeling-1_fixednulls` was made from the raw Roboflow export.

In that corrected copy, all annotations were removed for files whose names start with:

`S2A_MSIL1C_20161107T141402_N0500_R053_T24WWT_20230921T211238`

Those chips should be treated as null examples with no annotations. The images remain in the dataset, but their COCO annotation lists are empty.

### Malformed annotation handling

At least one chip in the older export contained inconsistent COCO segmentation formats:

- most instances were stored as RLE masks
- one instance used malformed polygon coordinates

To make the review visualizations reliable, the review code was updated to:

- decode RLE masks correctly
- rasterize polygon segmentations when valid
- skip malformed polygon coordinate lists instead of drawing them

This prevented visualization artifacts from being mistaken for thresholding errors.

## Review image format

Each saved review image is a three-panel triptych:

1. the original black-and-white NIR chip
2. the same chip with existing annotations in blue
3. the same chip with existing annotations in blue and new threshold candidates in red

This format was chosen to make manual review readable without hiding the underlying NIR texture.

## Fixed-threshold workflow

The fixed-threshold workflow starts from the NIR band of each chip and builds red candidate annotations using connected components.

### Candidate logic

1. Read the NIR band from the chip `.tif`
2. Convert the existing COCO annotations into a chip-sized mask
3. Identify bright candidate pixels using a fixed NIR reflectance threshold
4. Remove any candidate pixels that overlap existing annotations
5. Group remaining candidate pixels into connected components
6. Keep only components whose sizes fall in the chosen small-object range

### Large-bright-region guard

To reduce false positives on land-like bright features and large bright surfaces, a separate blocking mask is built before the final red candidates are accepted.

1. Find connected bright regions above a secondary brightness threshold
2. Keep only regions larger than a chosen minimum size
3. Buffer those large bright regions by a few pixels
4. Forbid new red annotations inside that buffered mask

This guard proved important when looser threshold settings began adding red detections on bright land or broad bright coastal texture.

### Current preferred settings

The current preferred fixed-threshold settings are:

- `nir_threshold = 0.30`
- `min_pixels = 2`
- `max_pixels = 32`
- `large_bright_threshold = 0.22`
- `large_region_min_pixels = 100`
- `large_region_buffer = 2`

Manual review favored `0.30` as the most stable fixed threshold. Lower values recovered more candidates but also introduced more obvious false positives.

## Otsu workflow

An Otsu-based version of the same workflow was built for comparison.

The Otsu variant keeps the same:

- existing-annotation exclusion
- connected-component filtering
- large-bright-region guard
- triptych output format

The only difference is how the brightness threshold is chosen:

- instead of using one fixed NIR threshold, it computes a per-chip Otsu threshold on the pixels that are not already annotated and not blocked by the large-bright-region guard
- the computed threshold is clipped to a user-defined floor and ceiling

This makes Otsu useful for sensitivity checks, but visual review so far has favored the fixed `0.30` threshold for consistency and interpretability.

## Why the fixed threshold is preferred right now

The fixed-threshold workflow is easier to explain and more stable across chips:

- it follows the same basic physical intuition as the NIR threshold literature
- it avoids chip-to-chip threshold swings
- it was easier to tune by eye with the advisor
- `0.30` performed better than the tested Otsu variants during the current qualitative review stage

## Relationship to retraining

This workflow is still a review and refinement stage. It is not yet a finalized training-data rebuild. The intended order is:

1. finish visual QA on tiny-iceberg candidates
2. decide which corrected or added labels should be kept
3. rebuild training data if the additions are accepted
4. retrain or fine-tune the segmentation model on the revised dataset
