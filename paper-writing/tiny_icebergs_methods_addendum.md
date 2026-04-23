# Tiny-Iceberg Methods Addendum

This note summarizes the small-iceberg annotation-refinement workflow developed after the main v2 retraining dataset was assembled.

## Purpose

During manual review we found that some existing annotations likely miss very small icebergs. We therefore built a separate review workflow to propose candidate additions without modifying the raw Roboflow export directly.

## Data handling

The newer Roboflow project export (`final-labeling-1`) was downloaded as a raw snapshot and kept unchanged. A second working copy (`final-labeling-1_fixednulls`) was created for corrections. In that corrected copy, all annotations were removed for one known null scene prefix:

`S2A_MSIL1C_20161107T141402_N0500_R053_T24WWT_20230921T211238*`

The new Roboflow export contains only the higher-SZA bins (`sza_65_70`, `sza_70_75`, `sza_gt75`). The `sza_lt65` bin continues to use the older annotation source.

## Review output

Each reviewed chip is saved as a three-panel image:

1. black-and-white NIR chip
2. NIR chip with existing annotations in blue
3. NIR chip with existing annotations in blue and new threshold candidates in red

This format was used for rapid visual inspection and advisor feedback.

## Fixed-threshold candidate generation

Candidate tiny-iceberg annotations are generated from the NIR band of each chip. Existing annotations are first rasterized to a mask and excluded so that new candidates cannot overlap old labels. The remaining bright pixels are grouped into connected components, and only small components within a chosen size range are kept as candidate additions.

To suppress obvious false positives on broad bright land-like or shoreline features, a large-bright-region guard is applied before the final candidate mask is created. Bright connected regions above a second reflectance threshold are identified, filtered by minimum size, buffered outward by a few pixels, and excluded from the red-candidate search space.

The current preferred settings from qualitative review are:

- fixed NIR threshold: `0.30`
- candidate size range: `2–32` pixels
- large-bright-region threshold: `0.22`
- large-bright-region minimum size: `100` pixels
- large-bright-region buffer: `2` pixels

At this stage, the fixed-threshold workflow is preferred over Otsu because it produced more stable and easier-to-interpret candidate additions during visual review.

## Otsu comparison

A parallel Otsu-based workflow was retained for comparison. It uses the same existing-label exclusion, component-size filtering, and large-bright-region guard, but replaces the fixed NIR threshold with a per-chip Otsu threshold bounded by user-defined floor and ceiling values. The Otsu results are currently treated as a sensitivity analysis rather than the primary label-refinement path.

## Annotation-format handling

Some exported COCO annotations were stored as RLE masks rather than polygon coordinate lists, and at least one polygon record contained malformed coordinate strings. The review code was updated to decode RLE masks directly and to skip malformed polygon entries during visualization. This prevented visualization artifacts from being misread as thresholding errors.

## Current status

The tiny-iceberg workflow has been used for qualitative review only. No retrained model has yet been produced from these candidate additions. The next decision point is whether the reviewed additions should be merged into a revised training dataset for a new training run.
