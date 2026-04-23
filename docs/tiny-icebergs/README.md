# Tiny-Icebergs Workflow

This folder documents the local-first review workflow used to find likely missed tiny icebergs in Sentinel-2 chips without rewriting the raw Roboflow export in place.

## Purpose

- Review existing annotations and add candidate tiny icebergs in a transparent way.
- Keep the raw Roboflow export unchanged and perform corrections in a separate working copy.
- Compare a fixed NIR threshold workflow against an Otsu-based variant on the same chips.

## Current recommendation

The current preferred review setup is the fixed-threshold workflow, not Otsu.

- Threshold method: fixed NIR threshold
- `nir_threshold = 0.30`
- `min_pixels = 2`
- `max_pixels = 32`
- `large_bright_threshold = 0.22`
- `large_region_min_pixels = 100`
- `large_region_buffer = 2`

This setup performed best in manual visual review because it recovered small bright iceberg candidates while avoiding many false positives on broad bright land-like features.

## Folder roles

- `fixed-threshold/`
  Working notes and scripts for the fixed-threshold review workflow.
- `otsu/`
  Matching workflow that swaps the fixed threshold for a per-chip Otsu threshold.
- `METHODOLOGY.md`
  Full description of data sources, corrections, and review logic.

## Important data notes

- The new Roboflow export `final-labeling-1` is a high-SZA subset only.
- It contains `588` chips, all from `sza_65_70`, `sza_70_75`, and `sza_gt75`.
- There are no `sza_lt65` chips in that download.
- A corrected working copy called `final-labeling-1_fixednulls` was created to preserve the raw download while fixing known null-label errors.

## Output style

The review outputs are triptychs:

1. Original black-and-white NIR chip
2. Original chip with existing annotations in blue
3. Original chip with existing annotations in blue and new threshold candidates in red

That format is intended for fast visual QA with an advisor or annotator.
