# Locator Workflow

This is the recommended Python-only workflow for creating final example panels
with a Greenland locator map.

## Single example

```bash
python3 roboflow/preview_mask_classes.py \
  --aoi-gpkg /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/S2-iceberg-areas/aois_greenland_area_distributions.gpkg \
  --original-image /path/to/chip.tif \
  --image /path/to/roboflow_export.png \
  --mask /path/to/chip_ground_truth.tif \
  --out /path/to/output_preview.png \
  --label-map 2:shadow 3:iceberg \
  --show-values 2 3
```

## Batch mode

Generate a batch of previews directly from the AOI GeoPackage:

```bash
python3 roboflow/make_example_previews.py \
  --pngs S2-iceberg-areas/roboflow_manual_upload_fixed/*.png \
  --aoi-gpkg S2-iceberg-areas/aois_greenland_area_distributions.gpkg \
  --out-dir roboflow/roboflow_examples
```

If you ever want to override the Python-generated locator panel with a
pre-rendered one, you can still use:

```bash
python3 roboflow/make_example_previews.py \
  --pngs S2-iceberg-areas/roboflow_manual_upload_fixed/*.png \
  --locator-dir roboflow/locator_maps \
  --out-dir roboflow/roboflow_examples
```

## Resulting panel order

- locator map
- original color
- shadow
- iceberg
