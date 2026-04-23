# Example Chip Workflow

This is the exact end-to-end workflow for the example chip
`S2A_MSIL1C_20160725T145922_N0500_R125_T22WES_20231004T134617_pB5_9_13_`.

## 1. Infer the fjord prefix

```bash
python3 roboflow/infer_chip_region.py \
  --chip "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/S2-iceberg-areas/S2UnetPlusPlus/imgs/S2A_MSIL1C_20160725T145922_N0500_R125_T22WES_20231004T134617_pB5_9_13_.tif" \
  --aoi "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/S2-iceberg-areas/aois_greenland_area_distributions.gpkg"
```

Expected output:

```text
matching_regions=NK-UF
region_prefixes=NK
```

## 2. Create the locator map in MATLAB

```matlab
make_locator_map( ...
    "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/S2-iceberg-areas/S2UnetPlusPlus/imgs/S2A_MSIL1C_20160725T145922_N0500_R125_T22WES_20231004T134617_pB5_9_13_.tif", ...
    "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/S2-iceberg-areas/aois_greenland_area_distributions.gpkg", ...
    "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/roboflow/locator_map_example.png", ...
    "NK");
```

## 3. Build the final preview with the MATLAB locator panel

```bash
python3 roboflow/preview_mask_classes.py \
  --locator-image "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/roboflow/locator_map_example.png" \
  --original-image "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/S2-iceberg-areas/S2UnetPlusPlus/imgs/S2A_MSIL1C_20160725T145922_N0500_R125_T22WES_20231004T134617_pB5_9_13_.tif" \
  --image "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/S2-iceberg-areas/roboflow_manual_upload_fixed/S2A_MSIL1C_20160725T145922_N0500_R125_T22WES_20231004T134617_pB5_9_13_.png" \
  --mask "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/S2-iceberg-areas/S2UnetPlusPlus/masks/S2A_MSIL1C_20160725T145922_N0500_R125_T22WES_20231004T134617_pB5_9_13__ground_truth.tif" \
  --out "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/roboflow/roboflow_examples/S2A_MSIL1C_20160725T145922_N0500_R125_T22WES_20231004T134617_pB5_9_13__preview.png" \
  --label-map 2:shadow 3:iceberg \
  --show-values 2 3
```

## Final panel order

- locator map
- original color
- shadow
- iceberg
