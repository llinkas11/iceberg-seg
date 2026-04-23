# MATLAB Locator Map

Use `make_locator_map.m` to generate a fjord locator panel for a georeferenced
chip.

Example:

```matlab
make_locator_map( ...
    "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/S2-iceberg-areas/S2UnetPlusPlus/imgs/S2A_MSIL1C_20160725T145922_N0500_R125_T22WES_20231004T134617_pB5_9_13_.tif", ...
    "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/roboflow/aoi_boxes.csv", ...
    "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/roboflow/locator_map_example.png", ...
    "NK");
```

Suggested region prefixes:

- `KQ`
- `SK`
- `NK`
- `UQ`
- `DB`
- `II`
- `SQ`

If you are unsure which prefix to use for a chip, infer it with:

```bash
python3 roboflow/infer_chip_region.py \
  --chip /path/to/chip.tif \
  --aoi /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/S2-iceberg-areas/aois_greenland_area_distributions.gpkg
```

This script:

- opens a topographic basemap
- draws the fjord AOI outlines for the selected region
- outlines the chip in red
- labels the region
- writes the locator panel to a PNG

For the example chip above, the inferred match is:

- `matching_regions=NK-UF`
- `region_prefixes=NK`
