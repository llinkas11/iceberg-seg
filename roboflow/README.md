# Roboflow Workflow

This folder is a starter workspace for Step 3 of the low-sun-angle iceberg project:

1. Run a pretrained model locally to generate predicted masks.
2. Convert those masks into a Roboflow upload package.
3. Upload the package into a Roboflow semantic segmentation project.
4. Use Roboflow's SAM-powered annotation tools to refine masks.
5. Export the corrected labels for evaluation or fine-tuning.

## Verification Tool

Use `check_mask_labels.py` before exporting if you need to verify the source mask
encoding:

```bash
python3 roboflow/check_mask_labels.py \
  --mask /path/to/example_ground_truth.tif
```

Or inspect several masks from a folder:

```bash
python3 roboflow/check_mask_labels.py \
  --mask-dir /path/to/masks \
  --limit 10
```

This is especially useful when checking whether:

- `1 = iceberg, 2 = shadow`
- or `1 = shadow, 2 = iceberg`

If you need a visual check, generate an overlay preview:

```bash
python3 roboflow/preview_mask_classes.py \
  --image /path/to/chip.tif \
  --mask /path/to/chip_ground_truth.tif \
  --out roboflow/mask_preview.png
```

## Recommended Project Setup in Roboflow

- Project type: `Semantic Segmentation`
- Class list:
  - `iceberg`

This workflow now assumes an iceberg-only annotation project. If your source masks
still contain shadow or other auxiliary values, leave them out of `--class-map`
when exporting and only iceberg polygons will be written into the COCO JSON.

## Folder Convention

This script expects ordinary image files and their corresponding predicted masks:

```text
roboflow/
  README.md
  prepare_coco_from_masks.py

some_prediction_run/
  images/
    scene_001.png
    scene_002.png
  masks/
    scene_001.png
    scene_002.png
```

Each mask should:

- have the same width and height as its image
- use integer pixel values for class IDs
- share the same base filename as the source image

Example:

- `images/scene_001.png`
- `masks/scene_001.png`

## Quick Start

If your prediction masks are already saved as PNG/TIF label rasters:

```bash
python3 roboflow/prepare_coco_from_masks.py \
  --images-dir path/to/images \
  --masks-dir path/to/masks \
  --out-dir roboflow/upload_batch_01 \
  --class-map 1:iceberg
```

If you run `predict_tifs.py`, use the original chip folder as `--images-dir` and the
`geotiffs/` output folder as `--masks-dir`, with `--mask-suffix _pred`:

```bash
python3 roboflow/prepare_coco_from_masks.py \
  --images-dir S2-iceberg-areas/chips/KQ/sza_70_75/tifs \
  --masks-dir S2-iceberg-areas/area_comparison/KQ/sza_70_75/unet/geotiffs \
  --out-dir roboflow/sep_oct_bin_70_75_batch_01 \
  --class-map 1:iceberg \
  --mask-suffix _pred
```

If your masks contain both iceberg and shadow classes but you only want iceberg in Roboflow:

```bash
python3 roboflow/prepare_coco_from_masks.py \
  --images-dir path/to/images \
  --masks-dir path/to/masks \
  --out-dir roboflow/upload_batch_01 \
  --class-map 1:iceberg
```

You can omit `--class-map` entirely and the script will default to `1:iceberg`.

If iceberg and shadow live in separate mask values but you want Roboflow to see
their full combined footprint as a single `iceberg` class, merge them during export:

```bash
python3 roboflow/prepare_coco_from_masks.py \
  --images-dir path/to/images \
  --masks-dir path/to/masks \
  --out-dir roboflow/upload_batch_01 \
  --merge-values 3 4 \
  --merge-name iceberg
```

Then zip the output folder and upload it to Roboflow as a `COCO Segmentation`
dataset:

```bash
cd roboflow
zip -r upload_batch_01.zip upload_batch_01
```

## Output Layout

The export script creates:

```text
roboflow/upload_batch_01/
  images/
  _annotations.coco.json
```

This is ready to zip and upload.

## Existing Project-Specific Exporter

There is already a project-specific script at:

- `/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/S2-iceberg-areas/export_roboflow.py`

Use that script when your predictions are stored as:

- `predicted_masks.npy`
- pickled chip arrays such as `x_test.pkl`

Use `prepare_coco_from_masks.py` when you have normal image files plus normal mask
files on disk.

## Suggested Step 3 Annotation Workflow

1. Generate predicted masks from the pretrained CNN or UNet++ model.
2. Export a modest first batch, such as 20 to 50 scenes, into COCO format.
3. Upload to Roboflow.
4. Refine masks with Smart Polygon in Enhanced mode.
5. Manually correct missed bergs and false positives.
6. Export the corrected labels and keep a versioned folder per solar-zenith bin.

## Suggested Local Batch Naming

Use one folder per annotation batch:

- `roboflow/sep_oct_bin_65_70_batch_01`
- `roboflow/sep_oct_bin_70_75_batch_01`
- `roboflow/sep_oct_bin_75_plus_batch_01`

That makes it much easier to compare model behavior by solar zenith bin later.
