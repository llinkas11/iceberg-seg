# Upload Notes

## Confirmed Manual Annotation Package

This file is ready to use as the starting labeled dataset for Roboflow:

- `/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/S2-iceberg-areas/roboflow_manual_upload.zip`

Verified contents on 2026-03-22:

- 402 images
- 2820 segmentation annotations
- classes:
  - `iceberg`
  - `shadow`
- format:
  - COCO Segmentation

## Best Upload Path

For this labeled segmentation dataset, the cleanest approach is to upload the zip
through the Roboflow web import flow into project `icebergseg`.

Why:

- the archive already contains both images and `_annotations.coco.json`
- this matches the standard Roboflow dataset import workflow for segmentation
- the CLI flow quoted earlier is best thought of as image upload oriented

## Recommended Project Settings

- Project ID: `icebergseg`
- Task type: `Semantic Segmentation`
- Classes:
  - `iceberg`
  - `shadow`

## After Upload

1. Confirm the masks render correctly on a few images.
2. Generate a new version in Roboflow.
3. Train a baseline Roboflow model.
4. Use that model to assist annotation of September-October imagery.
