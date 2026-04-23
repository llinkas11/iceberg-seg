# Student Annotation Guide

## Project Overview

This project uses satellite images to identify and outline icebergs in Arctic
waters. Our goal is to improve a computer vision model so it can better detect
icebergs in low-illumination images, where shadows are longer and the iceberg
boundaries are harder to see.

You will help by:

- setting up a Roboflow account and workspace
- training an initial model from an existing labeled dataset
- using that model to help annotate new low-illumination images
- checking and correcting the model's predictions

No prior experience with Roboflow is required.

## What You Will Label

There are two classes in this project:

- `Iceberg`: the visible iceberg body
- `Shadow`: the dark shadow cast by the iceberg

Do not label:

- open water
- land
- unrelated dark water patches
- sea ice or melange unless specifically instructed

## Why This Matters

Accurate labels help us train a better model. The model can then suggest
annotations on new images, which saves time, but those suggestions still need
human review.

Your job is to make sure the labels are correct.

## Session 1: Classroom Setup and Model Training

### Goal

Create your Roboflow account, join the project workflow, upload the labeled
dataset, and train a first model.

### What You Will Do

1. Create a Roboflow account.
2. Join or create your assigned workspace.
3. Name your workspace or group clearly:
   - `group1`
   - `group2`
   - `group3`
4. Upload the provided labeled dataset.
5. Share workspace access with Lulu's email.
6. Start model training in Roboflow.

### Support During Session 1

- Lulu will share her screen and demonstrate each step.
- Shibali will circulate and help with technical issues.

## Session 2: Schiller Labeling Session

### Goal

Use the trained model to help annotate new low-illumination satellite images.

### What You Will Do

1. Open your assigned image set in Roboflow.
2. Review the model-generated annotations.
3. Correct any mistakes.
4. Add missing iceberg or shadow labels.
5. Remove labels that are wrong.

### Important Rule

The model is a starting point only. Every annotation must be checked by a
person.

## How To Label Correctly

### Label as `Iceberg`

Use the `Iceberg` label for the visible above-water portion of the iceberg.

### Label as `Shadow`

Use the `Shadow` label for the dark region cast by an iceberg when it is clearly
associated with that iceberg.

### Do Not Label

Do not label:

- open water
- shoreline or land
- dark regions that are not true iceberg shadows
- sea ice, melange, or other confusing surface texture unless instructed

## Common Confusions

### Iceberg vs. Shadow

- The iceberg body is usually brighter and more structured.
- The shadow is darker and attached to or adjacent to the iceberg.

### Shadow vs. Open Water

- Open water may appear dark, but it should not be labeled unless it is clearly
  the shadow cast by an iceberg.

### Iceberg vs. Sea Ice / Melange

- If the object is not a distinct iceberg, do not label it as one.

## Reference Materials

A PowerPoint will be provided during the labeling session showing examples of:

- iceberg labels
- shadow labels
- confusing examples
- correct vs. incorrect annotations

Use the PowerPoint as your visual guide whenever you are unsure.

## Group Organization

Each student pair will be assigned a folder or image set.

Please keep track of:

- your group name
- your assigned folder
- which images you completed
- any uncertainties or edge cases

A shared tracking sheet will be used to record progress.

## When You Are Unsure

If you are uncertain about a label:

- check the PowerPoint reference
- compare the object to nearby examples
- ask Lulu or Shibali
- leave a note if needed rather than guessing

It is better to ask than to introduce incorrect labels into the dataset.

## Quick Checklist

Before you finish an image, ask:

- Did I label only iceberg bodies and shadows?
- Did I avoid labeling open water or unrelated dark regions?
- Did I correct obvious model mistakes?
- Did I add any missing iceberg or shadow regions?
- Did I record my progress in the tracking sheet?

## Thank You

Your annotations directly improve the quality of the dataset and the model. This
work makes it possible to study iceberg detection more accurately in challenging
low-light conditions.
