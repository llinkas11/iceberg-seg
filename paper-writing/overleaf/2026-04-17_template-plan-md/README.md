# 2026-04-17_template-plan-md

**Date:** 17 April 2026
**Summary:** Fresh placeholder-only IGS-class template for the restarted paper. All prior session folders are archived under `../../_archive/overleaf-sessions/`. Structure primarily follows the IGS class demo; section organisation follows Fisser (2024). All content is placeholder comments with inline references to `plan.md` + `reference/*.md` + `iceberg-rework-README.md`.

## Contents

| File | Size | Purpose |
|------|------|---------|
| `main.tex` | ~9 KB | Full IGS-class skeleton with placeholder comments only. Zero prose. |
| `references.bib` | <1 KB | Empty BibTeX file with format-guidance comments |
| `igs.cls` | 4.3 KB | JGlac/AOG class file (loads natbib automatically) |
| `igs.bst` | 29 KB | IGS BibTeX style |

## Document class and options

```latex
\documentclass[review,jog]{igs}
```

- `review`: single-column submission layout.
- `jog`: Journal of Glaciology (use `aog` for Annals).
- Remove `review` temporarily to check equation width against two-column production layout.

## Section structure

Unnumbered by default. Follows Fisser (2024) organisation:

1. Introduction
2. Study area and data
3. Methods
4. Results
5. Discussion
6. Conclusion
+ Acknowledgements, References

Each section body is a block of `% TODO` comments that:

- point to authoritative sources for numbers (`plan.md`, `reference/*.md`, `iceberg-rework-README.md`)
- flag which subsection of each scaffolding file to pull from (`introduction_draft.md`, `methods_draft.md`, `introduction_outline.md`, `tiny_icebergs_methods_addendum.md`)
- warn when not to copy numeric claims without reconciling first

## What authors must still fill in

1. **Title** (both long and running-head short form).
2. **Author block**: names with surnames in CAPS, affiliation footnotes, corresponding-author email.
3. **Abstract** (≤200 words, counted with `wc -w` on the body).
4. **All section bodies** (Introduction through Conclusion).
5. **`references.bib`** populated from `../../introduction_outline.md` reference list.
6. **Acknowledgements**.
7. Any **figures/tables** (pending results from `plan.md` Steps 9-10).

The template does NOT lift prose from the existing `introduction_draft.md` / `methods_draft.md` / `introduction_outline.md`. Those scaffolding files carry 2026-04-15/16 language that predates the `plan.md` pivot (binary segmentation, 40 m filter, `v3_balanced`). Each has a reconciliation header at the top flagging which claims are superseded. Read those headers before copying prose.

## Authoritative sources (referenced throughout `main.tex`)

| Source | What it provides |
|---|---|
| `../../plan.md` | Project state, methodological decisions, dataset splits, remaining pipeline steps |
| `../../iceberg-rework-README.md` | Folder layout, data sources, v3_clean / v3_balanced tables, pipeline diagram |
| `../../reference/descriptive_stats_results_discussion.md` | Dataset composition, size distribution, meteorological/temperature characterisation, Fisser (2025) comparison |
| `../../reference/b08_analysis_results_discussion.md` | B08 reflectance by SZA bin, iceberg/ocean/contrast characterisation, annotation-aware IC filter justification (§3.1-3.6) |

## Deploy to Overleaf

1. Open the existing Overleaf project.
2. Delete current `main.tex`, `references.bib`, and any existing `igs.cls` / `igs.bst` in Overleaf.
3. Drag-drop all four files from this folder into the Overleaf file tree.
4. Click Recompile.

Expected: clean compile, one-column review layout, title/author/abstract placeholders visible, empty sections show just their headings.

## Changes from previous Overleaf sessions

Both prior Overleaf session folders (`2026-04-17_initial-import`, `2026-04-17_restart-with-igs-class`) are archived at `../../_archive/overleaf-sessions/`. They carried old 3-class / 323-chip / 240-scene / Otsu-NDWI / DenseCRF-live framings that predate `plan.md`. This template starts from scratch against the authoritative sources.

## Compile status

Not yet tested. Update this section after first Overleaf recompile.

## Changes (2026-04-28)

- Loaded `booktabs` for the Results tables.
- Replaced the `% TODO` placeholders in the Results section with six
  populated tables and accompanying narrative:
  - Table 1: per-pair root-length MAE on v4_clean baseline (4 SZA bins).
  - Table 2: per-pair mean IoU on v4_clean baseline.
  - Table 3: detection statistics on v4_clean baseline.
  - Table 4: per-pair root-length MAE on v4_raw baseline (parallel paper
    table, no 40 m filter, no IC mask).
  - Table 5: per-pair mean IoU on v4_raw baseline (paired with Table 4).
  - Table 6: A0 vs A2 head-to-head, isolating the preprocessing pipeline
    on the lt65 split.

## Changes (2026-04-30)

- Added the top-hat small-iceberg recovery subsection (replaces its prior
  `% TODO`). Three new tables fed by `tophat_recover.py` over the
  baseline_v1 trained checkpoint:
  - Table 7: per-pair root-length MAE on the six +TH method outputs.
  - Table 8: per-pair mean IoU on the +TH outputs.
  - Table 9: detection statistics on the +TH outputs.
- Numbers traceable to
  `iceberg-rework/runs/exp_baseline_v1/20260424_185158/per_iceberg_TH/`.

- Inserted five paper figures via `\begin{figure}` blocks in `main.tex`
  pointing at PNGs under `paper-writing/figures/fig-archive/`:
  - **Fig. 1** (`fig01_annotation_difficulty`) at §2.3 Chip filtering.
    Three rows / four columns showing image, preliminary annotation,
    cleaned binary annotation, and a per-row note for one Fisser low-SZA
    chip, one Roboflow high-SZA chip, and one IC-masked ambiguous chip.
  - **Fig. 2** (`fig02_dataset_workflow`) at §2.4 Annotation sources.
    Two-source pipeline with shadow merge, 40 m component filter, IC
    pixel mask, and binary train / validation / test split.
  - **Fig. 3** (`fig03_method_schematic`) at the top of §3 Methods.
    Branching tree of the six segmentation methods on one test chip.
  - **Fig. 4** (`fig04_evaluation_schematic`) at §3.7 Evaluation.
    Hungarian-matching schematic with synthetic GT and pred components,
    matched-pair IoU labels, FN/FP markers, and a results table.
  - **Fig. 5** (`fig05_progression`) at the top of §4 Results, in a
    `figure*` (full text width). Phase A dataset progression chain plus
    Phase B six-method sweep on the frozen Phase A winner.
- Diagrammatic figures (2-5) generated by four new scripts under
  `iceberg-rework/scripts/`: `make_figure02_dataset_workflow.py`,
  `make_figure03_method_schematic.py`,
  `make_figure04_evaluation_schematic.py`,
  `make_figure05_progression.py`. Shared box / arrow primitives in
  `_diagram_helpers.py`. All five figures route through
  `_fig_registry.write` into the consolidated
  `paper-writing/figures/fig-archive/` directory.
- Numbers traceable to `iceberg-rework/runs/exp_baseline_v1/20260424_185158/`,
  `runs/exp_baseline_v1_raw/20260428_092402/`,
  `runs/exp_A0_fisser_lt65_original/20260428_094028/`, and
  `runs/exp_A2_our_lt65/20260428_094654/` per_iceberg/ outputs.

## Changes (2026-05-05)

- Added a "Supplementary material" section before Acknowledgements, declaring
  Fig. S1 (`figS01_otsu_floor_distribution`): per-chip Otsu threshold
  distribution on Sentinel-2 B08 over all 23,981 study chips, with skip-rates
  at floors 0.10 / 0.15 / 0.20 (6.08 % / 41.02 % / 50.54 %) and the
  noise-floor spike from the L1C +0.10 offset.
- Expanded the OT method paragraph in §3 to cite Fig. S1 and justify the
  0.10 floor in offset-uncorrected reflectance over the alternative 0.20
  (offset-corrected) interpretation.
- Source script for Fig. S1:
  `iceberg-rework/scripts/script_check_answers/q07_otsu_floor_distribution.py`.
  Output PNG archived under
  `paper-writing/figure_review/script_check_answers/q07_otsu_floor_distribution/`.

## Next steps

1. Deploy to Overleaf and confirm clean compile.
2. Populate author block and running-head title.
3. Convert `../../introduction_outline.md` reference list into `references.bib` entries.
4. Draft Introduction section (CARS structure per outline).
5. Draft Study area and data, Methods sections pulling from reconciled scaffolding.
6. Populate Results after `plan.md` Steps 9-10 complete.
