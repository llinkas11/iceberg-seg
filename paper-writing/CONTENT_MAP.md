# Paper content map: main vs supplementary

This file is the index of where every piece of paper content lives, split by main-text vs supplementary material. Existing files stay in their current locations; this map is the navigation layer. Update whenever a new figure, table, or prose block is added.

For Journal of Glaciology (IGS) Article submission. Style rules in `paper-writing/CLAUDE.md`. Source LaTeX in `overleaf/2026-04-17_template-plan-md/main.tex`.

## Main paper

The "final paper" surface that goes into the IGS submission PDF. The Overleaf folder is the source of record for the rendered output; the prose drafts in `paper-writing/*.md` are the working documents that feed `main.tex`.

### Prose

| File | Section | Status |
|------|---------|--------|
| `introduction_draft.md`, `introduction_outline.md` | 1. Introduction | Draft |
| `methods_draft.md` | 2. Methods | Scaffolding (per `CLAUDE.md`); reconcile against `plan.md` and `reference/*.md` before lifting into LaTeX |
| `results.md` | 3. Results | Draft (per-pair tables on baseline_v1, preprocessing as controlled variable) |
| `model_progression.md` | 2.13 Experimental progression (referenced from Methods) | Authoritative within scope |
| `overleaf/2026-04-17_template-plan-md/main.tex` | Final LaTeX | Source of record for compilation |

### Figures (main text)

| Slug | Caption summary | Source script |
|------|------------------|----------------|
| `fig01_annotation_difficulty` | Annotation cleaning examples (shadow merge, 40 m filter, IC mask) | `iceberg-rework/scripts/make_figure01_annotation_difficulty.py` |
| `fig02_dataset_workflow` | Dataset construction workflow (Roboflow + Fisser to v4_clean) | `iceberg-rework/scripts/make_figure02_dataset_workflow.py` |
| `fig03_method_schematic` | Six-method comparison schematic on one chip | `iceberg-rework/scripts/make_figure03_method_schematic.py` |
| `fig04_evaluation_schematic` | Per-iceberg Hungarian matching diagram | `iceberg-rework/scripts/make_figure04_evaluation_schematic.py` |
| `fig05_progression` | Two-phase Phase A x Phase B progression diagram | `iceberg-rework/scripts/make_figure05_progression.py` |
| `mae_rootlen_vs_sza` | Per-pair MAE on root length, six methods x four SZA bins | `iceberg-rework/scripts/make_fig_mae_vs_sza.py` |
| `area_scatter` | Per-pair predicted vs reference area, four SZA panels | `iceberg-rework/scripts/make_fig_area_scatter.py` |
| `bias_delta_by_area` | Bias delta by reference area, six methods | `iceberg-rework/scripts/make_fig_bias_by_area.py` |
| `re_by_area_bin` | Relative error by area bin, six methods | `iceberg-rework/scripts/make_fig_re_by_area.py` |
| `outline_examples` | Per-SZA-bin outline examples (worst-pos, near-zero, worst-neg) | `iceberg-rework/scripts/make_fig_outline_examples.py` |

Index of record: `figures/figures.md` (live PNGs in `figures/fig-archive/`).

### Tables (main text)

| Table | Content | Source |
|-------|---------|--------|
| 1 | Per-pair root-length MAE, six methods x four SZA bins (baseline_v1) | `results.md` Section 3.3.1 |
| 2 | Per-pair IoU, six methods x four SZA bins (baseline_v1) | `results.md` Section 3.3.2 |
| 3 | Detection statistics, six methods (baseline_v1) | `results.md` Section 3.3.3 |
| 4 | Phase A 2x2 (preprocessing x null injection) on best val IoU | `results.md` Section 3.1.3 |
| 5 | Per-pair MAE on v4_raw (preprocessing-impact companion table) | `results.md` Section 3.4 |

## Supplementary material

Content that lives in the supplementary appendix (S1, S2, ...) rather than the main text. Per Journal of Glaciology style: figures and tables prefixed `S` (Fig. S2, Table S1), cited in the main text and grouped under a SUPPLEMENTARY MATERIAL heading near References.

Authoritative source for the dataset / backbone progression story: `shib_end_to_end/phase_a_higher_sza_t1_t4.md`. The supplementary prose in this paper-writing tree should track that artifact's T1, T1b, T2, T3, T4, T5.

### Prose

| File | Section | Status |
|------|---------|--------|
| `paper-writing/supplementary.md` | S1-S6 prose draft, all sections | Working Markdown draft |
| `paper-writing/overleaf/git-mirror/supplementary.tex` | LaTeX skeleton mirroring supplementary.md, Tables S1-S5 + Figs S1-S3 | Compiles standalone; embeds figures from `figures/supplementary/` |
| `tiny_icebergs_methods_addendum.md` | S2 (deferred branch) Tiny-iceberg annotation recovery | Scoped but not included in Phase B comparison |
| `shib_end_to_end/phase_a_higher_sza_t1_t4.md` | Authoritative source for T1-T5 | Single source of truth |
| `shib_end_to_end/phase_a_cleanup_audit.md` | Phase A lt65 audit (C1/C2 ablation) | Background, supplementary |
| `reference/descriptive_stats_results_discussion.md` | Dataset composition reference | Cited from Methods, not in supplementary appendix |
| `reference/b08_analysis_results_discussion.md` | B08 / IC filter justification | Cited from Methods, not in supplementary appendix |

### Figures (supplementary)

| Slug | Caption summary | Source script |
|------|------------------|----------------|
| `figS01_otsu_floor_distribution` | Per-chip Otsu floor distribution, 23,981 chips | `iceberg-rework/scripts/script_check_answers/q07_otsu_floor_distribution.py` |
| `figs02_phase_a_heatmap` | Phase A per-SZA-bin x 18-experiment IoU + MAE heatmap | `iceberg-rework/scripts/make_figS02_phase_a_heatmap.py` |
| `figs03_a0_vs_a1_by_sza` | Backbone comparison A0 vs A1 vs A7b paired bars | `iceberg-rework/scripts/make_figS03_a0_vs_a1_by_sza.py` |

Clean-named copies of the latest PNG of each supplementary figure live under `paper-writing/figures/supplementary/` and are mirrored into `paper-writing/overleaf/git-mirror/figures/supplementary/` for LaTeX compilation. The full timestamped archive remains in `figures/fig-archive/`.

Index of record: same `figures/figures.md` "Supplemental figures" section. The supplementary slug convention is lowercase `figs<N>` (registry constraint) but in-paper labelling is `Figure S<N>` per IGS style.

### Tables (supplementary)

| Table | Content | Source |
|-------|---------|--------|
| S1 | T1: Phase A 10 originals + 8 A1-anchored variants, per-bin per-experiment | `shib_end_to_end/phase_a_higher_sza_t1_t4.md` Sections T1, T1b |
| S2 | T2 revised: best non-A0 per bin (A7b winner) | Same artifact, Section T2 (revised) |
| S3 | T3 base + TH: Phase B backbone comparison (A0 vs A1 vs A7b, 12 methods, 4 SZA bins) | Same artifact, Section T3 |
| S4 | T4: Recommended retrieval pipeline per SZA bin (per-bin best vs cross-bin pick) | Same artifact, Section T4 |
| S5 | T5: Top-hat effect deltas (MAE and IoU, base to TH) | Same artifact, Section T5 |

## What is NOT in either bucket

- `plan.md`: project-state ledger; never appears in the paper.
- `handoff.md`: cross-session handoff document; never appears in the paper.
- `refactor_plan.md`, `iceberg-rework-README.md`: project-internal notes.
- `figure_review/`: review-cycle materials (checklists, pre-meeting decks).
- `_archive/`: frozen old session content.

## Editing protocol

- A figure or table's status (main vs supplementary) is decided here; do not infer from filename alone.
- When promoting a supplementary item to main text (or vice versa), update this file in the same commit that touches the figure / table source.
- Slugs follow the registry regex `^[a-z0-9_]+$`. The `S` prefix in IGS labelling is applied at LaTeX caption time (`\caption{\textbf{Fig. S2.} ...}`), not in the slug.
