---
description: Systematically update every IDS2026 artifact for a new finding (model run, threshold, dataset, eval, writing change, figure)
---

# /update-icebergs

When a new finding lands in the IDS2026 Sentinel-2 iceberg-area project (a new model run, a new threshold or parameter, a new dataset version, a new evaluation result, a writing-only edit, or a new figure or table), walk this checklist top to bottom to keep every artifact in sync. The project keeps state in many loosely-coupled places: `paper-writing/plan.md` (canonical state ledger), eight other source-of-truth markdowns, two `CLAUDE.md` guardrail files, an Overleaf LaTeX manuscript, ~14 pipeline scripts in `S2-iceberg-areas/`, the `iceberg-rework/` experimental tree, `shib_end_to_end/` audit files, figures + figure-archive + figure-review CSVs. Without a structured pass, stale checkpoint IDs, IoU values, chip counts, and threshold constants drift independently across the tree.

User-supplied finding description (free text, may be empty):

$ARGUMENTS

---

## Step 0: Parse the finding

Extract three things from the user's free-text above:

1. **Finding type** (exactly one):
   - `new-model` (new training run, new checkpoint, new val IoU)
   - `new-threshold-or-param` (B08 threshold, Otsu floor/ceil, prob_threshold, etc.)
   - `new-dataset` (new split, new chip set, new labeling pass, new manifest)
   - `new-eval` (new MAE / IoU numbers without retraining)
   - `writing-only` (prose edit, citation update, no numerical change)
   - `new-figure-or-table` (new figure added, table reorganized, slug renamed)
   - `other` (anything else; ask the user to clarify before proceeding)
2. **New values** (be exact: checkpoint ID, IoU, MAE, chip count, threshold, run date, units).
3. **Old values being replaced** (needed for Step 10's grep). If the user did not say what is being replaced, infer from `CLAUDE.md` §Project Context (current values: checkpoint `s2_20260227_231556`, val IoU `0.4398`, training chips `323`) but flag the inference and ask the user to confirm.

If `$ARGUMENTS` is empty, OR any of the three extractions are missing or ambiguous, ask **one** structured `AskUserQuestion` follow-up. Do not proceed past Step 0 until type, new values, and old values are all explicit.

## Step 1: Update `paper-writing/plan.md` first (always)

`paper-writing/plan.md` is the **authoritative state ledger**. Every other doc cross-references it. Edit it before anything else, regardless of finding type.

Convention (matches existing entries):
- **Append** a dated update block to the project-state header (top of file), after the most recent `**2026-MM-DD update:**` entry. Do not overwrite history.
- Format: `**YYYY-MM-DD update:** <one-paragraph summary stating what changed, what now wins, and where the supporting evidence lives>`.
- Update the `**Last verified:** YYYY-MM-DD` line.
- Touch tables in §"Resolved Prerequisites", §"Key Methodological Decisions", or §"Pipeline Stages" only if the finding actually changes that table's content.

After editing, re-read the head of `plan.md` to confirm the new dated block reads cleanly and `Last verified` is current.

## Step 2: Pick the section map

Use the finding type from Step 0 to pick which downstream sections to walk. Sections marked `yes` are required. Sections not listed are skipped. The user may override; print the planned scope before walking.

| Finding type | plan.md | CLAUDE.md | other prose | Overleaf | code | figures | review CSVs | runs_summaries | HPC sync | findings_log |
|---|---|---|---|---|---|---|---|---|---|---|
| new-model | yes | yes | results.md, methods_draft.md, model_progression.md, handoff.md | yes | yes (defaults) | re-render | yes | yes | yes | yes |
| new-threshold-or-param | yes | yes | methods_draft.md, results.md | yes | yes (constants) | re-render | yes | yes | yes | yes |
| new-dataset | yes | yes | methods_draft.md, model_progression.md, handoff.md | yes | maybe (paths) | maybe | yes | yes | maybe | yes |
| new-eval | yes | no | results.md | yes | no | re-render | yes | yes | no | yes |
| writing-only | yes (note only) | no | named docs only | yes | no | no | no | no | no | yes |
| new-figure-or-table | yes | no | CONTENT_MAP.md, results.md or methods_draft.md | yes | yes (fig source script) | yes | yes | no | no | yes |
| other | ask | ask | ask | ask | ask | ask | ask | ask | ask | yes |

Print the planned scope to the user before walking. Steps 1, 10, 12, and 13 always run.

## Step 3: Walk source-of-truth prose

Read each before editing. Edit only the docs the section map flagged.

Master list of prose docs:
- `paper-writing/plan.md` (already updated in Step 1)
- `paper-writing/CONTENT_MAP.md`: paper structure index. Update when figures/tables move main↔supplementary or when source-script paths change.
- `paper-writing/methods_draft.md`: methods scaffolding.
- `paper-writing/results.md`: per-pair tables, baseline_v1 results.
- `paper-writing/model_progression.md`: Phase A × B grid, experimental progression.
- `paper-writing/handoff.md`: session handoff summary.
- `paper-writing/supplementary.md`: S1 to S6 supplementary prose.
- `paper-writing/introduction_draft.md`, `paper-writing/introduction_outline.md`: introduction.
- `paper-writing/iceberg-rework-README.md`, `paper-writing/refactor_plan.md`: project-internal notes.
- `paper-writing/tiny_icebergs_methods_addendum.md`: tiny-iceberg branch (deferred from main Phase B).
- `shib_end_to_end/phase_a_higher_sza_t1_t4.md`: single source of truth for supplementary Tables T1 to T5.
- `shib_end_to_end/phase_a_cleanup_audit.md`, `shib_end_to_end/results.md`: cross-bin audit.
- `iceberg-rework/reference/descriptive_stats_results_discussion.md`, `iceberg-rework/reference/b08_analysis_results_discussion.md`: cited from Methods.

For `writing-only` findings, restrict to the docs the user named. For `new-eval`, results.md is the primary edit target. For `new-figure-or-table`, CONTENT_MAP.md is mandatory.

## Step 4: Update guardrails

- `CLAUDE.md` (project root): §"Project Context" block. Update model checkpoint, val IoU, training chip count, SZA bin definitions if changed.
- `paper-writing/CLAUDE.md`: only if writing conventions or IGS journal style changed.
- `iceberg-rework/plan.md`, `iceberg-rework/README.md`: secondary state docs; sync if dataset definition or methodology changed.

## Step 5: Update Overleaf LaTeX (do NOT push)

Edit locally. **Never** `git push` to Overleaf without explicit user approval; print the push command and stop.

Files (verified during planning, 2026-05-07):
- `paper-writing/overleaf/git-mirror/main.tex`: main paper.
- `paper-writing/overleaf/git-mirror/supplementary.tex`: supplementary appendix.
- `paper-writing/overleaf/git-mirror/references.bib`: bibliography.
- `paper-writing/overleaf/git-mirror/figures/supplementary/`: supplementary PNGs (mirrored from `paper-writing/figures/supplementary/`).
- Style files: `igs.cls`, `igs.bst`. Almost never edited.

**Known inconsistency (flag on first walk):** `paper-writing/CONTENT_MAP.md` line 19 references `overleaf/2026-04-17_template-plan-md/main.tex`, but only `overleaf/git-mirror/` exists today. On the first invocation per session that walks Overleaf, flag this to the user and ask whether to update CONTENT_MAP.md to point to `overleaf/git-mirror/main.tex`.

If a figure was updated in Step 7, refresh the mirrored copy under `paper-writing/overleaf/git-mirror/figures/supplementary/`.

After all Overleaf edits, print (do not run) the push command:

```bash
cd paper-writing/overleaf/git-mirror
git status
# review diff, then:
git add -A && git commit -m "<message>" && git push
```

## Step 6: Update code constants

Grep first for the OLD value, edit second. Common stale-value sites:

`S2-iceberg-areas/`:
- `threshold_tifs.py`: B08 threshold (currently `0.12`).
- `otsu_threshold_tifs.py`: `OTSU_FLOOR=0.10`, `OTSU_CEIL=0.50`, `SEA_ICE_FRAC=0.15`.
- `compare_areas.py`: SZA bin labels and ordering.
- `train.py`: default training config.
- `predict_tifs.py`, `predict.py`: default checkpoint path.
- `export_onnx.py`: checkpoint path docstring (currently `s2_20260227_231556/best_model.pth`).
- `chip_sentinel2.py`: `runs/s2_*/best_model.pth` glob in help text.
- `run_pipeline.sh`: any pinned run IDs.

`iceberg-rework/`:
- `configs/*.yaml`: `prob_threshold`, `seed`, backbone, augmentation, balance scheme.
- `scripts/run_experiment.py`: stage defaults.
- `scripts/run_methods.sh`: flag forwarding (PR-11 lesson: drift here caused the UNet_TR baseline issue; verify `--prob_threshold` is passed through to `threshold_probs.py`).

For every constant updated, grep the project for the OLD value to confirm no docstring or help-text reference is left behind.

## Step 7: Update figures + figure source scripts

- `paper-writing/figures/figures.md`: figure index of record. Update slugs and captions.
- `paper-writing/figures/`: clean named PNGs.
- `paper-writing/figures/supplementary/`: figS01, figS02, figS03 PNGs.
- `paper-writing/figures/fig-archive/`: **move** the previous-version PNG/SVG here with a date suffix (`<slug>_YYYY-MM-DD.png`). Do not delete in place.
- Source scripts (print commands; do not auto-run unless the user approves):
  - `iceberg-rework/scripts/make_figure01_annotation_difficulty.py` through `make_figure05_progression.py`: main-text figures.
  - `iceberg-rework/scripts/make_fig_mae_vs_sza.py`, `make_fig_area_scatter.py`, `make_fig_bias_by_area.py`, `make_fig_re_by_area.py`, `make_fig_outline_examples.py`: main-text result figures.
  - `iceberg-rework/scripts/make_figS02_phase_a_heatmap.py`, `make_figS03_a0_vs_a1_by_sza.py`: supplementary.
  - `iceberg-rework/scripts/script_check_answers/q07_otsu_floor_distribution.py`: figS01.
- For each updated figure, refresh the mirrored copy under `paper-writing/overleaf/git-mirror/figures/supplementary/`.

## Step 8: Update review materials

- `paper-writing/figure_review/figure_review_checklist.csv`
- `paper-writing/figure_review/figure_review_checklist__meeting_review.csv`
- `paper-writing/figure_review/results_figs_table_previews/*.csv`: summary CSVs and PNG previews.
- `paper-writing/figure_review/script_check_answers/*.csv`: q01 to q20 parameter-sweep answer tracking.
- `paper-writing/figure_review/table_previews/`: table PNG previews.

## Step 9: Reconcile experiment results dirs

- `iceberg-rework/runs_summaries/exp_*/*/evaluation/eval_results.csv`, `eval_summary.csv`: flag stale runs the new finding supersedes; mark superseded ones with a note in plan.md (do not delete files).
- `iceberg-rework/sweeps/ndwi_threshold_sweep.csv`, `cloud_fractions_kq.csv`: parameter sweeps; check whether the sweep grid still spans the new value.
- `iceberg-rework/reference/*.csv`: usually stable. Re-check `descriptive_stats.csv`, `fisser_quality_filter.csv`, `ic_filter_10km.csv` only if dataset definition changed.

## Step 10: Stale-reference sweep (always)

Grep the **whole project** for each OLD value collected in Step 0. This catches docstrings, help text, captions, and prose blocks that earlier steps missed.

For each OLD value, run separately:

```bash
cd /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026
grep -rn -F "<OLD_VALUE>" \
  --include="*.md" --include="*.py" --include="*.tex" --include="*.bib" \
  --include="*.csv" --include="*.yaml" --include="*.sh" \
  --exclude-dir=".git" --exclude-dir=".venv" --exclude-dir=".venv_ads" \
  --exclude-dir="fig-archive" --exclude-dir="_archive" --exclude-dir="IDS2026_archive" \
  --exclude-dir="node_modules" --exclude-dir=".claude" .
```

Report every hit. Hits inside `fig-archive/`, `_archive/`, `IDS2026_archive/`, or dated update blocks in `plan.md` are intentionally historical: do not touch. For all other hits, ask the user to confirm before updating.

## Step 11: HPC sync (if code changed)

For every `.py` edited under `S2-iceberg-areas/`, **print** (do not run) the rsync per project CLAUDE.md convention:

```bash
rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/S2-iceberg-areas/<file>.py smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
```

For `iceberg-rework/scripts/` edits, the equivalent target is `smishra@moosehead.bowdoin.edu:~/iceberg-rework/scripts/`. If the user has a different HPC layout, ask once and note the answer in CLAUDE.md.

Print the rsync block as a copy-paste-ready bash fence. Do **not** execute.

## Step 12: Append to `paper-writing/findings_log.md`

If `paper-writing/findings_log.md` does not exist, create it with this header:

```markdown
# Findings log

Append-only chronological record of project findings. Each entry summarizes one finding and lists every artifact touched. Maintained by `/update-icebergs`.
```

Then append:

```markdown
## YYYY-MM-DD: <one-line finding name>

- **Type:** <category from Step 0>
- **Old:** <values>
- **New:** <values>
- **Files updated:**
  - <relative/path/one>
  - <relative/path/two>
- **Pending:**
  - <rsync commands not yet run>
  - <Overleaf push not yet pushed>
  - <figures not yet re-rendered>
- **Manual follow-up:**
  - <re-train, re-evaluate, re-annotate, etc.>
```

## Step 13: Final report

Print a structured summary to the user:

1. **Files touched**: count and grouped list by section (plan / guardrails / prose / Overleaf / code / figures / review / runs_summaries / findings_log).
2. **Stale references still present**: from Step 10, with the reason each was kept (archived in fig-archive/, historical entry in plan.md, etc.).
3. **Pending HPC rsync**: copy-pasteable block.
4. **Pending Overleaf push**: copy-pasteable block.
5. **Pending figure re-renders**: copy-pasteable commands.
6. **Manual follow-up**: re-training, re-evaluation, re-annotation, etc.

End with: "Review the diff before `git add`. The HPC rsync, Overleaf push, and figure re-renders above are not yet executed."

---

## Operating principles (apply throughout)

- **plan.md is canonical.** When other docs disagree, plan.md wins.
- **Append, don't overwrite, dated history.** Findings get added to plan.md and findings_log.md as new dated blocks; previous entries stay.
- **Archive, don't delete.** Old figure PNGs go to `figures/fig-archive/<slug>_YYYY-MM-DD.png`. Stale eval CSVs are noted as superseded in plan.md, not removed.
- **Never push or rsync without approval.** Print commands; let the user run them.
- **Confirm before bulk-updating stale grep hits.** Some are intentionally historical.
- **Read before edit.** Always Read a file before editing it (project CLAUDE.md File Editing Protocol).
- **No em dashes in any prose written into project files** (project CLAUDE.md Writing and Response Style). Use commas or colons.
- **No AI filler language.** No "certainly", "absolutely", "great question", etc. (project CLAUDE.md).
