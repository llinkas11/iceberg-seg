# Paper-writing Instructions (scoped)

Supplements the root CLAUDE.md. Applies to all `.tex`, `.bib`, and figure files in `paper-writing/` and its subdirectories. Root rules remain in force (no em dashes, no AI language, short and direct, fact-check every claim).

## Target journal

Journal of Glaciology (IGS). Paper type: **Article**.

Source of record for style: `paper-writing/jglac-instructionsforauthors-11Apr2019.pdf`. Re-read if unsure.

## Pre-edit checklist (run before EVERY .tex or .bib edit)

1. Working inside the current session folder (latest `overleaf/YYYY-MM-DD_*`), not a past snapshot.
2. Abstract will remain ≤200 words after the edit (Article limit).
3. Section hierarchy preserved: BOLD CAPS (section), Bold sentence case (subsection), Italic sentence case (subsubsection). If using IGS class file, normal LaTeX markup; the class applies formatting.
4. Every new citation in text has a matching entry in `references.bib`.
5. Units are SI, superscript notation (`m s$^{-1}$`, not `m/s`).
6. New equations are numbered and cited elsewhere.
7. No em dashes; no hypertext in prose.

## Abstract

- Article: ≤200 words. Letter: ≤150. Communications: no abstract.
- Current `main.tex` abstract is ~458 words, over limit. Trim before any round of submission.
- Count words with a real count, not an estimate.

## Structure

Required order: title, authors + affiliations, abstract, numbered sections, Acknowledgments, References, Appendices (if any), Supplementary material citation (if any).

## Typography and language

- No em dashes anywhere.
- One space between sentences.
- British OR American spelling, consistent throughout.
- Numbers <10 spelled as words in ordinary text; numerals OK in tables, figures, equations, with units, or ≥10.
- Italicize all algebraic symbols, including in prose: `$\gamma$`, `$\theta_s$`.
- Indent 2nd+ paragraphs, not the first, in each subsection.
- Footnotes only for author address changes and table notes.
- No hypertext in prose; hyperlinks rendered as ordinary text.
- Hyphenate compound adjectives consistently (`mass-balance measurements`, but `measurement of mass balance`).
- Treat `criteria` and `data` as plurals.

## Acronyms

- Minimize use.
- Spell out on first use in both abstract AND main text.
- Familiar (no spell-out needed): NASA, NATO, VHF, GIS, DEM.
- Spell out once: SZA, NIR, IoU, RMSE, UNet++, DenseCRF, CRF, NDWI, AAR, FWHM, etc.

## Units (SI, superscript notation)

- km (not Km), K (not °K), kg m$^{-3}$ for density.
- `m s$^{-1}$`, never `m/s`.
- Year as unit: `a` (annus). Year as noun: spell out.
- Water equivalent: `mm w.e.`, `m w.e.` (always include `w.e.`).

## Dates

- Prose: `27 November 2008` or `November 2008`.
- Figures/tables only: `YYYYMMDD` (e.g., `20081127`) or three-letter month.
- Year range: `2006–08` with en dash (`--` in LaTeX), not `2006–2008`.
- Mass-balance year: `2006/07` (solidus).

## Equations

- Numbered in order of appearance.
- Text reference form: `(1)`, `(2a)`, `(2b)`, `(3–5)`.
- Every equation cited somewhere in text.

## In-text citations

- `(Smith, 2000)` single author.
- `(Wang and Smith, 2000)` two authors.
- `(Wang and others, 2000)` three or more authors. **Never "et al."**
- In-sentence: `Smith (2000) showed that...`, `Wang and others (2000) reported...`.
- Groups: chronological, alphabetical within same year: `(Zaremba, 1973; Colbeck, 1979, 1991; Gow and others, 1979, 1987)`.
- Same author same year: `(Wang, 2009a,b,c, 2010)` no spaces between letters.
- `(Wang, in press a)` has a space before the letter.
- Current `main.tex` abstract says `Fisser et al. (2024)`. Change to `Fisser and others (2024)` on first revision.

## References (.bib format)

- Authors: surname first, initials after, no periods, no spaces in initials: `Smith JA` not `J.A. Smith`.
- Connector between last two authors: `and`.
- Journal names: ISO 4 abbreviated, italicized: `J. Geophys. Res.` not `Journal of Geophysical Research`.
- Always include volume AND issue number: `113(B11)`.
- Include DOI when available: `(doi: 10.1029/2008JB005751)`.
- Order in list: alphabetical by first author surname.
- Among same first author: single-author works first, then double-authored, then three+; within each, most recent first; same year, letters a, b, c by citation order in text.
- Author list: all if ≤6; else `First ZZ and 6 others`.
- Include too much info rather than too little.
- Cite `in press` for accepted, not-yet-published works.
- Do NOT include personal communications, unpublished data, or web-only data in reference list; cite inline in text instead.

Example format:
```
Castelnau O, Duval P, Montagnat M and Brenner R (2008) Elastoviscoplastic micromechanical modeling of the transient creep of ice. J. Geophys. Res., 113(B11), B11203 (doi: 10.1029/2008JB005751)
```

## Figures

- Not in boxes.
- Strong black lines; avoid tinting.
- Labels: Optima, Arial, Calibri, or similar sans-serif; ≥8–10 pt in final print.
- Max widths: 85 mm single-column, 179 mm double-column.
- Caption below, roman font, bold prefix: `\caption{\textbf{Fig. n.} ...}` (IGS class may auto-generate prefix).
- Panel labels: `a`, `b`, etc. No period, no parentheses.
- Axis labels: upper case first letter of first word only.
- Axis units in parentheses with superscript: `Reflectance (m s$^{-1}$)`.
- In-text ref: `Fig. 1`, not `Figure 1`.
- Final file formats: TIFF (600 dpi line / 300 dpi color), EPS, or PDF preferred; JPEG/PNG acceptable for raster.

## Tables

- Caption ABOVE body, roman font, bold prefix: `\caption{\textbf{Table n.} ...}`.
- Column headings in roman.
- Use `booktabs`: `\toprule`, `\midrule`, `\bottomrule`. Horizontal rule above header, below header, bottom only.
- Footnote markers: superscript letters `a, b, c, ...` or symbols in order `*, †, ‡, §, ‖, #`, then doubled (`**`, etc.).
- In-text ref: `Table 1`.

## Appendices

- Placed after References, ordered with upper-case letters.
- Heading: `APPENDIX A – [TITLE]` in bold caps.
- Equations numbered `(A1), (A2)` in A, `(B1)` in B, etc.
- Figures and tables numbered continuously with main text.

## Supplementary material

- Cite in text. Cite again at end of paper under `SUPPLEMENTARY MATERIAL` heading before Acknowledgments.
- Supplementary figures/tables: prefix `S`. `Table S1`, `Figure S3`.
- 50 MB file size cap.

## After every .tex/.bib edit

1. If abstract was touched: recount words (use `wc -w` on the abstract body) and confirm ≤200.
2. If new citations added: grep the `.bib` for matching keys.
3. Update the current session's `README.md` "Changes" bullet list.
4. If a numbered/lettered convention was violated (e.g., added a citation but not the bib entry), fix immediately, do not defer.

## File conventions specific to this project

- Main paper source: `main.tex` inside the current `overleaf/YYYY-MM-DD_*/` session folder.
- Target document class: `\documentclass[review,jog]{igs}` using the IGS class file (`igs.cls`, `igs.bst`) sourced from the AOG Overleaf template zip. `jog` selects Journal of Glaciology formatting.
- Bibliography file: `references.bib`. BibTeX entries; the IGS `.bst` handles output formatting. Do not pre-format author names.
- Prior Overleaf sessions from before the plan.md pivot live under `_archive/overleaf-sessions/` and are frozen references only.

## Authoritative project state

Four live reference files form the single source of truth for methodology, numbers, and results prose (all 2026-04-16):

- `paper-writing/plan.md`: current project-state document. Completed steps, resolved prerequisites, methodological decisions (shadow merge, 40 m RL filter, annotation-aware IC, DN offset), `v3_balanced` split (364 / 137 / 228), remaining steps (retrain → 6-method inference → per-iceberg eval). Path references into the iceberg-rework codebase.
- `paper-writing/iceberg-rework-README.md`: project-level README with folder layout, data-source inventory (`smishra/rework/` paths), v3_clean and v3_balanced tables, processing pipeline diagram, key methodological decisions with justification references.
- `paper-writing/reference/descriptive_stats_results_discussion.md`: dataset composition (984 chips by SZA bin), 40 m filter effect on Roboflow and Fisser component counts, iceberg size distribution (Tables 3/6 with mean/median/max area and root length), temporal/meteorological characterization, comparison to Fisser (2025) reference values.
- `paper-writing/reference/b08_analysis_results_discussion.md`: per-iceberg and pixel-level B08 reflectance by SZA bin (Tables 1–3), iceberg/ocean/contrast characterization, annotation-aware IC filter (Table 4, §3.1–3.6), justification for 15 % threshold + masking (rather than discarding) + no-dynamic-threshold decision.

Any methodology claim added to the paper must match these four sources. `introduction_draft.md`, `introduction_outline.md`, and `methods_draft.md` are kept as scaffolding but carry older 3-class / pre-40m-filter language; do NOT lift prose from them without reconciling against `plan.md` and `reference/*` first.

## Fact-checking reminder (from root CLAUDE.md)

Every number in the abstract or results must be traceable to actual script output, not estimated. Flag unverified numbers with `% UNVERIFIED` inline before leaving them in place.
