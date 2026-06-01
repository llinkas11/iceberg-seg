# Handoff: paper-integration session, end of 2026-05-01

This file is the single document the next context window reads to pick up cleanly. The authoritative project state remains `paper-writing/plan.md`; this handoff covers only what changed in this session and what's next.

> **2026-05-05 addendum:** Phase A higher-SZA re-eval (Slurm 60293, A0..A9 x 4 SZA bins, UNet only) and Phase B backbone comparison (Slurm 60296 + 60297, six methods x 4 SZA bins, A0 and A1) complete. Original headline: A1 wins every higher-SZA bin on per-pair IoU and root-length MAE; A0 still wins lt65. Full T1-T4 in `shib_end_to_end/phase_a_higher_sza_t1_t4.md`. Pushed in commits `4b50d78` (CSVs + scripts) and `2c7b33b` (docs). The user's framing direction: keep Phase A and the backbone-comparison story in supplementary material; main text should focus on the headline that the learned pipeline beats Fisser's baseline at higher SZA.
>
> **2026-05-05 evening + 2026-05-06 update:** 8 A1-anchored Phase A variants trained (Slurm 60309-60316) and re-evaluated (60318); A7b (= A8b == A9b by collapse, A1 manifest + size oversample + augmentation) is the new higher-SZA champion across all 18 backbones (mean MAE 27.24 m vs A1's 28.01 m, mean IoU 0.531 vs 0.499). Phase B re-run with A7b (Slurm 60323) + top-hat companions (60328) shifts the cross-bin pipeline recommendation from A1 + UNet_CRF to **A7b + UNet_CRF** (higher-SZA mean IoU 0.616 vs 0.602 at tied MAE; A7b wins sza_gt75 outright on both metrics). Top-hat lifts UNet_OT recall but degrades UNet_CRF cross-bin, so A7b + UNet_CRF base (no TH) is the cleanest recommendation. T3 / T4 in the artifact updated; supplementary fig S03 now shows the three-backbone comparison. Co-authored by Claude Sonnet 4.6.

## What just landed

### Phase A leaderboard (all 10 experiments, A0-A9, completed)

Best val IoU on lt65: **A0** (`v4_raw_lt65`, Fisser preprocessing, no nulls): val IoU 0.613, test IoU 0.577, UNet match rate 51.2%. A1 second at 0.503; A2-A9 cluster at 0.22-0.27 because the 40 m + IC mask preprocessing degrades calibration in low-data regimes (PR-12 in plan.md). 2x3 balancing grid empirically collapsed to a 1x3 progression on `v4_clean_lt65_plus_nulls` (D ≡ I, J ≡ K ≡ L). Numbers and discussion already in `plan.md` Phase A leaderboard section and `paper-writing/results.md` §3.1.

### Indexing audit (eight numerical checks, all PASS)

Confirmed no bug in v4_clean's IC-mask + annotation pipeline. The training pkls, model checkpoint, evaluation pipeline, and every number in `plan.md` and `main.tex` Tables 1-11 are correct. Audit log in `~/.claude/plans/curried-booping-wilkinson.md`.

### Figure 1 fix (annotation-cleaning visual)

Found and fixed a bug in `make_figure01_annotation_difficulty.py` that was rendering mismatched paired annotations for Fisser chips (it indexed `data/fisser_filtered/` by v4_clean's redrawn split, but those pkls preserve Fisser's original split). Patch is in commit `f10898d`. Also unified the colour convention across all three rows (gold = kept, red outline = removed by 40 m filter, red fill = pixels zeroed by IC mask) and added a legend at the bottom of the figure.

Final figure: `paper-writing/figures/fig-archive/20260501_202440__fig01_annotation_difficulty.png`. Row (a) is now `fisser_0112` (sparser: 11 kept icebergs, 2,184 shadow px, max root length 601 m, median 133 m). Caption rewritten in main.tex.

### Git state

- Branch: `paper-figures-and-results`. Latest commit `d155cc6`. Pushed.
- HPC and local repo are in sync for `iceberg-rework/`.
- The Overleaf folder (`paper-writing/overleaf/2026-04-17_template-plan-md/`) is **not** in git. Changes to `main.tex` and the figures dir under it need a manual sync to overleaf.com.

## What's left

In priority order:

1. **Overleaf sync.** Upload the latest `main.tex` and `figures/fig-archive/20260501_202440__fig01_annotation_difficulty.png` to overleaf.com. The repo doesn't track the Overleaf side.

2. **Discussion + Conclusion.** Currently TODO comments in `main.tex` lines ~474, 482, 489, 493, 497, 509, 513. Section structure: SZA-dependent contrast compression, learned-segmentation robustness, operational implications, limitations, then a one-paragraph conclusion. Pull from `methods_draft.md` section comments and `paper-writing/reference/b08_analysis_results_discussion.md` §1-2.

3. **Frontmatter.** Title, running-head short title, author list, affiliations (lines 24-36 of main.tex are TODO).

4. **Acknowledgements.** Line 513 TODO. Supervisors, Fisser et al. for shared data, Bowdoin HPC, Roboflow annotators.

5. **`references.bib` already populated** (19 entries; none missing).

6. **Compile check** in Overleaf to confirm `figures/fig-archive/...` paths resolve and `\citep` calls don't break.

## Key file paths

| | Path |
|---|---|
| Project state (canonical) | `paper-writing/plan.md` |
| Methods draft (already integrated into main.tex) | `paper-writing/methods_draft.md` |
| Results draft (mirrors §3 of main.tex) | `paper-writing/results.md` |
| Phase A narrative + 2x3 grid | `paper-writing/model_progression.md` |
| Repository design + audit history | `paper-writing/refactor_plan.md` |
| Latest plan + audit log (this session) | `~/.claude/plans/curried-booping-wilkinson.md` |
| Overleaf main.tex | `paper-writing/overleaf/2026-04-17_template-plan-md/main.tex` |
| Overleaf figures | `paper-writing/overleaf/2026-04-17_template-plan-md/figures/fig-archive/` |
| Source of truth figures (registry) | `paper-writing/figures/fig-archive/` |
| HPC working tree | `/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/` |
| Trained baseline checkpoint | `runs/exp_baseline_v1/20260424_185158/model/best_model.pth` |

## Trained checkpoints already on HPC

- `exp_baseline_v1/20260424_185158/` (canonical, all 4 SZA bins, val IoU 0.323 / test IoU 0.314)
- `exp_baseline_v1_raw/20260428_092402/` (v4_raw companion, all 4 bins)
- `exp_A0` through `exp_A9` (lt65 only, ten variants)

All have per_iceberg eval CSVs at `runs/<exp>/<ts>/per_iceberg/eval_per_iceberg_summary.csv`.

## Recent commits (paper-figures-and-results branch)

```
d155cc6 Switch fig01 row (a) to fisser_0112 (sparser)
f10898d Fix fig01 annotation-cleaning visual: pkl pairing + colour convention
08ca083 Paper-writing: figures + tables + Methods/Results prose snapshot
99c2684 Add paper-figure scripts + outline / results helpers
c0b2760 Wire UNet_TR prob_threshold from YAML through to threshold_probs.py
```

## How to resume

1. Read this file end-to-end.
2. Open `paper-writing/plan.md` for the canonical project state and headline tables.
3. Skim `paper-writing/results.md` for the current Results-section content.
4. If picking up the paper write-up: jump to `main.tex` line 474 (Discussion section TODOs).
5. If picking up the figure work: row (a) of Fig. 1 is now `fisser_0112`. Other candidates listed in this session's audit (e.g. `fisser_0090` with max root 1.4 km if you want a single very large berg); the figure script accepts any `chip_stem`.

## Process note

For any future investigation, write the plan to `~/.claude/plans/<plan-file>.md` first, ExitPlanMode for approval, THEN execute. The figure-1 bug investigation skipped the plan step, which got flagged. The retrospective audit log is in `curried-booping-wilkinson.md`.
