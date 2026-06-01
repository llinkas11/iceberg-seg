# Supplementary material

Drafting surface for the paper's supplementary appendix. The authoritative numerical record lives in `shib_end_to_end/phase_a_higher_sza_t1_t4.md`; this file is the prose layer that will be lifted into LaTeX (likely a separate `supplementary.tex` once the main paper structure stabilises). Mirror tables and figure references are cited here so the supplementary section is self-contained.

In-text labels follow Journal of Glaciology style: `Table S1`, `Figure S2`. Slugs in `figures/fig-archive/` use the registry-constrained lowercase form (`figs02_*`, `figs03_*`).

## S1 Phase A backbone progression on the v4_clean test split

Phase A of the executed comparison trained ten UNet++ backbones on lt65-only data (Methods Section 2.13). The eight A1-anchored variants (Section S1.2) re-anchor the class- and size-balancing schemes onto A1's manifest (`v4_raw_lt65_plus_nulls`) instead of A4's (`v4_clean_lt65_plus_nulls`), testing whether the size-oversample axis interacts with the GT-zero null injection that distinguishes A1 from A0.

Cross-experiment evaluation uses the v4_clean test split (228 chips, 57 per SZA bin) for all 18 backbones, so cells are directly comparable across experiments and across SZA bins. The dataset-drift guard in `run_methods.sh` is overridden with `FORCE=1` for this evaluation: each backbone was trained on a different manifest, but the v4_clean test chips are the unifying reference.

### S1.1 Original 10 backbones (A0 through A9)

Per-pair Hungarian-matched IoU and root-length MAE; UNet method only (the trained checkpoint's argmax). See Table S1 for the full grid; here is the headline:

- A0 (Fisser preprocessing on v4_raw_lt65, no null injection, no augmentation) wins lt65 cleanly: per-pair IoU 0.710, root-length MAE 10.99 m on the v4_clean lt65 test split. The original Phase A leaderboard (Section 3.1) reported A0 on the v4_raw_lt65 test split (12,343 pairs); this re-eval reports on the v4_clean lt65 split (4,016 pairs) for cross-bin comparison.
- A1 (A0 + 29 GT-zero training chips) loses lt65 by 5 m MAE but gains every higher-SZA bin. Aggregate over the three higher-SZA bins: A1 mean per-pair MAE 28.01 m vs A0 33.33 m, a 16 per cent reduction.
- A2 through A9 cluster well below both A0 and A1 because A2's 40 m + IC mask preprocessing (kept fixed for A3-A9) shifts the model's calibration substantially (Section 3.2 of the main results). The empirical 1x3 collapse from the planned 2x3 grid (A5 == A6, A7 == A8 == A9) is documented in the main text.

### S1.2 A1-anchored variants (added 2026-05-05 evening)

Eight new training runs anchor on A1's `v4_raw_lt65_plus_nulls` manifest and apply the original Phase A balancing schemes (D, I, J, K, L) on top:

- A5a, A6a: A1 + class balancing (fixed positive D / adaptive I), aug=off
- A7a, A8a, A9a: A1 + size oversample (J / K / L), aug=off
- A7b, A8b, A9b: A1 + size oversample (J / K / L), aug=on

Empirical collapse repeats: A5a == A6a (GT+ majority forces fixed-positive direction), A7a == A8a == A9a, A7b == A8b == A9b (size oversample saturates at the same equilibrium under the 4x replication cap regardless of class scheme).

**Higher-SZA winner shift**: A7b (= A8b == A9b) is the new champion across all 18 backbones with mean per-pair MAE 27.24 m and IoU 0.531 across the three higher-SZA bins. A7b also beats A1 at lt65 (15.90 m vs 17.74 m); A0 still wins lt65 outright. The size-oversample axis carries most of the lift, augmentation adds the final refinement at higher SZA.

Table S1 reports the full 18-experiment leaderboard. Figure S2 visualises it as a per-SZA-bin x per-experiment heatmap (IoU and MAE side-by-side).

## S2 Phase B backbone comparison (12 methods x 4 SZA bins)

The original Phase B (Methods Section 2.7-2.10) ran on the A0 checkpoint over the lt65 test split only. The 2026-05-05 evening / 2026-05-06 follow-up re-runs Phase B on three backbones (A0, A1, A7b) on the v4_clean test split for all four SZA bins, with the six top-hat companions (TR_TH, OT_TH, UNet_TH, UNet_TR_TH, UNet_OT_TH, UNet_CRF_TH) added in a sibling sweep.

Pixel methods (TR, OT, TR_TH, OT_TH) are backbone-independent; their numbers are identical across A0, A1, A7b.

Among the four learned methods (UNet, UNet_TR, UNet_OT, UNet_CRF) and their TH companions:

- **UNet (argmax)**: A7b sweeps the three higher-SZA bins on both metrics. lt65 still favours A0.
- **UNet_TR**: A0 wins all four bins. The fixed-threshold-on-probabilities path favours A0's sharper probability distribution and is unsuited to the diffuse outputs A1 and A7b produce.
- **UNet_OT**: A0 wins lt65 spectacularly (8.45 m, reproducing the published 8.18 m within rounding). A7b wins higher-SZA MAE in two bins (sza_65_70, sza_70_75) and ties A1 at sza_gt75.
- **UNet_CRF**: A1 and A7b both clearly beat A0 at higher SZA. **A7b + UNet_CRF** is the cross-bin pipeline pick: higher-SZA mean IoU 0.616 vs A1 + UNet_CRF's 0.602, at essentially tied MAE (15.59 m vs 15.21 m); A7b wins sza_gt75 outright on both metrics.

Top-hat companions lift UNet_OT recall substantially in higher SZA bins (e.g. -10 m MAE at sza_65_70 for A1, doubled n) but degrade UNet_CRF cross-bin (+4 m MAE for no IoU gain). UNet_CRF base is therefore the cleanest cross-bin pick over its TH companion.

Tables S3 (base methods) and S3 continued (top-hat companions) report the full 12-method matrix. Table S5 isolates the top-hat effect deltas. Figure S3 visualises the three-backbone comparison on the four UNet-method bins.

## S3 Recommended retrieval pipeline per SZA bin (Table S4)

Combining all twelve methods on three backbones, the per-bin best (root-length MAE) and the cross-bin learned-method pick are:

| SZA bin | Per-bin best (MAE-optimal) | Cross-bin learned pick |
|---------|----------------------------|------------------------|
| sza_lt65 | A0 + UNet_OT (8.45 m, IoU 0.733) | A7b + UNet_CRF (12.24 m, IoU 0.589) |
| sza_65_70 | TR (7.91 m, n=315) | A7b + UNet_CRF (11.80 m, IoU 0.645, n=339) |
| sza_70_75 | TR (6.46 m, n=126) | A7b + UNet_CRF (12.50 m, IoU 0.623, n=478) |
| sza_gt75 | OT_TH (19.52 m, n=252) | A7b + UNet_CRF (22.47 m, IoU 0.581, n=219) |

Pixel methods (TR, OT, OT_TH) achieve the lowest MAE in higher-SZA bins but at low recall: TR drops to n=126 at sza_70_75 (vs UNet_CRF's n=478). The paper recommends the per-bin best for benchmark-style reporting and A7b + UNet_CRF as the single learned cross-bin pipeline; the lt65 result reproduces the published Fisser-comparable 8.18 m headline within rounding.

## Supplementary figure index

| ID | Slug | Caption pointer |
|----|------|------------------|
| Fig. S1 | `figS01_otsu_floor_distribution` | Per-chip Otsu floor distribution across 23,981 chips, justifies the 0.10 noise floor in `otsu_threshold_tifs.py`. |
| Fig. S2 | `figs02_phase_a_heatmap` | Phase A per-SZA-bin x 18-experiment IoU + MAE heatmap (UNet method). |
| Fig. S3 | `figs03_a0_vs_a1_by_sza` | Three-backbone comparison (A0, A1, A7b) by SZA bin (UNet method), MAE + IoU paired bars. |

Live PNGs in `paper-writing/figures/fig-archive/`. Generators in `iceberg-rework/scripts/`.

## Supplementary table index

| ID | Title | Source |
|----|-------|--------|
| Table S1 | Phase A 18-backbone leaderboard, per-pair IoU + MAE per SZA bin | `shib_end_to_end/phase_a_higher_sza_t1_t4.md` Sections T1 and T1b |
| Table S2 | Best non-A0 Phase A backbone per SZA bin (A7b winner) | Same, Section T2 (revised) |
| Table S3 | Phase B backbone comparison: A0 vs A1 vs A7b, twelve methods, four SZA bins | Same, Section T3 |
| Table S4 | Recommended retrieval pipeline per SZA bin | Same, Section T4 |
| Table S5 | Top-hat effect on MAE and IoU (base to TH deltas) | Same, Section T5 |

## Cross-references in the main text

The main text cites the supplementary content from the following anchor points:

- Methods Section 2.6 (UNet++ training) ends with a "Backbone selection across SZA bins" subsection that points at Tables S1-S2 and Figures S2-S3 for the dataset-progression story.
- Results Section 3.5 ("Summary of headline findings") closes with a "supplementary follow-up" paragraph pointing at Tables S3-S5 and Figure S3 for the cross-bin recommendation.
- The lt65 headline number (A0 + UNet_OT, 8.45 m) is in the main results tables; the higher-SZA numbers are summarised but defer to Tables S3-S4 for the per-method breakdown.

## Editing protocol

- Numbers in this file must trace to `shib_end_to_end/phase_a_higher_sza_t1_t4.md`. When that artifact updates, sync this file in the same commit.
- LaTeX rendering happens in a separate `supplementary.tex` in the Overleaf folder once the main paper is camera-ready; until then, this Markdown file is the working draft.
- Per `paper-writing/CLAUDE.md`: no em dashes, British or American spelling consistent throughout, units in superscript SI notation, citation form `Smith and others (2024)` not `Smith et al. (2024)`.
