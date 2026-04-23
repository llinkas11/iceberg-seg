<!--
RECONCILIATION NOTE (2026-04-17)

Paragraph 3 pipeline enumeration (the 7-item list around line 70) lists
"Image -> UNet++ -> probability map -> CRF -> segmentation" as live pipeline 7.
DenseCRF was tested on a 4-chip sandbox and rejected (IoU decreased 0.011 to 0.013);
not applied to the full dataset. See methods_draft.md §2.9 reconciliation note and
plan.md §PR-6. Keep DenseCRF as a negative-comparison point only; do NOT list it as a
live pipeline in the new template.

Other content (CARS structure, claim-by-claim citation mapping, complete reference list,
tone notes) remains valid. Reference list is the primary scaffolding source for building
references.bib in the new template.
-->

# Introduction Outline

Target length: ~500-600 words (10-15% of manuscript). Four paragraphs following the CARS model.

---

## Paragraph 1: Broad Background (Establish Territory)

**Purpose:** Why iceberg areas matter. Not a history lesson: start with the measurement problem.

**Claim-by-claim citation mapping:**

| # | Claim | Citation(s) |
|---|-------|-------------|
| 1a | Iceberg calving is a major component of Greenland ice sheet mass loss | Enderlin et al. (2014) *GRL* 41(3), 866-872; Alley et al. (2023) *Ann. Rev. Earth Planet. Sci.* 51, 189-215 |
| 1b | Iceberg areas are needed for freshwater flux estimates | Moon et al. (2018) *Nature Geoscience* 11, 49-54; Rezvanbehbahani et al. (2020) *Comms. Earth Environ.* 1, 31 |
| 1c | Iceberg areas needed for drift and deterioration modeling | Keghouche et al. (2010) *JGR Oceans* 115(C12); Marchenko et al. (2019) *Applied Ocean Research* 88, 210-222 |
| 2 | Satellite optical remote sensing (Sentinel-2, 10 m, NIR) is the primary tool for above-waterline iceberg area retrieval | Sulak et al. (2017) *Ann. Glaciol.* 58(74), 92-106; Scheick et al. (2019) *J. Glaciol.* 65(251), 468-480 |
| 3 | Small icebergs (<100 m) contribute disproportionately to freshwater but are underrepresented in inventories | Rezvanbehbahani et al. (2020) *Comms. Earth Environ.* 1, 31 (found small icebergs comprise a significant fraction of total freshwater budget; CNN-based detection revealed icebergs missed by coarser methods) |
| 4 | Accurate retrieval across seasonal illumination constrains calving budget closure | Moyer et al. (2019) *Ann. Glaciol.* (seasonal freshwater flux variability in Sermilik); Enderlin et al. (2014) |

**Opening sentence direction:** "Iceberg areas are essential inputs for estimating freshwater fluxes from marine-terminating glaciers in Greenland, yet the reliability of satellite-based area retrieval degrades under the low solar illumination that characterizes autumn and early winter at high latitudes."

---

## Paragraph 2: Literature Review and Gap (Establish Niche)

**Purpose:** Introduce threshold-based detection, Fisser 2024's SZA error characterization, and the two failure modes. Then identify the gap.

**Claim-by-claim citation mapping:**

| # | Claim | Citation(s) |
|---|-------|-------------|
| 1a | NIR reflectance thresholding is the standard optical iceberg detection method | Sulak et al. (2017); Moyer et al. (2019); Scheick et al. (2019); Rezvanbehbahani et al. (2020) (all in Fisser 2024's intro) |
| 1b | Icebergs may be confused with sea ice using threshold methods | Sulak et al. (2017) (noted in Fisser 2024 p.2) |
| 1c | Fisser et al. (2024) calibrated B08 >= 0.12 at SZA 56 degrees (Svalbard airborne reference), quantified SZA-dependent error across 14 KQ acquisitions at SZA 45-81 degrees | Fisser et al. (2024) *Ann. Glaciol.* 65, e38. DOI: 10.1017/aog.2024.39 |
| 1d | Up to SZA 65 degrees, standardized error stays between +5.9% and -5.67%; above 65 degrees, underestimation and inconsistency | Fisser et al. (2024), Results section |
| 2 | Physical mechanism: at high SZA, iceberg surfaces segregate into sun-facing slopes and shadow faces; shadow pixels fall below threshold | Fisser et al. (2024), Section "Iceberg area error by solar zenith angle" and Fig. 14; also Gardner and Sharp (2010) *Rev. Geophysics* (shadow and albedo at high SZA); Lhermitte et al. (2014) *The Cryosphere* 8(3), 1069-1086 (albedo over rough snow and ice surfaces) |
| 3 | Ocean NIR reflectance rises at high SZA, compressing ice-water spectral contrast | Pegau and Paulson (2001) *Ann. Glaciol.* 33, 221-224 (directly cited in Fisser 2024); also Robock (1980) *Monthly Weather Review* 108(3), 267-285 (seasonal albedo cycle) |
| 4a | Fisser 2024 used pre-filtered conditions: cloud <2%, sea ice concentration <~10% | Fisser et al. (2024), Data section p.3 |
| 4b | Fisser 2024 evaluated only large icebergs (>100 m root length) | Fisser et al. (2024), Methods: "we only delineated icebergs with a root length above 100 m" |
| 4c | ESA recommends SZA limit of 70 degrees for Sentinel-2 | European Space Agency (2019) "Copernicus Sentinel-2 observation in low illumination conditions over Northern Europe and Arctic Areas" (cited in Fisser 2024 p.1) |
| 4d | Some studies raised thresholds at high SZA but effect on area error remains unclear | Moyer et al. (2019) raised 0.13 threshold to 0.3 at SZA ~47 degrees (cited in Fisser 2024 p.2) |

**Gap statement (no citation needed, this is our claim):** It remains unknown whether (a) the same degradation patterns hold under unfiltered automated conditions, and (b) whether methods using spatial context rather than per-pixel reflectance can overcome both failure modes.

**Transition sentence:** "However, whether learned segmentation methods that incorporate spatial context can maintain retrieval accuracy where fixed thresholding fails has not been tested."

---

## Paragraph 3: Study Rationale, Research Question, and Approach (Occupy Niche)

**Purpose:** State what this study does, the hypothesis, and the experimental design.

**Claim-by-claim citation mapping:**

| # | Claim | Citation(s) |
|---|-------|-------------|
| 1 | CNNs have shown strong performance for pixel-wise segmentation in cryosphere remote sensing, including iceberg detection | Rezvanbehbahani et al. (2020) *Comms. Earth Environ.* (used CNN for iceberg delineation in Greenland fjords, found it less prone to false detections in melange than threshold); Baumhoer et al. (2019) *Remote Sensing of Environment* 235, 111400 (U-Net for Antarctic calving front delineation from SAR) |
| 2a | UNet++ architecture: nested dense skip connections for multi-scale feature aggregation | Zhou, Z., Siddiquee, M.M.R., Tajbakhsh, N., and Liang, J. (2018) "UNet++: A Nested U-Net Architecture for Medical Image Segmentation." In *DLMIA/ML-CDS 2018*, LNCS 11045, pp. 3-11. DOI: 10.1007/978-3-030-00889-5_1 |
| 2b | ResNet34 encoder pretrained on ImageNet | He, K., Zhang, X., Ren, S., and Sun, J. (2016) "Deep Residual Learning for Image Recognition." *CVPR*. |
| 2c | Binary iceberg segmentation: UNet++ has a 3-class output head (ocean, iceberg, shadow); only the iceberg class is extracted for area retrieval. Ocean and shadow serve as background. | Design decision, no external citation needed |
| 3 | Sentinel-2 L1C data, B08 at 10 m resolution, bands B04/B03/B08 | Drusch, M. et al. (2012) "Sentinel-2: ESA's Optical High-Resolution Mission for GMES Operational Services." *Remote Sensing of Environment* 120, 25-36 |
| 4a | Threshold baselines: Fisser's calibrated B08 >= 0.12, a fixed threshold, and per-chip Otsu adaptive thresholding | Fisser et al. (2024); Otsu, N. (1979) "A Threshold Selection Method from Gray-Level Histograms." *IEEE Trans. SMC* 9(1), 62-66 |
| 4b | Hybrid pipelines: UNet++ probability maps refined by thresholding (fixed, Otsu) or DenseCRF | Krahenbuhl, P. and Koltun, V. (2011) "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials." *NeurIPS 2011*, pp. 109-117 |
| 4c | Top-hat morphological filtering applied across all pipelines to recover small icebergs | No citation needed for top-hat itself (standard morphological operation); motivation from Rezvanbehbahani et al. (2020) re: small-iceberg underrepresentation |
| 5 | Unfiltered scene conditions as deliberate design choice | No citation needed (contrasted against Fisser 2024's filtering) |

**Full pipeline enumeration (for methods section, summarized here for reference):**

1. Image → fixed threshold → segmentation
2. Image → Otsu → segmentation
3. Image → Fisser calibrated threshold (0.12) → segmentation
4. Image → UNet++ → segmentation map
5. Image → UNet++ → probability map → fixed threshold → segmentation
6. Image → UNet++ → probability map → Otsu → segmentation
7. Image → UNet++ → probability map → CRF → segmentation
8. All of the above + top-hat morphological filtering

---

## Paragraph 4: Roadmap (brief)

**Purpose:** One or two sentences previewing paper structure. No citations needed.

Section 2 describes the study sites, imagery, and detection methods. Section 3 presents area retrieval results by SZA bin. Section 4 discusses implications for operational iceberg monitoring under low-illumination conditions.

---

## Complete Reference List for Introduction

Listed alphabetically. All confirmed with full publication details.

1. **Alley, R. and 8 others** (2023). Iceberg calving: regimes and transitions. *Annual Review of Earth and Planetary Sciences* 51, 189-215. DOI: 10.1146/annurev-earth-032320-110916
2. **Baumhoer, C.A., Dietz, A.J., Kneisel, C., and Kuenzer, C.** (2019). Automated extraction of Antarctic glacier and ice shelf fronts from Sentinel-1 imagery using deep learning. *Remote Sensing of Environment* 235, 111400.
3. **Drusch, M., Del Bello, U., Carlier, S., et al.** (2012). Sentinel-2: ESA's Optical High-Resolution Mission for GMES Operational Services. *Remote Sensing of Environment* 120, 25-36.
4. **Enderlin, E.M., Howat, I.M., Jeong, S., Noh, M.-J., van Angelen, J.H., and van den Broeke, M.R.** (2014). An improved mass budget for the Greenland ice sheet. *Geophysical Research Letters* 41(3), 866-872. DOI: 10.1002/2013GL059010
5. **European Space Agency** (2019). Copernicus Sentinel-2 observation in low illumination conditions over Northern Europe and Arctic Areas.
6. **Fisser, H., Doulgeris, A.P., and Hoyland, K.V.** (2024). Impact of varying solar angles on Arctic iceberg area retrieval from Sentinel-2 near-infrared data. *Annals of Glaciology* 65, e38. DOI: 10.1017/aog.2024.39
7. **Gardner, A.S. and Sharp, M.J.** (2010). A review of snow and ice albedo and the development of a new physically based broadband albedo parameterization. *Journal of Geophysical Research* 115, F01009. DOI: 10.1029/2009JF001444
8. **He, K., Zhang, X., Ren, S., and Sun, J.** (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
9. **Keghouche, I., Counillon, F., and Bertino, L.** (2010). Modeling dynamics and thermodynamics of icebergs in the Barents Sea from 1987 to 2005. *Journal of Geophysical Research: Oceans* 115(C12). DOI: 10.1029/2010JC006165
10. **Krahenbuhl, P. and Koltun, V.** (2011). Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials. *Advances in Neural Information Processing Systems (NeurIPS)* 24, 109-117.
11. **Marchenko, A., Diansky, N., and Fomin, V.** (2019). Modeling of iceberg drift in the marginal ice zone of the Barents Sea. *Applied Ocean Research* 88, 210-222. DOI: 10.1016/j.apor.2019.03.008
12. **Moon, T. and 5 others** (2018). Subsurface iceberg melt key to Greenland fjord freshwater budget. *Nature Geoscience* 11, 49-54. DOI: 10.1038/s41561-017-0018-z
13. **Moyer, A.N., Sutherland, D.A., Nienow, P.W., and Sole, A.J.** (2019). Seasonal variations in iceberg freshwater flux in Sermilik Fjord, Southeast Greenland from Sentinel-2 imagery. *Geophysical Research Letters* 46(15), 8903-8912. DOI: 10.1029/2019GL082309
14. **Otsu, N.** (1979). A Threshold Selection Method from Gray-Level Histograms. *IEEE Transactions on Systems, Man, and Cybernetics* 9(1), 62-66.
15. **Pegau, W.S. and Paulson, C.A.** (2001). The albedo of Arctic leads in summer. *Annals of Glaciology* 33, 221-224. DOI: 10.3189/172756401781818833
16. **Rezvanbehbahani, S., Stearns, L.A., Keramati, R., Shankar, S., and Van Der Veen, C.J.** (2020). Significant contribution of small icebergs to the freshwater budget in Greenland fjords. *Communications Earth & Environment* 1, 31. DOI: 10.1038/s43247-020-00032-3
17. **Scheick, J., Enderlin, E.M., and Hamilton, G.** (2019). Semi-automated open water iceberg detection from Landsat applied to Disko Bay, West Greenland. *Journal of Glaciology* 65(251), 468-480. DOI: 10.1017/jog.2019.23
18. **Sulak, D.J., Sutherland, D.A., Stearns, L.A., and Hamilton, G.S.** (2017). Iceberg properties and distributions in three Greenlandic fjords using satellite imagery. *Annals of Glaciology* 58(74), 92-106. DOI: 10.1017/aog.2017.5
19. **Zhou, Z., Siddiquee, M.M.R., Tajbakhsh, N., and Liang, J.** (2018). UNet++: A Nested U-Net Architecture for Medical Image Segmentation. In *DLMIA/ML-CDS 2018*, LNCS 11045, pp. 3-11. Springer. DOI: 10.1007/978-3-030-00889-5_1

---

## Notes on Tone

- No hedging. Write assertively.
- No em dashes. Use commas or colons.
- Do not repeat the abstract. The intro motivates and contextualizes; the abstract summarizes results.
- "The aim of this study is..." phrasing is fine for the research question sentence.
- Do not over-cite. Two to three citations per claim maximum.
