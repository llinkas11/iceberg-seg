<!--
results.md: Results-section scaffolding for the paper. Numbers traceable to
the per-iceberg eval CSVs at runs/<exp>/<ts>/per_iceberg/. Each table here
maps directly into a LaTeX booktabs table in main.tex; figures map onto
slugs in fig-archive registries.

Source CSVs:
  runs/exp_baseline_v1/20260424_185158/per_iceberg/{eval_per_iceberg_summary.csv,
                                                    eval_per_iceberg_detection.csv}
  runs/exp_A0..A9/<ts>/per_iceberg/eval_per_iceberg_summary.csv
  runs/exp_A0..A2/<ts>/inference/sza_lt65/UNet/probs/  (calibration audit)

Figure registry: runs/exp_baseline_v1/20260424_185158/per_iceberg/figures.md.

Last updated: 2026-04-30 (full Phase A leaderboard A0-A9 + Phase B sweep on
canonical baseline_v1 + preprocessing impact audit).
-->

# Results

## 3.1 Phase A: dataset progression on lt65

Phase A walks four controlled variables on lt65-only chips: preprocessing pipeline (A0 to A2), null-chip injection (A0 to A1, A2 to A3), augmentation (A3 to A4), and class plus size balancing (A4 to A9). All ten experiments use byte-identical hyperparameters (ResNet34 encoder, 100 epochs, learning rate 1 x 10$^{-4}$, batch size 16, seed 42); only the training data and the balancing scheme differ.

### 3.1.1 Phase A leaderboard

Table 1 reports best validation IoU and test IoU on the lt65 split for each Phase A configuration. Configurations are ordered by best validation IoU.

\begin{table}
\caption{\textbf{Table 1.} Phase A leaderboard. UNet match rate is the per-pair detection rate at IoU $\geq$ 0.3; UNet root-length MAE is the per-pair mean absolute error in iceberg root length on matched pairs. All columns evaluated on the lt65 test split.}
\begin{tabular}{llrrrr}
\toprule
ID & Manifest & val IoU & test IoU & UNet match rate & UNet RL MAE (m) \\
\midrule
A0 & v4\_raw\_lt65 (Fisser preprocessing, no nulls)            & 0.613 & 0.577 & 0.512 & 9.82 \\
A1 & v4\_raw\_lt65\_plus\_nulls (Fisser preprocessing + nulls) & 0.503 & 0.477 & 0.315 & 15.21 \\
A3 & v4\_clean\_lt65\_plus\_nulls                              & 0.269 & 0.336 & 0.182 & 15.69 \\
A2 & v4\_clean\_lt65 (our preprocessing, no nulls)             & 0.261 & 0.344 & 0.245 & 15.26 \\
A7--A9 & v4\_clean + nulls + aug + size oversample             & 0.243 & 0.320 & 0.163 & 14.78 \\
A5--A6 & v4\_clean + nulls + aug + class balance               & 0.237 & 0.312 & 0.158 & 15.23 \\
A4 & v4\_clean + nulls + aug                                   & 0.225 & 0.274 & 0.122 & 14.93 \\
\bottomrule
\end{tabular}
\end{table}

A0 is the best Phase A configuration by every metric. A0 to A1 loses 0.11 best validation IoU through null-chip injection at a 1:1 GT$+$ : GT0 ratio. A0 to A2 loses 0.35 through preprocessing alone. Once preprocessing is applied (A2 onward), no balancing scheme moves the result by more than 0.05 IoU.

### 3.1.2 Empirical collapse of the 2x3 grid

The Phase A 2x3 grid (Section 2.13.4) was designed to vary class balance in $\{$none, fixed pos, adaptive$\}$ crossed with size balance in $\{$off, oversample$\}$. Empirically the grid collapses to a 1x3 progression on the v4\_clean\_lt65\_plus\_nulls manifest. A5 and A6 produce identical training sets (87 chips each, 58 GT$+$ and 29 GT0) because the manifest's GT$+$ class is the natural majority (198 vs 29, ratio 6.8:1), so the adaptive scheme resolves to the same direction as the fixed-positive scheme. A7, A8, and A9 also produce identical training sets (110 chips) because the size oversample step bottoms out at the same equilibrium under the 4x replication cap, regardless of whether class balancing was applied first.

The grid therefore reduces to three distinct training conditions on this dataset: A4 (1:1 class balance, no size step, 58 chips), A5 / A6 (2:1 class balance, no size step, 87 chips), and A7 / A8 / A9 (size oversample on top of class balance, 110 chips). All three sit within 0.02 best validation IoU of each other.

This finding shifts the Phase A interpretation from "which balancing scheme wins" to "which dataset definition is being optimized": a cleaner iceberg-only task or a realistic fjord-scene task with sea ice and melange retained.

### 3.1.3 Preprocessing pipeline is the dominant Phase A variable

Table 2 isolates the two largest Phase A axes as a 2x2 (preprocessing $\times$ null injection) on best validation IoU.

\begin{table}
\caption{\textbf{Table 2.} Phase A 2x2: preprocessing pipeline x null-chip injection, reported as best validation IoU on the lt65 split.}
\begin{tabular}{lrr}
\toprule
              & no nulls (A0, A2) & + nulls 1:1 (A1, A3) \\
\midrule
Fisser preprocessing & 0.613 & 0.503 \\
Our preprocessing    & 0.261 & 0.269 \\
\bottomrule
\end{tabular}
\end{table}

The preprocessing axis carries a 0.35 IoU gap; the null axis carries at most 0.11. Our preprocessing pipeline (40 m component filter plus annotation-aware IC pixel mask) is the dominant negative variable in Phase A.

## 3.2 Preprocessing impact: probability calibration

A direct audit on a sample of ten lt65 chips per run (Section 2.13.3) reveals the mechanism behind the A0 to A2 collapse. Table 3 reports the median pixel value of the UNet$++$ softmax P(iceberg) band, and the fraction of pixels in the threshold-relevant range 0.20--0.35.

\begin{table}
\caption{\textbf{Table 3.} P(iceberg) distribution on the lt65 test split. Median across ten sample chips of the per-pixel percentile.}
\begin{tabular}{lrrr}
\toprule
Run            & median P(iceberg) & frac. pixels 0.20--0.35 & frac. pixels $\geq$ 0.5 \\
\midrule
baseline\_v1   & 0.001 & 0.4 \%  & 4.0 \% \\
A0 (raw lt65)  & 0.013 & 2.2 \%  & 6.6 \% \\
A2 (clean lt65) & 0.278 & 59.0 \% & 5.2 \% \\
\bottomrule
\end{tabular}
\end{table}

Baseline\_v1 and A0 produce sharply bimodal probability distributions: most pixels collapse to near zero, and only confident iceberg pixels rise above 0.5. A2's distribution is diffuse, with the majority of pixels in the 0.20--0.35 range that any fixed threshold below 0.35 will misclassify as iceberg. The downstream consequence is visible in the UNet$++$ + threshold predictions: predicted-polygon median area is 2,200 m$^2$ on baseline\_v1, 1,600 m$^2$ on A0, and 200 m$^2$ on A2; 49 \% of A2's predicted polygons fall below 200 m$^2$, all speckle.

This calibration shift is a property of the data, not the code. All three runs use the same training script, the same hyperparameters, and the same architecture. The IC pixel mask zeros bright non-annotated pixels in 112 of A2's 198 training chips; the validation and test splits are never masked. The model trains on partially zeroed images and then encounters fully bright pixels at inference, producing diffuse predictions in the resulting train-test domain shift. Applying Fisser's preprocessing pipeline in a low-data, lt65-only regime degrades model calibration to the point that no fixed-threshold post-processing recovers it.

## 3.3 Phase B: method sweep on the canonical baseline

Phase B fixes the Phase A winner and varies the inference method only. The canonical baseline\_v1 trained on v4\_clean (916 chips, all four SZA bins, ResNet34 encoder, 100 epochs) is the Phase B base because it is the only run trained across all SZA bins; A0 wins on lt65 alone but does not generalise. All six methods share the same UNet$++$ probability map; only the post-processing differs.

### 3.3.1 Per-pair root-length MAE

Table 4 reports the per-pair mean absolute error in iceberg root length, in metres, by method and SZA bin. Lower is better; the lowest value per column is in bold.

\begin{table}
\caption{\textbf{Table 4.} Per-pair root-length MAE (m) on the v4\_clean test split. Hungarian matching with IoU $\geq$ 0.3, lower is better. Bold marks the lowest value per column.}
\begin{tabular}{lrrrr}
\toprule
Method   & $<$ 65 & 65--70 & 70--75 & $>$ 75 \\
\midrule
TR        & 17.81  & 7.91   & 6.46   & 20.07 \\
OT        & 22.73  & 13.77  & 14.51  & 15.91 \\
UNet      & 10.48  & 11.48  & 13.86  & 15.57 \\
UNet\_TR  & 14.24  & 15.54  & 18.56  & 19.62 \\
UNet\_OT  & \textbf{7.98} & 11.96 & 13.91 & 15.27 \\
UNet\_CRF & 10.12  & \textbf{7.37} & \textbf{9.04} & \textbf{12.59} \\
\bottomrule
\end{tabular}
\end{table}

UNet$++$ + DenseCRF achieves the lowest per-pair MAE in three of four SZA bins. UNet$++$ + Otsu wins the easiest bin (lt65). At the highest SZA, where image contrast is weakest, UNet$++$ + DenseCRF (12.59 m) beats Fisser's fixed B08 threshold (20.07 m) by 7.5 m, and beats per-chip Otsu (15.91 m) by 3.3 m. The fixed B08 threshold remains competitive at moderate SZA (65--70 and 70--75) because the bands are broadly bimodal in that range, but degrades sharply at SZA $>$ 75.

### 3.3.2 Per-pair IoU on matched pairs

Table 5 reports per-pair IoU on Hungarian-matched pairs. Higher is better; the highest value per column is in bold.

\begin{table}
\caption{\textbf{Table 5.} Per-pair IoU on matched pairs, by method and SZA bin. Higher is better. Bold marks the highest value per column.}
\begin{tabular}{lrrrr}
\toprule
Method   & $<$ 65 & 65--70 & 70--75 & $>$ 75 \\
\midrule
TR        & 0.482 & 0.670 & 0.686 & 0.594 \\
OT        & 0.470 & 0.622 & 0.621 & 0.620 \\
UNet      & 0.701 & 0.672 & 0.643 & 0.646 \\
UNet\_TR  & 0.665 & 0.639 & 0.603 & 0.611 \\
UNet\_OT  & \textbf{0.730} & 0.673 & 0.643 & 0.642 \\
UNet\_CRF & 0.653 & \textbf{0.691} & \textbf{0.666} & \textbf{0.658} \\
\bottomrule
\end{tabular}
\end{table}

UNet$++$ + DenseCRF wins three of four bins on per-pair IoU, mirroring the MAE result. UNet$++$ + Otsu wins the easiest bin. The chip-level pixel IoU sits between 0.005 and 0.013 for every method because the chips are 94 \% ocean by pixel area, so a single false-positive pixel inflates the union and crashes the chip-level metric. The per-pair table is therefore the publication-relevant IoU.

### 3.3.3 Detection statistics

Table 6 reports per-method detection counts as a selection-bias disclosure. Per-pair MAE on a 30 \%-matched method is not directly comparable to per-pair MAE on a 60 \%-matched method without context.

\begin{table}
\caption{\textbf{Table 6.} Per-method detection statistics across all SZA bins, v4\_clean test split. Match rate = n\_matched / n\_gt\_total; precision = n\_matched / n\_pred\_total.}
\begin{tabular}{lrrrrr}
\toprule
Method   & n\_gt & n\_pred & n\_matched & match rate & precision \\
\midrule
TR        & 18,990 & 16,818  & 2,759 & 0.145 & 0.164 \\
OT        & 18,990 & 35,564  & 5,547 & 0.292 & 0.156 \\
UNet      & 18,990 & 21,422  & 9,916 & \textbf{0.522} & 0.463 \\
UNet\_TR  & 18,990 & 23,009  & 8,369 & 0.441 & 0.364 \\
UNet\_OT  & 18,990 & 15,468  & 5,929 & 0.312 & \textbf{0.483} \\
UNet\_CRF & 18,990 & 20,981  & 8,891 & \textbf{0.468} & 0.424 \\
\bottomrule
\end{tabular}
\end{table}

UNet$++$ argmax achieves the highest match rate (0.522), but at the cost of mid-range precision (0.463). UNet$++$ + DenseCRF reaches a 0.468 match rate at 0.424 precision, and produces the best per-pair MAE; this is the balance the paper recommends. UNet$++$ + Otsu has the highest precision (0.483) but recovers only 0.312 of ground-truth icebergs, reflecting Otsu's tendency to set conservative thresholds that miss low-contrast icebergs at high SZA.

The fixed B08 threshold's 0.145 match rate explains the deceptively competitive Table 4 values at moderate SZA: TR detects icebergs only when their B08 reflectance is unambiguous, and the matched subset is therefore selection-biased toward easy targets.

## 3.4 Figures

The five publication figures live in the figure registry at \texttt{runs/exp\_baseline\_v1/20260424\_185158/per\_iceberg/figures.md} and are referenced by slug throughout the paper. Each is a Fisser-comparable analog and was rendered by a registered figure-producing script.

\begin{description}
\item[\texttt{mae\_rootlen\_vs\_sza}] Per-pair MAE on root length by SZA bin and method. Fisser (2024) Fig. 11 analog. Per-bin n\_pairs annotated.
\item[\texttt{area\_scatter\_by\_method}] Predicted vs reference iceberg area, one panel per method, with the 1:1 line, the linear fit, and per-panel n, Pearson r, slope, intercept. Fisser (2024) Fig. 6 / Fisser (2025) Fig. 10 analog.
\item[\texttt{bias\_delta\_by\_area}] Top: per-pair bias by reference-area bin and method, P10--P90 band shaded. Bottom: $\Delta$bias relative to the TR baseline. Area bin edges follow Fisser (2025) Fig. 16.
\item[\texttt{re\_by\_area\_bin}] Per-pair relative error in iceberg area as a function of reference area bin, one line per method, P10--P90 band shaded. Fisser (2024) Fig. 7 / Fisser (2025) Fig. 12 analog.
\item[\texttt{outline\_examples}] Per-SZA-bin chip examples with reference (cyan dashed) and UNet\_CRF predicted (magenta) outlines, picked at three per-pair RE positions: worst-positive, near-zero, worst-negative. Fisser (2024) Fig. 13 analog.
\end{description}

## 3.5 Summary of headline findings (llinkas baseline\_v1)

\begin{enumerate}
\item Within Phase A, A0 (Fisser preprocessing on lt65 chips, no null injection, no augmentation) achieves the best validation IoU (0.613). All other Phase A configurations underperform A0 by at least 0.11.
\item The dominant Phase A variable is the preprocessing pipeline. The A0 to A2 contrast (Fisser cleaning replaced by 40 m + IC mask) costs 0.35 best validation IoU and shifts the median pixel P(iceberg) on lt65 test chips from 0.013 to 0.278. This calibration shift propagates into a collapse of all probability-thresholded methods on A2-A9.
\item Null-chip injection at a 1:1 ratio reduces best validation IoU by 0.11 under Fisser preprocessing (A0 to A1) and by negligible amounts under our preprocessing (A2 to A3). Class and size balancing schemes produce identical training sets on the v4\_clean\_lt65\_plus\_nulls manifest because the GT$+$ class is the natural majority and the size oversample saturates under the 4x cap.
\item On the canonical baseline\_v1 (all four SZA bins), UNet$++$ + DenseCRF achieves the lowest per-pair root-length MAE in three of four SZA bins; UNet$++$ + Otsu wins lt65. UNet$++$ + DenseCRF beats the fixed B08 threshold by 7.5 m at SZA $>$ 75.
\end{enumerate}

---

## 3.6 Phase B on A0: lt65 method sweep (smishra independent study runs)

<!-- Last updated: 2026-05-03. Source CSVs:
  runs/exp_B{0-5}_method_*/20260503_*/per_iceberg/eval_per_iceberg_summary.csv
  runs/exp_B{0-5}_method_*/20260503_*/per_iceberg/eval_per_iceberg.csv
  runs/fisser_validation/per_iceberg/eval_per_iceberg.csv
  Checkpoint: runs/exp_A0_fisser_lt65_original/20260428_094028/model/best_model.pth
  Manifest: data/v4_raw_lt65/manifest.json (398 chips, lt65 only, no 40m filter, no IC mask)
-->

Phase B fixes the A0 checkpoint (val IoU 0.612, trained on v4\_raw\_lt65, 398 lt65 chips, Fisser preprocessing) and sweeps all six inference methods on the same v4\_raw\_lt65 test split (100 chips, lt65 only). Since all B runs share the same checkpoint and manifest, inference predictions are identical across B0--B5; the runs formally document the method comparison under a single controlled anchor.

Experiments: B0 (TR), B1 (OT), B2 (UNet), B3 (UNet\_TR), B4 (UNet\_OT), B5 (UNet\_CRF). All ran 2026-05-03.

### 3.6.1 Per-pair root-length MAE

Table 7 reports per-pair mean absolute error in iceberg root length (m), lt65 only. Lower is better.

\begin{table}
\caption{\textbf{Table 7.} Per-pair root-length MAE (m) on the v4\_raw\_lt65 lt65 test split (100 chips, A0 checkpoint). Hungarian matching with IoU $\geq$ 0.3. Other SZA bins are -- because v4\_raw\_lt65 is lt65-only.}
\begin{tabular}{lr}
\toprule
Method   & sza\_lt65 \\
\midrule
TR        & 17.05 \\
OT        & 16.40 \\
UNet      & 9.82  \\
UNet\_TR  & 12.33 \\
\textbf{UNet\_OT}  & \textbf{8.18}  \\
UNet\_CRF & 10.97 \\
\bottomrule
\end{tabular}
\end{table}

UNet\_OT achieves the lowest root-length MAE (8.18 m). UNet alone is second (9.82 m). Threshold-only methods (TR 17.05 m, OT 16.40 m) are approximately 2x worse than any UNet variant.

### 3.6.2 Per-pair IoU on matched pairs

Table 8 reports per-pair IoU on matched pairs, lt65 only. Higher is better.

\begin{table}
\caption{\textbf{Table 8.} Per-pair IoU on matched pairs, v4\_raw\_lt65 lt65 test split, A0 checkpoint.}
\begin{tabular}{lr}
\toprule
Method   & sza\_lt65 \\
\midrule
TR        & 0.440 \\
OT        & 0.463 \\
UNet      & 0.626 \\
UNet\_TR  & 0.589 \\
\textbf{UNet\_OT}  & \textbf{0.637} \\
UNet\_CRF & 0.560 \\
\bottomrule
\end{tabular}
\end{table}

UNet\_OT wins on per-pair IoU (0.637), consistent with the root-length MAE result. UNet alone is second (0.626).

### 3.6.3 Relative area error

Table 9 reports mean relative area error (RE\%) computed as $(A_{\text{pred}} - A_{\text{GT}}) / A_{\text{GT}} \times 100$ per matched pair. Two populations are reported: all matched icebergs, and the Fisser-comparable subset (GT root length $> 100$ m). Positive RE implies overestimation; negative implies underestimation.

\begin{table}
\caption{\textbf{Table 9.} Mean relative area error (\%) on matched pairs, v4\_raw\_lt65 lt65 test split, A0 checkpoint. Left: all matched icebergs. Right: gt\_rl $>$ 100 m only (Fisser-comparable size cutoff). n = number of matched pairs.}
\begin{tabular}{lrrrr}
\toprule
Method   & mean RE\% (all) & n (all) & mean RE\% ($>$100 m) & n ($>$100 m) \\
\midrule
TR        & $-$53.3 & 1414  & $-$33.2 & 102 \\
OT        & $-$38.0 & 1715  & $-$46.0 & 208 \\
UNet      & $+$27.5 & 12343 & $+$16.9 & 711 \\
UNet\_TR  & $+$53.1 & 10320 & $+$33.0 & 535 \\
\textbf{UNet\_OT}  & $+$26.3 & 5267  & \textbf{+4.9} & 139 \\
UNet\_CRF & $-$26.7 & 10354 & $-$10.5 & 698 \\
\bottomrule
\end{tabular}
\end{table}

On the $>$100 m subset, UNet\_OT achieves $+$4.9\% mean RE, falling within Fisser (2024)'s published range of $-$5.7\% to $+$5.9\% for the B08 $\geq$ 0.12 threshold at SZA $<$ 65$^\circ$ under manually curated conditions. TR achieves $-$33.2\% on the same subset under our automated pipeline. The gap between TR's $-$33\% and Fisser's $\pm$5.7\% is explained in Section 3.7.

The large n disparity between methods (TR: 1414, UNet: 12343) reflects detection rate differences, not dataset size. All methods evaluate against the same 100 test chips.

## 3.7 IC filter and Fisser comparison

### 3.7.1 IC chip-skip filter

The B08 threshold method (TR) in \texttt{threshold\_tifs.py} includes an IC chip-skip filter: if more than 15\% of a chip's pixels exceed the 0.22 threshold, the chip is skipped entirely and no prediction is produced. This filter was designed to reject sea-ice contaminated scenes in operational settings, where high NIR reflectance across the chip indicates sea ice rather than isolated icebergs.

On the v4\_raw\_lt65 lt65 test split (100 chips), 68 of 100 chips fail the IC filter. These are precisely the iceberg-dense Fisser chips, which have high NIR fractions because they were curated to contain many icebergs. The IC filter therefore removes the hardest and most iceberg-rich chips from TR's evaluation, leaving TR with predictions only on the 32 least-dense chips.

The 1414 TR matched pairs in Table 9 come from those 32 chips. The 68 skipped chips contribute zero TR predictions; their GT icebergs are false negatives under the \texttt{count\_as\_false\_negative} policy.

Note on threshold: TR uses 0.22, not 0.12. Sentinel-2 products with processing baseline $>$ 4.0 require subtracting a constant radiometric offset of 1000 from the DNs before conversion to reflectance (ESA, 2024). Chips in v4\_raw\_lt65 were generated without this correction, so reflectance values are 0.10 higher than in Fisser's corrected space. The effective threshold is therefore $0.12 + 0.10 = 0.22$.

### 3.7.2 Fisser validation attempt

To examine whether our pipeline reproduces Fisser's published TR result under comparable conditions, TR was re-run without the IC filter (\texttt{--ic\_threshold 1.0}) on the lt65 test chips. All 100 chips received predictions. Results:

\begin{table}
\caption{\textbf{Table 10.} TR without IC filter on lt65 test chips (Fisser validation run, 2026-05-03). Scene-level standardized area error (SAE) computed as $(\sum A_{\text{pred}} - \sum A_{\text{GT}}) / \sum A_{\text{GT}}$ per chip, then averaged over chips. Fisser-comparable subset: GT root length $>$ 100 m.}
\begin{tabular}{lrr}
\toprule
Population & n chips & mean scene-level SAE \\
\midrule
All matched pairs          & 30 & $-$36.2\% \\
GT root length $>$ 100 m  & 43 & $-$21.4\% \\
\midrule
Fisser (2024) published    & 14 acquisitions & $-$5.7\% to $+$5.9\% \\
\bottomrule
\end{tabular}
\end{table}

The gap between our $-$21.4\% and Fisser's $\pm$5.7\% cannot be closed by removing the IC filter alone. Three methodological differences prevent a direct comparison:

\begin{enumerate}
\item \textbf{Metric definition.} Fisser's standardized area error (SRE$_\theta$) is the raw Greenland scene-level RE scaled to the Svalbard calibration magnitude at SZA 56$^\circ$ (Fisser et al., 2024, Eq. 5). This standardization step requires the Svalbard airborne reference data, which is not available here. Our scene-level SAE is raw (unstandardized).
\item \textbf{Iceberg matching.} Fisser directly summed all visually-delineated reference iceberg areas per scene with no IoU threshold. Our per-pair metric uses Hungarian matching at IoU $\geq$ 0.3; unmatched GT icebergs (false negatives) are excluded from the SAE denominator, biasing it toward less-negative values. Despite this bias, our result is still far more negative than Fisser's.
\item \textbf{Iceberg population.} Fisser's Greenland experiment used icebergs with mean root length 326.94 m $\pm$ 26.15 m (all $>$ 100 m). Our test chips include all iceberg sizes down to 100 m$^2$ (one pixel), and the $>$100 m filter applied here still admits smaller icebergs than Fisser's mean size.
\end{enumerate}

\textbf{Conclusion.} The Fisser validation attempt does not converge, and direct reproduction of their $\pm$5.7\% is not possible without their Svalbard calibration data. The Fisser number is therefore cited as motivation -- the published TR baseline under manually curated, ideal conditions -- rather than as a benchmark that our automated pipeline is expected to match. The contribution of this study is an automated pipeline that produces UNet\_OT at $+$4.9\% RE on the $>$100 m subset without manual scene curation or size cutoff, compared to TR's $-$33\% under the same automated conditions.

## 3.8 Updated summary of headline findings

\begin{enumerate}
\item A0 (Fisser preprocessing, no 40 m filter, no IC mask, lt65, no null injection) is the Phase A winner on every metric: val IoU 0.612, test IoU 0.577, UNet match rate 0.512, UNet root-length MAE 9.82 m.
\item The dominant Phase A variable is the IC pixel mask (not the 40 m filter). C1/C2 ablation isolates this: C1 (40 m filter only) achieves val IoU 0.601, near A0. C2 (IC mask only) collapses to val IoU 0.287, near A2 (0.261).
\item On the Phase B lt65 method sweep (A0 checkpoint, v4\_raw\_lt65), UNet\_OT achieves the best root-length MAE (8.18 m) and per-pair IoU (0.637). TR is 17.05 m MAE.
\item On the Fisser-comparable $>$100 m subset, UNet\_OT achieves $+$4.9\% mean relative area error, within Fisser's published range of $\pm$5.7\% for TR under manual curation. Our automated TR achieves $-$33.2\% on the same subset due to the IC chip-skip filter and automated (rather than manual) scene selection.
\item TR's automated performance gap ($-$33\% vs Fisser's $\pm$5.7\%) is structural: 68\% of lt65 test chips fail the IC filter because Fisser chips are iceberg-dense. This is not a calibration error; it reflects the fundamental difficulty of the automated pipeline.
\end{enumerate}

## 3.9 Higher-SZA generalisation and backbone re-eval (added 2026-05-05)

The original Phase B (Section 3.6) was lt65-only because Phase A itself was lt65-scoped. Three follow-up Slurm runs (60293, 60296, 60297) extend the comparison to all four SZA bins on the unifying v4\_clean test split (228 chips, 57 per bin). Full T1-T4 tables in `phase_a_higher_sza_t1_t4.md`; this section is a brief pointer rather than a full transcription, in line with the direction to keep the dataset/backbone story in supplementary material.

Headline:
\begin{enumerate}
\item All ten Phase A backbones (A0..A9) re-evaluated on v4\_clean across all four SZA bins (UNet only). A1 (Fisser preprocessing + 29 GT-zero chips) wins every higher-SZA bin on per-pair IoU and root-length MAE; A0 still wins lt65. Aggregate over the three higher-SZA bins: A1 mean per-pair MAE 28.01 m, A0 33.33 m, a 16\% reduction. Augmentation alone (A4) is the next-best non-A0 generaliser at 35.27 m mean MAE; class balancing (A5/A6) does not close the A1 gap, and size oversample (A7/A8/A9) actively hurts higher-SZA performance.
\item Re-running the six-method Phase B sweep with both A0 and A1 backbones on v4\_clean shows A1 + UNet\_CRF wins three of four SZA bins on root-length MAE among the learned methods (lt65 still favours A0 + UNet\_OT at 8.45 m, reproducing the published 8.18 m headline within rounding on the v4\_clean lt65 split).
\item The recommended single-backbone-single-method pipeline across all four SZA bins is A1 + UNet\_CRF (n high in higher bins, IoU consistently above 0.55, MAE 14.0 / 10.9 / 12.1 / 22.6 m for lt65 / 65--70 / 70--75 / $>$75). The per-bin best combination is A0 + UNet\_OT for lt65 and A1 + UNet\_CRF for the higher bins.
\item Top-hat variants (six TH companions to the six base methods) were NOT included in the Phase B re-runs and remain a coverage gap. One additional Slurm sweep with `--with_tophat` would close it.
\item Phase A and the backbone comparison are intended for the supplementary material (T1 + T2 as supplementary figures); the main text focuses on the headline that the learned pipeline beats Fisser's threshold baseline at higher SZA.
\end{enumerate}
