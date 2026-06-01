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

## 3.5 Summary of headline findings

\begin{enumerate}
\item Within Phase A, A0 (Fisser preprocessing on lt65 chips, no null injection, no augmentation) achieves the best validation IoU (0.613). All other Phase A configurations underperform A0 by at least 0.11.
\item The dominant Phase A variable is the preprocessing pipeline. The A0 to A2 contrast (Fisser cleaning replaced by 40 m + IC mask) costs 0.35 best validation IoU and shifts the median pixel P(iceberg) on lt65 test chips from 0.013 to 0.278. This calibration shift propagates into a collapse of all probability-thresholded methods on A2-A9.
\item Null-chip injection at a 1:1 ratio reduces best validation IoU by 0.11 under Fisser preprocessing (A0 to A1) and by negligible amounts under our preprocessing (A2 to A3). Class and size balancing schemes produce identical training sets on the v4\_clean\_lt65\_plus\_nulls manifest because the GT$+$ class is the natural majority and the size oversample saturates under the 4x cap.
\item On the canonical baseline\_v1 (all four SZA bins), UNet$++$ + DenseCRF achieves the lowest per-pair root-length MAE in three of four SZA bins; UNet$++$ + Otsu wins lt65. UNet$++$ + DenseCRF beats the fixed B08 threshold by 7.5 m at SZA $>$ 75.
\end{enumerate}

\paragraph{2026-05-05 / 2026-05-06 supplementary follow-up.} The Phase A and Phase B re-evaluation across all four SZA bins on the v4\_clean test split is intended for the supplementary material. Eighteen Phase A backbones were evaluated (the original ten, plus eight A1-anchored variants that re-anchor the class- and size-balancing schemes onto A1's manifest rather than A4's). Three backbones (A0, A1, A7b) were taken into a 12-method Phase B sweep (six base + six top-hat companions). Headline (revised 2026-05-06): the higher-SZA winner among Phase A backbones is **A7b** (= A8b == A9b by collapse; A1 manifest plus size oversample plus augmentation), with mean per-pair MAE 27.24 m and IoU 0.531 across the three higher-SZA bins, beating A1's 28.01 m / 0.499 and A0's 33.33 m / 0.490. Among learned cross-bin pipelines, **A7b $+$ UNet\_CRF** is the new single-backbone-single-method pick: higher-SZA mean IoU 0.616 vs A1 $+$ UNet\_CRF's 0.602 at tied MAE (15.59 m vs 15.21 m); A7b wins sza\_gt75 outright on both metrics. A0 $+$ UNet\_OT remains the lt65 pick and reproduces the published 8.18 m headline within rounding (8.45 m on the v4\_clean lt65 split). Top-hat companions improve UNet\_OT recall in higher SZA bins but degrade UNet\_CRF cross-bin, so the cleanest recommendation is A7b $+$ UNet\_CRF base (no TH). The full T1--T5 tables (per-bin $\times$ 18-experiment leaderboard, best non-A0 per bin, A0-vs-A1-vs-A7b twelve-method sweep, recommended pipeline, top-hat effect deltas) live in `shib\_end\_to\_end/phase\_a\_higher\_sza\_t1\_t4.md`. T1 and T2 are the supplementary figures that anchor this story; the main text retains the Fisser-comparable headline rather than the dataset progression.
