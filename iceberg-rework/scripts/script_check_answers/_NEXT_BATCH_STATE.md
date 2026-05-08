# Script-check next-batch state

Stopped because the harness sandbox blocks `git commit` (and `git push`,
`rsync`, and direct `python3 …` invocations from the local Mac). SSH to
HPC, file reads/writes inside the project, and `git status / git log /
git add / git diff` all work, but every attempted `git commit` returns
"Permission to use Bash has been denied" with no explanation.

Until commits unblock, no further questions in this batch can land. The
work below has been done up to the staging step; running it locally
should be a single `git commit -m "..."` per question.

## Status by question

### Q2 (q02_polygonisation_artifacts) — DONE on HPC, files local, STAGED, NOT COMMITTED

- Script: `iceberg-rework/scripts/script_check_answers/q02_polygonisation_artifacts.py`
  (also pushed to HPC at the same path under
  `/mnt/research/.../iceberg-rework/scripts/script_check_answers/`).
- HPC run: 14,109 of 23,981 chips passed the IC filter; runtime ~14 min.
- Outputs synced to
  `paper-writing/figure_review/script_check_answers/q02_polygonisation_artifacts/`
  (CSV + 2 PNGs).
- Headline: 30.5% of pre-cutoff polygons are 1-pixel; 46.7% are <= 2-pixel;
  100 m² cutoff = 1 px so it keeps every polygon (not a salt-and-pepper
  filter). 1-iter binary_opening drops 71.9% of kept polygon count and
  13.6% of total area (1078.95 → 931.76 km²).
- Docs updated:
  - `paper-writing/figure_review/figure_review_checklist.csv` (row added)
  - `paper-writing/methods_draft.md` Section 2.14 (Q2 paragraph appended)
  - `iceberg-rework/script-check-README.md` (Pre-checked sub-bullet)
- All Q2 changes are staged via `git add` and ready to commit.

To finish Q2:
```
cd /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026
git commit -m "Add q02_polygonisation_artifacts empirical answer to script-check pack

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git push origin paper-figures-and-results
```

### Q3 (q03_connectivity) — script READY, not yet run

Script written + parse-OK on HPC at
`/mnt/research/.../iceberg-rework/scripts/script_check_answers/q03_connectivity.py`
and locally at the matching path. Runs `rasterio.features.shapes` at 4-
and 8-connectivity over the same 14k IC-passing chips and reports per-
chip count / area deltas plus a per-(region, SZA bin) bar chart.

To run:
```
ssh bowdoin 'cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/script_check_answers && nohup python3 -u q03_connectivity.py > /tmp/q03_out.log 2>&1 &'
```

Expected runtime: ~12-15 min.

### Q4 (q04_ndwi_sweep) — script READY, audits + extends existing CSV

Script written + parse-OK. Existing
`iceberg-rework/sweeps/ndwi_threshold_sweep.csv` (4.2 MB, 33,640 rows)
covers KQ at NDWI thresholds {-0.05, 0.0, 0.05, 0.1} for all four SZA
bins. The script appends SK and NDWI=0.2 rows in place, then emits a
summary CSV + figures.

The CSV has already been pushed to HPC at
`/mnt/research/.../iceberg-rework/sweeps/ndwi_threshold_sweep.csv`.

To run:
```
ssh bowdoin 'cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/script_check_answers && nohup python3 -u q04_ndwi_sweep.py > /tmp/q04_out.log 2>&1 &'
```

### Q5–Q10 (q08, q09, q10) — scripts READY, not yet run

- `q08_otsu_log.py` — per-chip Otsu on B08 vs log(B08+eps). Iterates the
  full 24k chip pool. Expected ~10-15 min.
- `q09_ic_fixed_vs_otsu.py` — fixed-IC vs Otsu-IC test. Expected ~10-15 min.
- `q10_otsu_ic_order.py` — Otsu-then-IC vs IC-then-Otsu. Expected ~10-15 min.

### Q12–Q14 (q12, q13, q14) — scripts in pack, Q14 not yet written

- `q12_tophat_se_radius.py` — disk(r) sweep on v4_clean test chips
  (228 chips, 4 radii). Expected fast, ~1-3 min.
- `q13_tophat_threshold.py` — sigma / Otsu of top-hat response on v4_clean
  test chips. Expected fast.
- **q14_tophat_crs_audit.py — NOT YET WRITTEN**. Per spec: enumerate chip
  CRS values across the 24k chip pool; flag any base-method
  GeoDataFrames whose CRS differs from the chip CRS in
  `tophat_recover.py`'s rasterisation path. Metadata-only audit, no
  per-chip math.

### Q20 (q20_crf_iterations) — NOT YET WRITTEN

Per spec: run `densecrf_tifs` with iterations in {1, 3, 5, 10, 20} on a
chip subset (use existing UNet probs at the canonical probs root);
measure mask-delta between successive iterations. Reuse
`densecrf_tifs.py` / `crf_utils.py`.

## Other unstaged tree changes (out of scope for this batch)

The working tree already had unrelated edits before this batch:
- `iceberg-rework/slurm/re_phase_b_tophat.slurm`
- `paper-writing/figure_review/figure_review_checklist__meeting_review.csv`
- `paper-writing/figures/figures.md`
- a number of new untracked configs / make_figS scripts under
  `iceberg-rework/`.

These are NOT staged in the Q2 commit and should be committed separately
according to their own workstreams.
