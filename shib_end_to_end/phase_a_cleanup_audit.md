# Phase A Cleanup Audit

**Date:** 2026-05-02 (lt65 audit); 2026-05-05 follow-up linked below.
**Scope:** Phase A lt65 runs, A0-A9
**Status:** CSV audit confirms A0 is the lt65 Phase A winner under validation and per-iceberg metrics. A5/A6 only win one narrow chip-level diagnostic metric. The follow-up C1/C2 ablation shows the 40 m filter alone is not the main cause of the A2 collapse; the IC-only intervention causes the validation collapse.

> **2026-05-05 update:** Re-evaluating every Phase A backbone on the v4_clean test split for all four SZA bins (Slurm 60293) shows A1 wins every higher-SZA bin, while A0 still wins lt65. Aggregate over the three higher-SZA bins: A1 mean per-pair MAE 28.01 m vs A0's 33.33 m (16% lower). A1 + UNet_CRF is the strongest single-pipeline option across all four SZA bins. Full T1-T4 tables in [phase_a_higher_sza_t1_t4.md](phase_a_higher_sza_t1_t4.md). The lt65 conclusion below stands; the higher-SZA story is additive, not contradictory.

---

## Bottom Line

**Best lt65 Phase A run: A0 (`exp_A0_fisser_lt65_original`).**

A0 is best on the metrics that should drive the Fisser-reproduction / paper-facing Phase A decision:

| Metric | A0 value | Rank |
|---|---:|---|
| Best validation IoU | 0.612772 | best |
| UNet per-iceberg matched IoU | 0.6259 | best |
| UNet root-length MAE | 9.8212 m | best / lowest |
| UNet matched pairs | 12,343 | best / most |

A5/A6 should **not** be called the global Phase A winner. They only win the chip-level GT-positive pixel-IoU diagnostic:

| Metric | A5/A6 value | Caveat |
|---|---:|---|
| Chip-level GT-positive IoU | 0.0373 | tiny absolute value; tiny margin over A4; conflicts with validation and per-iceberg metrics |

Recommended wording:

> A0 is the Phase A winner for the lt65/Fisser-reproduction experiment. A5/A6 are only the best cleaned-family runs under chip-level GT-positive pixel IoU, and should not be used as the global Phase A winner.

---

## CSV Evidence

Run location:

```text
/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/runs
```

Audit output from moosehead:

```text
exp,best_val_iou,chip_gtpos_iou,per_iceberg_iou,rl_mae,n_pairs
A0,0.612772,0.0313,0.6259,9.8212,12343
A1,0.502823,0.0345,0.5338,15.2064,7597
A2,0.260566,0.0268,0.4736,15.2627,2077
A3,0.269109,0.035,0.4851,15.6876,1543
A4,0.22482,0.0368,0.4592,14.9327,1035
A5,0.236857,0.0373,0.4999,15.2348,1340
A6,0.236857,0.0373,0.4999,15.2348,1340
A7,0.24326,0.0342,0.4812,14.7849,1382
A8,0.24326,0.0342,0.4812,14.7849,1382
A9,0.24326,0.0342,0.4812,14.7849,1382
```

Files read:

```text
runs/<exp>/<timestamp>/model/training_log.csv
runs/<exp>/<timestamp>/evaluation/eval_summary_gt_positive_only.csv
runs/<exp>/<timestamp>/per_iceberg/eval_per_iceberg_summary.csv
```

---

## C1/C2 Cleanup Ablation Results

Follow-up runs were submitted from the writable mirror:

```text
/mnt/research/v.gomezgilyaspik/students/smishra/llinkas-rework
```

Completed runs:

```text
C1  exp_C1_40m_only_lt65  20260502_231718
C2  exp_C2_ic_only_lt65   20260502_232344
```

Per-iceberg summaries for C1/C2 were added after the main run with `scripts/eval_per_iceberg.py`, because `run_experiment.py` only ran the chip-level evaluator.

Combined comparison:

```text
label,run,best_val_iou,chip_UNet_iou,chip_best_method,chip_best_iou,perpair_UNet_iou,UNet_rootlen_MAE,n_pairs
A0_no_cleanup,20260428_094028,0.612772,0.0313,UNet_CRF,0.0325,0.6259,9.8212,12343
C1_40m_only,20260502_231718,0.601926,0.0279,UNet_CRF,0.0288,0.6454,13.0100,5054
C2_ic_only,20260502_232344,0.287281,0.0331,UNet_TR,0.0370,0.6653,9.6857,6458
A2_40m_plus_ic,20260428_094654,0.260566,0.0268,UNet_TR,0.0483,0.4736,15.2627,2077
```

Interpretation:

- C1 is close to A0 on validation (`0.601926` vs `0.612772`), so the 40 m filter alone does not explain the A2 collapse.
- C2 collapses on validation (`0.287281`), close to A2 (`0.260566`), so the IC chip-drop / train-time IC mask is the primary suspect.
- C2's matched-pair UNet IoU and root-length MAE look good, but this is over far fewer matched pairs than A0 and after a changed test pool. Do not use that alone to declare C2 better.
- A2 is worst overall under the paper-facing combination: low validation, low per-pair IoU, high root-length MAE, and many fewer matched pairs.

Current conclusion:

> For lt65 reproduction, keep A0/no-cleanup as the anchor. The 40 m filter appears tolerable by itself; the IC intervention is the likely source of the training/evaluation collapse and needs a separate design review before being used as a training cleanup step.

---

## Interpretation

The disagreement is metric-specific, not mysterious:

- `eval_summary_gt_positive_only.csv` is a chip-level pixel-IoU aggregation. It makes A5/A6 look slightly better.
- `training_log.csv` best validation IoU strongly favors A0.
- `eval_per_iceberg_summary.csv` strongly favors A0 on matched-object IoU and root-length MAE.

For this project, per-iceberg metrics matter more for the paper because they are closer to the Fisser-style object/area/root-length comparison. The chip-level pixel IoU is useful as a diagnostic, but it should not select the Phase A winner by itself.

---

## What This Does And Does Not Mean

This does **not** mean the cleanup pipeline is invalid across the whole project.

It means:

- Do not claim cleanup improves the lt65/Fisser-reproduction UNet training result.
- Do not use A5/A6 as the Phase A winner unless the scope is explicitly "cleaned-family, chip-level GT-positive IoU."
- Treat cleanup as a methodological intervention with tradeoffs.

It does **not** mean:

- never do root-length filtering,
- never merge shadows,
- never use IC masking,
- or throw out `v4_clean`.

The suspicious part is the lt65-only training/calibration behavior, especially the train-only IC mask: training chips are partly zeroed, while validation/test chips are not. That can create a domain shift.

---

## Completed Follow-Up Run

This section records the run plan that was executed for C1/C2. The key result is now in the C1/C2 ablation section above.

Do **not** run Phase B from A5.

The completed cleanup ablation was designed to isolate whether the lt65 drop came from:

1. the 40 m root-length filter,
2. the IC chip-drop / train-time IC mask,
3. their combination,
4. or the split/pool change.

### Ablation Matrix

Use the same seed, training hyperparameters, balancing scheme, and augmentation setting as A0/A2:

- seed = 42
- augmentation off
- `scheme_A_fisser_original`
- lt65 only
- UNet focus

| Proposed ID | Size filter | IC chip-drop/mask | Build flags | Existing analog |
|---|---|---|---|---|
| `exp_C0_raw_lt65_locked` | off | off | `--skip_size_filter --skip_ic_mask --filter_sza_bin sza_lt65` | A0-style |
| `exp_C1_40m_only_lt65_locked` | on | off | `--skip_ic_mask --filter_sza_bin sza_lt65` | new |
| `exp_C2_ic_only_lt65_locked` | off | on | `--skip_size_filter --filter_sza_bin sza_lt65` | new |
| `exp_C3_40m_ic_lt65_locked` | on | on | `--filter_sza_bin sza_lt65` | A2-style |

Important: these must use a **locked split** if the goal is causal attribution. A0 vs A2 currently differs in both preprocessing and test pool. The ablation should keep the same chip identities wherever possible, or explicitly report which chips are dropped by IC before comparing metrics.

### Immediate Practical Plan

1. Build the two missing intermediate manifests. Do this on a compute node, `dover` / `foxcroft`, or through Slurm; do not run the build directly on the moosehead login node.

```bash
cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework

# 40 m only: size filter on, IC disabled
python3 scripts/build_clean_dataset.py \
  --skip_ic_mask \
  --filter_sza_bin sza_lt65 \
  --out_dir data/v4_lt65_40m_only \
  --manifest_id v4_lt65_40m_only

# IC only: raw annotations/components, IC enabled
python3 scripts/build_clean_dataset.py \
  --skip_size_filter \
  --filter_sza_bin sza_lt65 \
  --out_dir data/v4_lt65_ic_only \
  --manifest_id v4_lt65_ic_only
```

Slurm form from moosehead:

```bash
sbatch --job-name=build_40m_only --partition=gpu --gres=gpu:rtx3080:1 \
  --cpus-per-task=2 --mem=32G --time=01:00:00 \
  --output=logs/exp/build_40m_only_%j.out \
  --error=logs/exp/build_40m_only_%j.err \
  --wrap='cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework && /home/llinkas/.venvs/iceberg-unet312/bin/python scripts/build_clean_dataset.py --skip_ic_mask --filter_sza_bin sza_lt65 --out_dir data/v4_lt65_40m_only --manifest_id v4_lt65_40m_only'

sbatch --job-name=build_ic_only --partition=gpu --gres=gpu:rtx3080:1 \
  --cpus-per-task=2 --mem=32G --time=01:00:00 \
  --output=logs/exp/build_ic_only_%j.out \
  --error=logs/exp/build_ic_only_%j.err \
  --wrap='cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework && /home/llinkas/.venvs/iceberg-unet312/bin/python scripts/build_clean_dataset.py --skip_size_filter --filter_sza_bin sza_lt65 --out_dir data/v4_lt65_ic_only --manifest_id v4_lt65_ic_only'
```

2. Check their manifests before training:

```bash
python3 - <<'EOF'
import json
for mid in ["v4_raw_lt65", "v4_lt65_40m_only", "v4_lt65_ic_only", "v4_clean_lt65"]:
    p = f"data/{mid}/manifest.json"
    d = json.load(open(p))
    chips = d.get("chips", [])
    test = [c for c in chips if c.get("split") == "test"]
    gtpos = [c for c in test if c.get("has_iceberg")]
    print(mid, "total", d.get("total_chips") or len(chips),
          "test", len(test), "test_gtpos", len(gtpos),
          "filters", d.get("filters"))
EOF
```

3. Add experiment YAMLs by copying A0/A2 patterns:

```text
configs/experiments/exp_C1_40m_only_lt65.yaml
configs/experiments/exp_C2_ic_only_lt65.yaml
```

Both should inherit `baseline_v1`, set augmentation off, use `scheme_A_fisser_original`, and point to the new manifest ids.

4. Submit the two new experiments:

```bash
sbatch slurm/exp.slurm exp_C1_40m_only_lt65 manifest,balance,train,infer,evaluate
sbatch slurm/exp.slurm exp_C2_ic_only_lt65 manifest,balance,train,infer,evaluate
```

If `exp.slurm` expects env vars instead of positional args on the current branch, use:

```bash
EXP_ID=exp_C1_40m_only_lt65 STAGES=manifest,balance,train,infer,evaluate sbatch slurm/exp.slurm
EXP_ID=exp_C2_ic_only_lt65 STAGES=manifest,balance,train,infer,evaluate sbatch slurm/exp.slurm
```

5. Re-run the same CSV audit table and compare:

```text
A0 raw
C1 40m only
C2 IC only
A2 40m + IC
```

---

## Decision After Ablation

The ablation result matches this case:

> `C2` collapses but `C1` does not: IC chip-drop/masking is the main issue.

Original decision rule:

- If `C1` collapses but `C2` does not: the 40 m component filter is the main issue.
- If `C2` collapses but `C1` does not: IC chip-drop/masking is the main issue.
- If both are moderate but `C3/A2` collapses: the combination is the issue.
- If all cleaned variants underperform A0: report cleanup as a scientifically motivated but performance-costly preprocessing intervention for lt65 reproduction.

For Phase B:

- Paper/all-SZA method sweep: use `baseline_v1`.
- lt65/Fisser reproduction: use A0.
- cleaned-family diagnostic only: A5/A6 may be discussed, but label them narrowly.

---

## Remote Working Copy Recovery

Because `/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/configs/experiments` is owned by `llinkas` and not group-writable, the new C1/C2 experiment YAMLs cannot be written there by `smishra`.

Current writable mirror:

```text
/mnt/research/v.gomezgilyaspik/students/smishra/llinkas-rework
```

As of the post-rsync inspection, this mirror contains:

```text
configs/
scripts/
slurm/
data/v4_lt65_40m_only/manifest.json
data/v4_lt65_ic_only/manifest.json
runs/exp_A0...exp_A9
```

Before submitting jobs from this mirror, patch the Slurm root paths so jobs write into `smishra/llinkas-rework`, not Lulu's tree:

```bash
cd /mnt/research/v.gomezgilyaspik/students/smishra/llinkas-rework

perl -0pi -e 's#/mnt/research/v\\.gomezgilyaspik/students/llinkas/iceberg-rework#/mnt/research/v.gomezgilyaspik/students/smishra/llinkas-rework#g' \
  slurm/_common.sh slurm/exp.slurm

grep -R "llinkas/iceberg-rework" -n slurm/_common.sh slurm/exp.slurm || echo "slurm paths patched"
grep -n "ROOT=\\|SBATCH --output\\|SBATCH --error\\|source " slurm/_common.sh slurm/exp.slurm
```

Only the Slurm wrapper needs patching for the C1/C2 run. `run_experiment.py` resolves its repo root from its own file path through `validate_experiment.py`, and `run_methods.sh` computes `REPO_DIR` from its script location. Some older/helper scripts still contain hard-coded `llinkas` paths, but they are not on the C1/C2 critical path.

Then write the two YAMLs under the writable mirror:

```bash
cat > configs/experiments/exp_C1_40m_only_lt65.yaml <<'EOF'
# C1: lt65 Fisser chips with 40 m root-length filter only.
# IC chip-drop and train-time IC masking disabled.

id: exp_C1_40m_only_lt65
inherits: baseline_v1

change:
  data:
    manifest_id:      v4_lt65_40m_only
    balancing_scheme: scheme_A_fisser_original
  augmentation:
    enabled: false

controlled_variable: data

progression:
  phase: C
  step: C1
  compared_to: exp_A0_fisser_lt65_original

notes: |
  Uses v4_lt65_40m_only: lt65 Fisser chips with shadow merge and 40 m
  root-length component filtering, but no IC chip-drop and no train-time IC
  pixel mask. This isolates the size-filter effect from the IC effect.
EOF

cat > configs/experiments/exp_C2_ic_only_lt65.yaml <<'EOF'
# C2: lt65 Fisser chips with IC chip-drop / train-time IC mask only.
# 40 m root-length filtering disabled.

id: exp_C2_ic_only_lt65
inherits: baseline_v1

change:
  data:
    manifest_id:      v4_lt65_ic_only
    balancing_scheme: scheme_A_fisser_original
  augmentation:
    enabled: false

controlled_variable: data

progression:
  phase: C
  step: C2
  compared_to: exp_A0_fisser_lt65_original

notes: |
  Uses v4_lt65_ic_only: lt65 Fisser chips with raw component sizes retained,
  but IC chip-drop and train-only IC pixel masking enabled. This isolates the
  IC preprocessing effect from the 40 m size-filter effect.
EOF
```

Validate before submitting:

```bash
/home/llinkas/.venvs/iceberg-unet312/bin/python scripts/validate_experiment.py --exp exp_C1_40m_only_lt65
/home/llinkas/.venvs/iceberg-unet312/bin/python scripts/validate_experiment.py --exp exp_C2_ic_only_lt65
```

Submit from the writable mirror:

```bash
EXP_ID=exp_C1_40m_only_lt65 STAGES=manifest,balance,train,infer,evaluate sbatch slurm/exp.slurm
EXP_ID=exp_C2_ic_only_lt65 STAGES=manifest,balance,train,infer,evaluate sbatch slurm/exp.slurm
```

Monitor:

```bash
squeue -u smishra
tail -f logs/exp/ice_exp_*.out
```

After they finish, run the same CSV audit from the writable mirror and compare A0, C1, C2, A2.

---

## Cleanup Visuals

Ten cleanup visualization PNGs are now available locally after rsync:

```text
/Users/smishra/iceberg-seg/shib_audit/cleanup_viz_sample/
```

These visuals support manual inspection of what the cleanup pipeline does, but they do not override the quantitative result above.
