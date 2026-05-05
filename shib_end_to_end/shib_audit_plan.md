# Iceberg Segmentation: Human-in-the-Loop Re-Run Plan

**For:** Codex (hand this file as full context)
**Owner:** smishra@bowdoin.edu
**Date:** 2026-05-02
**Status:** Phase A already ran on moosehead. Phase B not started. This plan re-verifies Phase A, generates visualizations, and deliberately chooses the Phase B anchor. Do **not** treat this file's older A5/A6 recommendation as authoritative without the metric-scope review below.

---

## Project Summary (read this first)

Sentinel-2 iceberg segmentation for Greenland fjords (KQ + SK). Binary UNet++ (ResNet34) trained to detect icebergs (shadow merged into iceberg class). Six segmentation methods compared: TR, OT, UNet, UNet_TR, UNet_OT, UNet_CRF.

**Two-phase experimental design:**
- **Phase A** (10 experiments, A0-A9): dataset variant sweep. Each trains a separate UNet++ and evaluates all 6 methods. Controlled variable is which training data / augmentation / balancing scheme to use.
- **Phase B** (6 experiments, B0-B5): method sweep on a deliberately chosen frozen checkpoint. Depending on the metric scope, this may be A0, A5/A6, or the canonical all-bin `baseline_v1`; do not assume the anchor from the table below.

**Metric warning:** The chip-level `eval_summary_gt_positive_only.csv` metric and the Fisser-comparable per-iceberg metrics tell different stories. Mean IoU on sza_lt65 GT-positive test chips makes A5/A6 look best by a tiny margin (0.0373 vs 0.0368 for A4). Training validation IoU, per-iceberg matched IoU, and root-length MAE all favor A0. For paper/Fisser-comparable reporting, prefer the per-iceberg metrics unless there is a specific reason to optimize chip-level pixel IoU.

---

## Key Paths

```
# HPC (moosehead.bowdoin.edu, ssh smishra@moosehead.bowdoin.edu)
LLINKAS = /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework
RESEARCH = /mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas
AUDIT    = $RESEARCH/shib_audit

# Conda env on moosehead
conda activate iceberg-unet

# Local (this machine)
LOCAL    = /Users/smishra/iceberg-seg
SCRIPTS  = $LOCAL/iceberg-rework/scripts
```

**Critical files:**
```
$LLINKAS/data/v4_clean_lt65/manifest.json          # Clean lt65 manifest, not necessarily the Phase A winner
$LLINKAS/data/v4_clean_lt65_plus_nulls/manifest.json
$LLINKAS/data/v4_raw_lt65/manifest.json            # A0 anchor manifest
$LLINKAS/data/v3_clean/split_log.csv               # per-chip metadata (all splits)
$LLINKAS/scripts/run_experiment.py                 # Phase A/B orchestrator
$LLINKAS/scripts/make_figure01_annotation_difficulty.py  # cleanup viz
$LLINKAS/configs/experiments/exp_A*.yaml           # experiment configs
$LLINKAS/configs/experiments/exp_B*.yaml
```

---

## Phase A Results (already run, re-audited 2026-05-02)

All 10 Phase A experiments completed. The table below is **only** the chip-level GT-positive metric from `evaluation/eval_summary_gt_positive_only.csv`; it is useful as a diagnostic, but it should not by itself decide the paper-facing Phase A winner.

### Chip-level GT-positive metric

Results for **UNet, sza_lt65, GT-positive chips**:

| Exp | Description | Chip-level UNet IoU | n_chips | Notes |
|-----|-------------|----------|---------|-------|
| A0 | Fisser raw lt65, no filter, no aug | 0.0313 | 49 | v4_raw_lt65 (398 chips) |
| A1 | Fisser raw lt65 + nulls, no aug | 0.0345 | 49 | v4_raw_lt65_plus_nulls |
| A2 | Our lt65, 40m+IC filter, no nulls, no aug | 0.0268 | 47 | v4_clean_lt65 (330 chips) |
| A3 | Our lt65 + nulls, no aug | 0.0350 | 47 | v4_clean_lt65_plus_nulls |
| A4 | Our lt65 + nulls + aug | 0.0368 | 47 | v4_clean_lt65_plus_nulls |
| A5 | Our lt65 + nulls + aug + 2:1 balance | **0.0373** | 47 | tied best under this metric only |
| A6 | Our lt65 + nulls + aug + adaptive balance | **0.0373** | 47 | tied best under this metric only |
| A7 | Our lt65 + nulls + aug + size balance | 0.0342 | 47 | size balancing hurts |
| A8 | Our lt65 + nulls + aug + 2:1 + size | 0.0342 | 47 | no improvement over A7 |
| A9 | Our lt65 + nulls + aug + adaptive + size | 0.0342 | 47 | no improvement over A7 |

### Paper-facing / Fisser-comparable metrics

Read-only audit of the same run directories found the following for **UNet, sza_lt65**:

| Exp | best val IoU | per-iceberg mean IoU | per-iceberg root-length MAE (m) | Interpretation |
|-----|--------------|----------------------|----------------------------------|----------------|
| A0 | **0.6128** | **0.6259** | **9.82** | Best by training validation, matched IoU, and root-length MAE |
| A1 | 0.5028 | 0.5338 | 15.21 | Second by validation and matched IoU |
| A2 | 0.2606 | 0.4736 | 15.26 | Our cleanup pipeline hurts lt65-only calibration |
| A3 | 0.2691 | 0.4851 | 15.69 | Nulls help slightly within clean family |
| A4 | 0.2248 | 0.4592 | 14.93 | Augmentation does not recover A0-level behavior |
| A5 | 0.2369 | 0.4999 | 15.23 | Best clean-family chip-IoU run, not global winner |
| A6 | 0.2369 | 0.4999 | 15.23 | Same as A5 |
| A7 | 0.2433 | 0.4812 | 14.78 | Size balance slightly improves RL MAE within clean family |
| A8 | 0.2433 | 0.4812 | 14.78 | Same as A7 |
| A9 | 0.2433 | 0.4812 | 14.78 | Same as A7 |

**A0 vs A2 discrepancy (important):** A0 and A2 use different test pools (raw lt65 has 398 chips; clean lt65 has 330 chips), so their chip-level table is confounded. But the stronger finding is broader than that table: A0 also wins on training validation IoU and per-iceberg matched metrics. The cleanup pipeline is scientifically important, but it is not automatically performance-improving for lt65-only training.

**Current decision rule:** A5/A6 are only "winners" under the chip-level GT-positive metric and only by a tiny margin. For paper/Fisser-comparable claims, A0 is the Phase A lt65 winner. For Phase B across all SZA bins, the canonical `baseline_v1` checkpoint remains the safest anchor unless the human explicitly chooses a cleaned-family lt65-only experiment.

Candidate checkpoints:
```
# A5: best clean-family chip-level GT-positive IoU
$LLINKAS/runs/exp_A5_our_lt65_plus_nulls_aug_2pos/20260430_001810/model/best_model.pth

# canonical all-bin baseline, likely paper Phase B anchor
$LLINKAS/runs/exp_baseline_v1/20260424_185158/model/best_model.pth
```

---

## Step 0: Setup

### 0.1 Create audit directory

```bash
ssh smishra@moosehead.bowdoin.edu
mkdir -p /mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/shib_audit
mkdir -p $AUDIT/phase_a_results
mkdir -p $AUDIT/cleanup_viz
mkdir -p $AUDIT/phase_b_results
```

### 0.2 Confirm conda env and key imports

```bash
conda activate iceberg-unet
python3 -c "import torch, rasterio, pandas, numpy, matplotlib; print('OK')"
```

If any import fails: `pip install <package>` inside the env.

### 0.3 Rsync any updated local scripts to moosehead

```bash
rsync -av /Users/smishra/iceberg-seg/iceberg-rework/scripts/*.py \
    smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
```

---

## Step 1: Manifest Audit

**Goal:** Confirm the three manifests used in Phase A have correct chip counts and filter parameters. This verifies the data pipeline ran correctly before training.

### 1.1 Run the manifest check

```bash
ssh smishra@moosehead.bowdoin.edu
conda activate iceberg-unet
cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework

python3 - <<'EOF'
import json, sys

manifests = {
    "v4_raw_lt65":           "data/v4_raw_lt65/manifest.json",
    "v4_clean_lt65":         "data/v4_clean_lt65/manifest.json",
    "v4_clean_lt65_plus_nulls": "data/v4_clean_lt65_plus_nulls/manifest.json",
}

expected = {
    "v4_raw_lt65": {
        "total_chips": 398,
        "filters.shadow_merge": True,
        "filters.root_length_min_m": 0,
        "filters.ic_mask_scope": "none",
        "filters.filter_sza_bin": "sza_lt65",
    },
    "v4_clean_lt65": {
        "total_chips": 330,
        "filters.shadow_merge": True,
        "filters.root_length_min_m": 40,
        "filters.ic_threshold": 0.15,
        "filters.ic_mask_scope": "train_only",
        "filters.filter_sza_bin": "sza_lt65",
    },
    "v4_clean_lt65_plus_nulls": {
        "total_chips": 359,
        "base_manifest_id": "v4_clean_lt65",
        "n_added_nulls": 29,
    },
}

all_ok = True
for name, path in manifests.items():
    with open(path) as f:
        d = json.load(f)
    print(f"\n=== {name} ===")
    for key, exp_val in expected[name].items():
        parts = key.split(".")
        actual = d
        for p in parts:
            actual = actual.get(p, "MISSING") if isinstance(actual, dict) else "MISSING"
        ok = actual == exp_val
        if not ok:
            all_ok = False
        status = "OK" if ok else f"FAIL (got {actual!r}, want {exp_val!r})"
        print(f"  {key}: {status}")
    # Print counts_by_split
    counts = d.get("counts_by_split", {})
    for split, sc in counts.items():
        print(f"  {split}: GT+={sc.get('gt_positive',0)}  GT0={sc.get('gt_zero',0)}")

print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
EOF
```

### 1.2 Verify IC masking was train-only

```bash
python3 - <<'EOF'
import pandas as pd
log = pd.read_csv("data/v3_clean/split_log.csv")

# ic_masked should only be True for train split
for split in ["train", "val", "test"]:
    sub = log[log["split"] == split]
    n_masked = sub["ic_masked"].astype(str).str.lower().eq("true").sum()
    print(f"{split}: {n_masked} ic_masked chips (expected: only train > 0)")

# Counts
masked_train = log[(log["split"] == "train") & (log["ic_masked"].astype(str).str.lower() == "true")]
print(f"\nTotal training chips IC-masked: {len(masked_train)}")
print(f"Expected: ~193")
EOF
```

**HUMAN CHECKPOINT 1:** Review output. Confirm:
- v4_raw_lt65 has 398 chips, no filter, no IC mask
- v4_clean_lt65 has 330 chips, 40m filter, IC mask train-only
- v4_clean_lt65_plus_nulls has 359 chips (330 + 29 nulls), same test set as v4_clean_lt65
- Training: ~193 IC-masked chips. Val: 0. Test: 0.
- If anything fails: stop and investigate before continuing.

---

## Step 2: Phase A Results Verification

**Goal:** Confirm all 10 Phase A experiments have complete evaluation results. Pull into a single comparison table.

### 2.1 Collect all results

```bash
ssh smishra@moosehead.bowdoin.edu
conda activate iceberg-unet
cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework

python3 - <<'EOF'
import os, csv, sys

RUNS = "runs"
EXPS = [f"exp_A{i}" for i in range(10)]
EXP_DIRS = {
    "exp_A0": "exp_A0_fisser_lt65_original",
    "exp_A1": "exp_A1_fisser_lt65_plus_nulls",
    "exp_A2": "exp_A2_our_lt65",
    "exp_A3": "exp_A3_our_lt65_plus_nulls",
    "exp_A4": "exp_A4_our_lt65_plus_nulls_aug",
    "exp_A5": "exp_A5_our_lt65_plus_nulls_aug_2pos",
    "exp_A6": "exp_A6_our_lt65_plus_nulls_aug_adaptive",
    "exp_A7": "exp_A7_our_lt65_plus_nulls_aug_size",
    "exp_A8": "exp_A8_our_lt65_plus_nulls_aug_2pos_size",
    "exp_A9": "exp_A9_our_lt65_plus_nulls_aug_adaptive_size",
}

results = []
for short, exp_dir in EXP_DIRS.items():
    run_root = os.path.join(RUNS, exp_dir)
    if not os.path.isdir(run_root):
        print(f"MISSING: {exp_dir}")
        continue
    latest = sorted(os.listdir(run_root))[-1]
    eval_csv = os.path.join(run_root, latest, "evaluation", "eval_summary_gt_positive_only.csv")
    if not os.path.exists(eval_csv):
        print(f"NO EVAL: {exp_dir}/{latest}")
        continue
    with open(eval_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sza_bin"] == "sza_lt65" and row["method"] == "UNet":
                results.append({
                    "exp":    short,
                    "dir":    exp_dir,
                    "run":    latest,
                    "iou":    float(row["mean_iou"]),
                    "n_chips": int(row["n_chips"]),
                })

results.sort(key=lambda x: x["iou"], reverse=True)
print(f"\n{'Exp':<8} {'UNet IoU':>10} {'n_chips':>8}  Dir")
print("-" * 60)
for r in results:
    print(f"{r['exp']:<8} {r['iou']:>10.4f} {r['n_chips']:>8}  {r['dir']}")

out = "/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/shib_audit/phase_a_results/phase_a_unet_iou_sza_lt65.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["exp","dir","run","iou","n_chips"])
    w.writeheader()
    w.writerows(results)
print(f"\nSaved: {out}")
EOF
```

### 2.2 Verify A5/A6 training logs

```bash
# Check that A5 and A6 checkpoints exist and have training logs
RUNS=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/runs

for exp in exp_A5_our_lt65_plus_nulls_aug_2pos exp_A6_our_lt65_plus_nulls_aug_adaptive; do
  LATEST=$(ls "$RUNS/$exp/" | sort | tail -1)
  echo "=== $exp / $LATEST ==="
  ls "$RUNS/$exp/$LATEST/model/"
  echo "--- training_log.csv tail ---"
  tail -3 "$RUNS/$exp/$LATEST/model/training_log.csv" 2>/dev/null || echo "no training_log.csv"
done
```

### 2.3 Investigate A0 vs A2 discrepancy

```bash
python3 - <<'EOF'
import json, pandas as pd

# Load chip stems from each manifest's test split
for name, path in [
    ("v4_raw_lt65",   "data/v4_raw_lt65/manifest.json"),
    ("v4_clean_lt65", "data/v4_clean_lt65/manifest.json"),
]:
    with open(path) as f:
        d = json.load(f)
    test_chips = [c["chip_stem"] for c in d["chips"] if c.get("split") == "test"]
    gt_pos     = [c["chip_stem"] for c in d["chips"]
                  if c.get("split") == "test" and c.get("has_iceberg")]
    print(f"\n{name}: {len(test_chips)} test chips, {len(gt_pos)} GT-positive")

# Find the 19 chips in A0 test but not A2 (different pool sizes cause full reshuffle)
with open("data/v4_raw_lt65/manifest.json") as f:
    raw = json.load(f)
with open("data/v4_clean_lt65/manifest.json") as f:
    clean = json.load(f)

raw_test  = {c["chip_stem"] for c in raw["chips"] if c.get("split") == "test"}
clean_test = {c["chip_stem"] for c in clean["chips"] if c.get("split") == "test"}

only_raw   = sorted(raw_test - clean_test)
only_clean = sorted(clean_test - raw_test)

print(f"\nChips in A0 test but NOT in A2 test: {len(only_raw)}")
print(f"Chips in A2 test but NOT in A0 test: {len(only_clean)}")

# Check IC fraction for chips only in A0 (these are the IC-dropped ones in A2)
raw_by_stem = {c["chip_stem"]: c for c in raw["chips"]}
ic_fracs = [(s, raw_by_stem[s].get("ic_aware", "?")) for s in only_raw if s in raw_by_stem]
ic_fracs.sort(key=lambda x: float(x[1]) if isinstance(x[1], float) else 0, reverse=True)
print("\nTop 10 A0-only test chips by IC fraction (these were dropped in A2):")
for stem, ic in ic_fracs[:10]:
    print(f"  {stem}  ic_aware={ic:.3f}" if isinstance(ic, float) else f"  {stem}  ic={ic}")
EOF
```

**HUMAN CHECKPOINT 2:** Review Phase A table and discrepancy investigation. Confirm:
- All 10 experiments have evaluations. If any are missing: stop and re-run that experiment (see Step 2R below).
- A5 and A6 are tied at 0.0373 only for chip-level GT-positive IoU.
- A0 is the clear winner by best validation IoU and per-iceberg matched metrics.
- A0 > A2 is partly confounded by different test pools, but the broader A0 advantage is also visible in training validation and per-iceberg outputs.
- The A0-only chips have high IC fractions (>= 0.15), confirming they were excluded in A2.
- **Decision:** Choose the Phase B anchor deliberately:
  - Paper/all-SZA method sweep: use `baseline_v1`.
  - Low-SZA Fisser reproduction: use A0.
  - Cleaned-family-only diagnostic: use A5 or A6, with an explicit caveat that this is not the global Phase A winner.

### 2R: Re-run a Phase A experiment (only if eval is missing)

```bash
# Example: re-run A5 from scratch (all stages)
cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework
conda activate iceberg-unet
sbatch slurm/exp.slurm exp_A5_our_lt65_plus_nulls_aug_2pos manifest,balance,train,infer,evaluate

# OR run interactively (no GPU queue):
python3 scripts/run_experiment.py \
    --exp exp_A5_our_lt65_plus_nulls_aug_2pos \
    --stages manifest,balance,train,infer,evaluate
```

Note: `exp.slurm` sets `EXP_ID` and `STAGES` via sbatch `--export`. Check the slurm file to confirm the exact invocation before submitting.

---

## Step 3: Cleanup Visualization (20 chips per SZA bin)

**Goal:** Generate the "chip | preliminary annotation | cleaned annotation" 3-panel figures for 20 representative chips per SZA bin. This is Figure 1 in llinkas's paper draft. Pull the rendered PNGs locally for review.

This uses `make_figure01_annotation_difficulty.py`, which has two modes:
1. `--list_candidates`: prints candidate chip stems per scenario
2. Default: renders a 3-row figure for supplied stems

**Three annotation scenarios shown per figure:**
- Row (a): Fisser low-SZA chip, shadow-rich (shadow merge effect)
- Row (b): Roboflow high-SZA chip, 40m filter drops sub-16px specks
- Row (c): Roboflow IC-masked chip, bright background zeroed in training

### 3.1 Find candidates

```bash
ssh smishra@moosehead.bowdoin.edu
conda activate iceberg-unet
cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework

python3 scripts/make_figure01_annotation_difficulty.py \
    --list_candidates \
    --split_log   data/v3_clean/split_log.csv \
    --raw_fisser  data/raw_chips/fisser_pkls \
    --raw_coco    /mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/roboflow_download/raw_export/train/_annotations.coco.json \
    --filtered_coco data/annotations_filtered.coco.json \
    --chips_dir   /mnt/research/v.gomezgilyaspik/students/smishra/rework/chips
```

This prints 5 candidates per scenario type. Record the stems you want to use.

> **Note on `--raw_fisser`:** the raw Fisser pkl files are at `data/raw_chips/fisser_pkls/` (verify this path exists; if not, check `audit_fisser_provenance.py` output or `data/raw_chips/fisser/`).

**HUMAN CHECKPOINT 3:** Review the candidate list. Pick:
- 3-5 Fisser low-SZA stems for row (a)
- 3-5 Roboflow high-SZA stems for row (b)
- 3-5 Roboflow IC-masked stems for row (c)

### 3.2 Render figures for selected stems

Repeat the call below for each selected stem triple. The output is one PNG per call.

```bash
OUT=$AUDIT/cleanup_viz

# Replace with stems chosen in 3.1. One call = one 3-row figure.
python3 scripts/make_figure01_annotation_difficulty.py \
    --stem_a fisser_XXXX \
    --stem_b KQ_r0099_c0012 \
    --stem_c KQ_r0033_c0005 \
    --split_log   data/v3_clean/split_log.csv \
    --raw_fisser  data/raw_chips/fisser_pkls \
    --raw_coco    /mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/roboflow_download/raw_export/train/_annotations.coco.json \
    --filtered_coco data/annotations_filtered.coco.json \
    --chips_dir   /mnt/research/v.gomezgilyaspik/students/smishra/rework/chips \
    --out_dir     $OUT
```

Run this for enough stem combinations to get ~20 examples per SZA bin. In practice: run 7-8 calls covering all 4 SZA bins using row (b) or (c) stems from each bin.

### 3.3 Pull rendered figures locally

```bash
# Run locally:
rsync -av \
    smishra@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/shib_audit/cleanup_viz/ \
    /Users/smishra/iceberg-seg/shib_audit/cleanup_viz/
```

**HUMAN CHECKPOINT 4:** Open the PNGs locally. Confirm:
- Column 1 (chip) shows visible icebergs with correct RGB stretch
- Column 2 (preliminary) shows the raw Roboflow/Fisser annotation
- Column 3 (cleaned) shows annotation after shadow merge + 40m filter (or IC mask for row c)
- The cleaning step is not removing real icebergs (visually spot-check)
- If any chips look wrong: note the stem and flag for re-annotation review

---

## Step 4: Phase B Setup and Run

**Goal:** Run the 6 method experiments (B0-B5) using a deliberately selected frozen checkpoint. No new training, just inference + evaluation.

**STOP BEFORE RUNNING:** Do not submit Phase B jobs until the anchor has been chosen from the metric-scope decision in Human Checkpoint 2. The older A5 recommendation is no longer sufficient.

### 4.1 Confirm which checkpoint to use

Candidate anchors:

1. **Paper/all-SZA anchor, recommended unless the goal changed:** `baseline_v1`, trained on full `v4_clean` across all SZA bins.
2. **Low-SZA Fisser-reproduction anchor:** A0, best lt65 run by validation IoU and per-iceberg metrics.
3. **Cleaned-family diagnostic anchor:** A5 or A6, best only under chip-level GT-positive IoU within the cleaned-family framing.

If using A5, the checkpoint is:
```
$LLINKAS/runs/exp_A5_our_lt65_plus_nulls_aug_2pos/20260430_001810/model/best_model.pth
```

If using baseline_v1, the checkpoint is:
```
$LLINKAS/runs/exp_baseline_v1/20260424_185158/model/best_model.pth
```

Verify the chosen checkpoint exists:
```bash
ssh smishra@moosehead.bowdoin.edu
ls -lh <chosen_checkpoint_path>
```

### 4.2 Check Phase B experiment configs

Phase B configs are at `$LLINKAS/configs/experiments/exp_B{0..5}.yaml`. Check what manifest they inherit:

```bash
cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework
for b in exp_B0_method_threshold exp_B1_method_otsu exp_B2_method_unet exp_B3_method_unet_threshold exp_B4_method_unet_otsu exp_B5_method_unet_crf; do
    echo "=== $b ==="
    grep -E "manifest_id|inherits|checkpoint" configs/experiments/${b}.yaml 2>/dev/null
done
```

**Key issue if using A5/A6:** Phase B configs inherit `baseline_v1`, which uses `manifest_id: v4_clean`. A5/A6 were trained on `v4_clean_lt65_plus_nulls`. The `run_methods.sh` drift guard checks that the checkpoint's training manifest matches the inference manifest. Options:
- **Option A:** Update a diagnostic copy of the Phase B configs to set `manifest_id: v4_clean_lt65_plus_nulls`. This makes inference use the same manifest as training, but it is lt65-scoped and should not be described as the all-SZA Phase B paper sweep.
- **Option B:** Set `FORCE=1` in the run environment to bypass the drift guard. Avoid this for paper claims unless the mismatch is explicitly justified.

If using `baseline_v1`, this mismatch should not occur because the checkpoint and inherited configs both use `v4_clean`.

**HUMAN CHECKPOINT 5:** Decide:
- Which checkpoint to use: `baseline_v1`, A0, A5, or A6.
- If using A5/A6, how to handle the manifest mismatch: update configs (Option A) or FORCE=1 (Option B).
- Confirm Phase B YAML configs point to the correct checkpoint path

### 4.3 Update Phase B configs (if Option A chosen)

If running an A5/A6 cleaned-family diagnostic and updating configs: open a diagnostic copy of each `exp_B*.yaml` and set:
```yaml
change:
  data:
    manifest_id: v4_clean_lt65_plus_nulls
  checkpoint: <full path to chosen A5/A6 best_model.pth>
```

If the YAMLs use a `checkpoint` field, set it explicitly. If `run_experiment.py` picks up checkpoint from a prior `train` stage in the same run, you may need to pass it via `--checkpoint` argument or skip the train stage.

Check `run_experiment.py`'s `stage_infer` function: it takes a `checkpoint` argument. For a diagnostic A5 run, pass it explicitly:

```bash
python3 scripts/run_experiment.py \
    --exp exp_B2_method_unet \
    --stages manifest,infer,evaluate \
    --checkpoint /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/runs/exp_A5_our_lt65_plus_nulls_aug_2pos/20260430_001810/model/best_model.pth
```

Check if `run_experiment.py` accepts a `--checkpoint` flag; if not, add it.

### 4.4 Run Phase B (all 6 experiments)

```bash
ssh smishra@moosehead.bowdoin.edu
conda activate iceberg-unet
cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework

CHOSEN_CKPT=<chosen_checkpoint_path>

for b in exp_B0_method_threshold exp_B1_method_otsu exp_B2_method_unet \
          exp_B3_method_unet_threshold exp_B4_method_unet_otsu exp_B5_method_unet_crf; do
    echo "Submitting $b ..."
    sbatch slurm/exp.slurm $b manifest,infer,evaluate
    # OR interactively:
    # python3 scripts/run_experiment.py --exp $b --stages manifest,infer,evaluate
done
```

Monitor jobs:
```bash
squeue -u smishra
```

**HUMAN CHECKPOINT 6:** Wait for all Phase B jobs to complete. Then run the verification below.

### 4.5 Verify Phase B results

```bash
python3 - <<'EOF'
import os, csv

RUNS = "runs"
PHASE_B = [
    "exp_B0_method_threshold",
    "exp_B1_method_otsu",
    "exp_B2_method_unet",
    "exp_B3_method_unet_threshold",
    "exp_B4_method_unet_otsu",
    "exp_B5_method_unet_crf",
]

print(f"{'Exp':<35} {'sza_lt65 method':<15} {'IoU':>7} {'n_chips':>8}")
print("-" * 70)
for exp_dir in PHASE_B:
    run_root = os.path.join(RUNS, exp_dir)
    if not os.path.isdir(run_root):
        print(f"  MISSING: {exp_dir}")
        continue
    latest = sorted(os.listdir(run_root))[-1]
    eval_csv = os.path.join(run_root, latest, "evaluation", "eval_summary_gt_positive_only.csv")
    if not os.path.exists(eval_csv):
        print(f"  NO EVAL: {exp_dir}")
        continue
    with open(eval_csv) as f:
        for row in csv.DictReader(f):
            if row["sza_bin"] == "sza_lt65":
                print(f"{exp_dir:<35} {row['method']:<15} {float(row['mean_iou']):>7.4f} {row['n_chips']:>8}")
EOF
```

---

## Step 5: Full Phase A+B Comparison

**Goal:** Generate the final heatmap comparing all methods across all SZA bins for the selected anchor vs the baseline. Only do this after the anchor and metric scope are written down.

### 5.1 Identify baseline eval dir

The baseline is `exp_baseline_v1` (trained on full v4_clean, all SZA bins, no augmentation). Check if it ran:
```bash
ls /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/runs/exp_baseline_v1/
```

If it ran, note the latest timestamp and its eval dir.

### 5.2 Run compare_model_eval.py

```bash
cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework

# Fill in the actual timestamps from your run dirs.
# If the selected anchor is baseline_v1, this comparison may be unnecessary.
BASELINE_EVAL=runs/exp_baseline_v1/<TIMESTAMP>/evaluation
STAGE1_EVAL=<selected_anchor_eval_dir>
OUT=$AUDIT/phase_a_results/model_comparison

python3 scripts/compare_model_eval.py \
    --out_root       $OUT \
    --split_log      data/v3_clean/split_log.csv \
    --baseline_dir   $BASELINE_EVAL \
    --stage1_dir     $STAGE1_EVAL \
    --baseline_label baseline_v1 \
    --stage1_label   <selected_anchor_label> \
    --training_summary <selected_anchor_training_summary_if_available>
```

### 5.3 Pull comparison outputs locally

```bash
rsync -av \
    smishra@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/shib_audit/phase_a_results/ \
    /Users/smishra/iceberg-seg/shib_audit/phase_a_results/
```

**HUMAN CHECKPOINT 7:** Review `summary_sheet.md` and heatmap PNGs. Confirm:
- The selected anchor matches the metric scope written in this audit plan.
- If using A5/A6, all outputs are labeled as a cleaned-family or lt65-scoped diagnostic, not as the global Phase A winner.
- If using `baseline_v1`, the Phase B results are interpreted as the all-SZA paper method sweep.
- Phase B results make sense (UNet-based methods should generally outperform threshold-only, with detection-rate caveats).

---

## Step 6: shib_audit/ Final Structure

After all steps, the audit directory should contain:

```
$AUDIT/
  phase_a_results/
    phase_a_unet_iou_sza_lt65.csv          # Table from Step 2.1
    model_comparison/
      summary_sheet.md                     # Headline + tables from Step 5
      iou_delta_heatmap_gt_positive_only.png
      iou_delta_heatmap_all_chips.png
      lt65_gt_positive_iou_comparison.csv
  cleanup_viz/
    fig-archive/
      figure01_annotation_difficulty_*.png # Figures from Step 3
  phase_b_results/                         # Populated after Step 4
```

Local mirror at `/Users/smishra/iceberg-seg/shib_audit/`.

---

## Known Issues and Flags

| Issue | Severity | Resolution |
|-------|----------|------------|
| A5/A6 "winner" claim depends on chip-level GT-positive IoU | High: can misdirect Phase B | Treat A5/A6 as cleaned-family diagnostic winners only. A0 wins validation/per-iceberg lt65 metrics; `baseline_v1` remains the likely all-SZA Phase B anchor. |
| A0 vs A2 test sets differ (49 vs 47 chips) | Low: known, documented | Different pools, different splits. Not a bug. |
| Cleanup pipeline hurts lt65-only training metrics | High: interpretation risk | Keep shadow merge / 40 m / IC as defensible preprocessing for data hygiene and Fisser-style rules, but do not claim it improves lt65 UNet performance. |
| A5 and A6 identical results | Informational | Both converged to same chip-level IoU. Use A5 only for a cleaned-family diagnostic if that scope is chosen. |
| Phase B manifest mismatch (v4_clean vs v4_clean_lt65_plus_nulls) | High if using A5/A6 | Prefer `baseline_v1` for all-SZA paper Phase B. If running A5/A6 diagnostics, update diagnostic configs or explicitly justify FORCE=1. |
| A7/A8/A9 size balancing hurts (0.0342 < 0.0368 for A4) | Informational | Drop size-balanced experiments from final paper table |
| `--raw_fisser` path for make_figure01 | Verify | Check `data/raw_chips/fisser_pkls/` exists before running Step 3 |
| UNet_TR recall is very high (0.80+ in A5/A6 sza_lt65) | Informational | Fixed threshold too low relative to calibration, causes over-prediction |

---

## Commands Cheat Sheet

```bash
# SSH
ssh smishra@moosehead.bowdoin.edu

# Conda
conda activate iceberg-unet

# Key dirs (set these once per session)
LLINKAS=/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework
RESEARCH=/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas
AUDIT=$RESEARCH/shib_audit

# Candidate Phase B checkpoints
BASELINE_CKPT=$LLINKAS/runs/exp_baseline_v1/20260424_185158/model/best_model.pth
A5_DIAGNOSTIC_CKPT=$LLINKAS/runs/exp_A5_our_lt65_plus_nulls_aug_2pos/20260430_001810/model/best_model.pth

# Run one experiment (interactive)
cd $LLINKAS
python3 scripts/run_experiment.py --exp <exp_id> --stages manifest,balance,train,infer,evaluate

# Pull any dir locally
rsync -av smishra@moosehead.bowdoin.edu:<remote_path>/ <local_path>/

# Check SLURM queue
squeue -u smishra

# Check GPU availability
sinfo -p gpu
```
