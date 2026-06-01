# Sentinel-2 Iceberg Segmentation across SZA bins

Louisa Linkas and Shibali Mishra, Independent Study, Bowdoin College, Spring 2026.

Segmentation of icebergs in Sentinel-2 L1C imagery across four solar zenith angle (SZA) bins in Kangerlussuaq (KQ) and Sermilik (SK) fjords on the east coast of Greenland. Compares six retrieval methods (TR, OT, UNet++, UNet+TR, UNet+OT, UNet+CRF) on a single shared dataset and reports Fisser-comparable per-pair MAE on iceberg area.

## Where things live

| Folder | Purpose |
|---|---|
| `iceberg-rework/` | Live code mirror (HPC: `/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/`). All scripts, configs, slurm wrappers, and small-text data. |
| `paper-writing/` | Project documentation. Start with `plan.md` (state) and `methods_draft.md` (methodology); see also `model_progression.md` (Phase A and B), `iceberg-rework-README.md` (folder layout), `refactor_plan.md` (deep audit). |
| `iceberg-labeler/` | Companion FastAPI labeling app for hand-validating predictions. |
| `roboflow/` | Roboflow annotation workflow (instructions, locator maps, batch upload scripts). |
| `S2-iceberg-areas/` | Pre-rework codebase. Archive only; do not edit. |
| `notebooks/` | Exploration. |

## Read this first

- **`paper-writing/plan.md`**: project state, completed steps, remaining work, critical file paths. The single source of truth for what is done and what is in flight.
- **`paper-writing/methods_draft.md`**: methodology section draft.
- **`paper-writing/model_progression.md`**: experimental progression. Phase A walks the dataset (Fisser reproduction -> our lt65 + nulls + aug + balancing). Phase B walks the method (TR -> CRF) on the Phase A winner.
- **`paper-writing/iceberg-rework-README.md`**: project README with folder layout and data tables.
- **`iceberg-rework/README.md`**: HPC quick-start (how to run experiments).

## Reproducing the paper from a fresh machine

See [REPRODUCE.md](REPRODUCE.md) for the post-graduation recipe: clone, pull the bulk-data tarballs from the `archive-v1` GitHub release, set up a conda env, run a smoke test on the canonical baseline. [DATA_ARCHIVE.md](DATA_ARCHIVE.md) lists every tarball with its source path on HPC and SHA-256.

## How to run an experiment (during active work)

From a moosehead shell:

```
ssh moosehead
cd /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework
sbatch slurm/baseline_v1.slurm
```

Validate any experiment first:

```
python scripts/validate_experiment.py --exp <experiment_id>
```

Available experiments under `iceberg-rework/configs/experiments/`: `exp_baseline_v1`, `exp_A0` through `exp_A6`, `exp_B0` through `exp_B5`, `exp_ablation_no_aug`. See `paper-writing/model_progression.md` for the progression.

## Data and code separation

- **Code, configs, small text data**: github.com/llinkas11/iceberg-seg.
- **Materialised pkls, model checkpoints, inference outputs, large reference rasters**: HPC working tree only. Too big for git.
- **Source chip data**: `/mnt/research/v.gomezgilyaspik/students/smishra/rework/` (read-only).

## Hardware

Training and inference on `moosehead.bowdoin.edu` via Slurm. Default request: 1 RTX 3080, 4 CPUs, 32 GB RAM, 10 hours wall time.
