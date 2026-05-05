# Reproducing the paper

End-to-end recipe to fetch this repo, pull the bulk data, set up an environment, and re-run inference + evaluation. Aims at reproducing the canonical baseline numbers in the paper, not the full Phase A grid (those are re-runnable but slow).

## 1. Clone the repo

Repository requires Git LFS for the model checkpoint and a couple of binary artifacts.

```bash
brew install git-lfs           # macOS; or apt install git-lfs / yum install git-lfs
git lfs install
git clone git@github.com:llinkas11/iceberg-seg.git
cd iceberg-seg
git lfs pull                   # fetches *.pth, *.onnx, *.gpkg
```

## 2. Pull the tier-3 data archive

The chips, predictions, and run outputs are too large for the repo (see [DATA_ARCHIVE.md](DATA_ARCHIVE.md)). The helper script downloads tarballs from the `archive-v1` GitHub release and extracts them into `data/`. It uses the `gh` CLI for auth (so `gh auth login` once first).

```bash
brew install gh && gh auth login       # one-time
bash download_data.sh                  # fetches all tarballs (~45 GB compressed)
# or, for a smaller subset (just what's needed for the smoke test below):
bash download_data.sh chips_v4_clean exp_A0_full
```

After extraction:
```
data/
  v4_clean/                    # canonical training/val/test split (manifest.json + chips/)
  raw_chips/                   # uncropped Sentinel-2 chips (incl. fisser/)
  exp_A0_fisser_lt65_original/ # full inference outputs for the Phase A leaderboard winner
  ...
```

## 3. Set up Python environment

The pipeline targets PyTorch 2.x with CUDA 12.1.

```bash
conda env create -f environment.yml
conda activate iceberg-unet
# or:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch is installed separately (the wheel index depends on your CUDA). On a CUDA 12.1 box:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CPU-only is fine for inference; training at scale wants a GPU.

## 4. Smoke test (~2 minutes)

Run inference on 5 chips with the canonical baseline checkpoint:

```bash
cd iceberg-rework/scripts
python predict.py \
  --manifest ../../data/v4_clean/manifest.json \
  --checkpoint ../canonical_models/baseline_v1__best_model.pth \
  --out_dir /tmp/smoke \
  --limit 5
ls /tmp/smoke/
```

The two canonical checkpoints kept in the repo (via Git LFS) are:
- `iceberg-rework/canonical_models/baseline_v1__best_model.pth` — produces every published baseline metric.
- `iceberg-rework/canonical_models/exp_A0_fisser_lt65_original__best_model.pth` — Phase A leaderboard winner.

All other Phase A and S2-iceberg-areas checkpoints live in `checkpoints_other.tar.zst` in tier 3 (see DATA_ARCHIVE.md).

Expected: 5 prediction GeoTIFFs.

## 5. Full reproduction

The full pipeline is five stages, all driven by `iceberg-rework/scripts/run_experiment.py`:

1. `manifest` — verify chip list + split
2. `train` — UNet++ training (seed required, 100 epochs, ~6h on a single A100)
3. `infer` — six inference methods (B08 thresh, Otsu, UNet, UNet+thresh, UNet+Otsu, UNet+CRF)
4. `evaluate` — chip-level + per-pair iceberg metrics
5. `figures` — re-render the paper figures

Re-run the canonical baseline:
```bash
bash run_experiment.sh exp_baseline_v1   # config: configs/baselines/baseline_v1.yaml
```

Re-run the Phase A leaderboard winner:
```bash
bash run_experiment.sh exp_A0_fisser_lt65_original
```

See [iceberg-rework/plan.md](iceberg-rework/plan.md) for the canonical project state, methodological decisions, and Phase A leaderboard.

## 6. Render the paper

```bash
cd paper-writing/overleaf/2026-04-17_template-plan-md
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Or upload that folder to Overleaf and compile there.

## Help

- Script reference: `iceberg-rework/scripts/` (each `.py` has a docstring).
- Methodology: [paper-writing/methods_draft.md](paper-writing/methods_draft.md), [paper-writing/plan.md](paper-writing/plan.md).
- Reference experiments + audit logs: [iceberg-rework/reference/](iceberg-rework/reference/).
