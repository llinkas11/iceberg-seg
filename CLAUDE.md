# Project Guardrails

## BEFORE DOING ANYTHING

Read this file in full at the start of every session and before responding to any task or query. Do not proceed until these rules are loaded. This applies to every interaction, including simple questions, edits, and code tasks.

## Writing and Response Style

- NO EM DASHES. Use commas or colons instead.
- NO AI LANGUAGE. Do not use: "certainly", "absolutely", "great question", "of course", "happy to help", "I'd be glad to", or any filler affirmations.
- No repetition. Do not restate what the user just said. Do not summarize what you just did at the end of a response.
- Short and direct. User asks terse questions; respond in kind. Do not over-explain.
- Thorough, concise, simple, eloquent, structured.
- When writing academic content (abstracts, methods, results), avoid hedging language and passive constructions. Write assertively.

## Fact-Checking

- Ground every claim in actual script output, paper text, or confirmed filesystem data.
- Never put unverified numbers into an abstract or methods section. Flag uncertainty explicitly: "unverified" or "not yet confirmed."
- When reviewer feedback challenges an assumption, re-examine step by step. Do not accept corrections at face value without checking.

## Technical Constraints

### File Paths
- ALL data output goes to: `/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/`
- Scripts live in: `~/S2-iceberg-areas/` (home dir, scripts only)
- NEVER default output paths to `~/` or relative paths. Every `--out_dir`, `--chips_dir`, `--log_csv`, or any output argument must default to the full `/mnt/research/...` path.
- Data includes: chips, downloads, predictions, CSVs, GeoPackages, figures.

### Commands
- Always include the rsync command when writing a new script: `rsync -av /Users/smishra/S2-iceberg-areas/<script>.py smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/`
- Give exact commands to copy-paste. Do not describe commands in prose when a code block is more useful.
- For JupyterLab instructions, use full absolute paths on moosehead.
- Conda environment on moosehead: `iceberg-unet`

## Project Context

- Study: Sentinel-2 L1C iceberg area comparison across solar zenith angle (SZA) bins
- Regions: Kangerlussuaq (KQ) and Sermilik (SK) fjords, Greenland
- SZA bins: sza_lt65, sza_65_70, sza_70_75, sza_gt75
- Methods: UNet++ (ResNet34, 3-class output: ocean/iceberg/shadow; iceberg channel extracted for area retrieval), B08 threshold, Otsu threshold, DenseCRF
- Imagery: bands B04/B03/B08, 10m resolution, 256x256 chips
- HPC: moosehead.bowdoin.edu, SLURM scheduler
- Model checkpoint: `runs/s2_20260227_231556/best_model.pth` (val IoU 0.4398, 323 training chips)

## Abstract Constraints

- Character limit: 2500 characters including spaces
- Target: 2400-2500 characters (do not trim well under the limit)
- No em dashes
- All numbers must be confirmed from actual data before inclusion

## Code Style

- Simple over clever. Before writing any step, ask: is this the simplest, most efficient, and correct way to do this?
- Comment every logical block with a numbered step header, e.g.:
  ```python
  # 1. Load chip paths
  # 2. Run inference
  # 3. Write outputs
  ```
- All functions and scripts get a docstring: what it does, inputs, outputs. No filler prose.
- No unnecessary abstractions. If a loop does the job, use a loop. Do not wrap simple operations in classes or helper functions unless reuse is certain.
- Variable and argument names should be self-explanatory. No single-letter names outside of loop indices.

## File Editing Protocol

- Before editing any file, always `Read` it first to capture manual changes made since the last Claude edit. Never overwrite from a cached version.
- After any code change, run a minimal test to confirm the change produces the expected output before considering the task done (e.g. run on 5 chips, print shape/dtype/range, assert key properties).
- After any code change, update the methodology document to reflect what changed: inputs, outputs, logic, or parameters affected. Methodology file: `/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/paper-writing/methods_draft.md`. Note: `methods_draft.md` still carries some 3-class / 790-chip language; the authoritative project-state source is `/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/paper-writing/plan.md`, which reflects binary segmentation, the 40 m root-length filter, and the `v3_balanced` (364 / 137 / 228) split.

## Project-Specific Code Constraints

- Do not change existing `*_pred.tif` outputs when modifying scripts. New outputs are additive.
- Flag if ever about to use a relative or `~/` path for data output.
- DenseCRF requires softmax probability maps (`*_prob.tif`); `predict_tifs.py` must be modified before DenseCRF can run.
