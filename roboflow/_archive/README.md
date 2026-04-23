# `roboflow/_archive/` — superseded scripts

Archived on 2026-04-18.

## Contents

### `older-upload-script-draft.py`

Originally named `from __future__ import annotations.py` — the filename was accidentally saved as the first line of the file (a Python `__future__` import), likely a shell-redirection mishap.

This is an earlier 69-line draft of the Roboflow upload script. The active, more complete 99-line version is `../upload_include3_fill4_dataset.py`, which adds proper argparse for `--project-id`, `--dataset-dir`, `--batch-name`, and better error messages.

The two scripts share the same hard-coded `API_KEY`, `WORKSPACE_ID`, `PROJECT_ID` (`icebergseg`), `BATCH_NAME` (`restart_2026_03_25`), and `DATASET_DIR` (`S2-iceberg-areas/roboflow_manual_upload_include3_fill4_iceberg`).

**Do NOT run this script.** Use the active version at `../upload_include3_fill4_dataset.py`.
