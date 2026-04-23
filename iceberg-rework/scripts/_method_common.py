"""
_method_common.py: helpers shared across the six method scripts, the manifest
builder, the trainer, and the evaluator.

Provenance writers for method runs:
  write_method_config  emit method_config.json with every parameter used
  write_skipped_chips  emit skipped_chips.csv, one row per refused chip

Skip-reason constants:
  SKIP_TOO_FEW_BANDS, SKIP_TOO_FEW_PROB_BANDS, SKIP_IC_BLOCK_FILTER,
  SKIP_OTSU_FLOOR, SKIP_FLAT_PROB, SKIP_CHIP_TIF_MISSING

Hashing + manifest + git helpers (used by build_clean_dataset + train):
  load_manifest, sha256_of_file, sha256_of_text, get_git_sha
"""

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone


# Reason strings written into skipped_chips.csv. Method scripts import these
# rather than hand-writing the literals, so downstream filters can match on
# constants and a typo does not split one bucket into two.
SKIP_TOO_FEW_BANDS      = "too_few_bands"
SKIP_TOO_FEW_PROB_BANDS = "too_few_prob_bands"
SKIP_IC_BLOCK_FILTER    = "ic_block_filter"
SKIP_OTSU_FLOOR         = "otsu_floor"
SKIP_FLAT_PROB          = "flat_prob"
SKIP_CHIP_TIF_MISSING   = "chip_tif_not_found"


def load_manifest(path):
    """Load a manifest.json; raise if the expected keys are missing."""
    with open(path) as f:
        m = json.load(f)
    for key in ("manifest_id", "chips_sha", "chips"):
        if key not in m:
            raise ValueError(f"{path}: manifest missing required key '{key}'")
    return m


def sha256_of_text(text):
    """Hex sha256 of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_file(path, chunk=1 << 20):
    """Hex sha256 of file bytes. Returns None if path is empty or missing."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def get_git_sha(repo_dir):
    """Return short git SHA for repo_dir, or None if not a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def write_method_config(out_dir, method_name, params, extra=None):
    """
    Write method_config.json into out_dir. Captures the method name, the
    parameter block (min_area_m2, per-method thresholds, etc.), the chips_dir
    that was processed, the script path, and the git SHA of the scripts/ tree.

    params is any json-serialisable dict; extra is optional additional context
    (for example the checkpoint path and its training_config.json contents
    for UNet-based methods).
    """
    os.makedirs(out_dir, exist_ok=True)
    repo_dir = os.path.dirname(os.path.abspath(__file__))

    config = {
        "run_kind":     "inference_method",
        "method":       method_name,
        "run_utc":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "out_dir":      os.path.abspath(out_dir),
        "git_sha":      get_git_sha(repo_dir),
        "params":       dict(params),
    }
    if extra is not None:
        config["extra"] = extra

    path = os.path.join(out_dir, "method_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return path


def write_skipped_chips(out_dir, skipped):
    """
    Write skipped_chips.csv into out_dir. skipped is a list of dicts with
    keys at least {'chip_stem', 'reason'}; any extra keys are preserved.
    Always writes the header row even if the list is empty, so downstream
    code can rely on the file's existence.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "skipped_chips.csv")

    # Canonical column order; any extra keys get appended alphabetically.
    base_cols = ["chip_stem", "reason"]
    extra_cols = sorted({k for row in skipped for k in row.keys()} - set(base_cols))
    fieldnames = base_cols + extra_cols

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in skipped:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path
