"""
run_experiment.py: drive one experiment through the five pipeline stages.

Stages:
  manifest   ensure the data manifest named in the experiment config exists
             (builds it via build_clean_dataset.py if missing)
  train      run train.py with ICEBERG_EXPERIMENT=1 so the seed guard fires
  infer      run run_methods.sh on the trained checkpoint + manifest
  evaluate   run eval_methods.py with --manifest and --skipped_chip_policy
  figures    stub for the figure registry (Phase 6)

Each stage stamps the experiment id, the resolved config_sha, and the
chips_sha into its outputs so cross-experiment comparison can join them.

Usage:
    python scripts/run_experiment.py --exp exp_07_method_otsu
    python scripts/run_experiment.py --exp exp_07_method_otsu --stages manifest,evaluate
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone

from validate_experiment import (
    BASELINES_DIR,
    EXPERIMENTS_DIR,
    REPO_DIR,
    config_sha,
    enforce_single_variable,
    resolve_experiment,
    validate_baseline,
)

SCRIPTS_DIR = os.path.join(REPO_DIR, "scripts")
DEFAULT_PY  = os.environ.get("PY", "/home/llinkas/.venvs/iceberg-unet312/bin/python")
if not os.path.exists(DEFAULT_PY):
    DEFAULT_PY = sys.executable

STAGE_ORDER = ["manifest", "train", "infer", "evaluate", "figures"]


def run(cmd, env=None):
    """Echo-and-run a subprocess command; raise on non-zero exit."""
    print(f"\n>>> {cmd if isinstance(cmd, str) else ' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.run(cmd, shell=isinstance(cmd, str), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"stage failed: exit {proc.returncode}")


def stage_manifest(cfg, run_dir):
    """
    Ensure manifest file exists. If not, call build_clean_dataset.py with the
    baseline's manifest_id as --out_dir basename. Writes manifest_stamp.json
    into run_dir.
    """
    manifest_id = cfg["data"]["manifest_id"]
    manifest_path = os.path.join(REPO_DIR, "data", manifest_id, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"Manifest {manifest_id} not found, building...")
        run([
            DEFAULT_PY, os.path.join(SCRIPTS_DIR, "build_clean_dataset.py"),
            "--manifest_id", manifest_id,
            "--out_dir",     os.path.join(REPO_DIR, "data", manifest_id),
            "--seed",        str(cfg["data"]["split"]["seed"]),
        ])
    with open(manifest_path) as f:
        chips_sha = json.load(f).get("chips_sha", "")

    stamp = {
        "stage":       "manifest",
        "manifest_id": manifest_id,
        "manifest":    manifest_path,
        "chips_sha":   chips_sha,
    }
    with open(os.path.join(run_dir, "manifest_stamp.json"), "w") as f:
        json.dump(stamp, f, indent=2)
    return manifest_path


def stage_train(cfg, run_dir, manifest_path):
    """
    Train a model under ICEBERG_EXPERIMENT=1 so the seed guard fires.
    Output goes to run_dir/model/.
    """
    out_dir  = os.path.join(run_dir, "model")
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.dirname(manifest_path)

    env = os.environ.copy()
    env["ICEBERG_EXPERIMENT"] = "1"

    cmd = [
        DEFAULT_PY, os.path.join(SCRIPTS_DIR, "train.py"),
        "--mode",       "s2",
        "--data_dir",   data_dir,
        "--out_dir",    out_dir,
        "--encoder",    cfg["model"]["encoder"],
        "--epochs",     str(cfg["training"]["epochs"]),
        "--batch_size", str(cfg["training"]["batch_size"]),
        "--lr",         str(cfg["training"]["lr"]),
        "--workers",    str(cfg["training"]["workers"]),
        "--seed",       str(cfg["training"]["seed"]),
    ]
    if not cfg["augmentation"].get("enabled", True):
        cmd.append("--no_augment")
    run(cmd, env=env)
    return os.path.join(out_dir, "best_model.pth")


def stage_infer(cfg, run_dir, manifest_path, checkpoint):
    out_base = os.path.join(run_dir, "inference")
    cmd = [
        "bash", os.path.join(SCRIPTS_DIR, "run_methods.sh"),
        "--manifest",   manifest_path,
        "--checkpoint", checkpoint,
        "--out_base",   out_base,
    ]
    run(cmd)
    return out_base


def stage_evaluate(cfg, run_dir, manifest_path, infer_dir):
    eval_out = os.path.join(run_dir, "evaluation")
    os.makedirs(eval_out, exist_ok=True)
    cmd = [
        DEFAULT_PY, os.path.join(SCRIPTS_DIR, "eval_methods.py"),
        "--manifest",            manifest_path,
        "--test_dir",            infer_dir,
        "--out_dir",             eval_out,
        "--skipped_chip_policy", cfg["evaluation"].get("skipped_chip_policy", "count_as_false_negative"),
    ]
    run(cmd)
    return eval_out


def stage_figures(cfg, run_dir, eval_dir):
    """
    Placeholder for Phase 6 figure registry. Currently just records that the
    stage ran; a future iceberg.figures.registry will generate + archive.
    """
    print(f"[figures] stub; will archive under fig-archive/ once registry lands.")


def write_run_stamp(run_dir, exp_id, baseline_id, resolved_cfg, sha, stages_run):
    """Top-level run_stamp.json with identity info everyone else can read."""
    stamp = {
        "experiment_id":     exp_id,
        "baseline_id":       baseline_id,
        "config_sha":        sha,
        "run_utc":           datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stages_run":        stages_run,
    }
    with open(os.path.join(run_dir, "run_stamp.json"), "w") as f:
        json.dump(stamp, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True,
                        help="Experiment id (without .yaml)")
    parser.add_argument("--stages", default=",".join(STAGE_ORDER),
                        help=f"Comma-separated stage names. Default: {','.join(STAGE_ORDER)}")
    parser.add_argument("--runs_root", default=os.path.join(REPO_DIR, "runs"),
                        help="Root for runs/<exp>/<timestamp>/")
    args = parser.parse_args()

    exp_path = os.path.join(EXPERIMENTS_DIR, f"{args.exp}.yaml")
    if not os.path.exists(exp_path):
        raise SystemExit(f"experiment not found: {exp_path}")

    baseline_id, merged, change = resolve_experiment(exp_path)
    validate_baseline(baseline_id)
    enforce_single_variable(change, merged.get("declared_controlled_variable"))

    # Strip fields that should not hash into config identity.
    core_cfg = {k: v for k, v in merged.items()
                if k not in {"change", "notes", "declared_controlled_variable"}}
    sha = config_sha(core_cfg)
    print(f"experiment {args.exp}: validated (config_sha={sha[:16]}...)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs_root, args.exp, ts)
    os.makedirs(run_dir, exist_ok=True)
    print(f"run dir: {run_dir}")

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGE_ORDER:
            raise SystemExit(f"unknown stage: {s}; valid: {STAGE_ORDER}")

    manifest_path = None
    checkpoint    = None
    infer_dir     = None
    eval_dir      = None
    for s in stages:
        print(f"\n=== stage: {s} ===")
        if s == "manifest":
            manifest_path = stage_manifest(merged, run_dir)
        elif s == "train":
            if manifest_path is None:
                manifest_path = stage_manifest(merged, run_dir)
            checkpoint = stage_train(merged, run_dir, manifest_path)
        elif s == "infer":
            if manifest_path is None:
                manifest_path = stage_manifest(merged, run_dir)
            if checkpoint is None:
                # Allow re-using an existing checkpoint when skipping train.
                checkpoint = os.path.join(run_dir, "model", "best_model.pth")
                if not os.path.exists(checkpoint):
                    raise SystemExit(
                        "infer requested but no checkpoint found; pass --stages "
                        "manifest,train,infer or drop in a checkpoint at "
                        f"{checkpoint}"
                    )
            infer_dir = stage_infer(merged, run_dir, manifest_path, checkpoint)
        elif s == "evaluate":
            if manifest_path is None:
                manifest_path = stage_manifest(merged, run_dir)
            if infer_dir is None:
                infer_dir = os.path.join(run_dir, "inference")
            eval_dir = stage_evaluate(merged, run_dir, manifest_path, infer_dir)
        elif s == "figures":
            stage_figures(merged, run_dir, eval_dir)

    write_run_stamp(run_dir, args.exp, baseline_id, merged, sha, stages)
    print(f"\nrun done: {run_dir}")


if __name__ == "__main__":
    main()
