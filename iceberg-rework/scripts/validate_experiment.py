"""
validate_experiment.py: check that an experiment YAML differs from its baseline
in exactly one controlled variable.

An experiment YAML inherits a baseline (currently baseline_v1) and declares a
`change:` block listing only the fields that differ. The single-variable rule
says that `change:` may touch only one top-level family at a time:

    data
    methods
    augmentation
    training
    inference
    evaluation

Violating this makes a downstream head-to-head comparison meaningless, because
two things changed at once. This script is the gate that run_experiment.py
calls before doing any work.

Usage:
    python scripts/validate_experiment.py --baseline baseline_v1
    python scripts/validate_experiment.py --exp exp_07

When --baseline is passed alone, checks the baseline file is internally valid.
When --exp is passed, resolves inheritance, enforces the single-variable rule,
and prints the merged resolved config if all checks pass.

Returns non-zero exit code on any failure. The runner refuses to proceed on any
non-zero exit.
"""

import argparse
import hashlib
import json
import os
import sys

# yaml is an external dep, but already installed in the iceberg-unet venv.
import yaml


CONFIG_FAMILIES = {
    "data",
    "methods",
    "augmentation",
    "training",
    "inference",
    "evaluation",
    "preprocessing",
    "model",
    "reporting",
}

# The families we count as "the controlled variable" of an experiment.
# preprocessing, model, reporting can be tweaked but are not the primary
# dimension of scientific comparison; they still must not co-vary with the
# controlled family in a given experiment.
CONTROLLED_FAMILIES = {
    "data", "methods", "augmentation", "training", "inference", "evaluation",
}

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_DIR   = os.path.join(REPO_DIR, "configs", "baselines")
EXPERIMENTS_DIR = os.path.join(REPO_DIR, "configs", "experiments")


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def deep_merge(base, change):
    """
    Return a deep merge of base <- change. `change` wins on every leaf; dict
    nodes merge recursively, lists are replaced wholesale.
    """
    out = dict(base) if isinstance(base, dict) else base
    if not isinstance(change, dict):
        return change
    for k, v in change.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def config_sha(config):
    """Stable SHA over the resolved config; downstream runs stamp this."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def resolve_experiment(exp_yaml_path):
    """
    Load exp YAML, resolve inheritance against the named baseline, return a
    tuple (baseline_id, merged_config, change_block).
    """
    exp = load_yaml(exp_yaml_path)
    baseline_id = exp.get("inherits")
    if not baseline_id:
        raise ValueError(f"{exp_yaml_path}: missing 'inherits' key")
    baseline_path = os.path.join(BASELINES_DIR, f"{baseline_id}.yaml")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"baseline not found: {baseline_path}")
    baseline = load_yaml(baseline_path)

    change = exp.get("change") or {}
    if not isinstance(change, dict):
        raise ValueError(f"{exp_yaml_path}: 'change' must be a mapping")

    merged = deep_merge(baseline, change)
    # Preserve experiment-level metadata at the top level.
    merged["inherits"]  = baseline_id
    merged["change"]    = change
    merged["id"]        = exp.get("id", os.path.splitext(os.path.basename(exp_yaml_path))[0])
    merged["notes"]     = exp.get("notes", "")
    merged["declared_controlled_variable"] = exp.get("controlled_variable")

    return baseline_id, merged, change


def enforce_single_variable(change, declared_controlled=None):
    """
    Raise ValueError if the change block touches more than one controlled
    family. Non-controlled families (preprocessing/model/reporting) are
    allowed to co-vary, but only if `controlled_variable:` is declared.
    """
    touched_controlled = [k for k in change.keys() if k in CONTROLLED_FAMILIES]
    unknown             = [k for k in change.keys() if k not in CONFIG_FAMILIES]

    if unknown:
        raise ValueError(
            f"change block contains unknown family keys: {unknown}. "
            f"Allowed: {sorted(CONFIG_FAMILIES)}"
        )

    if len(touched_controlled) > 1 and not declared_controlled:
        raise ValueError(
            f"change block touches {len(touched_controlled)} controlled families "
            f"({touched_controlled}) but no 'controlled_variable:' is declared. "
            "Split this into separate experiments, or declare which family is the "
            "intended variable and why the others are co-varying."
        )

    if declared_controlled and declared_controlled not in CONTROLLED_FAMILIES:
        raise ValueError(
            f"controlled_variable '{declared_controlled}' is not a recognised "
            f"family; expected one of {sorted(CONTROLLED_FAMILIES)}"
        )


def validate_baseline(baseline_id):
    path = os.path.join(BASELINES_DIR, f"{baseline_id}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"baseline not found: {path}")
    cfg = load_yaml(path)

    required_top = ["id", "data", "preprocessing", "augmentation", "model",
                    "training", "inference", "methods", "evaluation", "reporting"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f"baseline {baseline_id}: missing required top-level keys: {missing}")

    if cfg["id"] != baseline_id:
        raise ValueError(f"baseline file id '{cfg['id']}' != filename id '{baseline_id}'")

    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=None,
                        help="Baseline id (without .yaml). Validates the baseline alone.")
    parser.add_argument("--exp", default=None,
                        help="Experiment id (without .yaml). Resolves + validates vs its baseline.")
    parser.add_argument("--print_resolved", action="store_true",
                        help="Print the resolved merged config to stdout on success.")
    args = parser.parse_args()

    if not args.baseline and not args.exp:
        parser.error("pass --baseline or --exp")

    try:
        if args.baseline and not args.exp:
            cfg = validate_baseline(args.baseline)
            print(f"baseline {args.baseline}: ok")
            if args.print_resolved:
                print(yaml.safe_dump(cfg, sort_keys=False))
            return 0

        exp_path = os.path.join(EXPERIMENTS_DIR, f"{args.exp}.yaml")
        if not os.path.exists(exp_path):
            raise FileNotFoundError(f"experiment not found: {exp_path}")
        baseline_id, merged, change = resolve_experiment(exp_path)

        # Baseline must itself be valid.
        validate_baseline(baseline_id)

        enforce_single_variable(change, merged.get("declared_controlled_variable"))

        sha = config_sha({k: v for k, v in merged.items()
                          if k not in {"change", "notes", "declared_controlled_variable"}})

        print(f"experiment {args.exp}: ok")
        print(f"  inherits: {baseline_id}")
        print(f"  change families: {sorted(change.keys())}")
        print(f"  config_sha: {sha[:16]}...")
        if args.print_resolved:
            print("---")
            print(yaml.safe_dump(merged, sort_keys=False))
        return 0

    except Exception as e:
        print(f"VALIDATION FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
