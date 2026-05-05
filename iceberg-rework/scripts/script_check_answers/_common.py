"""
_common.py: helpers shared across the script-check answer scripts.

Resolves dual Mac/HPC paths so the same script can run in either environment,
parses region + SZA bin from a chip path, and stamps a per-slug output dir
with a UTC-second timestamp so re-runs do not clobber prior outputs.

The scripts under this directory are review aids that empirically answer
parameter-choice questions raised in script-check-README.md. They are not
production methods and they are not paper figures: their outputs land in
paper-writing/figure_review/script_check_answers/<slug>/ for review.
"""

import os
from datetime import datetime
from pathlib import Path


# Canonical SZA bin list and region list, in the order figures use.
SZA_BINS = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]
REGIONS  = ["KQ", "SK"]


# HPC roots (per CLAUDE.md). chips and area_comparison live under smishra;
# the v4_clean manifest and iceberg-rework outputs live under llinkas.
#
# The probs root points at the canonical model-comparison test predictions
# (model_comparison_20260423_stage1_vs_baseline). Each chip's *_probs.tif
# lives at .../test/<sza_bin>/UNet/probs/<chip_stem>_probs.tif. As of
# 2026-05-05 no val-split predictions have been emitted, so Q15 defaults
# to the test split via --split test (see q15_unet_threshold_f1.py).
_HPC_CHIPS_ROOT       = Path("/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/chips")
_HPC_PROBS_ROOT       = Path("/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/results/model_comparison_20260423_stage1_vs_baseline/area_comparison/baseline_v3_balanced_aug/test")
_HPC_MANIFEST_PATH    = Path("/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/data/v4_clean/manifest.json")
_HPC_OUT_ROOT         = Path("/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/figure_review/script_check_answers")

# Mac fallbacks (OneDrive). Kept in sync with the actual local mirror.
_MAC_REPO_ROOT        = Path("/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026")
_MAC_CHIPS_ROOT       = _MAC_REPO_ROOT / "S2-iceberg-areas" / "chips"
_MAC_PROBS_ROOT       = _MAC_REPO_ROOT / "S2-iceberg-areas" / "area_comparison"  # placeholder; probs are HPC-only
_MAC_OUT_ROOT         = _MAC_REPO_ROOT / "paper-writing" / "figure_review" / "script_check_answers"


def resolve_chips_root():
    """Return the chips root that exists on this machine (HPC preferred)."""
    if _HPC_CHIPS_ROOT.exists():
        return _HPC_CHIPS_ROOT
    return _MAC_CHIPS_ROOT


def resolve_probs_root():
    """
    Return the probs root that exists on this machine. Probs are typically
    only on HPC; on Mac we return the placeholder (which may not exist) so
    argparse default printing stays sensible.
    """
    if _HPC_PROBS_ROOT.exists():
        return _HPC_PROBS_ROOT
    return _MAC_PROBS_ROOT


def resolve_manifest_path():
    """Return the v4_clean manifest path on this machine (HPC only in practice)."""
    if _HPC_MANIFEST_PATH.exists():
        return _HPC_MANIFEST_PATH
    return _MAC_REPO_ROOT / "iceberg-rework" / "data" / "v4_clean" / "manifest.json"


def resolve_out_root():
    """
    Return the script_check_answers output root for this machine. Anchors
    the HPC probe on the iceberg-rework repo dir, which is always present
    on HPC; the figure_review/ subtree is created on first run.
    """
    if Path("/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework").is_dir():
        return _HPC_OUT_ROOT
    return _MAC_OUT_ROOT


def make_slug_dir(slug, out_root=None):
    """
    Ensure <out_root>/<slug>/ exists and return its Path. Used by every
    answer script so all artifacts for a question stay co-located.
    """
    root = Path(out_root) if out_root else resolve_out_root()
    slug_dir = root / slug
    slug_dir.mkdir(parents=True, exist_ok=True)
    return slug_dir


def stamp(slug, suffix=""):
    """
    Build a `<YYYYMMDD_HHMMSS>__<slug>[__<suffix>]` filename body. Suffix is
    appended without an extension so the caller controls .csv vs .png.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if suffix:
        return f"{ts}__{slug}__{suffix}"
    return f"{ts}__{slug}"


def parse_region_sza(chip_path):
    """
    Parse (region, sza_bin) from a chip path like
    `.../chips/<REGION>/<SZA_BIN>/tifs/<stem>.tif`. Returns (None, None) if
    the path does not match the expected layout.
    """
    parts = Path(chip_path).parts
    # Walk from the right so paths with arbitrary roots still match.
    for i, part in enumerate(parts):
        if part == "tifs" and i >= 2:
            return parts[i - 2], parts[i - 1]
    return None, None


def list_chip_tifs(chips_root, regions=None, sza_bins=None):
    """
    Glob every chip TIF under chips_root that matches the canonical
    region/sza_bin layout. Returns a list of (Path, region, sza_bin) tuples
    sorted by (region, sza_bin, stem) for stable iteration.
    """
    chips_root = Path(chips_root)
    regions = regions or REGIONS
    sza_bins = sza_bins or SZA_BINS

    rows = []
    for region in regions:
        for sza_bin in sza_bins:
            tifs_dir = chips_root / region / sza_bin / "tifs"
            if not tifs_dir.is_dir():
                continue
            for tif in sorted(tifs_dir.glob("*.tif")):
                rows.append((tif, region, sza_bin))
    return rows
