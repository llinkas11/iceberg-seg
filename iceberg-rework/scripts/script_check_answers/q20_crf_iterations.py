"""
q20_crf_iterations.py: empirical answer to script-check question 20.

Question (from script-check-README.md, densecrf_tifs.py):
  "Argmax over five iterations. Standard mean-field schedule. We have not
  measured whether more iterations meaningfully change the result. Is five
  enough?"

What this script does:
  1. Iterate the v4_clean test chips that have UNet++ probs and source
     chip TIFs.
  2. For each chip, run DenseCRF mean-field inference at each iteration
     count in {1, 3, 5, 10, 20} via crf_utils.apply_densecrf, holding all
     other CRF parameters at the production defaults.
  3. Measure the per-chip mask delta between successive iteration counts
     (e.g. delta(3 vs 1), delta(5 vs 3), delta(10 vs 5), delta(20 vs 10)),
     where delta is the fraction of pixels whose binary label changes.
  4. Aggregate across chips: mean per-chip mask delta vs iteration step,
     plus the residual change at iter=20 vs iter=5 (the production
     setting) as the headline number.
  5. Emit a per-chip CSV plus an overview convergence plot.

Inputs:
  --probs_root   Root containing *_probs.tif (recursively searched).
  --chips_root   Root containing source chip TIFs (for the bilateral term).
  --iters        Comma list of iteration counts to compare (default 1,3,5,10,20).
  --out_root     Parent directory; a slug subfolder is created under it.

Outputs (under <out_root>/q20_crf_iterations/):
  <ts>__q20_crf_iterations.csv               one row per (chip, iter)
  <ts>__q20_crf_iterations__overview.png     mean mask-delta vs iter step

Usage on moosehead:
  python3 q20_crf_iterations.py

Deploy:
  rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/script_check_answers/ \
    bowdoin:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/script_check_answers/
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

# Add the parent scripts/ dir so crf_utils + _method_common imports work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crf_utils import apply_densecrf

from _common import (
    make_slug_dir, resolve_probs_root, resolve_out_root, stamp,
)


SLUG = "q20_crf_iterations"
DEFAULT_ITERS = [1, 3, 5, 10, 20]
DEFAULT_PARAMS = {
    "sxy_gaussian":    3,
    "compat_gaussian": 3,
    "sxy_bilateral":   40,
    "srgb_bilateral":  3,
    "compat_bilateral": 4,
    # iterations is overridden per call.
}


def _hpc_chips_root():
    """v4_clean test chips on HPC; falls back to the per-region/SZA pool."""
    candidates = [
        Path("/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/data/v4_clean/test_chips"),
        Path("/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/chips"),
        Path("/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/S2-iceberg-areas/chips"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--probs_root", default=str(resolve_probs_root()),
                   help="Root containing *_probs.tif (recursively searched)")
    p.add_argument("--chips_root", default=str(_hpc_chips_root()),
                   help="Root containing chip TIFs for the bilateral term")
    p.add_argument("--iters", default=",".join(str(x) for x in DEFAULT_ITERS),
                   help="Comma list of iteration counts to compare")
    p.add_argument("--out_root", default=str(resolve_out_root()),
                   help="Parent directory; a slug subfolder is created under it")
    return p.parse_args()


def index_probs(probs_root):
    """Scan probs_root recursively and return {chip_stem: Path}."""
    index = {}
    for p in Path(probs_root).rglob("*_probs.tif"):
        stem = p.name[:-len("_probs.tif")]
        index[stem] = p
    return index


def find_chip_tif(chips_root, stem):
    """Locate the chip TIF by stem; recursive glob since layout varies."""
    for p in Path(chips_root).rglob(f"{stem}.tif"):
        return p
    return None


def main():
    args = parse_args()
    iters = [int(x) for x in args.iters.split(",") if x.strip()]
    if iters != sorted(iters):
        raise SystemExit("--iters must be sorted ascending")

    # 1. Resolve output dir, index probs, sanity-check chip root
    out_dir = make_slug_dir(SLUG, args.out_root)
    probs_idx = index_probs(args.probs_root)
    print(f"Probs found: {len(probs_idx)}")
    print(f"Chips root: {args.chips_root}")
    print(f"Iter sweep: {iters}")

    # 2. Per-chip CRF runs at each iteration count
    rows = []
    skipped = 0
    for stem, probs_path in sorted(probs_idx.items()):
        chip_tif = find_chip_tif(args.chips_root, stem)
        if chip_tif is None:
            skipped += 1
            continue

        # Load 2-band probs (P_ocean, P_iceberg).
        with rasterio.open(probs_path) as src:
            if src.count < 2:
                skipped += 1
                continue
            prob = src.read().astype(np.float32)  # (2, H, W)

        # Load 3-band chip for the bilateral term.
        with rasterio.open(chip_tif) as src:
            chip = src.read().astype(np.float32)  # (3, H, W)

        masks_by_iter = {}
        for n_iter in iters:
            params = dict(DEFAULT_PARAMS, iterations=n_iter)
            try:
                mask = apply_densecrf(prob, chip, params)
            except Exception as exc:
                print(f"  [{stem}] iter={n_iter} failed: {exc}")
                continue
            masks_by_iter[n_iter] = mask.astype(np.uint8)

        if len(masks_by_iter) < 2:
            skipped += 1
            continue

        # Per-chip per-step mask-delta as fraction of pixels that flipped.
        prev_iter = None
        n_pix = 0
        for n_iter in iters:
            if n_iter not in masks_by_iter:
                continue
            mask = masks_by_iter[n_iter]
            n_pix = mask.size
            if prev_iter is None:
                rows.append({
                    "chip_stem": stem,
                    "iter":      n_iter,
                    "prev_iter": "",
                    "delta_frac": "",
                    "ice_frac":  float((mask == 1).mean()),
                })
            else:
                prev_mask = masks_by_iter[prev_iter]
                flipped = int((mask != prev_mask).sum())
                rows.append({
                    "chip_stem": stem,
                    "iter":      n_iter,
                    "prev_iter": prev_iter,
                    "delta_frac": flipped / n_pix,
                    "ice_frac":  float((mask == 1).mean()),
                })
            prev_iter = n_iter

    if skipped:
        print(f"Skipped {skipped} chips (no source TIF, no probs, or CRF failure)")
    n_chips = len({r["chip_stem"] for r in rows})
    print(f"Processed {n_chips} chips at {len(iters)} iter levels each")

    # 3. Write per-chip CSV
    csv_path = out_dir / f"{stamp(SLUG)}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "chip_stem", "iter", "prev_iter", "delta_frac", "ice_frac",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {csv_path}")

    if not rows:
        print("Empty result set; nothing to plot.")
        return

    # 4. Aggregate mean delta per (prev_iter -> iter) step + report headline
    print("\nMean mask delta per iteration step (fraction of pixels flipped):")
    step_means = []
    step_labels = []
    for k in range(1, len(iters)):
        prev_iter, n_iter = iters[k - 1], iters[k]
        deltas = [float(r["delta_frac"])
                  for r in rows
                  if r["iter"] == n_iter and r["prev_iter"] == prev_iter
                  and r["delta_frac"] != ""]
        if not deltas:
            continue
        m = float(np.mean(deltas))
        p99 = float(np.percentile(deltas, 99))
        step_means.append(m)
        step_labels.append(f"{prev_iter}->{n_iter}")
        print(f"  {prev_iter:2d} -> {n_iter:2d}: mean={m:.4%}  p99={p99:.4%}  n={len(deltas)}")

    # 5. Convergence plot: mean delta vs iteration step
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(step_labels))
    ax.bar(x, step_means, color="#1976D2", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, fontsize=10)
    ax.set_ylabel("Mean fraction of pixels flipped")
    ax.set_xlabel("CRF iteration step")
    ax.set_title(f"Q20: per-chip mask-delta convergence across CRF iterations (n={n_chips})")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    overview_path = out_dir / f"{stamp(SLUG, 'overview')}.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {overview_path}")


if __name__ == "__main__":
    main()
