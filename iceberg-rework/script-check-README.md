# Classical iceberg-segmentation methods: code review pack

Outlined below: The segmentation scripts I would like a second pair of eyes on. 
Goal: confirm that the parameter choices, edge-case handling, and inter-method integration are scientifically defensible. 
Scope: Unet++ training and the evaluation pipeline are not included here; only the per-chip segmentation logic.

## Scripts under review

All scripts live in [`iceberg-rework/scripts/`](scripts/), the same directory the production pipeline runs from. The links below jump straight to the live source on GitHub. The full source of each is also pasted at the end of this README so you can read it inline without clicking through every file.

| Script | Method ID | Role |
|---|---|---|
| [`scripts/_method_common.py`](scripts/_method_common.py) | (shared) | Provenance writers + skip-reason constants imported by every method script |
| [`scripts/threshold_tifs.py`](scripts/threshold_tifs.py) | TR | Fixed B08 NIR threshold on raw chip |
| [`scripts/threshold_masked_tifs.py`](scripts/threshold_masked_tifs.py) | TR (NDWI) | Same threshold but restricted to NDWI > 0 open-water pixels (sensitivity branch) |
| [`scripts/otsu_threshold_tifs.py`](scripts/otsu_threshold_tifs.py) | OT | Per-chip Otsu threshold on raw B08 |
| [`scripts/tophat_recover.py`](scripts/tophat_recover.py) | TH | White top-hat post-processor stacked on any base method |
| [`scripts/densecrf_tifs.py`](scripts/densecrf_tifs.py) | UNet+CRF | DenseCRF refinement of UNet++ softmax probabilities |
| [`scripts/crf_utils.py`](scripts/crf_utils.py) | (shared) | DenseCRF wrapper used by `densecrf_tifs.py` |
| [`scripts/threshold_probs.py`](scripts/threshold_probs.py) | UNet+TR | Fixed threshold applied to UNet++ P(iceberg) instead of B08 |
| [`scripts/otsu_probs.py`](scripts/otsu_probs.py) | UNet+OT | Per-chip Otsu applied to UNet++ P(iceberg) |

If you want to leave inline edits, the GitHub web UI lets you click the pencil icon on any file page to propose a change in a new branch and pull request.

## Project context

- **Imagery.** Sentinel-2 Level-1C, 10 m resolution, three bands per chip stacked in this order: B04 (red, 665 nm) at index 0, B03 (green, 560 nm) at index 1, B08 (NIR, 842 nm) at index 2. Chips are 256 x 256 pixels (2.56 km on a side) with the source scene's CRS and affine preserved.
- **Reflectance offset.** All scenes use ESA processing baseline N0500 or later, which adds a +1000 DN offset before distribution. Our chip extractor scales by 1e-4 without subtracting that offset, so every reflectance value in our chips is uniformly +0.10 higher than in the offset-corrected space used by Fisser and others (2024). Every threshold value quoted in this code is in the offset-uncorrected space, i.e. our 0.22 corresponds to Fisser's calibrated 0.12. Because the offset is identical across all our chips, it cancels out of any relative comparison between methods or SZA bins.
- **Reference baseline.** Fisser and others (2024) is the published Sentinel-2 reference for this region and the source of the `B08 >= 0.12` threshold and the 15 % ice-coverage (IC) chip-rejection rule we adopt.
- **Study aim.** Compare six methods for retrieving iceberg area at four SZA bins (lt65, 65 to 70, 70 to 75, gt75) in two Greenland fjords (Kangerlussuaq, Sermilik). Two purely classical methods (TR, OT), one learned (UNet++), and three hybrids that consume the UNet++ softmax (UNet_TR, UNet_OT, UNet_CRF). The white top-hat is in the pack as a sensitivity branch, not a headline method.
- **Outputs.** Each method writes per-chip GeoPackages plus an aggregate `all_icebergs.gpkg`, a `method_config.json` with the parameter block for provenance, and a `skipped_chips.csv` listing every chip the method refused with a reason code. Output schema is the same across methods so the evaluator can join on `chip_stem`.

## Methods at a glance

| Method | Input | Operation | Key parameters | Min polygon area |
|---|---|---|---|---|
| TR | B08 | `B08 >= 0.22` per pixel | `threshold=0.22`, `ic_threshold=0.15` | 100 m² |
| TR (NDWI) | B03, B08 | `(B08 >= 0.22) AND (NDWI > 0)` | `nir_threshold=0.22`, `ndwi_threshold=0.0` | 100 m² |
| OT | B08 | per-chip Otsu | `min_otsu_thresh=0.10` floor, `ic_threshold=0.15` | 100 m² |
| TH | B08 + base mask | `white_tophat(B08, disk(10)) >= 0.05`, then subtract base mask | `se_radius=10` px (100 m), `th_thresh=0.05`, `min_area_px=16` (40 m root length) | 16 px |
| UNet+CRF | UNet++ softmax + chip RGB | DenseCRF (gaussian + bilateral) | `sxy_g=3`, `compat_g=3`, `sxy_b=40`, `srgb=3`, `compat_b=4`, `iterations=5` | 100 m² |
| UNet+TR | UNet++ softmax band 1 | `P(iceberg) >= 0.22` | `threshold=0.22` | 100 m² |
| UNet+OT | UNet++ softmax band 1 | per-chip Otsu on `P(iceberg)` | flat-prob and IC guards | 100 m² |

The IC chip-rejection rule (Fisser 2025) is applied identically in TR, TR (NDWI), OT, and UNet+OT: if more than 15 % of pixels in a chip exceed the iceberg-defining threshold for that method, the chip is logged in `skipped_chips.csv` with reason `ic_block_filter` and excluded from the per-method aggregate. Skipped chips never silently disappear from the comparison; they are counted alongside accuracy metrics.

## What I would like you to check

The questions below are deliberately specific. If anything looks fine, a one-liner is enough; the value of this review is in catching the things I have not flagged.

### `threshold_tifs.py` (TR)

- **Threshold value.** We use 0.22 in offset-uncorrected reflectance space, equivalent to Fisser's 0.12 after the +0.10 baseline-N0500 correction. Is the offset reasoning correct, and is it acceptable to leave the chip extractor with the offset in place rather than subtract it before thresholding?
- **IC block filter.** A chip is skipped if more than 15 % of pixels exceed 0.22. The 15 % cutoff is taken from Fisser (2025). Is this the right operating point for our SZA range?
- **Connected-component polygonisation.** `rasterio.features.shapes` on the binary mask, no morphological cleanup before vectorisation. Any concerns about salt-and-pepper artifacts inflating the polygon count? The 100 m² min-area cutoff (~10 x 10 m) does most of the cleanup downstream.
- **No separation of touching icebergs.** Two icebergs whose pixels are 8-connected through one bright pixel become one polygon. Is that acceptable here, or should we use 4-connectivity, watershed, or distance-transform splitting?

### `threshold_masked_tifs.py` (TR with NDWI water mask)

- **NDWI threshold = 0.** NDWI = (B03 - B08) / (B03 + B08 + ε). We treat any pixel with NDWI > 0 as open water. Is 0 the right cutoff for icebergs in fjord water, or should it be slightly positive (e.g. 0.05) to exclude land-edge pixels?
- **AND with B08 >= 0.22.** A pixel is an iceberg only if both criteria pass. This kills sea ice (B08 high, NDWI low) and clouds (NDWI low) but also kills very bright icebergs whose NDWI may dip below 0 if B08 saturates. Is the AND defensible, or should it be a soft union with a higher B08 threshold for non-water pixels?
- **Sensitivity branch only.** This file is not in the headline six-method comparison. Is there a reason to promote it, e.g. if the IC filter is too crude for high-SZA chips?

### `otsu_threshold_tifs.py` (OT)

- **Otsu on raw B08.** No log-transform, no contrast stretch, no exclusion of saturated pixels. Otsu finds the threshold maximising inter-class variance on the chip's 256 x 256 pixel histogram. For ocean-dominated chips the histogram is highly skewed; does Otsu still give a sensible threshold, or should we be doing a log-transform first?
- **Flat-chip floor.** `min_otsu_thresh = 0.10`. Chips where Otsu returns a threshold below 0.10 are skipped. The floor exists because a featureless ocean chip will let Otsu find a "threshold" near zero on noise. Is 0.10 the right floor in offset-uncorrected reflectance, or should it be 0.20 (i.e. 0.10 after the +0.10 offset)?
- **IC filter on the Otsu result.** Same 15 % rule as TR but evaluated against the Otsu threshold for that chip. Order of operations: compute Otsu, then test whether 15 % of pixels exceed it. Is this the right order, or should the IC test be against a fixed reference threshold so that "sea ice contamination" means the same thing across chips?
- **Visualisation.** Three-panel diagnostic PNG (false-color RGB, B08 histogram with threshold marked, mask overlay). Used for QA only, not in the area pipeline. Any obvious thing missing from the QA panel?

### `tophat_recover.py` (TH)

- **Structuring element.** `disk(10)` at 10 m pixels, i.e. SE radius 100 m. Drawn from Fisser's reported small-iceberg cap. Is the disk shape and that radius defensible for the 40 m to 100 m size range we want to recover, or should we sweep over multiple radii and combine?
- **Top-hat threshold.** `th_thresh = 0.05` reflectance units. Honestly chosen by eye on a few chips. If there is a more principled estimator (e.g., chip-wise σ of the top-hat response, or an Otsu of the response), I would adopt it.
- **Min component size.** `min_area_px = 16` (~40 m root length). Matches the 40 m root-length filter we apply globally on the annotation side. Reasonable, or should the cutoff be smaller for a recovery method whose entire job is to find missed small icebergs?
- **Subtraction of base mask.** Recovered candidates are `(response >= th_thresh) AND NOT base_mask`. Base mask is built from `<stem>_pred.tif` if present, else by rasterising the base method's polygons. Any concerns about the rasterised-polygon path on per-chip CRS mismatches?
- **Cross-chip aggregation.** The combined `all_icebergs.gpkg` is written `try/except` because some Fisser synthetic chips have CRS = None and our real chips span UTM 24N + 25N. Per-chip gpkgs always succeed; the cross-chip concat is best-effort. Is this the right call?

### `densecrf_tifs.py` + `crf_utils.py` (UNet+CRF)

- **CRF parameters.** `sxy_gaussian = 3`, `compat_gaussian = 3`, `sxy_bilateral = 40`, `srgb_bilateral = 3`, `compat_bilateral = 4`, `iterations = 5`. Drawn from a 2-point sandbox sweep on a holdout subset. The bilateral-spatial scale (40 px = 400 m) feels generous for our 256 x 256 chip; is this defensible for icebergs at this resolution, or should `sxy_bilateral` be closer to 5 to 15 px?
- **Bilateral image.** The pairwise bilateral term consumes the chip after `scale_chip_to_uint8()`, which independently percentile-stretches each band (2/98) and casts to uint8. So the bilateral term is acting on a uint8 RGB-like rendering of B04/B03/B08, not on raw reflectance and not on grayscale. Is per-band 2/98 stretching appropriate here, or does it overweight bright outliers and pull boundaries onto noise?
- **Two-class CRF.** Ocean vs iceberg only. Shadow has been merged into iceberg upstream during training so the model never emits a third channel. Is the two-class CRF the right setup for this problem given that bright sea ice and dark shadow can both touch real iceberg edges?
- **Argmax over five iterations.** Standard mean-field schedule. We have not measured whether more iterations meaningfully change the result. Is five enough?

### `threshold_probs.py` and `otsu_probs.py` (UNet+TR, UNet+OT)

- **Threshold = 0.22 on probability.** UNet+TR uses 0.22 as the cutoff on `P(iceberg)`. The number was inherited from the Fisser reflectance threshold deliberately, so `UNet_TR` and `TR` use a numerically identical cutoff in different spaces. Is reusing 0.22 the right choice, or should we calibrate the probability threshold separately, e.g. to the F1 optimum on the validation set?
- **Flat-prob skip.** `otsu_probs.py` skips chips where `P(iceberg).max() - .min() < 0.01`. Otsu cannot find a threshold on a flat distribution; this guard exists to avoid a degenerate split. Is 0.01 a reasonable flatness floor on softmax space?
- **No `min_otsu_thresh` floor on probabilities.** OT on B08 has a floor at 0.10 reflectance; OT on probability does not. Should OT on probability also have a floor (e.g. 0.5) to avoid carving icebergs out of low-confidence ocean?

## How the methods plug together

```
chip .tif  (B04/B03/B08, 10 m)
   |
   |--> threshold_tifs.py        --> TR/all_icebergs.gpkg
   |--> threshold_masked_tifs.py --> TR_NDWI/all_icebergs.gpkg
   |--> otsu_threshold_tifs.py   --> OT/all_icebergs.gpkg + diagnostic PNGs
   |
   |--> predict_tifs.py [not in this pack] --> *_probs.tif (2 bands: ocean, iceberg)
   |        |
   |        |--> threshold_probs.py  --> UNet_TR/all_icebergs.gpkg
   |        |--> otsu_probs.py       --> UNet_OT/all_icebergs.gpkg
   |        |--> densecrf_tifs.py    --> UNet_CRF/all_icebergs.gpkg
   |
   |--> tophat_recover.py [stacked on any base] --> <base>_TH/all_icebergs.gpkg
```

`predict_tifs.py` is the UNet++ inference step and is not in this review pack. It writes a 2-band float32 GeoTIFF per chip with band 0 = `P(ocean)` and band 1 = `P(iceberg)`; that is the input to the three `*_probs.py` and to `densecrf_tifs.py`.

## Dependencies

- numpy, rasterio, geopandas, shapely, pandas
- scikit-image (`threshold_otsu`, `disk`, `white_tophat`, `label`)
- pydensecrf or pydensecrf2 (only needed for `densecrf_tifs.py`)
- matplotlib (only for `otsu_threshold_tifs.py` diagnostic PNGs)

## Provenance

- Branch: `paper-figures-and-results`. Live scripts under [`iceberg-rework/scripts/`](scripts/).
- The inline source pasted at the end of this README is a read-through copy of the live scripts at the time of the last commit to this README. If you are about to leave a comment on a specific line, prefer clicking through to the live file via the table above so the line numbers match.

---

# Inline source code

The full source of each script is pasted below as a single read-through. The same code is in the live files linked in the [Scripts under review](#scripts-under-review) table. Click a heading to expand.

<details>
<summary><strong>_method_common.py</strong> (shared helpers, 124 lines)</summary>

```python
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
```

</details>

<details>
<summary><strong>threshold_tifs.py</strong> — TR, fixed B08 threshold (184 lines)</summary>

```python
"""
threshold_tifs.py: Apply Fisser B08 >= 0.12 NIR threshold to S2 chip .tifs.

Mirrors the output format of predict_tifs.py so compare_areas.py can load both.

Usage:
  python threshold_tifs.py \\
      --chips_dir chips/KQ/sza_65_70/tifs \\
      --out_dir   georef_predictions/KQ/sza_65_70

Output:
  out_dir/all_icebergs.gpkg  iceberg polygons with area_m2
  out_dir/method_config.json parameters used by this run
  out_dir/skipped_chips.csv  chips excluded, with a reason

Note:
  --b08_idx is the 0-indexed band position of B08 in the chip stack.
  Default is 2 (i.e. bands were stacked as B04, B03, B08 by chip_sentinel2.py).
  If you used a different band order, adjust accordingly.
"""

import os
import argparse
import warnings
from glob import glob

import numpy as np
import rasterio as rio
from rasterio.features import shapes as rio_shapes
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd

from _method_common import (
    write_method_config, write_skipped_chips,
    SKIP_TOO_FEW_BANDS, SKIP_IC_BLOCK_FILTER,
)

warnings.filterwarnings("ignore")

THRESHOLD   = 0.22   # Fisser 2024 B08 NIR reflectance threshold (0.12) + 0.10 DN offset correction
                     # All scenes have processing baseline >=4.0 (N0500/N0510), which adds +1000 DN
                     # chip_sentinel2.py does not subtract this offset, so reflectances are +0.1 high
                     # 0.22 here = 0.12 in Fisser's corrected reflectance space
MIN_AREA_M2 = 100    # minimum polygon area in m2 (~10x10 m)
IC_THRESHOLD = 0.15  # Fisser 2025 IC block filter: skip chip if >15% of pixels exceed NIR threshold
                     # Flags chips dominated by sea ice rather than open water with icebergs


def apply_threshold(chips_dir, out_dir, b08_idx=2, threshold=THRESHOLD, min_area_m2=MIN_AREA_M2, ic_threshold=IC_THRESHOLD):
    os.makedirs(out_dir, exist_ok=True)

    tif_files = sorted(glob(os.path.join(chips_dir, "*.tif")))
    if not tif_files:
        print(f"No .tif files found in {chips_dir}")
        return

    print(f"Found {len(tif_files)} chips  threshold={threshold}  b08_idx={b08_idx}  ic_threshold={ic_threshold}")

    all_gdfs   = []
    skipped    = []   # one row per chip we refused to score

    for i, tif_path in enumerate(tif_files):
        stem = os.path.splitext(os.path.basename(tif_path))[0]

        with rio.open(tif_path) as src:
            chip = src.read().astype(np.float32)  # (C, H, W)
            meta = src.meta.copy()

        if chip.shape[0] <= b08_idx:
            print(f"  [{i+1}/{len(tif_files)}] SKIP {stem}: only {chip.shape[0]} bands")
            skipped.append({"chip_stem": stem, "reason": SKIP_TOO_FEW_BANDS,
                            "n_bands": chip.shape[0]})
            continue

        b08 = chip[b08_idx]

        # IC block filter (Fisser 2025): skip sea-ice-dominated chips
        ic_frac = float((b08 >= threshold).mean())
        if ic_frac > ic_threshold:
            print(f"  [{i+1:>4}/{len(tif_files)}] IC   {stem[:60]}  ic_frac={ic_frac:.2f}")
            skipped.append({"chip_stem": stem, "reason": SKIP_IC_BLOCK_FILTER,
                            "ic_frac": f"{ic_frac:.4f}"})
            continue

        iceberg_mask = (b08 >= threshold).astype(np.uint8)

        records = []
        for geom_dict, val in rio_shapes(iceberg_mask, transform=meta["transform"]):
            if val == 0:
                continue
            geom = shape(geom_dict)
            if geom.is_empty or geom.area < min_area_m2:
                continue
            records.append({
                "geometry"   : geom,
                "class_id"   : 1,
                "class_name" : "iceberg",
                "area_m2"    : round(geom.area, 2),
                "source_file": os.path.basename(tif_path),
            })

        print(f"  [{i+1:>4}/{len(tif_files)}] {stem[:60]}  icebergs={len(records)}")

        if records:
            gdf = gpd.GeoDataFrame(records, crs=meta["crs"])
            all_gdfs.append(gdf)

    # Write provenance files before the empty-run short-circuit below, so an
    # empty result still produces method_config.json + skipped_chips.csv the
    # evaluator can join on.
    cfg_path = write_method_config(
        out_dir, "TR",
        params={
            "chips_dir":    os.path.abspath(chips_dir),
            "threshold":    threshold,
            "min_area_m2":  min_area_m2,
            "b08_idx":      b08_idx,
            "ic_threshold": ic_threshold,
        },
    )
    skip_path = write_skipped_chips(out_dir, skipped)

    n_ic      = sum(1 for r in skipped if r["reason"] == SKIP_IC_BLOCK_FILTER)
    n_skipped = sum(1 for r in skipped if r["reason"] == SKIP_TOO_FEW_BANDS)

    if not all_gdfs:
        print("\nNo icebergs detected across all chips.")
        if n_ic:
            print(f"IC-filtered: {n_ic} chips (sea ice contamination)")
        if n_skipped:
            print(f"Skipped:     {n_skipped} chips (too few bands)")
        print(f"Method config : {cfg_path}")
        print(f"Skipped chips : {skip_path}")
        return

    target_crs = all_gdfs[0].crs
    reprojected = [gdf.to_crs(target_crs) if gdf.crs != target_crs else gdf for gdf in all_gdfs]
    merged = gpd.GeoDataFrame(pd.concat(reprojected, ignore_index=True), crs=target_crs)
    merged["iceberg_id"] = range(1, len(merged) + 1)

    icebergs = merged[merged["class_name"] == "iceberg"]
    print(f"\n{'-'*50}")
    print(f"Total iceberg polygons : {len(icebergs)}")
    if len(icebergs) > 0:
        print(f"  min  = {icebergs['area_m2'].min():.1f} m2")
        print(f"  mean = {icebergs['area_m2'].mean():.1f} m2")
        print(f"  max  = {icebergs['area_m2'].max():.1f} m2")
    if n_ic:
        print(f"IC-filtered: {n_ic} chips (sea ice contamination)")
    if n_skipped:
        print(f"Skipped:     {n_skipped} chips (too few bands)")
    print(f"{'-'*50}")

    out_path = os.path.join(out_dir, "all_icebergs.gpkg")
    merged.to_file(out_path, driver="GPKG")
    print(f"\nSaved         : {out_path}")
    print(f"Method config : {cfg_path}")
    print(f"Skipped chips : {skip_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply Fisser B08 >= 0.12 NIR threshold to S2 chip .tifs"
    )
    parser.add_argument("--chips_dir", required=True,
                        help="Directory of .tif chip files (same dir used by predict_tifs.py --imgs_dir)")
    parser.add_argument("--out_dir",   required=True,
                        help="Output directory, use the same path as predict_tifs.py --out_dir")
    parser.add_argument("--b08_idx",   type=int,   default=2,
                        help="0-indexed band position of B08 in chip stack (default: 2 for B04/B03/B08 order)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"NIR reflectance threshold (default: {THRESHOLD})")
    parser.add_argument("--min_area",     type=float, default=MIN_AREA_M2,
                        help=f"Min iceberg area in m2 (default: {MIN_AREA_M2})")
    parser.add_argument("--ic_threshold", type=float, default=IC_THRESHOLD,
                        help=f"IC block filter: skip chip if fraction of bright pixels exceeds this (default: {IC_THRESHOLD})")
    args = parser.parse_args()

    apply_threshold(args.chips_dir, args.out_dir, args.b08_idx, args.threshold, args.min_area, args.ic_threshold)


if __name__ == "__main__":
    main()
```

</details>

<details>
<summary><strong>threshold_masked_tifs.py</strong> — TR with NDWI water mask (197 lines)</summary>

```python
"""
threshold_masked_tifs.py: Apply Fisser B08 >= 0.12 NIR threshold restricted to
open-water pixels identified by an NDWI water mask.

NDWI = (B03 - B08) / (B03 + B08 + e)
Pixels where NDWI > ndwi_threshold are classified as open water.
B08 >= 0.12 is then applied ONLY within open-water pixels.

This prevents bright sea ice, clouds, and snow from being counted as icebergs,
giving a fairer comparison to UNet++ which learned to distinguish these classes.

Chip band order (set by chip_sentinel2.py): B04=0, B03=1, B08=2

Usage:
  python threshold_masked_tifs.py \\
      --chips_dir chips/KQ/sza_gt75/tifs \\
      --out_dir   area_comparison/KQ/sza_gt75/threshold_masked

Output:
  out_dir/all_icebergs_threshold_masked.gpkg
"""

import os
import argparse
import warnings
from glob import glob

import numpy as np
import rasterio as rio
from rasterio.features import shapes as rio_shapes
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd

warnings.filterwarnings("ignore")

NIR_THRESHOLD  = 0.22   # Fisser 2024 B08 threshold (0.12) + 0.10 DN offset correction
                        # All scenes baseline >=4.0: chip_sentinel2.py does not subtract +1000 DN offset
NDWI_THRESHOLD = 0.0    # NDWI > 0 -> open water (negative = ice/land/cloud)
MIN_AREA_M2    = 100    # ~10x10 m minimum polygon
IC_THRESHOLD   = 0.15   # Fisser 2025 IC block filter: skip chip if >15% of pixels exceed NIR threshold
                        # Flags chips dominated by sea ice rather than open water with icebergs


def apply_masked_threshold(
    chips_dir,
    out_dir,
    b03_idx=1,
    b08_idx=2,
    nir_threshold=NIR_THRESHOLD,
    ndwi_threshold=NDWI_THRESHOLD,
    min_area_m2=MIN_AREA_M2,
    ic_threshold=IC_THRESHOLD,
):
    os.makedirs(out_dir, exist_ok=True)

    tif_files = sorted(glob(os.path.join(chips_dir, "*.tif")))
    if not tif_files:
        print(f"No .tif files found in {chips_dir}")
        return

    print(f"Found {len(tif_files)} chips")
    print(f"  NIR threshold : B08 >= {nir_threshold}")
    print(f"  NDWI mask     : NDWI > {ndwi_threshold} (open water only)")
    print(f"  IC filter     : skip chip if bright-pixel fraction > {ic_threshold}")
    print(f"  Min area      : {min_area_m2} m^2\n")

    all_gdfs  = []
    n_skipped = 0
    n_ic      = 0

    for i, tif_path in enumerate(tif_files):
        stem = os.path.splitext(os.path.basename(tif_path))[0]

        with rio.open(tif_path) as src:
            chip = src.read().astype(np.float32)   # (C, H, W)
            meta = src.meta.copy()

        n_bands = chip.shape[0]
        if n_bands <= max(b03_idx, b08_idx):
            print(f"  [{i+1}/{len(tif_files)}] SKIP {stem} - only {n_bands} band(s)")
            n_skipped += 1
            continue

        b03 = chip[b03_idx]
        b08 = chip[b08_idx]

        # IC block filter (Fisser 2025): skip sea-ice-dominated chips
        ic_frac = float((b08 >= nir_threshold).mean())
        if ic_frac > ic_threshold:
            print(
                f"  [{i+1:>4}/{len(tif_files)}] IC   {stem[:55]}  ic_frac={ic_frac:.2f}"
            )
            n_ic += 1
            continue

        # NDWI water mask: open water has positive NDWI
        ndwi       = (b03 - b08) / (b03 + b08 + 1e-6)
        water_mask = (ndwi > ndwi_threshold).astype(np.uint8)

        # Apply NIR threshold restricted to open-water pixels
        iceberg_mask = ((b08 >= nir_threshold) & (water_mask == 1)).astype(np.uint8)

        records = []
        for geom_dict, val in rio_shapes(iceberg_mask, transform=meta["transform"]):
            if val == 0:
                continue
            geom = shape(geom_dict)
            if geom.is_empty or geom.area < min_area_m2:
                continue
            records.append({
                "geometry"   : geom,
                "class_id"   : 1,
                "class_name" : "iceberg",
                "area_m2"    : round(geom.area, 2),
                "source_file": os.path.basename(tif_path),
            })

        n_water_px   = int(water_mask.sum())
        n_iceberg_px = int(iceberg_mask.sum())
        print(
            f"  [{i+1:>4}/{len(tif_files)}] {stem[:55]}  "
            f"water_px={n_water_px:>6}  icebergs={len(records)}"
        )

        if records:
            gdf = gpd.GeoDataFrame(records, crs=meta["crs"])
            all_gdfs.append(gdf)

    if not all_gdfs:
        print("\nNo icebergs detected across all chips.")
        if n_ic:
            print(f"IC-filtered: {n_ic} chips (sea ice contamination)")
        if n_skipped:
            print(f"Skipped:     {n_skipped} chips (too few bands)")
        return

    target_crs = all_gdfs[0].crs
    reprojected = [gdf.to_crs(target_crs) if gdf.crs != target_crs else gdf for gdf in all_gdfs]
    merged = gpd.GeoDataFrame(pd.concat(reprojected, ignore_index=True), crs=target_crs)
    merged["iceberg_id"] = range(1, len(merged) + 1)

    icebergs = merged[merged["class_name"] == "iceberg"]
    print(f"\n{'-'*50}")
    print(f"Total iceberg polygons : {len(icebergs)}")
    if len(icebergs) > 0:
        print(f"  min    = {icebergs['area_m2'].min():.1f} m^2")
        print(f"  median = {icebergs['area_m2'].median():.1f} m^2")
        print(f"  mean   = {icebergs['area_m2'].mean():.1f} m^2")
        print(f"  max    = {icebergs['area_m2'].max():.1f} m^2")
        print(f"  total  = {icebergs['area_m2'].sum()/1e6:.4f} km^2")
    if n_ic:
        print(f"IC-filtered: {n_ic} chips (sea ice contamination)")
    if n_skipped:
        print(f"Skipped:     {n_skipped} chips (too few bands)")
    print(f"{'-'*50}")

    out_path = os.path.join(out_dir, "all_icebergs_threshold_masked.gpkg")
    merged.to_file(out_path, driver="GPKG")
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply NDWI-masked Fisser B08 threshold to S2 chip .tifs"
    )
    parser.add_argument("--chips_dir",      required=True,
                        help="Directory of .tif chip files")
    parser.add_argument("--out_dir",        required=True,
                        help="Output directory for all_icebergs_threshold_masked.gpkg")
    parser.add_argument("--b03_idx",        type=int,   default=1,
                        help="0-indexed band position of B03 in chip stack (default: 1)")
    parser.add_argument("--b08_idx",        type=int,   default=2,
                        help="0-indexed band position of B08 in chip stack (default: 2)")
    parser.add_argument("--threshold",      type=float, default=NIR_THRESHOLD,
                        help=f"NIR reflectance threshold (default: {NIR_THRESHOLD})")
    parser.add_argument("--ndwi_threshold", type=float, default=NDWI_THRESHOLD,
                        help=f"NDWI cutoff for open-water mask (default: {NDWI_THRESHOLD})")
    parser.add_argument("--min_area",       type=float, default=MIN_AREA_M2,
                        help=f"Min polygon area in m^2 (default: {MIN_AREA_M2})")
    parser.add_argument("--ic_threshold",   type=float, default=IC_THRESHOLD,
                        help=f"IC block filter: skip chip if bright-pixel fraction exceeds this (default: {IC_THRESHOLD})")
    args = parser.parse_args()

    apply_masked_threshold(
        chips_dir      = args.chips_dir,
        out_dir        = args.out_dir,
        b03_idx        = args.b03_idx,
        b08_idx        = args.b08_idx,
        nir_threshold  = args.threshold,
        ndwi_threshold = args.ndwi_threshold,
        min_area_m2    = args.min_area,
        ic_threshold   = args.ic_threshold,
    )


if __name__ == "__main__":
    main()
```

</details>

<details>
<summary><strong>otsu_threshold_tifs.py</strong> — OT, per-chip Otsu on B08 (319 lines)</summary>

```python
"""
otsu_threshold_tifs.py: Apply per-chip Otsu threshold on B08 to S2 chip .tifs.

Threshold is computed unsupervised via skimage.filters.threshold_otsu on the
B08 band of each chip independently, so it adapts to local illumination/scene
conditions rather than using a fixed reflectance cutoff.

Mirrors the output format of threshold_tifs.py / predict_tifs.py so
compare_areas.py can load all methods from the same directory.

Usage:
  python otsu_threshold_tifs.py \\
      --chips_dir chips/KQ/sza_65_70/tifs \\
      --out_dir   area_comparison/KQ/sza_65_70

Output:
  out_dir/
    otsu_thresholding/
      all_icebergs_otsu.gpkg     : all iceberg polygons with area_m2, otsu_thresh
      pngs/
        <chip>_otsu.png          : 3-panel false-color RGB, B08 histogram, mask overlay

Note:
  --b08_idx is the 0-indexed band position of B08 in the chip stack.
  Default is 2 (i.e. bands stacked as B04/B03/B08 by chip_sentinel2.py).

  Chips where Otsu yields a threshold below --min_otsu_thresh are skipped
  (all-ocean chips with no bright targets; Otsu would fire on noise).
"""

import os
import argparse
import warnings
from glob import glob

import numpy as np
import rasterio as rio
from rasterio.features import shapes as rio_shapes
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd
from skimage.filters import threshold_otsu
import matplotlib
matplotlib.use("Agg")

from _method_common import (
    write_method_config, write_skipped_chips,
    SKIP_TOO_FEW_BANDS, SKIP_OTSU_FLOOR, SKIP_IC_BLOCK_FILTER,
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

MIN_AREA_M2     = 100   # minimum polygon area in m^2 (~10x10 m)
IC_THRESHOLD    = 0.15  # skip chip if >15% of pixels exceed the Otsu threshold
MIN_OTSU_THRESH = 0.10  # skip chip if Otsu threshold < this (flat/featureless chips)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def percentile_stretch(band, lo=2, hi=98):
    """Stretch a 2D array to [0, 1] using percentile clipping."""
    p_lo, p_hi = np.percentile(band, [lo, hi])
    if p_hi == p_lo:
        return np.zeros_like(band, dtype=np.float32)
    return np.clip((band - p_lo) / (p_hi - p_lo), 0, 1).astype(np.float32)


def make_false_color(chip, b08_idx=2):
    """
    B04 to R, B03 to G, B08 to B, matches the UNet++ training chip rendering.
    Ocean appears dark blue (low NIR in blue channel); icebergs appear bright white.
    chip shape: (C, H, W)
    """
    red = percentile_stretch(chip[0])        # B04 -> R
    grn = percentile_stretch(chip[1])        # B03 -> G
    nir = percentile_stretch(chip[b08_idx])  # B08 -> B
    return np.stack([red, grn, nir], axis=-1)  # (H, W, 3)


def save_png(stem, chip, b08, otsu_thresh, iceberg_mask, n_icebergs, out_path, b08_idx=2):
    """Save 3-panel diagnostic PNG for one chip."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Panel 1: False-color RGB (NIR-R-G)
    fc = make_false_color(chip, b08_idx)
    axes[0].imshow(fc)
    axes[0].set_title("False color (NIR-R-G)", color="white", fontsize=11)
    axes[0].axis("off")

    # Panel 2: B08 histogram with Otsu threshold
    flat = b08.ravel()
    axes[1].hist(flat, bins=80, color="#4a9eff", alpha=0.8, edgecolor="none")
    axes[1].axvline(otsu_thresh, color="#ff6b6b", linewidth=2,
                    label=f"Otsu = {otsu_thresh:.4f}")
    axes[1].set_title("B08 histogram", color="white", fontsize=11)
    axes[1].set_xlabel("Reflectance", color="white", fontsize=9)
    axes[1].set_ylabel("Pixel count", color="white", fontsize=9)
    axes[1].tick_params(colors="white", labelsize=8)
    axes[1].legend(fontsize=9, facecolor="#2a2a4e", labelcolor="white",
                   edgecolor="#444")

    # shade the "iceberg" side of the threshold
    x_max = flat.max()
    axes[1].axvspan(otsu_thresh, x_max, alpha=0.15, color="#ff6b6b")

    # Panel 3: Mask overlay on B08
    b08_disp = percentile_stretch(b08)
    axes[2].imshow(b08_disp, cmap="gray")

    # overlay iceberg mask in semi-transparent cyan
    overlay = np.zeros((*iceberg_mask.shape, 4), dtype=np.float32)
    overlay[iceberg_mask == 1] = [0.0, 1.0, 0.9, 0.55]   # cyan, 55% opacity
    axes[2].imshow(overlay)

    patch = mpatches.Patch(color=(0.0, 1.0, 0.9, 0.8), label=f"Icebergs ({n_icebergs})")
    axes[2].legend(handles=[patch], fontsize=9, facecolor="#2a2a4e",
                   labelcolor="white", edgecolor="#444", loc="lower right")
    axes[2].set_title("Otsu mask on B08", color="white", fontsize=11)
    axes[2].axis("off")

    fig.suptitle(
        f"{stem}\n"
        f"otsu_thresh={otsu_thresh:.4f}   icebergs={n_icebergs}",
        color="white", fontsize=10, y=1.01
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def apply_otsu(chips_dir, out_dir, b08_idx=2, min_area_m2=MIN_AREA_M2,
               ic_threshold=IC_THRESHOLD, min_otsu_thresh=MIN_OTSU_THRESH):

    png_dir = os.path.join(out_dir, "pngs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    tif_files = sorted(glob(os.path.join(chips_dir, "*.tif")))
    if not tif_files:
        print(f"No .tif files found in {chips_dir}")
        return

    print(f"Found {len(tif_files)} chips")
    print(f"  b08_idx={b08_idx}  ic_threshold={ic_threshold}  "
          f"min_otsu_thresh={min_otsu_thresh}")
    print(f"  Output dir : {out_dir}")
    print(f"  PNGs dir   : {png_dir}\n")

    all_gdfs = []
    skipped  = []

    for i, tif_path in enumerate(tif_files):
        stem = os.path.splitext(os.path.basename(tif_path))[0]

        with rio.open(tif_path) as src:
            chip = src.read().astype(np.float32)   # (C, H, W)
            meta = src.meta.copy()

        if chip.shape[0] <= b08_idx:
            print(f"  [{i+1:>4}/{len(tif_files)}] SKIP {stem}: only {chip.shape[0]} bands")
            skipped.append({"chip_stem": stem, "reason": SKIP_TOO_FEW_BANDS,
                            "n_bands": chip.shape[0]})
            continue

        b08 = np.nan_to_num(chip[b08_idx], nan=0.0)

        otsu_thresh = float(threshold_otsu(b08))

        if otsu_thresh < min_otsu_thresh:
            print(f"  [{i+1:>4}/{len(tif_files)}] FLAT {stem[:55]}  "
                  f"otsu={otsu_thresh:.4f}")
            skipped.append({"chip_stem": stem, "reason": SKIP_OTSU_FLOOR,
                            "otsu_thresh": f"{otsu_thresh:.4f}"})
            continue

        ic_frac = float((b08 >= otsu_thresh).mean())
        if ic_frac > ic_threshold:
            print(f"  [{i+1:>4}/{len(tif_files)}] IC   {stem[:55]}  "
                  f"otsu={otsu_thresh:.4f}  ic_frac={ic_frac:.2f}")
            skipped.append({"chip_stem": stem, "reason": SKIP_IC_BLOCK_FILTER,
                            "otsu_thresh": f"{otsu_thresh:.4f}",
                            "ic_frac":     f"{ic_frac:.4f}"})
            continue

        iceberg_mask = (b08 >= otsu_thresh).astype(np.uint8)

        # Polygonize
        records = []
        for geom_dict, val in rio_shapes(iceberg_mask, transform=meta["transform"]):
            if val == 0:
                continue
            geom = shape(geom_dict)
            if geom.is_empty or geom.area < min_area_m2:
                continue
            records.append({
                "geometry"    : geom,
                "class_id"    : 1,
                "class_name"  : "iceberg",
                "area_m2"     : round(geom.area, 2),
                "otsu_thresh" : round(otsu_thresh, 4),
                "source_file" : os.path.basename(tif_path),
            })

        print(f"  [{i+1:>4}/{len(tif_files)}] {stem[:55]}  "
              f"otsu={otsu_thresh:.4f}  icebergs={len(records)}")

        # Save PNG
        png_path = os.path.join(png_dir, f"{stem}_otsu.png")
        save_png(stem, chip, b08, otsu_thresh, iceberg_mask,
                 len(records), png_path, b08_idx)

        if records:
            gdf = gpd.GeoDataFrame(records, crs=meta["crs"])
            all_gdfs.append(gdf)

    # -----------------------------------------------------------------------
    # Save merged GeoPackage + provenance
    # -----------------------------------------------------------------------
    cfg_path = write_method_config(
        out_dir, "OT",
        params={
            "chips_dir":       os.path.abspath(chips_dir),
            "b08_idx":         b08_idx,
            "min_area_m2":     min_area_m2,
            "ic_threshold":    ic_threshold,
            "min_otsu_thresh": min_otsu_thresh,
        },
    )
    skip_path = write_skipped_chips(out_dir, skipped)

    n_skipped = sum(1 for r in skipped if r["reason"] == SKIP_TOO_FEW_BANDS)
    n_flat    = sum(1 for r in skipped if r["reason"] == SKIP_OTSU_FLOOR)
    n_ic      = sum(1 for r in skipped if r["reason"] == SKIP_IC_BLOCK_FILTER)

    if not all_gdfs:
        print("\nNo icebergs detected across all chips.")
        if n_flat:    print(f"  Flat chips skipped : {n_flat}")
        if n_ic:      print(f"  IC chips skipped   : {n_ic}")
        if n_skipped: print(f"  Too few bands      : {n_skipped}")
        print(f"Method config : {cfg_path}")
        print(f"Skipped chips : {skip_path}")
        return

    target_crs = all_gdfs[0].crs
    reprojected = [gdf.to_crs(target_crs) if gdf.crs != target_crs else gdf for gdf in all_gdfs]
    merged = gpd.GeoDataFrame(pd.concat(reprojected, ignore_index=True),
                               crs=target_crs)
    merged["iceberg_id"] = range(1, len(merged) + 1)

    icebergs = merged[merged["class_name"] == "iceberg"]
    print(f"\n{'-'*55}")
    print(f"Total iceberg polygons : {len(icebergs)}")
    if len(icebergs) > 0:
        print(f"  min  area = {icebergs['area_m2'].min():.1f} m2")
        print(f"  mean area = {icebergs['area_m2'].mean():.1f} m2")
        print(f"  max  area = {icebergs['area_m2'].max():.1f} m2")
        print(f"  Otsu thresh range : [{merged['otsu_thresh'].min():.4f}, "
              f"{merged['otsu_thresh'].max():.4f}]")
    if n_flat:    print(f"Flat chips skipped  : {n_flat}")
    if n_ic:      print(f"IC chips skipped    : {n_ic}")
    if n_skipped: print(f"Too few bands       : {n_skipped}")
    print(f"{'-'*55}")

    gpkg_path = os.path.join(out_dir, "all_icebergs.gpkg")
    merged.to_file(gpkg_path, driver="GPKG")

    print(f"\nSaved GeoPackage : {gpkg_path}")
    print(f"Method config    : {cfg_path}")
    print(f"Skipped chips    : {skip_path}")
    print(f"Saved PNGs       : {png_dir}/  ({len(tif_files) - n_skipped - n_flat - n_ic} files)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply per-chip Otsu threshold on B08 to S2 chip .tifs"
    )
    parser.add_argument("--chips_dir",       required=True,
                        help="Directory of .tif chip files")
    parser.add_argument("--out_dir",         required=True,
                        help="Output directory (gpkg + pngs/ subfolder)")
    parser.add_argument("--b08_idx",         type=int,   default=2,
                        help="0-indexed band position of B08 (default: 2 for B04/B03/B08)")
    parser.add_argument("--min_area",        type=float, default=MIN_AREA_M2,
                        help=f"Min iceberg area in m^2 (default: {MIN_AREA_M2})")
    parser.add_argument("--ic_threshold",    type=float, default=IC_THRESHOLD,
                        help=f"IC block filter fraction (default: {IC_THRESHOLD})")
    parser.add_argument("--min_otsu_thresh", type=float, default=MIN_OTSU_THRESH,
                        help=f"Skip chip if Otsu threshold < this (default: {MIN_OTSU_THRESH})")
    args = parser.parse_args()

    apply_otsu(
        args.chips_dir, args.out_dir, args.b08_idx,
        args.min_area, args.ic_threshold, args.min_otsu_thresh,
    )


if __name__ == "__main__":
    main()
```

</details>

<details>
<summary><strong>tophat_recover.py</strong> — TH, white top-hat post-processor (293 lines)</summary>

```python
"""
tophat_recover.py: small-iceberg recovery via white top-hat on B08.

For each input chip in --chips_dir, reads the B08 band, applies a white
top-hat morphological filter (disk structuring element of radius
--se_radius pixels), thresholds the response at --th_thresh, drops
connected components below --min_area_px, subtracts pixels already
covered by the base method's per-chip prediction, and writes a per-chip
gpkg containing the union of base polygons + recovered polygons.

Outputs (per --out_dir):
  gpkgs/<chip_stem>_icebergs.gpkg   per-chip merged polygons (base + TH)
  all_icebergs.gpkg                 concat across chips
  recovery_stats.csv                per-chip counts of base / TH / total
  method_config.json                provenance: SE radius, threshold, base method id

Usage:
  python scripts/tophat_recover.py \
      --chips_dir   data/v4_clean/test_chips/sza_lt65 \
      --base_dir    runs/exp_baseline_v1/<ts>/inference/sza_lt65/UNet \
      --out_dir     runs/exp_baseline_v1/<ts>/inference/sza_lt65/UNet_TH

Notes:
- Designed as a post-processor; does not need a UNet checkpoint.
- Input chip and prediction tifs must share pixel grid (256x256, 10 m).
- Idempotent at the same parameters: rerunning overwrites outputs.
"""

import argparse
import csv
import json
import os
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.features import rasterize as rio_rasterize, shapes as rio_shapes
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape as shapely_shape
from skimage.morphology import disk, white_tophat
from skimage.measure import label

from _method_common import get_git_sha, sha256_of_file

# 1. Defaults
DEFAULT_SE_RADIUS = 10       # 100 m at 10 m pixels (Fisser cap on small icebergs)
DEFAULT_TH_THRESH = 0.05     # response threshold in reflectance units
DEFAULT_MIN_AREA_PX = 16     # 40 m root length, matches the global cutoff
PIXEL_AREA_M2 = 100.0


def read_b08(tif_path):
    """Return the B08 band (band 3) as a float32 array."""
    with rasterio.open(tif_path) as src:
        if src.count < 3:
            raise ValueError(f"{tif_path}: expected >= 3 bands, got {src.count}")
        b08 = src.read(3).astype(np.float32)
        transform = src.transform
        crs = src.crs
    return b08, transform, crs


def read_base_polygons(base_dir, stem, all_polys_cache=None):
    """
    Load base method polygons for one chip. Three fallbacks in order:
      1. per-chip gpkgs/<stem>_icebergs.gpkg
      2. all_icebergs.gpkg sliced by source_file == <stem>.tif
      3. None
    """
    p = os.path.join(base_dir, "gpkgs", f"{stem}_icebergs.gpkg")
    if os.path.exists(p):
        return gpd.read_file(p)

    if all_polys_cache is None:
        all_p = os.path.join(base_dir, "all_icebergs.gpkg")
        if not os.path.exists(all_p):
            return None
        all_polys_cache = gpd.read_file(all_p)

    src_name = f"{stem}.tif"
    sub = all_polys_cache[all_polys_cache.get("source_file") == src_name]
    return sub if len(sub) > 0 else None


def read_base_mask(base_dir, stem, ref_shape, ref_transform, all_polys_cache=None):
    """
    Build a binary mask of the base method's polygon footprint on the chip's
    pixel grid. Prefers <stem>_pred.tif under geotiffs/, falls back to
    rasterising the polygons read by read_base_polygons.
    Returns a uint8 array shaped like ref_shape, or None when no base
    information exists for this chip.
    """
    # 1. Direct geotiff path (UNet only writes this; others do not)
    pred_tif = os.path.join(base_dir, "geotiffs", f"{stem}_pred.tif")
    if os.path.exists(pred_tif):
        with rasterio.open(pred_tif) as src:
            return (src.read(1) > 0).astype(np.uint8)

    # 2. Rasterise polygons (per-chip gpkg or all_icebergs slice)
    polys = read_base_polygons(base_dir, stem, all_polys_cache=all_polys_cache)
    if polys is None or len(polys) == 0:
        return None
    geoms = [(g, 1) for g in polys.geometry if g is not None and not g.is_empty]
    if not geoms:
        return None
    return rio_rasterize(
        geoms,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0,
        dtype="uint8",
    )


def recover_tophat(b08, base_mask, se_radius, th_thresh, min_area_px):
    """
    Run the white top-hat recovery step.
    Returns a binary mask of NEW iceberg pixels (not already in base_mask).
    """
    # 1. White top-hat highlights bright spots smaller than the SE
    se = disk(se_radius)
    response = white_tophat(b08, se)

    # 2. Threshold + drop pixels already covered by the base method
    candidate = (response >= th_thresh).astype(np.uint8)
    if base_mask is not None:
        candidate &= (base_mask == 0).astype(np.uint8)

    # 3. Filter connected components below the size cutoff
    labels, n = label(candidate, connectivity=2, return_num=True)
    if n == 0:
        return np.zeros_like(candidate)
    sizes = np.bincount(labels.ravel())
    keep = np.zeros_like(sizes, dtype=bool)
    keep[1:] = sizes[1:] >= min_area_px
    return keep[labels].astype(np.uint8)


def mask_to_polygons(mask, transform, source_file):
    """Vectorise a binary mask into shapely polygons; returns list of records."""
    if mask.sum() == 0:
        return []
    rows = []
    iceberg_id = 1
    for geom, val in rio_shapes(mask, mask=mask.astype(bool), transform=transform):
        if val == 0:
            continue
        poly = shapely_shape(geom)
        rows.append({
            "class_id":    1,
            "class_name":  "iceberg",
            "area_m2":     poly.area,
            "source_file": source_file,
            "iceberg_id":  iceberg_id,
            "geometry":    poly,
        })
        iceberg_id += 1
    return rows


def main():
    parser = argparse.ArgumentParser(description="White top-hat small-iceberg recovery on a base method's outputs")
    parser.add_argument("--chips_dir", required=True, help="dir of input *.tif (B04/B03/B08)")
    parser.add_argument("--base_dir",  required=True, help="dir of base method outputs (must have geotiffs/ + gpkgs/)")
    parser.add_argument("--out_dir",   required=True, help="dir to write the recovered method outputs")
    parser.add_argument("--base_method_id", default="UNet", help="label for the base method (provenance only)")
    parser.add_argument("--se_radius", type=int, default=DEFAULT_SE_RADIUS,
                        help="disk SE radius in pixels (default 10 = 100 m)")
    parser.add_argument("--th_thresh", type=float, default=DEFAULT_TH_THRESH,
                        help="top-hat response threshold in reflectance units")
    parser.add_argument("--min_area_px", type=int, default=DEFAULT_MIN_AREA_PX,
                        help="drop recovered components below this pixel area")
    args = parser.parse_args()

    # 1. Stage output directories
    os.makedirs(os.path.join(args.out_dir, "gpkgs"), exist_ok=True)

    # 2. Iterate chips
    chip_paths = sorted(
        os.path.join(args.chips_dir, f)
        for f in os.listdir(args.chips_dir) if f.endswith(".tif")
    )
    if not chip_paths:
        raise SystemExit(f"no tifs found under {args.chips_dir}")

    stats_rows = []
    all_polys = []

    # Cache the cross-chip all_icebergs.gpkg once: TR/OT only emit this file,
    # so re-reading it per chip would dominate runtime.
    all_polys_cache = None
    all_icebergs_path = os.path.join(args.base_dir, "all_icebergs.gpkg")
    if os.path.exists(all_icebergs_path):
        all_polys_cache = gpd.read_file(all_icebergs_path)

    for chip_path in chip_paths:
        stem = os.path.splitext(os.path.basename(chip_path))[0]
        b08, transform, _crs = read_b08(chip_path)
        base_mask = read_base_mask(
            args.base_dir, stem, b08.shape, transform, all_polys_cache,
        )

        # 3. Run recovery on this chip
        recovered = recover_tophat(
            b08=b08,
            base_mask=base_mask,
            se_radius=args.se_radius,
            th_thresh=args.th_thresh,
            min_area_px=args.min_area_px,
        )

        recovered_polys = mask_to_polygons(recovered, transform,
                                            source_file=os.path.basename(chip_path))

        # 4. Merge with base polygons
        base_gdf = read_base_polygons(args.base_dir, stem, all_polys_cache)
        n_base = len(base_gdf) if base_gdf is not None else 0
        n_recov = len(recovered_polys)

        if base_gdf is not None and n_base > 0:
            recov_gdf = gpd.GeoDataFrame(recovered_polys, crs=base_gdf.crs) if recovered_polys else None
            merged = pd.concat([base_gdf, recov_gdf], ignore_index=True) if recov_gdf is not None else base_gdf
        elif recovered_polys:
            merged = gpd.GeoDataFrame(recovered_polys)
        else:
            merged = gpd.GeoDataFrame(columns=["class_id", "class_name", "area_m2",
                                                "source_file", "iceberg_id", "geometry"])

        # Re-id sequentially after the merge
        if len(merged) > 0:
            merged["iceberg_id"] = list(range(1, len(merged) + 1))

        out_gpkg = os.path.join(args.out_dir, "gpkgs", f"{stem}_icebergs.gpkg")
        if len(merged) > 0:
            merged.to_file(out_gpkg, driver="GPKG")
        all_polys.append(merged)

        stats_rows.append({
            "chip_stem":        stem,
            "n_base_polygons":  n_base,
            "n_recovered":      n_recov,
            "n_total":          n_base + n_recov,
        })

    # 5. Concatenated gpkg + stats CSV
    # Fisser synthetic chips have CRS = None, real Roboflow chips span UTM 24N
    # and 25N, so a single CRS for the cross-chip concat does not exist. The
    # eval script reads per-chip gpkgs, so the combined file is informational
    # only; warn but continue if the concat fails.
    try:
        combined = pd.concat(all_polys, ignore_index=True) if all_polys else pd.DataFrame()
        if len(combined) > 0:
            combined.to_file(os.path.join(args.out_dir, "all_icebergs.gpkg"), driver="GPKG")
    except Exception as exc:
        print(f"WARN: cross-chip all_icebergs.gpkg skipped ({exc.__class__.__name__}: {exc})")

    stats_path = os.path.join(args.out_dir, "recovery_stats.csv")
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["chip_stem", "n_base_polygons", "n_recovered", "n_total"]
        )
        writer.writeheader()
        writer.writerows(stats_rows)

    # 6. Method config (provenance)
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = {
        "method":           f"{args.base_method_id}_TH",
        "base_method":      args.base_method_id,
        "base_dir":         os.path.abspath(args.base_dir),
        "chips_dir":        os.path.abspath(args.chips_dir),
        "se_radius":        args.se_radius,
        "th_thresh":        args.th_thresh,
        "min_area_px":      args.min_area_px,
        "n_chips_processed": len(chip_paths),
        "n_polygons_base":  int(sum(r["n_base_polygons"] for r in stats_rows)),
        "n_polygons_recov": int(sum(r["n_recovered"] for r in stats_rows)),
        "n_polygons_total": int(sum(r["n_total"] for r in stats_rows)),
        "git_sha":          get_git_sha(repo_dir),
        "created_utc":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(args.out_dir, "method_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"top-hat recovery written to {args.out_dir}/")
    print(f"  base polygons:      {cfg['n_polygons_base']}")
    print(f"  recovered polygons: {cfg['n_polygons_recov']}")
    print(f"  total:              {cfg['n_polygons_total']}")


if __name__ == "__main__":
    main()
```

</details>

<details>
<summary><strong>densecrf_tifs.py</strong> — UNet+CRF, DenseCRF on UNet++ softmax (179 lines)</summary>

```python
"""
densecrf_tifs.py: UNet + CRF method.

Applies DenseCRF post-processing to UNet++ softmax probabilities.
Uses the chip image for the bilateral pairwise term (boundary-preserving smoothing).
Core CRF logic reused from partner's crf_utils.py, do not duplicate.

Inputs:
  - *_probs.tif  : 2-band float32 softmax probs from predict_tifs.py (ocean, iceberg)
  - *tif chips   : original chips (for bilateral term)
Output:
  all_icebergs.gpkg

Requires crf_utils.py in the same directory (copied from partner's sandbox).

Usage:
  python densecrf_tifs.py \\
      --probs_dir area_comparison/KQ/sza_70_75/UNet/probs \\
      --chips_dir /mnt/research/.../chips/KQ/sza_70_75/tifs \\
      --out_dir   area_comparison/KQ/sza_70_75/UNet_CRF

rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/densecrf_tifs.py smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import os
import sys
import argparse
import warnings
from glob import glob

import numpy as np
import rasterio as rio
from rasterio.features import shapes as rio_shapes
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd

# crf_utils.py must be in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crf_utils import apply_densecrf
from _method_common import (
    write_method_config, write_skipped_chips, SKIP_CHIP_TIF_MISSING,
)

warnings.filterwarnings("ignore", category=UserWarning)

MIN_AREA_M2 = 100.0

# Default CRF params from partner's sandbox run_001 (best of 2 tested)
DEFAULT_PARAMS = {
    "sxy_gaussian":    3,
    "compat_gaussian": 3,
    "sxy_bilateral":   40,
    "srgb_bilateral":  3,
    "compat_bilateral":4,
    "iterations":      5,
}


def find_chip(chips_dir, stem):
    """Find chip .tif matching a prob stem (strip _probs suffix if present)."""
    chip_stem = stem.replace("_probs", "")
    path = os.path.join(chips_dir, f"{chip_stem}.tif")
    if os.path.exists(path):
        return path
    # fallback: glob
    matches = glob(os.path.join(chips_dir, f"{chip_stem}*.tif"))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(
        description="UNet+CRF: DenseCRF post-processing on UNet++ softmax probs"
    )
    parser.add_argument("--probs_dir",    required=True,
        help="Directory of *_probs.tif from predict_tifs.py")
    parser.add_argument("--chips_dir",    required=True,
        help="Directory of original chip .tifs (for CRF bilateral term)")
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--min_area_m2",  type=float, default=MIN_AREA_M2)
    # CRF params (override defaults if needed)
    parser.add_argument("--sxy_gaussian",    type=float, default=DEFAULT_PARAMS["sxy_gaussian"])
    parser.add_argument("--compat_gaussian", type=float, default=DEFAULT_PARAMS["compat_gaussian"])
    parser.add_argument("--sxy_bilateral",   type=float, default=DEFAULT_PARAMS["sxy_bilateral"])
    parser.add_argument("--srgb_bilateral",  type=float, default=DEFAULT_PARAMS["srgb_bilateral"])
    parser.add_argument("--compat_bilateral",type=float, default=DEFAULT_PARAMS["compat_bilateral"])
    parser.add_argument("--iterations",      type=int,   default=DEFAULT_PARAMS["iterations"])
    args = parser.parse_args()

    params = {
        "sxy_gaussian":     args.sxy_gaussian,
        "compat_gaussian":  args.compat_gaussian,
        "sxy_bilateral":    args.sxy_bilateral,
        "srgb_bilateral":   args.srgb_bilateral,
        "compat_bilateral": args.compat_bilateral,
        "iterations":       args.iterations,
    }

    gpkg_dir = os.path.join(args.out_dir, "gpkgs")
    os.makedirs(gpkg_dir, exist_ok=True)

    prob_files = sorted(glob(os.path.join(args.probs_dir, "*_probs.tif")))
    print(f"Found {len(prob_files)} prob .tifs")
    print(f"CRF params: {params}\n")

    all_gdfs = []
    skipped  = []

    for i, prob_path in enumerate(prob_files):
        stem = os.path.basename(prob_path).replace("_probs.tif", "")

        chip_path = find_chip(args.chips_dir, stem)
        if chip_path is None:
            print(f"  [{i+1:>4}/{len(prob_files)}] NO CHIP  {stem[:60]}")
            skipped.append({"chip_stem": stem, "reason": SKIP_CHIP_TIF_MISSING})
            continue

        with rio.open(prob_path) as src:
            probs = src.read().astype(np.float32)   # (3, H, W)
            meta  = src.meta.copy()

        with rio.open(chip_path) as src:
            chip = src.read().astype(np.float32)    # (3, H, W)

        refined = apply_densecrf(probs, chip, params)  # (H, W) uint8

        mask = (refined == 1).astype(np.uint8)   # iceberg class

        records = []
        for geom_dict, val in rio_shapes(mask, transform=meta["transform"]):
            if val == 0:
                continue
            geom = shape(geom_dict)
            if geom.is_empty or geom.area < args.min_area_m2:
                continue
            records.append({"geometry": geom, "area_m2": geom.area,
                            "source_file": stem})

        n = len(records)
        print(f"  [{i+1:>4}/{len(prob_files)}] {stem[:60]}  icebergs={n}")

        if records:
            gdf = gpd.GeoDataFrame(records, crs=meta["crs"])
            gdf.to_file(os.path.join(gpkg_dir, f"{stem}_icebergs.gpkg"), driver="GPKG")
            all_gdfs.append(gdf)

    cfg_path = write_method_config(
        args.out_dir, "UNet_CRF",
        params={
            "probs_dir":   os.path.abspath(args.probs_dir),
            "chips_dir":   os.path.abspath(args.chips_dir),
            "min_area_m2": args.min_area_m2,
            "crf":         params,
        },
    )
    skip_path = write_skipped_chips(args.out_dir, skipped)

    if all_gdfs:
        target_crs = all_gdfs[0].crs
        reprojected = [gdf.to_crs(target_crs) if gdf.crs != target_crs else gdf for gdf in all_gdfs]
        merged = gpd.GeoDataFrame(pd.concat(reprojected, ignore_index=True),
                                  crs=target_crs)
        merged["area_m2"] = merged["area_m2"].round(2)
        out = os.path.join(args.out_dir, "all_icebergs.gpkg")
        merged.to_file(out, driver="GPKG")
        print(f"\nTotal icebergs  : {len(merged)}")
        print(f"Saved           : {out}")
    else:
        print("\nNo icebergs detected.")

    n_no_chip = sum(1 for r in skipped if r["reason"] == SKIP_CHIP_TIF_MISSING)
    if n_no_chip:
        print(f"Skipped (no chip): {n_no_chip}")
    print(f"Method config    : {cfg_path}")
    print(f"Skipped chips    : {skip_path}")


if __name__ == "__main__":
    main()
```

</details>

<details>
<summary><strong>crf_utils.py</strong> — DenseCRF wrapper (238 lines)</summary>

```python
"""
crf_utils.py: DenseCRF helpers used by densecrf_tifs.py and the earlier
phase1/phase2 sandbox sweeps.

apply_densecrf() is the only entry point used in the production pipeline.
It takes a (C, H, W) softmax probability stack plus the chip's (C, H, W)
reflectance image and returns a (H, W) hard label map after running
DenseCRF mean-field inference with a gaussian (location-only) and a
bilateral (location + reflectance) pairwise term.

The remaining helpers in this file are sandbox scaffolding: synthetic
probability generation from labels, IoU and area-bias diagnostics, and a
small CLI param-grid builder used by the offline tuning sweep. They are
left here so the file is self-contained and the sandbox scripts in
_archive/old-experiments/test-crf still import from one location.
"""

import argparse
import itertools
import json
import os
import pickle
from typing import Dict, List, Sequence

import numpy as np


DEFAULT_PARAM_GRID = {
    "sxy_gaussian": [3],
    "compat_gaussian": [3],
    "sxy_bilateral": [40, 80],
    "srgb_bilateral": [3, 5],
    "compat_bilateral": [4],
    "iterations": [5, 10],
}

DEFAULT_NOAUG_CHECKPOINT = (
    "/mnt/research/v.gomezgilyaspik/students/smishra/"
    "S2-iceberg-areas/runs/s2_v2_noaug/best_model.pth"
)

DEFAULT_V2_NORM_SOURCE = (
    "/mnt/research/v.gomezgilyaspik/students/smishra/"
    "S2-iceberg-areas/S2UnetPlusPlus/train_validate_test_v2/X_train.pkl"
)


def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def default_parent_crf_dir() -> str:
    # The scripts live in test-crf/, and the original pickles sit one level up.
    return os.path.abspath(os.path.join(script_dir(), ".."))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_pickle(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_pickle(path: str, obj) -> None:
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def squeeze_mask_channel(labels: np.ndarray) -> np.ndarray:
    # The training data stores masks as (N,1,H,W), but the CRF code wants (N,H,W).
    labels = np.asarray(labels)
    if labels.ndim == 4 and labels.shape[1] == 1:
        return labels[:, 0]
    return labels


def one_hot_from_labels(labels: np.ndarray, n_classes: int) -> np.ndarray:
    # This is handy for debugging because it makes perfectly confident probabilities from the labels.
    flat = np.eye(n_classes, dtype=np.float32)[labels]
    return np.moveaxis(flat, -1, 1)


def synthetic_probs_from_labels(
    labels: np.ndarray,
    n_classes: int = 3,
    true_prob: float = 0.90,
) -> np.ndarray:
    # This gives us "realistic enough" softmax maps for testing the CRF pipeline without a checkpoint.
    if not (0.0 < true_prob < 1.0):
        raise ValueError("true_prob must be between 0 and 1")
    off_prob = (1.0 - true_prob) / (n_classes - 1)
    probs = np.full((labels.shape[0], n_classes, labels.shape[1], labels.shape[2]), off_prob, dtype=np.float32)
    for cls in range(n_classes):
        probs[:, cls][labels == cls] = true_prob
    return probs


def uniform_probs(n_samples: int, height: int, width: int, n_classes: int = 3) -> np.ndarray:
    return np.full((n_samples, n_classes, height, width), 1.0 / n_classes, dtype=np.float32)


def compute_channel_stats(chips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # This is optional and only used if we ever need to reproduce a normalized training run.
    if chips.ndim != 4:
        raise ValueError(f"Expected chips shape (N,C,H,W), got {chips.shape}")
    mean = chips.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = chips.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def normalize_chips(chips: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(1, -1, 1, 1)
    return ((chips.astype(np.float32) - mean) / std).astype(np.float32)


def compute_iou(pred: np.ndarray, target: np.ndarray, classes: Sequence[int] = (1,)) -> float:
    # We ignore the ocean class here and focus on the classes we care about comparing.
    scores: List[float] = []
    for cls in classes:
        pred_mask = pred == cls
        target_mask = target == cls
        union = np.logical_or(pred_mask, target_mask).sum()
        if union == 0:
            continue
        inter = np.logical_and(pred_mask, target_mask).sum()
        scores.append(float(inter) / float(union))
    return float(np.mean(scores)) if scores else 0.0


def compute_class_area_bias(pred: np.ndarray, target: np.ndarray, cls: int = 1) -> float:
    # This tells us whether CRF changes total iceberg area, not just boundary overlap.
    pred_area = float((pred == cls).sum())
    target_area = float((target == cls).sum())
    if target_area == 0:
        return 0.0 if pred_area == 0 else float("inf")
    return 100.0 * (pred_area - target_area) / target_area


def scale_chip_to_uint8(chip: np.ndarray) -> np.ndarray:
    # The bilateral term expects image-like values, so we stretch each band into uint8 just for CRF.
    if chip.ndim != 3:
        raise ValueError(f"Expected chip shape (C,H,W), got {chip.shape}")
    channels: List[np.ndarray] = []
    for band in chip:
        lo = float(np.nanpercentile(band, 2))
        hi = float(np.nanpercentile(band, 98))
        scaled = np.clip((band - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        channels.append((scaled * 255.0).astype(np.uint8))
    return np.stack(channels, axis=-1)


def try_import_densecrf():
    # Some systems have the original package, others have the maintained fork.
    try:
        import pydensecrf.densecrf as dcrf  # type: ignore
        from pydensecrf.utils import unary_from_softmax  # type: ignore
        return dcrf, unary_from_softmax
    except ImportError:
        try:
            import pydensecrf2.densecrf as dcrf  # type: ignore
            from pydensecrf2.utils import unary_from_softmax  # type: ignore
            return dcrf, unary_from_softmax
        except ImportError as exc:
            raise ImportError(
                "DenseCRF dependency not found. Install `pydensecrf` or `pydensecrf2` to run Phase 2."
            ) from exc


def apply_densecrf(prob: np.ndarray, chip: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    dcrf, unary_from_softmax = try_import_densecrf()
    n_classes, height, width = prob.shape

    # The chip only affects the pairwise appearance term; the class probabilities drive the unary term.
    image_uint8 = scale_chip_to_uint8(chip)
    unary = unary_from_softmax(np.ascontiguousarray(prob))

    dense = dcrf.DenseCRF2D(width, height, n_classes)
    dense.setUnaryEnergy(unary)
    # This term only cares about nearby pixels agreeing in space.
    dense.addPairwiseGaussian(
        sxy=float(params["sxy_gaussian"]),
        compat=float(params["compat_gaussian"]),
    )
    # This term also looks at band values, so sharp image changes can preserve boundaries.
    dense.addPairwiseBilateral(
        sxy=float(params["sxy_bilateral"]),
        srgb=float(params["srgb_bilateral"]),
        rgbim=np.ascontiguousarray(image_uint8),
        compat=float(params["compat_bilateral"]),
    )

    # After a few mean-field updates, we convert the refined distribution back to hard labels.
    q = dense.inference(int(params["iterations"]))
    q = np.asarray(q, dtype=np.float32).reshape((n_classes, height, width))
    return np.argmax(q, axis=0).astype(np.uint8)


def parse_int_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def build_param_grid(args: argparse.Namespace) -> List[Dict[str, int]]:
    if getattr(args, "param_json", None):
        with open(args.param_json, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            grid = loaded
        elif isinstance(loaded, list):
            return loaded
        else:
            raise ValueError("param_json must contain either a dict grid or a list of param sets")
    else:
        # The CLI takes comma-separated values so a tiny grid search is easy to launch from the shell.
        grid = {
            "sxy_gaussian": parse_int_list(args.sxy_gaussian),
            "compat_gaussian": parse_int_list(args.compat_gaussian),
            "sxy_bilateral": parse_int_list(args.sxy_bilateral),
            "srgb_bilateral": parse_int_list(args.srgb_bilateral),
            "compat_bilateral": parse_int_list(args.compat_bilateral),
            "iterations": parse_int_list(args.iterations),
        }

    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    param_sets = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    max_param_sets = getattr(args, "max_param_sets", None)
    if max_param_sets:
        param_sets = param_sets[:max_param_sets]
    return param_sets


def summarize_array(name: str, arr: np.ndarray) -> str:
    return f"{name}: shape={arr.shape}, dtype={arr.dtype}"
```

</details>

<details>
<summary><strong>threshold_probs.py</strong> — UNet+TR, threshold on UNet++ softmax (125 lines)</summary>

```python
"""
threshold_probs.py: UNet + TR method.

Applies a fixed NIR threshold to the UNet++ iceberg probability band.
Instead of argmax, labels a pixel as iceberg if P(iceberg) >= threshold.

Input:  softmax prob .tifs from predict_tifs.py
        (2-band float32 GeoTIFF: band 1=ocean, band 2=iceberg)
Output: all_icebergs.gpkg  (same format as threshold_tifs.py)

Usage:
  python threshold_probs.py \\
      --probs_dir area_comparison/KQ/sza_70_75/UNet/probs \\
      --out_dir   area_comparison/KQ/sza_70_75/UNet_TR

rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/threshold_probs.py smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import os
import argparse
import warnings
from glob import glob

import numpy as np
import rasterio as rio
from rasterio.features import shapes as rio_shapes
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd

from _method_common import (
    write_method_config, write_skipped_chips, SKIP_TOO_FEW_PROB_BANDS,
)

warnings.filterwarnings("ignore", category=UserWarning)

ICEBERG_BAND = 1   # 0-indexed band in the prob .tif (ocean=0, iceberg=1)
THRESHOLD    = 0.22  # matches the Fisser threshold used in threshold_tifs.py
MIN_AREA_M2  = 100.0


def main():
    parser = argparse.ArgumentParser(
        description="UNet+TR: threshold UNet++ iceberg probability band"
    )
    parser.add_argument("--probs_dir",   required=True,
        help="Directory of *_probs.tif files from predict_tifs.py")
    parser.add_argument("--out_dir",     required=True,
        help="Output directory")
    parser.add_argument("--threshold",   type=float, default=THRESHOLD,
        help=f"P(iceberg) threshold (default {THRESHOLD})")
    parser.add_argument("--min_area_m2", type=float, default=MIN_AREA_M2)
    args = parser.parse_args()

    gpkg_dir = os.path.join(args.out_dir, "gpkgs")
    os.makedirs(gpkg_dir, exist_ok=True)

    prob_files = sorted(glob(os.path.join(args.probs_dir, "*_probs.tif")))
    print(f"Found {len(prob_files)} prob .tifs  |  threshold={args.threshold}")

    all_gdfs = []
    skipped  = []

    for i, prob_path in enumerate(prob_files):
        stem = os.path.basename(prob_path).replace("_probs.tif", "")

        with rio.open(prob_path) as src:
            probs = src.read().astype(np.float32)   # (C, H, W)
            meta  = src.meta.copy()

        if probs.shape[0] <= ICEBERG_BAND:
            skipped.append({"chip_stem": stem, "reason": SKIP_TOO_FEW_PROB_BANDS,
                            "n_bands": probs.shape[0]})
            continue

        iceberg_prob = probs[ICEBERG_BAND]
        mask = (iceberg_prob >= args.threshold).astype(np.uint8)

        records = []
        for geom_dict, val in rio_shapes(mask, transform=meta["transform"]):
            if val == 0:
                continue
            geom = shape(geom_dict)
            if geom.is_empty or geom.area < args.min_area_m2:
                continue
            records.append({"geometry": geom, "area_m2": geom.area,
                            "source_file": stem})

        n = len(records)
        print(f"  [{i+1:>4}/{len(prob_files)}] {stem[:60]}  icebergs={n}")

        if records:
            gdf = gpd.GeoDataFrame(records, crs=meta["crs"])
            gdf.to_file(os.path.join(gpkg_dir, f"{stem}_icebergs.gpkg"), driver="GPKG")
            all_gdfs.append(gdf)

    cfg_path = write_method_config(
        args.out_dir, "UNet_TR",
        params={
            "probs_dir":    os.path.abspath(args.probs_dir),
            "iceberg_band": ICEBERG_BAND,
            "threshold":    args.threshold,
            "min_area_m2":  args.min_area_m2,
        },
    )
    skip_path = write_skipped_chips(args.out_dir, skipped)

    if all_gdfs:
        target_crs = all_gdfs[0].crs
        reprojected = [gdf.to_crs(target_crs) if gdf.crs != target_crs else gdf for gdf in all_gdfs]
        merged = gpd.GeoDataFrame(pd.concat(reprojected, ignore_index=True),
                                  crs=target_crs)
        merged["area_m2"] = merged["area_m2"].round(2)
        out = os.path.join(args.out_dir, "all_icebergs.gpkg")
        merged.to_file(out, driver="GPKG")
        print(f"\nTotal icebergs : {len(merged)}")
        print(f"Saved          : {out}")
    else:
        print("\nNo icebergs detected.")
    print(f"Method config  : {cfg_path}")
    print(f"Skipped chips  : {skip_path}")


if __name__ == "__main__":
    main()
```

</details>

<details>
<summary><strong>otsu_probs.py</strong> — UNet+OT, Otsu on UNet++ softmax (145 lines)</summary>

```python
"""
otsu_probs.py: UNet + OT method.

Applies per-chip Otsu thresholding to the UNet++ iceberg probability band.
Finds the threshold that best separates the P(iceberg) histogram into
ocean-like vs iceberg-like regions within each chip.

Input:  softmax prob .tifs from predict_tifs.py
        (2-band float32 GeoTIFF: band 1=ocean, band 2=iceberg)
Output: all_icebergs.gpkg  (same format as otsu_threshold_tifs.py)

Usage:
  python otsu_probs.py \\
      --probs_dir area_comparison/KQ/sza_70_75/UNet/probs \\
      --out_dir   area_comparison/KQ/sza_70_75/UNet_OT

rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/otsu_probs.py smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import os
import argparse
import warnings
from glob import glob

import numpy as np
import rasterio as rio
from rasterio.features import shapes as rio_shapes
from skimage.filters import threshold_otsu
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd

from _method_common import (
    write_method_config, write_skipped_chips,
    SKIP_TOO_FEW_PROB_BANDS, SKIP_FLAT_PROB, SKIP_IC_BLOCK_FILTER,
)

warnings.filterwarnings("ignore", category=UserWarning)

ICEBERG_BAND  = 1      # 0-indexed: ocean=0, iceberg=1
MIN_AREA_M2   = 100.0
IC_THRESHOLD  = 0.15   # skip chip if >15% pixels exceed Otsu (sea-ice filter)


def main():
    parser = argparse.ArgumentParser(
        description="UNet+OT: Otsu threshold on UNet++ iceberg probability band"
    )
    parser.add_argument("--probs_dir",    required=True,
        help="Directory of *_probs.tif files from predict_tifs.py")
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--min_area_m2",  type=float, default=MIN_AREA_M2)
    parser.add_argument("--ic_threshold", type=float, default=IC_THRESHOLD,
        help="Skip chip if >this fraction exceeds Otsu (sea-ice filter)")
    args = parser.parse_args()

    gpkg_dir = os.path.join(args.out_dir, "gpkgs")
    os.makedirs(gpkg_dir, exist_ok=True)

    prob_files = sorted(glob(os.path.join(args.probs_dir, "*_probs.tif")))
    print(f"Found {len(prob_files)} prob .tifs")

    all_gdfs = []
    skipped  = []

    for i, prob_path in enumerate(prob_files):
        stem = os.path.basename(prob_path).replace("_probs.tif", "")

        with rio.open(prob_path) as src:
            probs = src.read().astype(np.float32)
            meta  = src.meta.copy()

        if probs.shape[0] <= ICEBERG_BAND:
            skipped.append({"chip_stem": stem, "reason": SKIP_TOO_FEW_PROB_BANDS,
                            "n_bands": probs.shape[0]})
            continue

        iceberg_prob = np.nan_to_num(probs[ICEBERG_BAND], nan=0.0)
        flat = iceberg_prob.flatten()

        # Otsu needs a bimodal distribution, so skip chips with flat prob
        if flat.max() - flat.min() < 0.01:
            print(f"  [{i+1:>4}/{len(prob_files)}] SKIP (flat prob)  {stem[:60]}")
            skipped.append({"chip_stem": stem, "reason": SKIP_FLAT_PROB,
                            "prob_range": f"{float(flat.max() - flat.min()):.4f}"})
            continue

        thresh  = float(threshold_otsu(flat))
        ic_frac = float((iceberg_prob >= thresh).mean())
        if ic_frac > args.ic_threshold:
            print(f"  [{i+1:>4}/{len(prob_files)}] SKIP (IC-filtered ic_frac={ic_frac:.2f})  {stem[:50]}")
            skipped.append({"chip_stem": stem, "reason": SKIP_IC_BLOCK_FILTER,
                            "otsu_thresh": f"{thresh:.4f}",
                            "ic_frac":     f"{ic_frac:.4f}"})
            continue

        mask = (iceberg_prob >= thresh).astype(np.uint8)

        records = []
        for geom_dict, val in rio_shapes(mask, transform=meta["transform"]):
            if val == 0:
                continue
            geom = shape(geom_dict)
            if geom.is_empty or geom.area < args.min_area_m2:
                continue
            records.append({"geometry": geom, "area_m2": geom.area,
                            "source_file": stem, "otsu_thresh": round(thresh, 4)})

        n = len(records)
        print(f"  [{i+1:>4}/{len(prob_files)}] {stem[:55]}  thr={thresh:.3f}  icebergs={n}")

        if records:
            gdf = gpd.GeoDataFrame(records, crs=meta["crs"])
            gdf.to_file(os.path.join(gpkg_dir, f"{stem}_icebergs.gpkg"), driver="GPKG")
            all_gdfs.append(gdf)

    cfg_path = write_method_config(
        args.out_dir, "UNet_OT",
        params={
            "probs_dir":    os.path.abspath(args.probs_dir),
            "iceberg_band": ICEBERG_BAND,
            "min_area_m2":  args.min_area_m2,
            "ic_threshold": args.ic_threshold,
        },
    )
    skip_path = write_skipped_chips(args.out_dir, skipped)

    if all_gdfs:
        target_crs = all_gdfs[0].crs
        reprojected = [gdf.to_crs(target_crs) if gdf.crs != target_crs else gdf for gdf in all_gdfs]
        merged = gpd.GeoDataFrame(pd.concat(reprojected, ignore_index=True),
                                  crs=target_crs)
        merged["area_m2"] = merged["area_m2"].round(2)
        out = os.path.join(args.out_dir, "all_icebergs.gpkg")
        merged.to_file(out, driver="GPKG")
        print(f"\nTotal icebergs : {len(merged)}")
        print(f"Saved          : {out}")
    else:
        print("\nNo icebergs detected.")
    print(f"Method config  : {cfg_path}")
    print(f"Skipped chips  : {skip_path}")


if __name__ == "__main__":
    main()
```

</details>
