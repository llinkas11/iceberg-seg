"""
make_figure01_annotation_difficulty.py: render Fig. 1, the annotation-cleaning
visual the paper opens with.

Three rows x four columns. Row (a) is a Fisser low-SZA chip with the original
3-class annotation versus the binary post shadow-merge + 40 m filter mask.
Row (b) is a high-SZA Roboflow chip where the 40 m filter drops sub-16 px
specks. Row (c) is a Roboflow chip whose IC fraction is high enough to
trigger the training-time pixel mask; panel 3 is the chip RGB after zeroing
bright non-annotated pixels (sea ice / cloud), not a separate mask layer.

Two modes:

  --list_candidates    print 5 candidate chip_stems per scenario from
                       split_log.csv and quit. Use this first; pick the
                       final three; rerun without the flag.

  default              render the 3x4 figure for the user-supplied stems and
                       route through _fig_registry.write into
                       <out_dir>/fig-archive/.

Reuses make_false_color (otsu_threshold_tifs.py), polygons_to_mask + find_tif
+ strip_rf_hash + B08_THRESHOLD + IC_THRESHOLD + CHIP_SIZE
(build_clean_dataset.py), and write (_fig_registry.py).
"""

import argparse
import json
import os
import pickle
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import label as cc_label
from skimage.measure import find_contours

from _fig_registry import write as fig_write
from build_clean_dataset import (
    B08_THRESHOLD,
    CHIP_RE,
    CHIP_SIZE,
    IC_THRESHOLD,
    find_tif,
    polygons_to_mask,
    strip_rf_hash,
)
from otsu_threshold_tifs import make_false_color, percentile_stretch


# 1. Constants
FISSER_PKL_OFFSETS = [
    ("Y_train.pkl", "X_train.pkl", 0, 323),
    ("Y_validation.pkl", "X_validation.pkl", 323, 362),
    ("y_test.pkl", "x_test.pkl", 362, 398),
]
FISSER_FILTERED_PKLS = {
    "train": "Y_train.pkl",
    "val":   "Y_validation.pkl",
    "test":  "y_test.pkl",
}
MIN_AREA_PX = 16


def _locate_fisser(stem_idx):
    """Map fisser_NNNN global index to (raw_y_pkl, raw_x_pkl, within_pkl_pos)."""
    for y_name, x_name, lo, hi in FISSER_PKL_OFFSETS:
        if lo <= stem_idx < hi:
            return y_name, x_name, stem_idx - lo
    raise ValueError(f"fisser stem index {stem_idx} out of range [0, 398)")


def _list_candidates(args):
    """Print 5 candidates per scenario from split_log + raw COCO/pkls."""
    log = pd.read_csv(args.split_log)
    print(f"Total chips in split_log: {len(log)}\n")

    # 1a. Fisser low-SZA candidates: shadow-rich, low IC, lots of icebergs
    cand_a = (
        log[(log["source"] == "fisser") &
            (log["sza_bin"] == "sza_lt65") &
            (log["n_icebergs"] >= 3) &
            (log["ic_aware"].astype(float) < 0.10)]
        .sort_values("n_icebergs", ascending=False)
        .head(5)
    )
    print("=== (a) Fisser low-SZA candidates (shadow-rich, low IC) ===")
    if len(cand_a):
        # Cross-check shadow presence
        shadow_counts = []
        for _, r in cand_a.iterrows():
            stem_idx = int(r["chip_stem"].split("_")[-1])
            try:
                y_name, _, pos = _locate_fisser(stem_idx)
                with open(os.path.join(args.raw_fisser, y_name), "rb") as f:
                    Y = pickle.load(f)
                if Y.ndim == 4:
                    Y = Y[:, 0]
                shadow_counts.append(int((Y[pos] == 2).sum()))
            except (FileNotFoundError, ValueError):
                shadow_counts.append(-1)
        cand_a = cand_a.assign(shadow_px=shadow_counts)
    print(cand_a[["chip_stem", "split", "n_icebergs", "ic_aware",
                  "shadow_px" if len(cand_a) else "ic_aware"]].to_string(index=False)
          if len(cand_a) else "  (none)")

    # 1b. Roboflow high-SZA candidates: dense scenes, dropped polygons
    cand_b_rough = (
        log[(log["source"] == "roboflow") &
            (log["sza_bin"].isin(["sza_70_75", "sza_gt75"])) &
            (log["n_icebergs"] >= 5)]
        .head(20)
    )
    print("\n=== (b) Roboflow high-SZA candidates (40 m filter drops >= 2) ===")
    if len(cand_b_rough):
        with open(args.raw_coco) as f:
            raw_coco = json.load(f)
        with open(args.filtered_coco) as f:
            filt_coco = json.load(f)

        def _ann_count_by_image(coco, target_stem):
            ice_cat = next((c["id"] for c in coco["categories"]
                             if c["name"] == "iceberg"), 2)
            for img in coco["images"]:
                fn = strip_rf_hash(img["file_name"])
                m = CHIP_RE.match(os.path.basename(fn))
                if not m:
                    continue
                stem = f"{m.group(1)}_r{int(m.group(2)):04d}_c{int(m.group(3)):04d}"
                if stem != target_stem:
                    continue
                return sum(1 for a in coco["annotations"]
                            if a.get("image_id") == img["id"]
                                and a.get("category_id") == ice_cat)
            return None

        rows = []
        for _, r in cand_b_rough.iterrows():
            n_raw = _ann_count_by_image(raw_coco, r["chip_stem"])
            n_clean = _ann_count_by_image(filt_coco, r["chip_stem"])
            if n_raw is None or n_clean is None:
                continue
            delta = n_raw - n_clean
            if delta >= 2:
                rows.append({
                    "chip_stem":  r["chip_stem"],
                    "sza_bin":    r["sza_bin"],
                    "split":      r["split"],
                    "n_icebergs": r["n_icebergs"],
                    "n_raw":      n_raw,
                    "n_clean":    n_clean,
                    "n_dropped":  delta,
                })
        if rows:
            print(pd.DataFrame(rows).head(5).to_string(index=False))
        else:
            print("  (none with n_dropped >= 2)")
    else:
        print("  (no chips matched the high-SZA + n_icebergs >= 5 prefilter)")

    # 1c. Roboflow ambiguous mixed-background candidates: ic_masked == True
    cand_c = (
        log[(log["source"] == "roboflow") &
            (log["ic_masked"].astype(str).str.lower() == "true") &
            (log["ic_aware"].astype(float) >= 0.18)]
        .sort_values("ic_aware", ascending=False)
        .head(5)
    )
    print("\n=== (c) Roboflow IC-masked candidates (ambiguous mixed-bg) ===")
    if len(cand_c):
        print(cand_c[["chip_stem", "sza_bin", "split", "n_icebergs",
                      "ic_aware"]].to_string(index=False))
    else:
        print("  (none)")


# 2. Per-row data loaders
def _load_fisser_row(chip_stem, split, args):
    """Return (chip_3band, prelim_mask_3class, clean_mask_binary, note_str).

    The ``split`` argument is unused: filter_small_icebergs.py preserves the
    Fisser-original pkl ordering (323 / 39 / 36) when it produces
    ``data/fisser_filtered/``, so the filtered pkl name and within-pkl
    position are the SAME as the raw pkl name and position resolved by
    ``_locate_fisser(stem_idx)``. v4_clean's redrawn split does not affect
    the filtered Fisser pkls, only the v4_clean train/val/test pkls.
    """
    stem_idx = int(chip_stem.split("_")[-1])
    y_name, x_name, pos = _locate_fisser(stem_idx)

    # Raw 3-class mask + raw chip data
    with open(os.path.join(args.raw_fisser, y_name), "rb") as f:
        Y_raw = pickle.load(f)
    if Y_raw.ndim == 4:
        Y_raw = Y_raw[:, 0]
    with open(os.path.join(args.raw_fisser, x_name), "rb") as f:
        X_raw = pickle.load(f)
    chip = X_raw[pos]
    prelim = Y_raw[pos]
    n_shadow = int((prelim == 2).sum())

    # Cleaned binary mask: use the SAME pkl name as the raw load. Indexing
    # the filtered fisser dir by v4_clean's redrawn split was a row-pairing
    # bug because the filtered pkls preserve Fisser-original ordering.
    with open(os.path.join(args.filtered_fisser, y_name), "rb") as f:
        Y_filt = pickle.load(f)
    if Y_filt.ndim == 4:
        Y_filt = Y_filt[:, 0]
    clean = (Y_filt[pos] > 0).astype(np.uint8)

    # Component drop count: post-shadow-merge raw vs cleaned.
    # 4-connectivity here matches scipy.ndimage.label's default in
    # filter_small_icebergs.py; using 8-connectivity would over-merge and
    # mis-count.
    merged = (prelim > 0).astype(np.int32)
    _, n_raw_cc = cc_label(merged)
    _, n_clean_cc = cc_label(clean)
    n_dropped = max(0, n_raw_cc - n_clean_cc)

    # Pair-correctness invariant: every component in the merged raw mask that
    # is >= MIN_AREA_PX must align pixel-for-pixel with the cleaned mask.
    labels_raw, _ = cc_label(merged)
    kept_from_raw = np.zeros_like(merged, dtype=np.uint8)
    for cid in range(1, n_raw_cc + 1):
        comp = (labels_raw == cid)
        if comp.sum() >= MIN_AREA_PX:
            kept_from_raw |= comp.astype(np.uint8)
    n_diff = int((kept_from_raw != clean).sum())
    if n_diff != 0:
        raise RuntimeError(
            f"PAIRING BUG: {chip_stem} raw kept components do not match "
            f"cleaned mask pixel-for-pixel ({n_diff} px differ). "
            f"Check that the filtered pkl name + position are aligned with "
            f"the raw pkl name + position."
        )

    # Dropped-pixel mask: union of components below the 40 m cutoff.
    # The cleaned panel renders kept_from_raw in gold and dropped in red.
    dropped = (merged.astype(np.uint8) & (~clean.astype(bool)).astype(np.uint8))

    note = (
        f"Fisser {split} #{pos}\n"
        f"{n_shadow:,} shadow px merged\n"
        f"{n_dropped} comp(s) <16 px dropped\n"
        f"{n_raw_cc} -> {n_clean_cc} components"
    )
    return chip, prelim, clean, dropped, note


def _resolve_roboflow_image_id(coco, target_stem):
    """Return (image_id, image_meta) for chip_stem in a COCO json, or (None, None)."""
    for img in coco["images"]:
        fn = strip_rf_hash(img["file_name"])
        m = CHIP_RE.match(os.path.basename(fn))
        if not m:
            continue
        stem = f"{m.group(1)}_r{int(m.group(2)):04d}_c{int(m.group(3)):04d}"
        if stem == target_stem:
            return img["id"], img
    return None, None


def _polygons_for_image(coco, image_id):
    ice_cat = next((c["id"] for c in coco["categories"]
                     if c["name"] == "iceberg"), 2)
    return [a for a in coco["annotations"]
            if a.get("image_id") == image_id
                and a.get("category_id") == ice_cat]


def _open_roboflow_chip(chip_stem, args):
    """Find the real Roboflow tif and return (chip_3band, sza_bin)."""
    m = CHIP_RE.match(f"{chip_stem}.tif")
    if not m:
        raise ValueError(f"chip_stem does not match CHIP_RE: {chip_stem}")
    stem = m.group(1)
    row = int(m.group(2))
    col = int(m.group(3))
    tif_path = find_tif(args.chips_root, stem, row, col)
    if tif_path is None:
        raise FileNotFoundError(f"no tif found under {args.chips_root} for {chip_stem}")
    with rasterio.open(tif_path) as src:
        chip = src.read().astype(np.float32)
    if chip.shape != (3, CHIP_SIZE, CHIP_SIZE):
        raise ValueError(f"unexpected chip shape {chip.shape} for {chip_stem}")
    return chip, tif_path


def _load_roboflow_row_b(chip_stem, args):
    """Row (b) loader: prelim polygons (gold outlines), cleaned polygons (cyan fill)."""
    with open(args.raw_coco) as f:
        raw_coco = json.load(f)
    with open(args.filtered_coco) as f:
        filt_coco = json.load(f)
    raw_id, raw_meta = _resolve_roboflow_image_id(raw_coco, chip_stem)
    clean_id, _ = _resolve_roboflow_image_id(filt_coco, chip_stem)
    if raw_id is None or clean_id is None:
        raise RuntimeError(f"chip_stem {chip_stem} not in one of the COCO jsons")

    prelim_anns = _polygons_for_image(raw_coco, raw_id)
    clean_anns = _polygons_for_image(filt_coco, clean_id)
    chip, _ = _open_roboflow_chip(chip_stem, args)

    # Compute the dropped polygon set: prelim minus clean, matched by
    # segmentation geometry (filter renumbers IDs, so id-based diff is unsafe).
    clean_seg_keys = set()
    for ca in clean_anns:
        seg = ca.get("segmentation")
        if isinstance(seg, list):
            clean_seg_keys.add(tuple(tuple(round(float(v), 3) for v in inner)
                                     for inner in seg if isinstance(inner, list)))
    dropped_anns = []
    for pa in prelim_anns:
        seg = pa.get("segmentation")
        if isinstance(seg, list):
            key = tuple(tuple(round(float(v), 3) for v in inner)
                        for inner in seg if isinstance(inner, list))
            if key not in clean_seg_keys:
                dropped_anns.append(pa)

    sza_bin = _resolve_sza_for_chip_stem(chip_stem, args)
    note = (
        f"Roboflow {sza_bin}\n"
        f"{len(prelim_anns)} prelim -> {len(clean_anns)} kept\n"
        f"{len(dropped_anns)} dropped <40 m"
    )
    return chip, prelim_anns, clean_anns, dropped_anns, note


def _resolve_sza_for_chip_stem(chip_stem, args):
    """Look up sza_bin for a chip from split_log."""
    log = pd.read_csv(args.split_log)
    row = log[log["chip_stem"] == chip_stem]
    return row.iloc[0]["sza_bin"] if len(row) else "unknown_sza"


def _load_roboflow_row_c(chip_stem, args):
    """
    Row (c) loader. Returns:
      chip          (3, H, W) reflectance
      prelim_anns   raw polygon annotations (gold col 2; pre-cleaning)
      clean_anns    kept polygon annotations (red col 3; post 40 m filter)
      ic_mask       (H, W) bool — pixels the training-time IC mask would zero
      note          multiline annotator note
    """
    with open(args.raw_coco) as f:
        raw_coco = json.load(f)
    with open(args.filtered_coco) as f:
        filt_coco = json.load(f)
    raw_id, _ = _resolve_roboflow_image_id(raw_coco, chip_stem)
    clean_id, _ = _resolve_roboflow_image_id(filt_coco, chip_stem)
    if raw_id is None or clean_id is None:
        raise RuntimeError(f"chip_stem {chip_stem} not in one of the COCO jsons")
    prelim_anns = _polygons_for_image(raw_coco, raw_id)
    clean_anns = _polygons_for_image(filt_coco, clean_id)

    chip, _ = _open_roboflow_chip(chip_stem, args)

    # Build cleaned annotation mask, then derive the IC pixel mask. The IC
    # mask is computed over the CLEANED annotation set, matching the
    # build_clean_dataset.py convention (training-time).
    segs = [s for ann in clean_anns for s in ann.get("segmentation", [])]
    ann_mask = polygons_to_mask(segs, CHIP_SIZE, CHIP_SIZE) > 0
    b08 = chip[2]
    ic_mask = (b08 >= B08_THRESHOLD) & (~ann_mask)
    n_masked = int(ic_mask.sum())

    log = pd.read_csv(args.split_log)
    row = log[log["chip_stem"] == chip_stem].iloc[0]
    note = (
        f"Roboflow {row['sza_bin']}\n"
        f"{len(prelim_anns)} prelim -> {len(clean_anns)} kept\n"
        f"IC fraction = {float(row['ic_aware']):.2f}\n"
        f"{n_masked:,} bright px masked\n"
        f"(sea ice / cloud)"
    )
    return chip, prelim_anns, clean_anns, ic_mask, note


# 3. Panel renderers
def _render_image_panel(ax, chip):
    """Column 1: false-color RGB."""
    rgb = make_false_color(chip, b08_idx=2)
    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])


def _render_fisser_prelim(ax, chip, prelim_3class):
    """Render Fisser preliminary annotation: ocean transparent, iceberg gold, shadow grey."""
    rgb = make_false_color(chip, b08_idx=2)
    ax.imshow(rgb)
    iceberg = np.zeros((*prelim_3class.shape, 4), dtype=np.float32)
    iceberg[prelim_3class == 1] = [1.0, 0.84, 0.0, 0.55]   # gold
    shadow = np.zeros((*prelim_3class.shape, 4), dtype=np.float32)
    shadow[prelim_3class == 2] = [0.45, 0.45, 0.45, 0.65]  # grey
    ax.imshow(shadow)
    ax.imshow(iceberg)
    ax.set_xticks([]); ax.set_yticks([])


RED_KEPT      = "#dc1c13"             # kept-iceberg outlines (rows a, b)
RED_MASK_RGBA = [0.86, 0.11, 0.08, 0.45]  # red translucent overlay for IC-masked pixels (row c)


def _render_outlines_from_mask(ax, chip, mask, *, color="gold", linewidth=1.2):
    """Trace contours from a binary mask and draw them on the chip RGB."""
    rgb = make_false_color(chip, b08_idx=2)
    ax.imshow(rgb)
    contours = find_contours(mask.astype(float), 0.5)
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], color=color, linewidth=linewidth)
    ax.set_xticks([]); ax.set_yticks([])


def _render_kept_and_dropped_from_masks(ax, chip, kept_mask, dropped_mask, *, linewidth=1.2):
    """
    Row (a) cleaned panel: gold outlines for kept components, red outlines
    for components below the 40 m cutoff that the cleaning step removed.
    """
    rgb = make_false_color(chip, b08_idx=2)
    ax.imshow(rgb)
    for c in find_contours(kept_mask.astype(float), 0.5):
        ax.plot(c[:, 1], c[:, 0], color="gold", linewidth=linewidth)
    for c in find_contours(dropped_mask.astype(float), 0.5):
        ax.plot(c[:, 1], c[:, 0], color=RED_KEPT, linewidth=linewidth * 0.9)
    ax.set_xticks([]); ax.set_yticks([])


def _render_polygon_outlines(ax, chip, anns, *, color="gold", linewidth=1.2):
    """Render polygon outlines (gold by default) from a COCO annotation list."""
    rgb = make_false_color(chip, b08_idx=2)
    ax.imshow(rgb)
    for ann in anns:
        for seg in ann.get("segmentation", []):
            if not isinstance(seg, list) or len(seg) < 6:
                continue
            pts = np.array(seg, dtype=float).reshape(-1, 2)
            ax.add_patch(plt.Polygon(pts, fill=False,
                                       edgecolor=color, linewidth=linewidth))
    ax.set_xticks([]); ax.set_yticks([])


def _render_kept_and_dropped_polygons(ax, chip, kept_anns, dropped_anns, *, linewidth=1.4):
    """
    Row (b) cleaned panel: gold outlines for the polygons kept after the
    40 m filter, red outlines for the polygons dropped by the filter.
    """
    rgb = make_false_color(chip, b08_idx=2)
    ax.imshow(rgb)
    for a in kept_anns:
        for seg in a.get("segmentation", []):
            if not isinstance(seg, list) or len(seg) < 6:
                continue
            pts = np.array(seg, dtype=float).reshape(-1, 2)
            ax.add_patch(plt.Polygon(pts, fill=False,
                                       edgecolor="gold", linewidth=linewidth))
    for a in dropped_anns:
        for seg in a.get("segmentation", []):
            if not isinstance(seg, list) or len(seg) < 6:
                continue
            pts = np.array(seg, dtype=float).reshape(-1, 2)
            ax.add_patch(plt.Polygon(pts, fill=False,
                                       edgecolor=RED_KEPT, linewidth=linewidth * 0.9))
    ax.set_xticks([]); ax.set_yticks([])


def _render_ic_overlay(ax, chip, ic_mask, anns, *, linewidth=1.4):
    """
    Row (c) cleaned panel: chip RGB underneath, RED semi-transparent overlay
    on IC-masked pixels (the pixels training will zero out), and GOLD
    polygon outlines for the kept iceberg annotations on top. The colour
    convention is: gold = kept iceberg (consistent with the preliminary
    panel), red = pixels removed by IC masking.
    """
    rgb = make_false_color(chip, b08_idx=2)
    ax.imshow(rgb)
    overlay = np.zeros((*ic_mask.shape, 4), dtype=np.float32)
    overlay[ic_mask] = RED_MASK_RGBA
    ax.imshow(overlay)
    for ann in anns:
        for seg in ann.get("segmentation", []):
            if not isinstance(seg, list) or len(seg) < 6:
                continue
            pts = np.array(seg, dtype=float).reshape(-1, 2)
            ax.add_patch(plt.Polygon(pts, fill=False,
                                       edgecolor="gold", linewidth=linewidth))
    ax.set_xticks([]); ax.set_yticks([])


def _render_note(ax, note_str):
    ax.axis("off")
    ax.text(0.0, 0.5, note_str, fontsize=13, family="monospace",
            va="center", ha="left", transform=ax.transAxes)


# 4. Main
def main():
    parser = argparse.ArgumentParser(description="Fig. 1 annotation-difficulty visual")
    parser.add_argument("--row_a", help="Fisser chip_stem for row (a)")
    parser.add_argument("--row_b", help="Roboflow chip_stem for row (b)")
    parser.add_argument("--row_c", help="Roboflow chip_stem for row (c)")
    parser.add_argument("--split_log", required=True)
    parser.add_argument("--raw_fisser", required=True,
                        help="dir of raw Fisser pkls (3-class)")
    parser.add_argument("--filtered_fisser", required=True,
                        help="dir of filtered Fisser pkls (binary)")
    parser.add_argument("--raw_coco", required=True)
    parser.add_argument("--filtered_coco", required=True)
    parser.add_argument("--chips_root", required=True,
                        help="root of real Roboflow chip tifs")
    parser.add_argument("--fisser_synth", default=None,
                        help="(unused; Fisser uses pkl arrays directly)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--list_candidates", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if args.list_candidates:
        _list_candidates(args)
        return

    for req in ("row_a", "row_b", "row_c"):
        if not getattr(args, req):
            parser.error(f"--{req} is required unless --list_candidates is set")

    log = pd.read_csv(args.split_log)
    split_a = log[log["chip_stem"] == args.row_a].iloc[0]["split"]

    # 4a. Load each row's data
    print(f"Loading row (a) Fisser: {args.row_a} ({split_a})")
    chip_a, prelim_a, clean_a, dropped_a, note_a = _load_fisser_row(args.row_a, split_a, args)
    print(f"  {note_a.replace(chr(10), ' | ')}")

    print(f"Loading row (b) Roboflow: {args.row_b}")
    chip_b, prelim_b, clean_b, dropped_b, note_b = _load_roboflow_row_b(args.row_b, args)
    print(f"  {note_b.replace(chr(10), ' | ')}")

    print(f"Loading row (c) Roboflow IC-masked: {args.row_c}")
    chip_c, prelim_c, clean_c, ic_mask_c, note_c = _load_roboflow_row_c(args.row_c, args)
    print(f"  {note_c.replace(chr(10), ' | ')}")

    # 4b. Compose 3x4 grid. Convention: in column 3, GOLD outlines = kept
    # iceberg annotation, RED = removed by the cleaning step (sub-40 m
    # components, dropped polygons, or pixels zeroed by the IC mask).
    fig, axes = plt.subplots(3, 4, figsize=(13, 9.5),
                              gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    fig.patch.set_facecolor("white")

    # Row (a) Fisser: cleaned panel = gold (kept components) + red (dropped <16 px)
    _render_image_panel(axes[0, 0], chip_a)
    _render_fisser_prelim(axes[0, 1], chip_a, prelim_a)
    _render_kept_and_dropped_from_masks(axes[0, 2], chip_a, clean_a, dropped_a)
    _render_note(axes[0, 3], note_a)

    # Row (b) Roboflow size filter: cleaned panel = gold (kept) + red (dropped)
    _render_image_panel(axes[1, 0], chip_b)
    _render_polygon_outlines(axes[1, 1], chip_b, prelim_b, color="gold")
    _render_kept_and_dropped_polygons(axes[1, 2], chip_b, clean_b, dropped_b)
    _render_note(axes[1, 3], note_b)

    # Row (c) Roboflow IC mask: preliminary panel = raw polygons in gold,
    # cleaned panel = gold polygons (kept) + red overlay (IC-masked pixels
    # that training will zero out).
    _render_image_panel(axes[2, 0], chip_c)
    _render_polygon_outlines(axes[2, 1], chip_c, prelim_c, color="gold")
    _render_ic_overlay(axes[2, 2], chip_c, ic_mask_c, clean_c)
    _render_note(axes[2, 3], note_c)

    # Column titles on row 0
    col_titles = [
        "Sentinel-2 chip",
        "preliminary",
        "cleaned",
        "annotator note",
    ]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=14, weight="bold")

    # Row labels on column 0
    row_labels = [
        "(a) low SZA / open water\nFisser shadow + 40 m",
        "(b) high SZA / dense\nRoboflow 40 m drop",
        "(c) ambiguous bg\nIC pixel mask",
    ]
    for i, lab in enumerate(row_labels):
        axes[i, 0].set_ylabel(lab, rotation=0, ha="right", va="center",
                                fontsize=13, labelpad=70)

    # Legend: show the colour convention used in column 3.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0], [0], color="gold", lw=2.4,
               label="Gold: iceberg annotation kept after cleaning"),
        Line2D([0], [0], color=RED_KEPT, lw=2.4,
               label="Red (outline): components / polygons removed by 40 m filter"),
        Patch(facecolor=RED_KEPT, alpha=RED_MASK_RGBA[3], edgecolor="none",
              label="Red (fill): pixels zeroed by IC mask (training only)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=1,
               frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.005))

    # No suptitle: figure caption (set in LaTeX) carries the figure title.
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    # 5. Route through fig registry
    caption = (
        "Annotation-difficulty examples for Sentinel-2 iceberg chips. "
        "(a) Low solar-zenith-angle Fisser chip showing the original "
        "three-class annotation (ocean / iceberg / shadow) and the cleaned "
        "binary mask after shadow merge and 40 m root-length filter. "
        "(b) High-SZA Roboflow chip where multiple sub-40 m annotations are "
        "removed by the size filter. (c) Mixed-background chip "
        "(sea ice and cloud edge) where the IC pixel mask zeros bright "
        "non-annotated pixels prior to training."
    )
    archive = fig_write(
        fig=fig,
        slug="fig01_annotation_difficulty",
        caption=caption,
        out_dir=args.out_dir,
        dpi=args.dpi,
    )
    plt.close(fig)
    print(f"\nFigure written: {archive}")


if __name__ == "__main__":
    main()
