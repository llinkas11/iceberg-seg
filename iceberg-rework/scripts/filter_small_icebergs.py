"""
filter_small_icebergs.py — Remove individual icebergs below a root-length cutoff.

Applies the Fisser 2025 minimum root-length threshold (40 m, i.e. 1600 m2 area,
i.e. 16 pixels at 10 m resolution) to both Roboflow COCO annotations and Fisser
pkl masks.  Produces filtered output files and sample visualizations.

For Fisser masks, shadow (class 2) is merged into iceberg (class 1) BEFORE
connected component analysis, so iceberg+shadow form single objects. This aligns
Fisser's 3-class annotations with Roboflow's binary annotations.

Usage:
  python scripts/filter_small_icebergs.py
  python scripts/filter_small_icebergs.py --min_rl 40 --n_viz 20
"""

import argparse
import csv
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import label as cc_label

# ── Paths ────────────────────────────────────────────────────────────────────
SMISHRA_REWORK = "/mnt/research/v.gomezgilyaspik/students/smishra/rework"
LLINKAS_REWORK = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"

COCO_JSON = os.path.join(
    SMISHRA_REWORK, "data/roboflow_export/train/train/_annotations.coco.json"
)
FISSER_PKL_DIR = os.path.join(
    SMISHRA_REWORK, "data/fisser_original/train_validate_test"
)

PIXEL_AREA_M2 = 100.0  # 10 m × 10 m


# ── COCO filtering ───────────────────────────────────────────────────────────

def filter_coco(coco_path, out_path, min_area_px, n_viz, viz_dir):
    """Filter COCO annotations, removing icebergs below min_area_px pixels."""
    with open(coco_path) as f:
        coco = json.load(f)

    # Only keep iceberg annotations (category_id == 2)
    iceberg_cat_id = None
    for cat in coco.get("categories", []):
        if cat["name"] == "iceberg":
            iceberg_cat_id = cat["id"]
            break
    if iceberg_cat_id is None:
        print("WARNING: No 'iceberg' category found in COCO JSON")
        iceberg_cat_id = 2

    # Group annotations by image
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    img_by_id = {img["id"]: img for img in coco["images"]}

    total_before = 0
    total_after = 0
    total_removed = 0
    removed_areas = []
    kept_areas = []
    per_image_stats = []

    filtered_annotations = []
    viz_count = 0

    for img_info in coco["images"]:
        img_id = img_info["id"]
        anns = ann_by_img.get(img_id, [])

        kept = []
        removed = []
        for ann in anns:
            if ann.get("category_id") != iceberg_cat_id:
                continue  # skip non-iceberg
            total_before += 1
            area_px = ann.get("area", 0)
            if area_px >= min_area_px:
                kept.append(ann)
                kept_areas.append(area_px)
                total_after += 1
            else:
                removed.append(ann)
                removed_areas.append(area_px)
                total_removed += 1

        filtered_annotations.extend(kept)

        per_image_stats.append({
            "image_id": img_id,
            "filename": img_info.get("file_name", ""),
            "n_before": len(kept) + len(removed),
            "n_after": len(kept),
            "n_removed": len(removed),
        })

        # Visualize a sample of chips
        if viz_count < n_viz and (len(removed) > 0 or len(kept) > 0):
            _viz_coco_chip(img_info, kept, removed, viz_dir, min_area_px)
            viz_count += 1

    # Write filtered COCO
    coco_out = dict(coco)
    coco_out["annotations"] = filtered_annotations
    # Re-assign annotation IDs
    for i, ann in enumerate(coco_out["annotations"]):
        ann["id"] = i + 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(coco_out, f)

    return {
        "total_before": total_before,
        "total_after": total_after,
        "total_removed": total_removed,
        "removed_areas": removed_areas,
        "kept_areas": kept_areas,
        "per_image_stats": per_image_stats,
    }


def _viz_coco_chip(img_info, kept, removed, viz_dir, min_area_px):
    """Render a visualization showing kept (gold) vs removed (red) annotations."""
    w = img_info.get("width", 256)
    h = img_info.get("height", 256)

    # Create blank gray image as background (we don't have B08 PNGs easily)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    bg = np.ones((h, w), dtype=np.uint8) * 180
    ax.imshow(bg, cmap="gray", vmin=0, vmax=255)

    # Draw kept annotations in gold
    for ann in kept:
        for seg in ann.get("segmentation", []):
            if isinstance(seg, list) and len(seg) >= 6:
                pts = np.array(seg).reshape(-1, 2)
                poly = plt.Polygon(pts, fill=False, edgecolor="gold", linewidth=1.2)
                ax.add_patch(poly)

    # Draw removed annotations in red
    for ann in removed:
        for seg in ann.get("segmentation", []):
            if isinstance(seg, list) and len(seg) >= 6:
                pts = np.array(seg).reshape(-1, 2)
                poly = plt.Polygon(pts, fill=False, edgecolor="red", linewidth=1.2, linestyle="--")
                ax.add_patch(poly)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()

    kept_patch = mpatches.Patch(edgecolor="gold", facecolor="none", linewidth=1.2, label=f"Kept ({len(kept)})")
    rem_patch = mpatches.Patch(edgecolor="red", facecolor="none", linewidth=1.2, linestyle="--", label=f"Removed ({len(removed)})")
    ax.legend(handles=[kept_patch, rem_patch], loc="upper right", fontsize=7)

    fname = img_info.get("file_name", f"img_{img_info['id']}").replace("/", "_")
    fname = os.path.splitext(fname)[0] + "_filter.png"
    os.makedirs(viz_dir, exist_ok=True)
    fig.savefig(os.path.join(viz_dir, fname), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Fisser pkl filtering ─────────────────────────────────────────────────────

def filter_fisser_masks(pkl_dir, out_dir, min_area_px, n_viz, viz_dir):
    """Filter Fisser pkl masks, zeroing connected components below min_area_px."""
    splits = [
        ("Y_train.pkl", "X_train.pkl"),
        ("Y_validation.pkl", "X_validation.pkl"),
        ("y_test.pkl", "x_test.pkl"),
    ]

    os.makedirs(out_dir, exist_ok=True)
    total_before = 0
    total_after = 0
    total_removed = 0
    removed_areas = []
    kept_areas = []
    viz_count = 0

    for y_file, x_file in splits:
        y_path = os.path.join(pkl_dir, y_file)
        x_path = os.path.join(pkl_dir, x_file)
        if not os.path.exists(y_path):
            print(f"  SKIP (not found): {y_file}")
            continue

        with open(y_path, "rb") as f:
            Y = np.array(pickle.load(f))
        with open(x_path, "rb") as f:
            X = np.array(pickle.load(f))

        if Y.ndim == 4:
            Y = Y[:, 0, :, :]  # (N, H, W)

        # Merge shadow (class 2) into iceberg (class 1) before analysis
        Y[Y == 2] = 1
        Y_filtered = Y.copy()
        split_before = 0
        split_after = 0
        split_removed = 0

        for i in range(len(Y)):
            iceberg_mask = (Y[i] == 1).astype(np.int32)
            if iceberg_mask.sum() == 0:
                continue

            labels, n_components = cc_label(iceberg_mask)
            comp_sizes = np.bincount(labels.ravel())  # index 0 = background

            for comp_id in range(1, n_components + 1):
                comp_pixels = int(comp_sizes[comp_id])
                split_before += 1

                if comp_pixels < min_area_px:
                    Y_filtered[i][labels == comp_id] = 0
                    split_removed += 1
                    removed_areas.append(comp_pixels)
                else:
                    split_after += 1
                    kept_areas.append(comp_pixels)

            # Visualize sample
            if viz_count < n_viz and split_removed > 0:
                _viz_fisser_chip(X[i], Y[i], Y_filtered[i], labels, min_area_px,
                                 viz_dir, f"fisser_{y_file}_{i:04d}")
                viz_count += 1

        total_before += split_before
        total_after += split_after
        total_removed += split_removed

        # Save filtered masks (keep original shape with channel dim)
        Y_out = Y_filtered[:, np.newaxis, :, :]
        out_path = os.path.join(out_dir, y_file)
        with open(out_path, "wb") as f:
            pickle.dump(Y_out, f)

        # Copy X files unchanged (avoid deserialize+reserialize of large arrays)
        import shutil
        x_src = os.path.join(pkl_dir, x_file)
        x_out_path = os.path.join(out_dir, x_file)
        shutil.copy2(x_src, x_out_path)

        print(f"  {y_file}: {split_before} icebergs → {split_after} kept, {split_removed} removed")

    return {
        "total_before": total_before,
        "total_after": total_after,
        "total_removed": total_removed,
        "removed_areas": removed_areas,
        "kept_areas": kept_areas,
    }


def _viz_fisser_chip(X, Y_orig, Y_filt, labels, min_area_px, viz_dir, name):
    """Visualize Fisser chip: B08 + original mask + filtered mask."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # B08 band (index 2)
    b08 = X[2] if X.shape[0] == 3 else X[0]
    axes[0].imshow(b08, cmap="gray")
    axes[0].set_title("B08 (NIR)")
    axes[0].set_axis_off()

    # Original mask with removed components highlighted
    viz = np.zeros((*Y_orig.shape, 3), dtype=np.uint8)
    viz[Y_orig == 0] = [30, 30, 80]    # ocean = dark blue
    viz[Y_orig == 2] = [80, 80, 80]    # shadow = gray

    # Kept icebergs in gold, removed in red
    for comp_id in range(1, labels.max() + 1):
        comp_mask = labels == comp_id
        comp_pixels = comp_mask.sum()
        if comp_pixels >= min_area_px:
            viz[comp_mask] = [218, 165, 32]  # gold
        else:
            viz[comp_mask] = [220, 50, 50]   # red

    axes[1].imshow(viz)
    axes[1].set_title("Gold=kept, Red=removed")
    axes[1].set_axis_off()

    # Filtered mask
    viz2 = np.zeros((*Y_filt.shape, 3), dtype=np.uint8)
    viz2[Y_filt == 0] = [30, 30, 80]
    viz2[Y_filt == 1] = [218, 165, 32]
    viz2[Y_filt == 2] = [80, 80, 80]
    axes[2].imshow(viz2)
    axes[2].set_title("Filtered mask")
    axes[2].set_axis_off()

    os.makedirs(viz_dir, exist_ok=True)
    fig.savefig(os.path.join(viz_dir, f"{name}_filter.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Filter icebergs below root-length cutoff"
    )
    parser.add_argument("--min_rl", type=float, default=40.0,
                        help="Minimum root length in meters (default: 40)")
    parser.add_argument("--n_viz", type=int, default=20,
                        help="Number of sample visualizations per source")
    parser.add_argument("--coco_json", default=COCO_JSON)
    parser.add_argument("--fisser_pkl_dir", default=FISSER_PKL_DIR)
    parser.add_argument("--out_dir", default=os.path.join(LLINKAS_REWORK, "data"))
    parser.add_argument("--viz_dir", default=os.path.join(LLINKAS_REWORK, "viz/filter_40m"))
    parser.add_argument("--summary_csv",
                        default=os.path.join(LLINKAS_REWORK, "reference/filter_40m_summary.csv"))
    args = parser.parse_args()

    min_area_m2 = args.min_rl ** 2  # 40^2 = 1600 m2
    min_area_px = int(min_area_m2 / PIXEL_AREA_M2)  # 16 pixels
    print(f"Root-length cutoff: {args.min_rl} m → {min_area_m2} m2 → {min_area_px} pixels")

    # ── Filter COCO annotations ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FILTERING ROBOFLOW COCO ANNOTATIONS")
    print(f"{'='*60}")

    coco_out = os.path.join(args.out_dir, "annotations_filtered.coco.json")
    coco_viz = os.path.join(args.viz_dir, "coco")
    coco_stats = filter_coco(args.coco_json, coco_out, min_area_px, args.n_viz, coco_viz)

    print(f"\nCOCO results:")
    print(f"  Annotations before: {coco_stats['total_before']}")
    print(f"  Annotations after:  {coco_stats['total_after']}")
    print(f"  Removed:            {coco_stats['total_removed']} ({coco_stats['total_removed']/max(1,coco_stats['total_before'])*100:.1f}%)")

    if coco_stats["kept_areas"]:
        kept = np.array(coco_stats["kept_areas"]) * PIXEL_AREA_M2
        print(f"  Kept area range:    {kept.min():.0f} - {kept.max():.0f} m2")
        print(f"  Kept RL range:      {np.sqrt(kept.min()):.1f} - {np.sqrt(kept.max()):.1f} m")
    if coco_stats["removed_areas"]:
        rem = np.array(coco_stats["removed_areas"]) * PIXEL_AREA_M2
        print(f"  Removed area range: {rem.min():.0f} - {rem.max():.0f} m2")

    print(f"  Filtered COCO saved: {coco_out}")
    print(f"  Visualizations: {coco_viz}/")

    # ── Filter Fisser masks ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FILTERING FISSER PKL MASKS")
    print(f"{'='*60}")

    fisser_out = os.path.join(args.out_dir, "fisser_filtered")
    fisser_viz = os.path.join(args.viz_dir, "fisser")
    fisser_stats = filter_fisser_masks(
        args.fisser_pkl_dir, fisser_out, min_area_px, args.n_viz, fisser_viz
    )

    print(f"\nFisser results:")
    print(f"  Icebergs before: {fisser_stats['total_before']}")
    print(f"  Icebergs after:  {fisser_stats['total_after']}")
    print(f"  Removed:         {fisser_stats['total_removed']} ({fisser_stats['total_removed']/max(1,fisser_stats['total_before'])*100:.1f}%)")

    if fisser_stats["kept_areas"]:
        kept = np.array(fisser_stats["kept_areas"]) * PIXEL_AREA_M2
        print(f"  Kept area range:    {kept.min():.0f} - {kept.max():.0f} m2")
    if fisser_stats["removed_areas"]:
        rem = np.array(fisser_stats["removed_areas"]) * PIXEL_AREA_M2
        print(f"  Removed area range: {rem.min():.0f} - {rem.max():.0f} m2")

    print(f"  Filtered pkls saved: {fisser_out}/")
    print(f"  Visualizations: {fisser_viz}/")

    # ── Combined summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMBINED SUMMARY")
    print(f"{'='*60}")
    total_b = coco_stats["total_before"] + fisser_stats["total_before"]
    total_a = coco_stats["total_after"] + fisser_stats["total_after"]
    total_r = coco_stats["total_removed"] + fisser_stats["total_removed"]
    print(f"  Total icebergs before: {total_b}")
    print(f"  Total icebergs after:  {total_a}")
    print(f"  Total removed:         {total_r} ({total_r/max(1,total_b)*100:.1f}%)")

    # Size distribution of removed icebergs
    all_removed = (
        [a * PIXEL_AREA_M2 for a in coco_stats["removed_areas"]] +
        [a * PIXEL_AREA_M2 for a in fisser_stats["removed_areas"]]
    )
    if all_removed:
        arr = np.array(all_removed)
        rls = np.sqrt(arr)
        print(f"\n  Removed iceberg size distribution:")
        for rl_max in [10, 20, 30, 40]:
            n = int((rls < rl_max).sum())
            print(f"    RL < {rl_max}m: {n}")

    # Write summary CSV
    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
    with open(args.summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "before", "after", "removed", "removed_pct"])
        writer.writerow([
            "coco", coco_stats["total_before"], coco_stats["total_after"],
            coco_stats["total_removed"],
            f"{coco_stats['total_removed']/max(1,coco_stats['total_before'])*100:.1f}"
        ])
        writer.writerow([
            "fisser", fisser_stats["total_before"], fisser_stats["total_after"],
            fisser_stats["total_removed"],
            f"{fisser_stats['total_removed']/max(1,fisser_stats['total_before'])*100:.1f}"
        ])
        writer.writerow([
            "total", total_b, total_a, total_r,
            f"{total_r/max(1,total_b)*100:.1f}"
        ])
    print(f"\n  Summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
