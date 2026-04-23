"""
descriptive_stats.py — Per-iceberg descriptive statistics and exploratory analysis.

Produces:
  1. Per-iceberg root-length histogram (one subplot per SZA bin)
  2. Relative abundance table (saved as figure)
  3. Histograms per SZA bin: count vs month, wind speed, temperature, iceberg area
  4. B08 reflectance table inside icebergs per SZA bin (saved as figure)
  5. Iceberg vs 100m neighborhood B08 histograms (cf. Fisser 2024 Fig. 9)
  6. Fisser 2025 comparison table (saved as figure)

Shadow (class 2) is merged into iceberg (class 1) before analysis.

Usage:
  python scripts/descriptive_stats.py
"""

import argparse
import csv
import json
import os
import pickle
import re
from collections import Counter
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label as cc_label, binary_dilation
from PIL import Image, ImageDraw
import rasterio

# ── Paths ────────────────────────────────────────────────────────────────────
SMISHRA = "/mnt/research/v.gomezgilyaspik/students/smishra/rework"
LLINKAS = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"

FILTERED_COCO = os.path.join(LLINKAS, "data/annotations_filtered.coco.json")
FISSER_FILTERED = os.path.join(LLINKAS, "data/fisser_filtered")
SPLIT_LOG = os.path.join(SMISHRA, "data/split_log.csv")
MET_CSV = os.path.join(LLINKAS, "reference/met_data.csv")
PROVENANCE_CSV = os.path.join(LLINKAS, "reference/fisser_provenance_audit.csv")
CHIPS_ROOT = os.path.join(SMISHRA, "chips")

PIXEL_AREA_M2 = 100.0
BUFFER_PX = 10  # 100m neighborhood at 10m resolution
SZA_BINS = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]
SZA_LABELS = {"sza_lt65": "SZA < 65°\n(Jul–Sep)", "sza_65_70": "SZA 65–70°\n(Sep–Oct)",
              "sza_70_75": "SZA 70–75°\n(Oct)", "sza_gt75": "SZA > 75°\n(Nov)"}
SZA_COLORS = {"sza_lt65": "#2196F3", "sza_65_70": "#4CAF50",
              "sza_70_75": "#FF9800", "sza_gt75": "#F44336"}

RF_HASH_RE = re.compile(r"_png\.rf\.[A-Za-z0-9]+\.png$")
CHIP_RE = re.compile(r"^(.+)_r(\d+)_c(\d+)(?:_B08)?\.(?:png|tif)$")
S2_DATE_RE = re.compile(r"S2[AB]_MSIL1C_(\d{8})T")


def strip_rf_hash(fn):
    return RF_HASH_RE.sub(".png", fn)


def save_table_as_figure(headers, rows, title, out_path, col_widths=None):
    """Render a table as a PNG figure."""
    n_cols = len(headers)
    n_rows = len(rows)
    fig_w = max(8, n_cols * 1.8)
    fig_h = max(2, 0.4 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=headers, loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f0f0")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_coco_per_iceberg(coco_path, split_log_path):
    with open(coco_path) as f:
        coco = json.load(f)
    stem_to_sza = {}
    with open(split_log_path) as f:
        for row in csv.DictReader(f):
            stem_to_sza[row.get("chip_stem", row["stem"])] = row["sza_bin"]
    img_sza = {}
    for img in coco["images"]:
        fn = strip_rf_hash(img["file_name"])
        m = CHIP_RE.match(os.path.basename(fn))
        if m:
            cs = f"{m.group(1)}_r{int(m.group(2)):04d}_c{int(m.group(3)):04d}"
            img_sza[img["id"]] = stem_to_sza.get(cs, "unknown")
    records = []
    for ann in coco["annotations"]:
        area_m2 = ann.get("area", 0) * PIXEL_AREA_M2
        sza = img_sza.get(ann["image_id"], "unknown")
        records.append({"area_m2": area_m2, "rl_m": np.sqrt(area_m2), "sza_bin": sza, "source": "roboflow"})
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)
    chip_counts = []
    for img in coco["images"]:
        n = len(ann_by_img.get(img["id"], []))
        sza = img_sza.get(img["id"], "unknown")
        chip_counts.append({"n_icebergs": n, "sza_bin": sza, "source": "roboflow"})
    return records, chip_counts, coco, img_sza, ann_by_img


def load_fisser_per_iceberg(pkl_dir):
    """Load Fisser masks with shadow merged into iceberg."""
    records = []
    chip_counts = []
    all_X = []
    all_Y = []
    for x_file, y_file in [("X_train.pkl", "Y_train.pkl"),
                            ("X_validation.pkl", "Y_validation.pkl"),
                            ("x_test.pkl", "y_test.pkl")]:
        y_path = os.path.join(pkl_dir, y_file)
        x_path = os.path.join(pkl_dir, x_file)
        if not os.path.exists(y_path):
            continue
        with open(y_path, "rb") as f:
            Y = np.array(pickle.load(f))
        with open(x_path, "rb") as f:
            X = np.array(pickle.load(f))
        if Y.ndim == 4:
            Y = Y[:, 0, :, :]
        # Merge shadow into iceberg
        Y[Y == 2] = 1
        all_X.append(X)
        all_Y.append(Y)
        for i in range(len(Y)):
            iceberg = (Y[i] == 1).astype(np.int32)
            labels, n_comp = cc_label(iceberg)
            n_ice = 0
            for c in range(1, n_comp + 1):
                px = int((labels == c).sum())
                if px < 16:
                    continue
                area_m2 = px * PIXEL_AREA_M2
                records.append({"area_m2": area_m2, "rl_m": np.sqrt(area_m2),
                                "sza_bin": "sza_lt65", "source": "fisser"})
                n_ice += 1
            chip_counts.append({"n_icebergs": n_ice, "sza_bin": "sza_lt65", "source": "fisser"})
    X_all = np.concatenate(all_X, axis=0) if all_X else np.array([])
    Y_all = np.concatenate(all_Y, axis=0) if all_Y else np.array([])
    return records, chip_counts, X_all, Y_all


def load_met_data(met_path):
    met = {}
    if not os.path.exists(met_path):
        return met
    with open(met_path) as f:
        for row in csv.DictReader(f):
            met[row["chip_stem"]] = row
    return met


def build_chip_months(split_log_path, provenance_path):
    chip_months = {}
    with open(split_log_path) as f:
        for row in csv.DictReader(f):
            m = S2_DATE_RE.search(row["stem"])
            if m:
                month = int(m.group(1)[4:6])
                chip_months[row.get("chip_stem", row["stem"])] = (month, row["sza_bin"])
    if os.path.exists(provenance_path):
        with open(provenance_path) as f:
            for row in csv.DictReader(f):
                if row["date"]:
                    month = int(row["date"].split("-")[1])
                    chip_months[f"fisser_{int(row['global_index']):04d}"] = (month, "sza_lt65")
    return chip_months


def main():
    parser = argparse.ArgumentParser(description="Descriptive statistics")
    parser.add_argument("--viz_dir", default=os.path.join(LLINKAS, "viz/descriptive_stats"))
    parser.add_argument("--out_csv", default=os.path.join(LLINKAS, "reference/descriptive_stats.csv"))
    args = parser.parse_args()
    os.makedirs(args.viz_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading data...")
    coco_records, coco_chips, coco, img_sza, ann_by_img = load_coco_per_iceberg(FILTERED_COCO, SPLIT_LOG)
    fisser_records, fisser_chips, fisser_X, fisser_Y = load_fisser_per_iceberg(FISSER_FILTERED)
    all_records = coco_records + fisser_records
    all_chips = coco_chips + fisser_chips
    met = load_met_data(MET_CSV)
    chip_months = build_chip_months(SPLIT_LOG, PROVENANCE_CSV)
    print(f"  COCO: {len(coco_records)} icebergs, {len(coco_chips)} chips")
    print(f"  Fisser: {len(fisser_records)} icebergs, {len(fisser_chips)} chips")

    # ══════════════════════════════════════════════════════════════════════
    # 1. Root-length histogram — one subplot per SZA bin, independent y-axes
    # ══════════════════════════════════════════════════════════════════════
    print("Plot 1: Root-length histograms...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, sza in zip(axes, SZA_BINS):
        rls = [r["rl_m"] for r in all_records if r["sza_bin"] == sza]
        if rls:
            ax.hist(rls, bins=40, range=(40, 800), color=SZA_COLORS[sza], edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Root Length (m)")
        ax.set_title(SZA_LABELS[sza], fontsize=10)
        ax.text(0.95, 0.92, f"n={len(rls):,}", transform=ax.transAxes, ha="right", fontsize=9)
    axes[0].set_ylabel("Count")
    fig.suptitle("Per-Iceberg Root Length Distribution (after 40m filter, shadow merged)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(args.viz_dir, "hist_root_length.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # 2. Relative abundance table → figure
    # ══════════════════════════════════════════════════════════════════════
    print("Table 1: Relative abundance...")
    headers = ["SZA Bin", "N chips", "n(0)", "n(≥1)", "n(≥5)", "n(0):n(≥1):n(≥5)"]
    rows = []
    for sza in SZA_BINS + ["all"]:
        chips = all_chips if sza == "all" else [c for c in all_chips if c["sza_bin"] == sza]
        n0 = sum(1 for c in chips if c["n_icebergs"] == 0)
        n1 = sum(1 for c in chips if c["n_icebergs"] >= 1)
        n5 = sum(1 for c in chips if c["n_icebergs"] >= 5)
        rows.append([sza, str(len(chips)), str(n0), str(n1), str(n5), f"{n0}:{n1}:{n5}"])
    save_table_as_figure(headers, rows, "Relative Abundance: Icebergs per Chip",
                         os.path.join(args.viz_dir, "table_relative_abundance.png"))

    # ══════════════════════════════════════════════════════════════════════
    # 3a. Count vs month — one subplot per SZA bin
    # ══════════════════════════════════════════════════════════════════════
    print("Plot 3a: Month histograms...")
    month_names = {7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov"}
    active_months = sorted(set(m for cs, (m, s) in chip_months.items()))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, sza in zip(axes, SZA_BINS):
        months = [m for cs, (m, s) in chip_months.items() if s == sza]
        if months:
            mc = Counter(months)
            bars = [mc.get(m, 0) for m in active_months]
            ax.bar([month_names.get(m, str(m)) for m in active_months], bars,
                   color=SZA_COLORS[sza], edgecolor="white", linewidth=0.3)
        else:
            ax.bar([month_names.get(m, str(m)) for m in active_months], [0]*len(active_months),
                   color=SZA_COLORS[sza])
        ax.set_xlabel("Month")
        ax.set_title(SZA_LABELS[sza], fontsize=10)
        ax.text(0.95, 0.92, f"n={len(months)}", transform=ax.transAxes, ha="right", fontsize=9)
    axes[0].set_ylabel("Chip Count")
    fig.suptitle("Chip Count by Acquisition Month", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(args.viz_dir, "hist_month.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # 3b. Count vs wind speed — one subplot per SZA bin
    # ══════════════════════════════════════════════════════════════════════
    print("Plot 3b: Wind histograms...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, sza in zip(axes, SZA_BINS):
        winds = [float(met[cs]["wind_speed_10m"]) for cs, (_, s) in chip_months.items()
                 if s == sza and cs in met and met[cs]["wind_speed_10m"]]
        if winds:
            ax.hist(winds, bins=20, color=SZA_COLORS[sza], edgecolor="white", linewidth=0.3)
        ax.axvline(15, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_title(SZA_LABELS[sza], fontsize=10)
        ax.text(0.95, 0.92, f"n={len(winds)}", transform=ax.transAxes, ha="right", fontsize=9)
    axes[0].set_ylabel("Chip Count")
    fig.suptitle("Chip Count by Wind Speed (ERA5)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(args.viz_dir, "hist_wind.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # 3c. Count vs temperature — one subplot per SZA bin
    # ══════════════════════════════════════════════════════════════════════
    print("Plot 3c: Temperature histograms...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, sza in zip(axes, SZA_BINS):
        temps = [float(met[cs]["temp_2m"]) for cs, (_, s) in chip_months.items()
                 if s == sza and cs in met and met[cs]["temp_2m"]]
        if temps:
            ax.hist(temps, bins=20, color=SZA_COLORS[sza], edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Temperature (C)")
        ax.set_title(SZA_LABELS[sza], fontsize=10)
        ax.text(0.95, 0.92, f"n={len(temps)}", transform=ax.transAxes, ha="right", fontsize=9)
    axes[0].set_ylabel("Chip Count")
    fig.suptitle("Chip Count by Temperature (ERA5)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(args.viz_dir, "hist_temp.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # 3d. Area distribution — one subplot per SZA bin + combined log-log
    # ══════════════════════════════════════════════════════════════════════
    print("Plot 3d: Area histograms...")
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, sza in zip(axes[:4], SZA_BINS):
        areas = [r["area_m2"] for r in all_records if r["sza_bin"] == sza]
        if areas:
            ax.hist(areas, bins=40, range=(1600, 100000), color=SZA_COLORS[sza],
                    edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Area (m²)")
        ax.set_title(SZA_LABELS[sza], fontsize=10)
        ax.text(0.95, 0.92, f"n={len(areas)}", transform=ax.transAxes, ha="right", fontsize=9)
    axes[0].set_ylabel("Count")
    # Combined log-log
    ax = axes[4]
    all_areas = [r["area_m2"] for r in all_records if r["area_m2"] > 0]
    bins_log = np.logspace(np.log10(1600), np.log10(max(all_areas)), 40)
    ax.hist(all_areas, bins=bins_log, color="#666", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Area (m²) [log]")
    ax.set_title("Power Law Check\n(all bins)", fontsize=10)
    fig.suptitle("Iceberg Area Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(args.viz_dir, "hist_area.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # 4. B08 inside icebergs table → figure
    # ══════════════════════════════════════════════════════════════════════
    print("Computing B08 inside icebergs per SZA bin...")
    b08_by_bin = {sza: [] for sza in SZA_BINS}
    ocean_by_bin = {sza: [] for sza in SZA_BINS}
    contrast_by_bin = {sza: [] for sza in SZA_BINS}

    # Roboflow
    for img_info in coco["images"]:
        fn = strip_rf_hash(img_info["file_name"])
        m = CHIP_RE.match(os.path.basename(fn))
        if not m:
            continue
        stem, row, col = m.group(1), int(m.group(2)), int(m.group(3))
        sza_bin = img_sza.get(img_info["id"], "unknown")
        if sza_bin not in SZA_BINS:
            continue
        anns = ann_by_img.get(img_info["id"], [])
        if not anns:
            continue
        fname = f"{stem}_r{row:04d}_c{col:04d}.tif"
        matches = glob(os.path.join(CHIPS_ROOT, "**", fname), recursive=True)
        if not matches:
            continue
        with rasterio.open(matches[0]) as src:
            b08 = src.read(3).astype(np.float32)
        mask_img = Image.new("L", (256, 256), 0)
        draw = ImageDraw.Draw(mask_img)
        for ann in anns:
            for seg in ann.get("segmentation", []):
                if isinstance(seg, list) and len(seg) >= 6:
                    coords = [(int(round(seg[i])), int(round(seg[i+1]))) for i in range(0, len(seg)-1, 2)]
                    draw.polygon(coords, fill=1)
        ice_mask = np.array(mask_img)
        ocean_mask = (ice_mask == 0) & (b08 < 0.22)
        if ice_mask.sum() > 0:
            b08_by_bin[sza_bin].append(float(b08[ice_mask > 0].mean()))
        if ocean_mask.sum() > 100:
            om = float(b08[ocean_mask].mean())
            ocean_by_bin[sza_bin].append(om)
            if ice_mask.sum() > 0:
                contrast_by_bin[sza_bin].append(float(b08[ice_mask > 0].mean()) - om)

    # Fisser (shadow already merged)
    for i in range(len(fisser_Y)):
        b08 = fisser_X[i][2]
        ice_mask = (fisser_Y[i] == 1).astype(np.uint8)
        if ice_mask.sum() == 0:
            continue
        ocean_mask = (fisser_Y[i] == 0) & (b08 < 0.22)
        b08_by_bin["sza_lt65"].append(float(b08[ice_mask > 0].mean()))
        if ocean_mask.sum() > 100:
            om = float(b08[ocean_mask].mean())
            ocean_by_bin["sza_lt65"].append(om)
            contrast_by_bin["sza_lt65"].append(float(b08[ice_mask > 0].mean()) - om)

    headers = ["SZA Bin", "N chips", "Iceberg B08", "Ocean B08", "Contrast", "% ice < 0.22"]
    tbl_rows = []
    for sza in SZA_BINS:
        ib = np.array(b08_by_bin[sza]) if b08_by_bin[sza] else np.array([0])
        ob = np.array(ocean_by_bin[sza]) if ocean_by_bin[sza] else np.array([0])
        ct = np.array(contrast_by_bin[sza]) if contrast_by_bin[sza] else np.array([0])
        tbl_rows.append([
            sza, str(len(b08_by_bin[sza])),
            f"{ib.mean():.3f}", f"{ob.mean():.3f}", f"{ct.mean():.3f}",
            f"{(ib < 0.22).mean()*100:.1f}%"
        ])
    save_table_as_figure(headers, tbl_rows,
                         "Mean B08 Reflectance Inside Icebergs vs Ocean per SZA Bin",
                         os.path.join(args.viz_dir, "table_b08_per_sza.png"))
    print("  Saved table_b08_per_sza.png")

    # ══════════════════════════════════════════════════════════════════════
    # 5. Iceberg vs 100m neighborhood histograms (Fisser Fig. 9 equivalent)
    # ══════════════════════════════════════════════════════════════════════
    print("Computing iceberg vs neighborhood B08 distributions...")
    ice_px_by_bin = {sza: [] for sza in SZA_BINS}
    neigh_px_by_bin = {sza: [] for sza in SZA_BINS}

    # Roboflow
    for img_info in coco["images"]:
        fn = strip_rf_hash(img_info["file_name"])
        m = CHIP_RE.match(os.path.basename(fn))
        if not m:
            continue
        stem, row, col = m.group(1), int(m.group(2)), int(m.group(3))
        sza_bin = img_sza.get(img_info["id"], "unknown")
        if sza_bin not in SZA_BINS:
            continue
        anns = ann_by_img.get(img_info["id"], [])
        if not anns:
            continue
        fname = f"{stem}_r{row:04d}_c{col:04d}.tif"
        matches = glob(os.path.join(CHIPS_ROOT, "**", fname), recursive=True)
        if not matches:
            continue
        with rasterio.open(matches[0]) as src:
            b08 = src.read(3).astype(np.float32)
        mask_img = Image.new("L", (256, 256), 0)
        draw = ImageDraw.Draw(mask_img)
        for ann in anns:
            for seg in ann.get("segmentation", []):
                if isinstance(seg, list) and len(seg) >= 6:
                    coords = [(int(round(seg[i])), int(round(seg[i+1]))) for i in range(0, len(seg)-1, 2)]
                    draw.polygon(coords, fill=1)
        ice_mask = np.array(mask_img)
        struct = np.ones((2*BUFFER_PX+1, 2*BUFFER_PX+1), dtype=bool)
        dilated = binary_dilation(ice_mask > 0, structure=struct)
        neigh_mask = dilated & (ice_mask == 0)
        if ice_mask.sum() > 0:
            ice_px_by_bin[sza_bin].append(b08[ice_mask > 0])
        if neigh_mask.sum() > 0:
            neigh_px_by_bin[sza_bin].append(b08[neigh_mask])

    # Fisser (shadow merged)
    for i in range(len(fisser_Y)):
        b08 = fisser_X[i][2]
        ice_mask = (fisser_Y[i] == 1).astype(np.uint8)
        if ice_mask.sum() == 0:
            continue
        struct = np.ones((2*BUFFER_PX+1, 2*BUFFER_PX+1), dtype=bool)
        dilated = binary_dilation(ice_mask > 0, structure=struct)
        neigh_mask = dilated & (ice_mask == 0)
        ice_px_by_bin["sza_lt65"].append(b08[ice_mask > 0])
        if neigh_mask.sum() > 0:
            neigh_px_by_bin["sza_lt65"].append(b08[neigh_mask])

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    bins_hist = np.linspace(0, 1.0, 80)
    for ax, sza in zip(axes, SZA_BINS):
        ice_all = np.concatenate(ice_px_by_bin[sza]) if ice_px_by_bin[sza] else np.array([])
        neigh_all = np.concatenate(neigh_px_by_bin[sza]) if neigh_px_by_bin[sza] else np.array([])
        if len(ice_all) > 0:
            ax.hist(ice_all, bins=bins_hist, density=True, alpha=0.6, color="#E53935",
                    label=f"Iceberg ({len(ice_all):,} px)")
        if len(neigh_all) > 0:
            ax.hist(neigh_all, bins=bins_hist, density=True, alpha=0.6, color="#1E88E5",
                    label=f"100m neigh. ({len(neigh_all):,} px)")
        ax.axvline(0.22, color="black", linestyle="--", linewidth=1.5, label="0.22 threshold")
        ax.set_xlabel("B08 NIR reflectance")
        ax.set_title(SZA_LABELS[sza], fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1.0)
    axes[0].set_ylabel("Density")
    fig.suptitle("Iceberg vs 100m Neighborhood B08 Reflectance\n(cf. Fisser 2024 Fig. 9, shadow merged into iceberg)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(args.viz_dir, "hist_iceberg_vs_neighborhood_b08.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved hist_iceberg_vs_neighborhood_b08.png")

    # ══════════════════════════════════════════════════════════════════════
    # 6. Fisser 2025 comparison table → figure
    # ══════════════════════════════════════════════════════════════════════
    areas_all = np.array([r["area_m2"] for r in all_records])
    rls_all = np.sqrt(areas_all)
    headers = ["Metric", "Our Dataset", "Fisser 2025"]
    tbl_rows = [
        ["N icebergs", f"{len(areas_all):,}", "—"],
        ["Mean area (m²)", f"{areas_all.mean():,.0f}", "2,468"],
        ["Median area (m²)", f"{np.median(areas_all):,.0f}", "—"],
        ["Max area (m²)", f"{areas_all.max():,.0f}", "399,700"],
        ["Mean RL (m)", f"{rls_all.mean():.0f}", "~50"],
        ["Max RL (m)", f"{rls_all.max():.0f}", "632"],
        ["> 400,000 m² (clumps?)", f"{int((areas_all > 400000).sum())}", "0 (by definition)"],
    ]
    save_table_as_figure(headers, tbl_rows, "Comparison with Fisser 2025 Reference Values",
                         os.path.join(args.viz_dir, "table_fisser_comparison.png"))
    print("  Saved table_fisser_comparison.png")

    # ══════════════════════════════════════════════════════════════════════
    # 7. Summary CSV
    # ══════════════════════════════════════════════════════════════════════
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sza_bin", "n_icebergs", "n_chips", "n_null", "n_ge1", "n_ge5",
                          "mean_area_m2", "median_area_m2", "max_area_m2",
                          "mean_rl_m", "median_rl_m",
                          "iceberg_b08_mean", "ocean_b08_mean", "contrast_mean"])
        for sza in SZA_BINS + ["all"]:
            recs = all_records if sza == "all" else [r for r in all_records if r["sza_bin"] == sza]
            chips = all_chips if sza == "all" else [c for c in all_chips if c["sza_bin"] == sza]
            areas = np.array([r["area_m2"] for r in recs]) if recs else np.array([0])
            rls = np.sqrt(areas)
            n0 = sum(1 for c in chips if c["n_icebergs"] == 0)
            n1 = sum(1 for c in chips if c["n_icebergs"] >= 1)
            n5 = sum(1 for c in chips if c["n_icebergs"] >= 5)
            ib = np.array(b08_by_bin.get(sza, [0]))
            ob = np.array(ocean_by_bin.get(sza, [0]))
            ct = np.array(contrast_by_bin.get(sza, [0]))
            writer.writerow([sza, len(recs), len(chips), n0, n1, n5,
                             f"{areas.mean():.0f}", f"{np.median(areas):.0f}", f"{areas.max():.0f}",
                             f"{rls.mean():.0f}", f"{np.median(rls):.0f}",
                             f"{ib.mean():.4f}", f"{ob.mean():.4f}", f"{ct.mean():.4f}"])
    print(f"\nSummary CSV: {args.out_csv}")
    print(f"All figures: {args.viz_dir}/")


if __name__ == "__main__":
    main()
