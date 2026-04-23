"""
Visualize UNet++ predictions on pkl test chips.

Two-panel layout per chip:
  [ RGB + ground truth overlay | RGB + prediction overlay ]

Iceberg pixels shown as semi-transparent gold fill on the actual image.
Only iceberg IoU is reported (ocean/shadow excluded — new chips have iceberg only).

If --split_log is provided, PNGs are saved into per-bin subdirs and
a per-bin iceberg IoU summary is printed.

Usage
-----
  CUDA_VISIBLE_DEVICES="" python visualize_predictions.py \\
      --checkpoint  runs/s2_v2_aug/best_model.pth \\
      --test_pkl    .../train_validate_test_v2/train_validate_test/x_test.pkl \\
      --split_log   .../train_validate_test_v2/split_log.csv \\
      --out_dir     viz_predictions/v2/
"""

import os
import csv
import argparse
import pickle
from collections import defaultdict, Counter

import numpy as np
import torch
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


SZA_BINS_ORDERED = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_split_log(csv_path):
    bins = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                bins.append(row["sza_bin"])
    return bins


def load_model(ckpt_path, device):
    ckpt    = torch.load(ckpt_path, map_location=device, weights_only=False)
    args    = ckpt.get("args", {})
    encoder = args.get("encoder", "resnet34")
    model   = smp.UnetPlusPlus(
        encoder_name    = encoder,
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Checkpoint : {ckpt_path}")
    print(f"Encoder    : {encoder}  |  val IoU : {ckpt.get('val_iou', 'n/a')}  |  epoch : {ckpt.get('epoch', 'n/a')}")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_batch(model, images_np, device, batch_size=32):
    preds  = []
    tensor = torch.from_numpy(images_np.astype(np.float32))
    for start in range(0, len(tensor), batch_size):
        batch  = tensor[start:start + batch_size].to(device)
        logits = model(batch)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def iceberg_iou(pred, target):
    """IoU for class 1 (iceberg) only."""
    tp = int(((pred == 1) & (target == 1)).sum())
    fp = int(((pred == 1) & (target != 1)).sum())
    fn = int(((pred != 1) & (target == 1)).sum())
    denom = tp + fp + fn
    return tp / denom if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def chip_to_grey(chip_chw):
    """B08 band (index 2) → (H,W) float, 2–98th percentile stretch to [0,1]."""
    b08 = chip_chw[2].copy()
    p2, p98 = np.percentile(b08, (2, 98))
    if p98 > p2:
        b08 = np.clip((b08 - p2) / (p98 - p2), 0, 1)
    return b08


def save_grid(grey, gt_mask, pred_mask, out_path, chip_idx, sza_bin, iou):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    bin_str = f"  [{sza_bin}]" if sza_bin else ""
    iou_str = f"  IoU={iou:.3f}" if not np.isnan(iou) else "  IoU=n/a"

    for ax, mask, title in [
        (axes[0], gt_mask,   f"Chip #{chip_idx}{bin_str}  — ground truth"),
        (axes[1], pred_mask, f"Prediction{iou_str}"),
    ]:
        ax.imshow(grey, cmap="gray", vmin=0, vmax=1)
        # draw iceberg contour outline
        ice = (mask == 1).astype(np.float32)
        if ice.any():
            ax.contour(ice, levels=[0.5], colors=["#ffd700"], linewidths=[1.2])
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    patch = mpatches.Patch(edgecolor="#ffd700", facecolor="none", label="iceberg outline")
    fig.legend(handles=[patch], loc="lower center", fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_pkl",   required=True)
    parser.add_argument("--split_log",  default=None,
                        help="split_log.csv for per-SZA-bin metrics + subdirs")
    parser.add_argument("--out_dir",    default="viz_predictions")
    parser.add_argument("--n",          type=int, default=0, help="Max chips (0=all)")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- data ----------------------------------------------------------------
    test_dir = os.path.dirname(args.test_pkl)
    X_test   = load_pkl(args.test_pkl)
    for y_name in ("y_test.pkl", "Y_test.pkl"):
        y_path = os.path.join(test_dir, y_name)
        if os.path.exists(y_path):
            Y_test = load_pkl(y_path)
            break
    else:
        raise FileNotFoundError(f"y_test.pkl not found in {test_dir}")

    n_chips = min(args.n, len(X_test)) if args.n > 0 else len(X_test)
    X = X_test[:n_chips].astype(np.float32)
    Y = Y_test[:n_chips].squeeze(1)
    print(f"Chips: {n_chips}")

    # ---- SZA bin labels ------------------------------------------------------
    sza_bins = None
    if args.split_log:
        sza_bins = load_split_log(args.split_log)[:n_chips]
        print("SZA bin counts:", dict(Counter(sza_bins)))
        for b in set(sza_bins):
            os.makedirs(os.path.join(args.out_dir, b), exist_ok=True)

    # ---- inference -----------------------------------------------------------
    model = load_model(args.checkpoint, device)
    print("Running inference...")
    preds = predict_batch(model, X, device, args.batch_size)

    # ---- save PNGs -----------------------------------------------------------
    bin_ious = defaultdict(list)

    for i in range(n_chips):
        sza_bin  = sza_bins[i] if sza_bins else None
        grey     = chip_to_grey(X[i])
        gt       = Y[i].astype(np.int32)
        pred     = preds[i].astype(np.int32)
        iou      = iceberg_iou(pred, gt)

        subdir   = os.path.join(args.out_dir, sza_bin) if sza_bin else args.out_dir
        out_path = os.path.join(subdir, f"chip_{i:04d}.png")
        save_grid(grey, gt, pred, out_path, i, sza_bin, iou)

        bin_ious[sza_bin or "all"].append(iou)

        if (i + 1) % 10 == 0 or i == n_chips - 1:
            print(f"  {i+1}/{n_chips}")

    # ---- per-bin summary -----------------------------------------------------
    print(f"\n{'─'*38}")
    print(f"{'Bin':<18} {'Iceberg IoU':>12}  {'N':>4}")
    print(f"{'─'*38}")

    report_bins = SZA_BINS_ORDERED if sza_bins else ["all"]
    all_vals = []
    for b in report_bins:
        if b not in bin_ious:
            continue
        vals = [v for v in bin_ious[b] if not np.isnan(v)]
        mean = np.mean(vals) if vals else float("nan")
        print(f"{b:<18} {mean:>12.4f}  {len(bin_ious[b]):>4}")
        all_vals.extend(vals)

    if len(report_bins) > 1 and all_vals:
        print(f"{'─'*38}")
        print(f"{'OVERALL':<18} {np.mean(all_vals):>12.4f}  {len(all_vals):>4}")

    print(f"{'─'*38}")
    print(f"\nPNGs saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()