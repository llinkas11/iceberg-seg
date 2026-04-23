"""
Inference script for the trained UNet++ iceberg segmentation model.

Runs on the test set and saves:
  1. PNG panels: [input image | ground truth | prediction] for every test chip
  2. .npy files: raw predicted masks (for further analysis)
  3. summary.csv: per-chip IoU, dice, iceberg pixel % for every test sample

Usage:
  python predict.py \
      --checkpoint runs/s2_exp1/best_model.pth \
      --data_dir   S2UnetPlusPlus \
      --out_dir    predictions/s2_exp1

  python predict.py \
      --checkpoint runs/s1_exp1/best_model.pth \
      --data_dir   S1UnetPlusPlus \
      --out_dir    predictions/s1_exp1
"""

import os
import pickle
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from train import IcebergDataset, compute_iou


# ---------------------------------------------------------------------------
# Colour schemes for mask overlay
# ---------------------------------------------------------------------------

# S1 binary:  0=ocean (transparent), 1=iceberg (cyan)
S1_COLORS = {
    0: (0.0, 0.0, 0.0, 0.0),   # ocean — transparent
    1: (0.0, 1.0, 1.0, 0.7),   # iceberg — cyan
}
S1_LABELS = {1: "Iceberg"}

# S2 3-class: 0=ocean (transparent), 1=iceberg (cyan), 2=shadow (orange)
S2_COLORS = {
    0: (0.0, 0.0, 0.0, 0.0),   # ocean — transparent
    1: (0.0, 1.0, 1.0, 0.7),   # iceberg — cyan
    2: (1.0, 0.5, 0.0, 0.6),   # shadow — orange
}
S2_LABELS = {1: "Iceberg", 2: "Shadow"}


def mask_to_rgba(mask_2d, color_map):
    """Convert (H, W) integer mask to (H, W, 4) RGBA array."""
    h, w = mask_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for cls, color in color_map.items():
        where = mask_2d == cls
        for c, v in enumerate(color):
            rgba[where, c] = v
    return rgba


def chip_to_rgb(chip):
    """
    Convert (3, H, W) float32 chip to (H, W, 3) uint8 for display.
    Each channel is stretched independently to [0, 255].
    """
    rgb = chip.cpu().numpy().transpose(1, 2, 0)          # (H, W, 3)
    lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def dice_score(pred, target, cls):
    p = (pred == cls)
    t = (target == cls)
    intersection = (p & t).sum()
    denom = p.sum() + t.sum()
    return (2 * intersection / (denom + 1e-6)).item()


def threshold_s2(chip, threshold=0.12, b08_idx=2):
    """Apply Fisser B08 >= threshold to produce a binary iceberg mask (0/1)."""
    b08 = chip[b08_idx].cpu().numpy()   # (H, W) in TOA reflectance [0, 1]
    return (b08 >= threshold).astype(np.int64)  # 1=iceberg, 0=ocean


# ---------------------------------------------------------------------------
# Load model from checkpoint
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt["args"]
    mode = args["mode"]
    num_classes = 1 if mode == "s1" else 3

    model = smp.UnetPlusPlus(
        encoder_name    = args["encoder"],
        encoder_weights = None,           # weights loaded from checkpoint
        in_channels     = 3,
        classes         = num_classes,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val IoU {ckpt['val_iou']:.4f}")
    return model, mode, num_classes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pth from train.py")
    parser.add_argument("--data_dir",   required=True,
                        help="S1UnetPlusPlus or S2UnetPlusPlus folder")
    parser.add_argument("--out_dir",    required=True,
                        help="Where to save outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--split",      default="test",
                        choices=["test", "validation", "train"],
                        help="Which split to run inference on")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir  = os.path.join(args.out_dir, "visualizations")
    mask_dir = os.path.join(args.out_dir, "predicted_masks")
    os.makedirs(vis_dir,  exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mode, num_classes = load_model(args.checkpoint, device)

    color_map = S1_COLORS if mode == "s1" else S2_COLORS
    cls_labels = S1_LABELS if mode == "s1" else S2_LABELS

    # -----------------------------------------------------------------------
    # Load split
    # -----------------------------------------------------------------------
    split_dir = os.path.join(args.data_dir, "train_validate_test")
    if args.split == "test":
        X = pickle.load(open(os.path.join(split_dir, "x_test.pkl"), "rb"))
        Y = pickle.load(open(os.path.join(split_dir, "y_test.pkl"), "rb"))
    elif args.split == "validation":
        X = pickle.load(open(os.path.join(split_dir, "X_validation.pkl"), "rb"))
        Y = pickle.load(open(os.path.join(split_dir, "Y_validation.pkl"), "rb"))
    else:
        X = pickle.load(open(os.path.join(split_dir, "X_train.pkl"), "rb"))
        Y = pickle.load(open(os.path.join(split_dir, "Y_train.pkl"), "rb"))

    dataset = IcebergDataset(X, Y, augment=False)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.workers, pin_memory=True)

    print(f"Running inference on {len(dataset)} chips ({args.split} split)...")

    # -----------------------------------------------------------------------
    # Inference loop
    # -----------------------------------------------------------------------
    all_preds   = []
    all_targets = []
    all_images  = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)

            if num_classes == 1:
                preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)  # (N, H, W)
            else:
                preds = torch.argmax(logits, dim=1)                       # (N, H, W)

            all_preds.append(preds.cpu())
            all_targets.append(masks.squeeze(1).cpu())
            all_images.append(images.cpu())

    all_preds   = torch.cat(all_preds,   dim=0)   # (N, H, W)
    all_targets = torch.cat(all_targets, dim=0)   # (N, H, W)
    all_images  = torch.cat(all_images,  dim=0)   # (N, 3, H, W)

    # -----------------------------------------------------------------------
    # Per-chip statistics
    # -----------------------------------------------------------------------
    rows = []
    iceberg_ious  = []
    iceberg_dices = []
    thresh_ious   = []
    thresh_dices  = []

    # Precompute threshold predictions for S2 (binary, class 1 = iceberg)
    thresh_preds = None
    if mode == "s2":
        thresh_preds = torch.from_numpy(
            np.stack([threshold_s2(all_images[i]) for i in range(len(all_images))])
        )  # (N, H, W) int64

    for i in range(len(all_preds)):
        pred   = all_preds[i]    # (H, W)
        target = all_targets[i]  # (H, W)

        iou_iceberg   = compute_iou(pred.unsqueeze(0), target.unsqueeze(0),
                                    max(num_classes, 2))
        dice_iceberg  = dice_score(pred, target, cls=1)
        iceberg_pct   = (pred == 1).float().mean().item() * 100
        gt_iceberg_pct = (target == 1).float().mean().item() * 100

        iceberg_ious.append(iou_iceberg)
        iceberg_dices.append(dice_iceberg)

        row = {
            "chip_idx"         : i,
            "iou_iceberg"      : round(iou_iceberg,  4),
            "dice_iceberg"     : round(dice_iceberg, 4),
            "pred_iceberg_pct" : round(iceberg_pct,  2),
            "gt_iceberg_pct"   : round(gt_iceberg_pct, 2),
        }

        if thresh_preds is not None:
            tp = thresh_preds[i]  # (H, W)
            t_iou  = compute_iou(tp.unsqueeze(0), target.unsqueeze(0), 2)
            t_dice = dice_score(tp, target, cls=1)
            t_pct  = (tp == 1).float().mean().item() * 100
            thresh_ious.append(t_iou)
            thresh_dices.append(t_dice)
            row["thresh_iou_iceberg"]  = round(t_iou,  4)
            row["thresh_dice_iceberg"] = round(t_dice, 4)
            row["thresh_iceberg_pct"]  = round(t_pct,  2)

        rows.append(row)

    # -----------------------------------------------------------------------
    # Save summary CSV
    # -----------------------------------------------------------------------
    import csv
    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    mean_iou  = np.mean(iceberg_ious)
    mean_dice = np.mean(iceberg_dices)
    print(f"\nMean iceberg IoU  (model)     : {mean_iou:.4f}")
    print(f"Mean iceberg Dice (model)     : {mean_dice:.4f}")
    if thresh_ious:
        print(f"Mean iceberg IoU  (threshold) : {np.mean(thresh_ious):.4f}")
        print(f"Mean iceberg Dice (threshold) : {np.mean(thresh_dices):.4f}")
    print(f"Summary saved to  : {csv_path}")

    # -----------------------------------------------------------------------
    # Visualizations — one PNG per chip:  [image | ground truth | prediction]
    # -----------------------------------------------------------------------
    print(f"\nSaving {len(all_preds)} visualizations...")

    n_panels = 4 if mode == "s2" else 3
    fig_w    = 16 if mode == "s2" else 12

    for i in range(len(all_preds)):
        chip   = all_images[i]            # (3, H, W)
        gt     = all_targets[i].numpy()   # (H, W)
        pred   = all_preds[i].numpy()     # (H, W)
        rgb    = chip_to_rgb(chip)        # (H, W, 3) uint8

        gt_rgba   = mask_to_rgba(gt,   color_map)
        pred_rgba = mask_to_rgba(pred, color_map)

        # Build title — include threshold metrics if available
        title = (
            f"Chip {i:03d}  |  "
            f"IoU={rows[i]['iou_iceberg']:.3f}  "
            f"Dice={rows[i]['dice_iceberg']:.3f}  "
            f"pred={rows[i]['pred_iceberg_pct']:.1f}%  "
            f"gt={rows[i]['gt_iceberg_pct']:.1f}%"
        )
        if "thresh_iou_iceberg" in rows[i]:
            title += (
                f"  |  thresh IoU={rows[i]['thresh_iou_iceberg']:.3f}  "
                f"Dice={rows[i]['thresh_dice_iceberg']:.3f}"
            )

        ch_labels = ("HH", "HV", "HH/HV") if mode == "s1" else ("B04", "B03", "B08")

        fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 4))
        fig.suptitle(title, fontsize=8)

        # — input image
        axes[0].imshow(rgb)
        axes[0].set_title(f"Input ({ch_labels[0]}/{ch_labels[1]}/{ch_labels[2]})")

        # — ground truth
        axes[1].imshow(rgb)
        axes[1].imshow(gt_rgba)
        axes[1].set_title("Ground truth")

        if mode == "s2":
            # — threshold (B08 >= 0.12)
            thresh_mask = thresh_preds[i].numpy()
            thresh_rgba = mask_to_rgba(thresh_mask, S2_COLORS)
            axes[2].imshow(rgb)
            axes[2].imshow(thresh_rgba)
            axes[2].set_title("Threshold (B08≥0.12)")

            # — model prediction
            axes[3].imshow(rgb)
            axes[3].imshow(pred_rgba)
            axes[3].set_title("UNet++ prediction")

            patches = [mpatches.Patch(color=color_map[c][:3], label=lbl)
                       for c, lbl in cls_labels.items()]
            axes[3].legend(handles=patches, loc="lower right", fontsize=7,
                           framealpha=0.7)
        else:
            # — prediction (S1 binary, no threshold panel)
            axes[2].imshow(rgb)
            axes[2].imshow(pred_rgba)
            axes[2].set_title("UNet++ prediction")

            patches = [mpatches.Patch(color=color_map[c][:3], label=lbl)
                       for c, lbl in cls_labels.items()]
            axes[2].legend(handles=patches, loc="lower right", fontsize=7,
                           framealpha=0.7)

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        fig.savefig(os.path.join(vis_dir, f"chip_{i:03d}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Save raw predicted masks as .npy
    # -----------------------------------------------------------------------
    np.save(os.path.join(mask_dir, "predicted_masks.npy"), all_preds.numpy())
    np.save(os.path.join(mask_dir, "ground_truth_masks.npy"), all_targets.numpy())
    print(f"Masks saved to   : {mask_dir}/predicted_masks.npy")
    print(f"                   shape = {all_preds.numpy().shape}")

    print(f"\nDone. All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
