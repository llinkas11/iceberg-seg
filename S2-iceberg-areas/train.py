"""
UNet++ training script for iceberg segmentation.

Supports two modes:
  --mode s1  : SAR (Sentinel-1),  binary segmentation     (0=ocean, 1=iceberg)
  --mode s2  : Optical (Sentinel-2), 3-class segmentation (0=ocean, 1=iceberg, 2=shadow)

Usage:
  python train.py --mode s2 --data_dir ./S2UnetPlusPlus --out_dir ./runs/s2_exp1
  python train.py --mode s1 --data_dir ./S1UnetPlusPlus --out_dir ./runs/s1_exp1
"""

import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import segmentation_models_pytorch as smp


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IcebergDataset(Dataset):
    def __init__(self, images, masks, augment=False):
        # images: (N, 3, 256, 256) float32
        # masks:  (N, 1, 256, 256) int64
        # These arrays are already stored in the form the model expects.
        self.images = torch.from_numpy(images)
        self.masks  = torch.from_numpy(masks.astype(np.int64))
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]   # (3, 256, 256)
        mask  = self.masks[idx]    # (1, 256, 256)

        if self.augment:
            # Augmentation happens on the fly here rather than through a separate transform pipeline.
            # random horizontal flip
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[2])
                mask  = torch.flip(mask,  dims=[2])
            # random vertical flip
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[1])
                mask  = torch.flip(mask,  dims=[1])
            # random 90° rotation
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                image = torch.rot90(image, k, dims=[1, 2])
                mask  = torch.rot90(mask,  k, dims=[1, 2])

        return image, mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_class_weights(masks, num_classes, device):
    """Inverse-frequency class weights computed from training masks."""
    flat = masks.flatten()
    counts = np.bincount(flat, minlength=num_classes).astype(np.float64)
    weights = 1.0 / (counts + 1e-6)
    weights /= weights.sum()          # normalize to sum=1
    weights *= num_classes            # scale so average weight = 1
    return torch.FloatTensor(weights).to(device)


def compute_iou(preds_cls, targets_cls, num_classes):
    """
    Mean IoU over non-background classes.
    preds_cls, targets_cls: (N, H, W) long tensors on same device.
    """
    ious = []
    for cls in range(1, num_classes):
        tp = ((preds_cls == cls) & (targets_cls == cls)).sum().float()
        fp = ((preds_cls == cls) & (targets_cls != cls)).sum().float()
        fn = ((preds_cls != cls) & (targets_cls == cls)).sum().float()
        denom = tp + fp + fn
        if denom > 0:
            ious.append((tp / denom).item())
    return float(np.mean(ious)) if ious else 0.0


# ---------------------------------------------------------------------------
# Train / validation loops
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, criterion_dice, criterion_ce,
              device, num_classes, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_iou  = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for images, masks in loader:
            images = images.to(device)           # (N, 3, H, W)
            masks  = masks.to(device)            # (N, 1, H, W) int64

            logits = model(images)               # (N, C, H, W)

            if num_classes == 1:
                # binary: masks float for BCE, keep (N, 1, H, W)
                loss_dice = criterion_dice(logits, masks.float())
                loss_ce   = criterion_ce(logits, masks.float())
                preds_cls = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
                targets_cls = masks.squeeze(1)
            else:
                # multiclass: masks (N, H, W) long for CE
                targets_flat = masks.squeeze(1)               # (N, H, W)
                loss_dice = criterion_dice(logits, targets_flat)
                loss_ce   = criterion_ce(logits, targets_flat)
                preds_cls   = torch.argmax(logits, dim=1)     # (N, H, W)
                targets_cls = targets_flat

            loss = loss_dice + loss_ce

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_iou  += compute_iou(preds_cls, targets_cls, max(num_classes, 2))

    n = len(loader)
    return total_loss / n, total_iou / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train UNet++ on iceberg dataset")
    parser.add_argument("--mode",       required=True, choices=["s1", "s2"],
                        help="s1=SAR binary, s2=optical 3-class")
    parser.add_argument("--data_dir",   required=True,
                        help="Path to S1UnetPlusPlus or S2UnetPlusPlus folder")
    parser.add_argument("--out_dir",    required=True,
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--encoder",    default="resnet34",
                        help="SMP encoder name (default: resnet34)")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--no_pretrain", action="store_true",
                        help="Disable ImageNet pretrained encoder weights")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Mode   : {args.mode.upper()}")
    print(f"Encoder: {args.encoder}")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    split_dir = os.path.join(args.data_dir, "train_validate_test")
    # The chips are loaded directly from pickle and passed into the model as float32 reflectance values.
    X_train = load_pkl(os.path.join(split_dir, "X_train.pkl"))
    Y_train = load_pkl(os.path.join(split_dir, "Y_train.pkl"))
    X_val   = load_pkl(os.path.join(split_dir, "X_validation.pkl"))
    Y_val   = load_pkl(os.path.join(split_dir, "Y_validation.pkl"))
    X_test  = load_pkl(os.path.join(split_dir, "x_test.pkl"))
    Y_test  = load_pkl(os.path.join(split_dir, "y_test.pkl"))

    print(f"Train  : {len(X_train)} samples")
    print(f"Val    : {len(X_val)} samples")
    print(f"Test   : {len(X_test)} samples")

    # -----------------------------------------------------------------------
    # Model config
    # -----------------------------------------------------------------------
    num_classes     = 1 if args.mode == "s1" else 3
    encoder_weights = None if args.no_pretrain else "imagenet"

    model = smp.UnetPlusPlus(
        encoder_name    = args.encoder,
        encoder_weights = encoder_weights,
        in_channels     = 3,
        classes         = num_classes,
    ).to(device)

    # -----------------------------------------------------------------------
    # Loss functions  (DiceLoss + CE/BCE, weighted for class imbalance)
    # -----------------------------------------------------------------------
    if num_classes == 1:
        # S1 binary — weight the positive (iceberg) class heavily
        pos_count = float(Y_train.sum())
        neg_count = float(Y_train.size - pos_count)
        pos_weight = torch.tensor([neg_count / (pos_count + 1e-6)]).to(device)
        print(f"BCE pos_weight: {pos_weight.item():.1f}  (iceberg={pos_count/Y_train.size*100:.2f}%)")

        criterion_dice = smp.losses.DiceLoss(mode="binary")
        criterion_ce   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        # S2 3-class — inverse-frequency weights
        class_weights = compute_class_weights(Y_train.flatten(), num_classes, device)
        print(f"CE class weights: {class_weights.cpu().numpy().round(3)}")

        criterion_dice = smp.losses.DiceLoss(mode="multiclass")
        criterion_ce   = nn.CrossEntropyLoss(weight=class_weights)

    # -----------------------------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------------------------
    train_ds = IcebergDataset(X_train, Y_train, augment=True)
    val_ds   = IcebergDataset(X_val,   Y_val,   augment=False)
    test_ds  = IcebergDataset(X_test,  Y_test,  augment=False)

    # The same dataset class is reused for all three splits; only augmentation changes.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    # -----------------------------------------------------------------------
    # Optimizer & scheduler
    # -----------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_iou   = -1.0
    best_ckpt_path = os.path.join(args.out_dir, "best_model.pth")
    log_path       = os.path.join(args.out_dir, "training_log.csv")

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,train_iou,val_iou,lr\n")

    print(f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  {'TrainIoU':>10}  {'ValIoU':>10}  {'LR':>8}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = run_epoch(
            model, train_loader, optimizer,
            criterion_dice, criterion_ce,
            device, num_classes, train=True
        )
        val_loss, val_iou = run_epoch(
            model, val_loader, optimizer,
            criterion_dice, criterion_ce,
            device, num_classes, train=False
        )
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{train_iou:>10.4f}  {val_iou:>10.4f}  {lr:>8.2e}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                    f"{train_iou:.6f},{val_iou:.6f},{lr:.2e}\n")

        # save best checkpoint
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_iou"    : val_iou,
                "args"       : vars(args),
            }, best_ckpt_path)

    # -----------------------------------------------------------------------
    # Test evaluation with best checkpoint
    # -----------------------------------------------------------------------
    print(f"\nBest val IoU: {best_val_iou:.4f} — evaluating on test set...")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_iou = run_epoch(
        model, test_loader, optimizer,
        criterion_dice, criterion_ce,
        device, num_classes, train=False
    )
    print(f"Test loss: {test_loss:.4f}  |  Test IoU: {test_iou:.4f}")

    with open(log_path, "a") as f:
        f.write(f"\ntest_loss,{test_loss:.6f}\ntest_iou,{test_iou:.6f}\n")

    print(f"\nCheckpoint : {best_ckpt_path}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
