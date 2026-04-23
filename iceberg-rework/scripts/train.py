"""
UNet++ training script for iceberg segmentation.

Supports two modes:
  --mode s1  : SAR (Sentinel-1),  binary segmentation (iceberg only)
  --mode s2  : Optical (Sentinel-2), binary segmentation (iceberg only, shadow merged)

Usage:
  python train.py --mode s2 --data_dir ./S2UnetPlusPlus --out_dir ./runs/s2_exp1
  python train.py --mode s1 --data_dir ./S1UnetPlusPlus --out_dir ./runs/s1_exp1
"""

import os
import json
import pickle
import random
import argparse
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import segmentation_models_pytorch as smp

from _method_common import get_git_sha


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def set_deterministic_seed(seed: int):
    """Seed Python, NumPy, PyTorch (CPU+CUDA), and cuDNN for matched runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seeded_worker_init_fn(worker_id):
    base = torch.initial_seed() % (2**32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IcebergDataset(Dataset):
    def __init__(self, images, masks, augment=False):
        # images: (N, 3, 256, 256) float32
        # masks:  (N, 1, 256, 256) int64
        self.images = torch.from_numpy(images)
        self.masks  = torch.from_numpy(masks.astype(np.int64))
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]   # (3, 256, 256)
        mask  = self.masks[idx]    # (1, 256, 256)

        if self.augment:
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
# Run provenance (get_git_sha comes from _method_common)
# ---------------------------------------------------------------------------

def get_manifest_id(data_dir):
    """Look for data_dir/manifest.json and return its manifest_id, else None."""
    manifest_path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path) as f:
            return json.load(f).get("manifest_id")
    except Exception:
        return None


def write_training_config(out_dir, args, best_val_iou, test_loss, test_iou):
    """
    Write training_config.json next to best_model.pth.

    Captures every hyperparameter, the data manifest id, the seed, the git
    SHA of the scripts/ directory at training time, and the final metrics.
    Downstream tooling reads this file instead of prying into the .pth.
    """
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_id = get_manifest_id(args.data_dir)

    config = {
        "run_kind":          "training",
        "run_started_utc":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script":            os.path.basename(__file__),
        "git_sha":           get_git_sha(repo_dir),
        "manifest_id":       manifest_id,
        "experiment_mode":   os.environ.get("ICEBERG_EXPERIMENT") == "1",
        "reproducible":      args.seed is not None,
        "args":              vars(args),
        "metrics": {
            "best_val_iou": float(best_val_iou),
            "test_loss":    float(test_loss),
            "test_iou":     float(test_iou),
        },
    }

    out_path = os.path.join(out_dir, "training_config.json")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train UNet++ on iceberg dataset")
    parser.add_argument("--mode",       required=True, choices=["s1", "s2"],
                        help="s1=SAR binary, s2=optical binary (iceberg only)")
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
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable training augmentation (flips, rotations)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed Python/NumPy/Torch for reproducibility; default None = non-deterministic")
    args = parser.parse_args()

    # When called via run_experiment.py, ICEBERG_EXPERIMENT=1 is set and a
    # seed is mandatory. Ad-hoc reruns without that env var can still omit
    # --seed, but results are then marked unreproducible in any downstream
    # comparison table.
    if os.environ.get("ICEBERG_EXPERIMENT") == "1" and args.seed is None:
        raise SystemExit(
            "Refusing to train without --seed under ICEBERG_EXPERIMENT=1. "
            "Pass --seed or unset ICEBERG_EXPERIMENT for an ad-hoc run."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    if args.seed is not None:
        set_deterministic_seed(args.seed)
        print(f"Seed  : {args.seed}  (deterministic cuDNN)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Mode   : {args.mode.upper()}")
    print(f"Encoder: {args.encoder}")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    split_dir = os.path.join(args.data_dir, "train_validate_test")
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
    num_classes     = 1   # binary segmentation: 0=ocean, 1=iceberg (shadow merged)
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
        # S1 binary: weight the positive (iceberg) class heavily
        pos_count = float(Y_train.sum())
        neg_count = float(Y_train.size - pos_count)
        pos_weight = torch.tensor([neg_count / (pos_count + 1e-6)]).to(device)
        print(f"BCE pos_weight: {pos_weight.item():.1f}  (iceberg={pos_count/Y_train.size*100:.2f}%)")

        criterion_dice = smp.losses.DiceLoss(mode="binary")
        criterion_ce   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        # S2 binary: inverse-frequency weights on the two classes
        class_weights = compute_class_weights(Y_train.flatten(), num_classes, device)
        print(f"CE class weights: {class_weights.cpu().numpy().round(3)}")

        criterion_dice = smp.losses.DiceLoss(mode="multiclass")
        criterion_ce   = nn.CrossEntropyLoss(weight=class_weights)

    # -----------------------------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------------------------
    train_ds = IcebergDataset(X_train, Y_train, augment=not args.no_augment)
    val_ds   = IcebergDataset(X_val,   Y_val,   augment=False)
    test_ds  = IcebergDataset(X_test,  Y_test,  augment=False)

    train_generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    train_worker_init = seeded_worker_init_fn if args.seed is not None else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True,
                              generator=train_generator, worker_init_fn=train_worker_init,
                              persistent_workers=args.workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)

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
                "ckpt_metric": "val_iou",
                "args"       : vars(args),
            }, best_ckpt_path)

    # -----------------------------------------------------------------------
    # Test evaluation with best checkpoint
    # -----------------------------------------------------------------------
    print(f"\nBest val IoU: {best_val_iou:.4f}, evaluating on test set...")
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

    # Write a JSON sidecar so downstream tools do not have to load the .pth.
    cfg_path = write_training_config(args.out_dir, args, best_val_iou, test_loss, test_iou)

    print(f"\nCheckpoint : {best_ckpt_path}")
    print(f"Training log    : {log_path}")
    print(f"Training config : {cfg_path}")


if __name__ == "__main__":
    main()
