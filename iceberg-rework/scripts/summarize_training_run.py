#!/usr/bin/env python3
"""Write a compact markdown summary for a training run."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from datetime import datetime, timezone


def parse_training_log(log_path: str):
    epoch_rows = []
    test_metrics = {}

    if not os.path.exists(log_path):
        return epoch_rows, test_metrics

    with open(log_path, newline="") as f:
        reader = csv.reader(f)
        header_seen = False
        for row in reader:
            if not row:
                continue
            if not header_seen:
                header_seen = True
                continue
            if row[0].isdigit() and len(row) >= 6:
                epoch_rows.append(
                    {
                        "epoch": int(row[0]),
                        "train_loss": float(row[1]),
                        "val_loss": float(row[2]),
                        "train_iou": float(row[3]),
                        "val_iou": float(row[4]),
                        "lr": row[5],
                    }
                )
            elif row[0] in {"test_loss", "test_iou"} and len(row) >= 2:
                test_metrics[row[0]] = float(row[1])

    return epoch_rows, test_metrics


def read_checkpoint_info(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        return {}

    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = ckpt.get("args") or {}
    return {
        "best_epoch": ckpt.get("epoch"),
        "best_val_iou": ckpt.get("val_iou"),
        "ckpt_metric": ckpt.get("ckpt_metric", "val_iou"),
        "seed": ckpt_args.get("seed"),
        "args": ckpt_args,
    }


def sacct_summary(job_id: str):
    if not job_id:
        return ""

    cmd = [
        "sacct",
        "-j",
        job_id,
        "--format=JobID,JobName,State,Elapsed,ExitCode",
        "-n",
        "-P",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def file_size_mb(path: str):
    if not os.path.exists(path):
        return None
    return os.path.getsize(path) / (1024 * 1024)


def tail_lines(path: str, max_lines: int = 20):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = [line.rstrip("\n") for line in f.readlines()]
    return lines[-max_lines:]


def fmt_metric(value):
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Summarize a completed training run")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--stdout-log", default=None)
    parser.add_argument("--stderr-log", default=None)
    args = parser.parse_args()

    if args.summary_path is None:
        args.summary_path = os.path.join(args.run_dir, "run_summary.md")

    training_log = os.path.join(args.run_dir, "training_log.csv")
    checkpoint_path = os.path.join(args.run_dir, "best_model.pth")

    epoch_rows, test_metrics = parse_training_log(training_log)
    checkpoint = read_checkpoint_info(checkpoint_path)
    sacct_text = sacct_summary(args.job_id)

    best_epoch_row = max(epoch_rows, key=lambda row: row["val_iou"]) if epoch_rows else None
    final_epoch_row = epoch_rows[-1] if epoch_rows else None

    lines = []
    lines.append("# Overnight Training Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).astimezone().isoformat()}")
    lines.append(f"Job ID: `{args.job_id}`")
    lines.append(f"Run directory: `{args.run_dir}`")
    lines.append(f"Data directory: `{args.data_dir}`")
    lines.append("")

    lines.append("## Status")
    if sacct_text:
        lines.append("```text")
        lines.extend(sacct_text.splitlines())
        lines.append("```")
    else:
        lines.append("No `sacct` summary available.")
    lines.append("")

    lines.append("## Metrics")
    lines.append(f"- Epochs logged: `{len(epoch_rows)}`")
    if best_epoch_row:
        lines.append(
            f"- Best validation IoU: `{best_epoch_row['val_iou']:.4f}` at epoch `{best_epoch_row['epoch']}`"
        )
        lines.append(
            f"- Best-epoch validation loss: `{best_epoch_row['val_loss']:.4f}`"
        )
    else:
        lines.append("- Best validation IoU: `n/a`")
    if final_epoch_row:
        lines.append(
            f"- Final logged epoch: `{final_epoch_row['epoch']}` with train IoU `{final_epoch_row['train_iou']:.4f}` and val IoU `{final_epoch_row['val_iou']:.4f}`"
        )
        lines.append(
            f"- Final logged losses: train `{final_epoch_row['train_loss']:.4f}`, val `{final_epoch_row['val_loss']:.4f}`"
        )
    else:
        lines.append("- Final logged epoch: `n/a`")
    lines.append(f"- Test IoU: `{fmt_metric(test_metrics.get('test_iou'))}`")
    lines.append(f"- Test loss: `{fmt_metric(test_metrics.get('test_loss'))}`")
    if checkpoint.get("best_epoch") is not None:
        lines.append(f"- Checkpoint best epoch: `{checkpoint['best_epoch']}`")
    if checkpoint.get("best_val_iou") is not None:
        lines.append(f"- Checkpoint best val IoU: `{checkpoint['best_val_iou']:.4f}`")
    if checkpoint:
        seed_val = checkpoint.get("seed")
        seed_str = seed_val if seed_val is not None else "non-deterministic"
        lines.append(f"- Seed: `{seed_str}` (checkpoint metric: `{checkpoint.get('ckpt_metric', 'val_iou')}`)")
    lines.append("")

    lines.append("## Artifacts")
    lines.append(f"- Training log: `{training_log}`")
    lines.append(f"- Best checkpoint: `{checkpoint_path}`")
    ckpt_mb = file_size_mb(checkpoint_path)
    if ckpt_mb is not None:
        lines.append(f"- Checkpoint size: `{ckpt_mb:.1f} MB`")
    lines.append("")

    lines.append("## Recent Training Log")
    if os.path.exists(training_log):
        lines.append("```text")
        for line in tail_lines(training_log, max_lines=15):
            lines.append(line)
        lines.append("```")
    else:
        lines.append("Training log not found.")
    lines.append("")

    if args.stderr_log:
        lines.append("## stderr tail")
        if os.path.exists(args.stderr_log):
            lines.append("```text")
            stderr_tail = tail_lines(args.stderr_log, max_lines=20)
            lines.extend(stderr_tail or ["<empty>"])
            lines.append("```")
        else:
            lines.append("stderr log not found.")
        lines.append("")

    os.makedirs(os.path.dirname(args.summary_path), exist_ok=True)
    with open(args.summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote summary to {args.summary_path}")


if __name__ == "__main__":
    main()
