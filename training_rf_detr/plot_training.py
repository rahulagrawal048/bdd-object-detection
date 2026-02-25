#!/usr/bin/env python3
"""
Read TensorBoard logs from an RF-DETR training run and save static plots:
  - Train loss vs step/epoch
  - Val mAP vs epoch
  - Train loss vs Val AP (dual-axis by epoch)

Requires: pip install tensorboard matplotlib
Run after training:
  python -m training_rf_detr.plot_training --logdir runs/rf_detr/bdd_5ep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_event_dirs(logdir: Path) -> list[Path]:
    """Find directories that contain TensorBoard event files."""
    event_dirs = []
    logdir = logdir.resolve()
    if not logdir.exists():
        return []
    for p in logdir.rglob("events.out.tfevents.*"):
        event_dirs.append(p.parent)
    return sorted(set(event_dirs))


def load_scalars(logdir: Path):
    """Load scalar events from TensorBoard logdir. Returns dict tag -> list of (step, value)."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        raise ImportError("Install tensorboard: pip install tensorboard") from None

    event_dirs = find_event_dirs(logdir)
    if not event_dirs:
        return {}

    # Load first event dir that has scalars (rfdetr often writes to logdir directly or a subdir)
    scalars = {}
    for edir in event_dirs:
        ea = event_accumulator.EventAccumulator(str(edir), size_guidance={"scalars": 5000})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            if events:
                scalars[tag] = [(e.step, e.value) for e in events]
        if scalars:
            break
    return scalars


def infer_epoch_from_steps(steps: list[int], num_epochs: int | None) -> np.ndarray | None:
    """If steps look like 0,1,2,... or 0, 437, 874,..., return epoch indices (0,1,2,...)."""
    if not steps or num_epochs is None:
        return None
    steps = np.array(steps)
    if steps.max() == 0:
        return np.zeros_like(steps)
    # Assume steps are linear in epoch: epoch = step / steps_per_epoch
    steps_per_epoch = (steps.max() + 1) / max(1, num_epochs)
    return steps / max(steps_per_epoch, 1)


def run_plots(logdir: Path, epochs: int | None = None, out_dir: Path | None = None) -> int:
    """Generate training plots from TensorBoard logs. Returns 0 on success."""
    logdir = Path(logdir).resolve()
    out_dir = Path(out_dir or logdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scalars = load_scalars(logdir)
    if not scalars:
        print(f"No TensorBoard scalar events found under {logdir}", file=sys.stderr)
        print("Ensure training was run with TensorBoard logging (pip install 'rfdetr[metrics]').", file=sys.stderr)
        return 1

    num_epochs = epochs

    # Prefer common tag names (rfdetr may use different conventions)
    loss_tags = [t for t in scalars if "loss" in t.lower() and "val" not in t.lower()]
    val_ap_tags = [t for t in scalars if any(x in t.lower() for x in ("map", "map_", "ap", "val")) and "loss" not in t.lower()]
    if not loss_tags:
        loss_tags = [t for t in scalars if "loss" in t.lower()]
    if not val_ap_tags:
        val_ap_tags = [t for t in scalars if t != loss_tags[0] if loss_tags else list(scalars)[:1]]

    loss_tag = loss_tags[0] if loss_tags else None
    ap_tag = val_ap_tags[0] if val_ap_tags else None

    # --- 1. Train loss vs step ---
    if loss_tag:
        steps, values = zip(*scalars[loss_tag])
        steps, values = np.array(steps), np.array(values)
        epochs_axis = infer_epoch_from_steps(list(steps), num_epochs)
        x = epochs_axis if epochs_axis is not None else steps
        xlabel = "Epoch" if epochs_axis is not None else "Step"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, values, color="C0")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Loss")
        ax.set_title("Training loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "train_loss.png", dpi=150)
        plt.close(fig)
        print(f"Saved {out_dir / 'train_loss.png'}")

    # --- 2. Val mAP vs epoch ---
    if ap_tag:
        steps, values = zip(*scalars[ap_tag])
        steps, values = np.array(steps), np.array(values)
        # Val metrics are usually once per epoch
        epochs_axis = infer_epoch_from_steps(list(steps), num_epochs)
        x = epochs_axis if epochs_axis is not None else np.arange(len(values))
        xlabel = "Epoch" if epochs_axis is not None else "Eval index"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, values, color="C1", marker="o", markersize=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mAP")
        ax.set_title("Validation mAP")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "val_map.png", dpi=150)
        plt.close(fig)
        print(f"Saved {out_dir / 'val_map.png'}")

    # --- 3. Train loss vs Val AP (dual axis by epoch) ---
    if loss_tag and ap_tag:
        loss_steps, loss_vals = zip(*scalars[loss_tag])
        ap_steps, ap_vals = zip(*scalars[ap_tag])
        loss_steps, loss_vals = np.array(loss_steps), np.array(loss_vals)
        ap_steps, ap_vals = np.array(ap_steps), np.array(ap_vals)

        # Align by epoch: for each epoch we want one loss value (e.g. mean over epoch) and one AP value
        if num_epochs and len(ap_vals) == num_epochs:
            # Resample loss to per-epoch (average loss in that epoch)
            steps_per_epoch = len(loss_steps) / num_epochs
            loss_per_epoch = []
            for e in range(num_epochs):
                lo = int(e * steps_per_epoch)
                hi = int((e + 1) * steps_per_epoch)
                if hi > len(loss_vals):
                    hi = len(loss_vals)
                if lo < hi:
                    loss_per_epoch.append(np.mean(loss_vals[lo:hi]))
                else:
                    loss_per_epoch.append(loss_vals[-1] if loss_vals.size else 0)
            loss_per_epoch = np.array(loss_per_epoch)
            epoch_idx = np.arange(num_epochs)
            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(epoch_idx, loss_per_epoch, color="C0", label="Train loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Train loss", color="C0")
            ax1.tick_params(axis="y", labelcolor="C0")
            ax1.grid(True, alpha=0.3)
            ax2 = ax1.twinx()
            ax2.plot(epoch_idx, ap_vals, color="C1", marker="o", markersize=5, label="Val mAP")
            ax2.set_ylabel("Val mAP", color="C1")
            ax2.tick_params(axis="y", labelcolor="C1")
            fig.legend(loc="upper right", bbox_to_anchor=(1, 1))
            fig.suptitle("Train loss vs Val mAP")
            fig.tight_layout()
            fig.savefig(out_dir / "train_loss_vs_val_ap.png", dpi=150)
            plt.close(fig)
            print(f"Saved {out_dir / 'train_loss_vs_val_ap.png'}")
        else:
            # Fallback: plot loss (left) and AP (right) with their native steps
            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(loss_steps, loss_vals, color="C0", label="Train loss")
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Train loss", color="C0")
            ax1.tick_params(axis="y", labelcolor="C0")
            ax2 = ax1.twinx()
            # Scale AP steps to same x range if different
            if ap_steps.max() > loss_steps.max():
                ap_x = np.linspace(loss_steps.min(), loss_steps.max(), len(ap_vals))
            else:
                ap_x = ap_steps
            ax2.plot(ap_x, ap_vals, color="C1", marker="o", markersize=4, label="Val mAP")
            ax2.set_ylabel("Val mAP", color="C1")
            ax2.tick_params(axis="y", labelcolor="C1")
            fig.legend(loc="upper right")
            fig.suptitle("Train loss vs Val mAP")
            fig.tight_layout()
            fig.savefig(out_dir / "train_loss_vs_val_ap.png", dpi=150)
            plt.close(fig)
            print(f"Saved {out_dir / 'train_loss_vs_val_ap.png'}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot RF-DETR training from TensorBoard logs")
    parser.add_argument("--logdir", type=Path, default=REPO_ROOT / "runs" / "rf_detr" / "bdd_5ep", help="Training output dir (with TB events)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (to convert steps to epoch on x-axis)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Where to save PNGs (default: same as logdir)")
    args = parser.parse_args()
    return run_plots(args.logdir, epochs=args.epochs, out_dir=args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
