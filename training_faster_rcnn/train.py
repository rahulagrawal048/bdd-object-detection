#!/usr/bin/env python3
"""
Training script for BDD object detection with Faster R-CNN.

Loads BDD via the data package, builds a Faster R-CNN (ResNet-50 FPN),
and runs training for one or more epochs. Supports a subset of data for
quick 1-epoch runs. Saves checkpoints optionally.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import time

import torch
from torch.utils.data import DataLoader

from data.config import DEFAULT_BDD_ROOT
from data.dataset import BDDDataset, collate_fn, get_default_transform
from training_faster_rcnn.model import get_model


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int = 10,
) -> float:
    """
    Run one epoch of training.

    Args:
        model: Faster R-CNN model.
        optimizer: Optimizer.
        data_loader: DataLoader yielding (list of images, list of targets).
        device: Device to run on.
        epoch: Current epoch index (for logging).
        print_freq: Log every N batches.

    Returns:
        Average loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    t0 = time.time()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        num_batches += 1

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if (batch_idx + 1) % print_freq == 0:
            elapsed = time.time() - t0
            speed = elapsed / num_batches
            eta = speed * (len(data_loader) - num_batches)
            print(
                f"  Epoch [{epoch}] Batch [{batch_idx + 1}/{len(data_loader)}] "
                f"Loss: {losses.item():.4f}  "
                f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]"
            )

    return total_loss / num_batches if num_batches else 0.0


def main() -> int:
    """Parse args, build dataset and model, run training loop."""
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN on BDD100K (subset or full)."
    )
    parser.add_argument(
        "--bdd-root",
        type=Path,
        default=DEFAULT_BDD_ROOT,
        help="Path to BDD dataset root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "val"),
        help="Split to train on.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use only first N samples (for 1-epoch demo). Example: 500.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (number of images per step).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader num_workers.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Learning rate.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1333,
        help="Max side length for resize (dataset transform).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=800,
        help="Min side length for resize (dataset transform).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Path to save checkpoint (e.g. checkpoints/subset1ep.pt).",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not use COCO-pretrained backbone.",
    )
    parser.add_argument(
        "--lr-step-epochs",
        type=int,
        nargs="*",
        default=None,
        help="Epochs at which to drop LR by 0.1 (e.g. --lr-step-epochs 3 5).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from this checkpoint (loads model, optimizer, epoch).",
    )
    args = parser.parse_args()

    if not args.bdd_root.exists():
        print(f"Error: BDD root not found: {args.bdd_root}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = get_default_transform(max_size=args.max_size, min_size=args.min_size)
    dataset = BDDDataset(
        args.bdd_root,
        split=args.split,
        subset=args.subset,
        transform=transform,
        require_detections=True,
    )
    if len(dataset) == 0:
        print("Error: No samples in dataset. Check BDD path and labels.", file=sys.stderr)
        return 1

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    # 10 BDD classes + background
    model = get_model(num_classes=11, pretrained=not args.no_pretrained)
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )

    start_epoch = 1
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.resume} (epoch {start_epoch - 1})")

    scheduler = None
    if args.lr_step_epochs:
        from torch.optim.lr_scheduler import MultiStepLR
        scheduler = MultiStepLR(optimizer, milestones=args.lr_step_epochs, gamma=0.1)
        for _ in range(1, start_epoch):
            scheduler.step()

    print(
        f"Training on {len(dataset)} samples, {len(loader)} batches/epoch, "
        f"epochs {start_epoch}..{args.epochs}, lr={optimizer.param_groups[0]['lr']}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss = train_one_epoch(
            model, optimizer, loader, device, epoch, print_freq=50
        )
        if scheduler is not None:
            scheduler.step()
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch [{epoch}] average loss: {avg_loss:.4f}  lr: {cur_lr:.6f}")
        else:
            print(f"Epoch [{epoch}] average loss: {avg_loss:.4f}")

        if args.save is not None:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "num_classes": 11,
            }
            torch.save(state, args.save)
            print(f"Checkpoint saved to {args.save} (epoch {epoch})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
