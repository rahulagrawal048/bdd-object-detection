#!/usr/bin/env python3
"""
CLI to evaluate a trained model on BDD validation set.

Usage:
  python -m evaluation.run_eval --checkpoint checkpoints/subset1ep.pt --bdd-root /path/to/BDD
  python -m evaluation.run_eval --bdd-root /path/to/BDD --subset 200 --save-viz --out-dir eval_out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.config import DEFAULT_BDD_ROOT
from data.dataset import BDDDataset, collate_fn, get_default_transform
from data.parser import image_path_for
from training_faster_rcnn.model import get_model
from evaluation.evaluate import run_evaluation
from evaluation.visualize import (
    plot_metrics,
    plot_pr_curves,
    save_qualitative_viz,
    cluster_failures,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate detection model on BDD val set")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to model checkpoint")
    parser.add_argument("--bdd-root", type=Path, default=DEFAULT_BDD_ROOT, help="BDD dataset root")
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"))
    parser.add_argument("--subset", type=int, default=None, help="Use first N samples (e.g. 500)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--out-dir", type=Path, default=Path("eval_output"))
    parser.add_argument("--save-viz", action="store_true", help="Save GT vs pred images")
    parser.add_argument("--max-viz", type=int, default=50)
    parser.add_argument("--no-pretrained", action="store_true", help="Model has no pretrained backbone")
    args = parser.parse_args()

    if not args.bdd_root.exists():
        print(f"Error: BDD root not found: {args.bdd_root}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = get_default_transform(max_size=1333, min_size=800)
    dataset = BDDDataset(
        args.bdd_root,
        split=args.split,
        subset=args.subset,
        transform=transform,
        require_detections=True,
    )
    if len(dataset) == 0:
        print("Error: No samples in dataset.", file=sys.stderr)
        return 1

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = get_model(num_classes=11, pretrained=not args.no_pretrained)
    if args.checkpoint and args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)
        print(f"Loaded checkpoint: {args.checkpoint}")
    model.to(device)

    print("Running evaluation...")
    metrics, all_preds, all_targets = run_evaluation(
        model, loader, device, score_thresh=args.score_thresh, num_classes=10
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Metrics ---")
    print(f"mAP@0.5:     {metrics['mAP_50']:.4f}")
    print(f"mAP@0.75:    {metrics['mAP_75']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['mAP_50_95']:.4f}")
    print("\nPer-class AP@0.5:")
    for name, ap in metrics["per_class_AP50"].items():
        print(f"  {name}: {ap:.4f}")

    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "mAP_50": metrics["mAP_50"],
                "mAP_75": metrics["mAP_75"],
                "mAP_50_95": metrics["mAP_50_95"],
                "per_class_AP50": {k: float(v) if v == v else None for k, v in metrics["per_class_AP50"].items()},
            },
            f,
            indent=2,
        )

    plot_metrics(metrics, save_path=args.out_dir / "metrics_plot.png")
    plot_pr_curves(metrics, save_path=args.out_dir / "pr_curve.png")
    print(f"\nSaved metrics and plot to {args.out_dir}")

    failure_clusters = cluster_failures(metrics)
    with open(args.out_dir / "failure_clusters.json", "w") as f:
        json.dump(
            {
                "worst_AP_classes": failure_clusters["worst_AP_classes"],
                "best_AP_classes": failure_clusters["best_AP_classes"],
            },
            f,
            indent=2,
        )

    if args.save_viz and len(dataset.frame_labels) > 0:
        image_paths = [
            image_path_for(args.bdd_root, fl.name, args.split)
            for fl in dataset.frame_labels
        ]
        saved = save_qualitative_viz(
            image_paths,
            all_preds,
            all_targets,
            args.out_dir / "qualitative",
            max_images=args.max_viz,
        )
        print(f"Saved {len(saved)} qualitative visualizations to {args.out_dir / 'qualitative'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
