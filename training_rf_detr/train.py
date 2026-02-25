#!/usr/bin/env python3
"""
Train RF-DETR on BDD100K (COCO format).

Usage:
  # 1. Prepare COCO-format data (one-time)
  python -m training_rf_detr.prepare_dataset --bdd-root /path/to/BDD

  # 2. Train (use conda env with rfdetr; run in a normal terminal, not sandbox)
  conda activate bdd-rfdetr
  export PYTHONNOUSERSITE=1
  python -m training_rf_detr.train --data datasets/bdd_coco --epochs 1

  # Effective batch 32 (e.g. for comparison)
  python -m training_rf_detr.train --data datasets/bdd_coco --epochs 1 --batch-size 16 --grad-accum-steps 2

  # With smaller batch for 24GB GPU
  python -m training_rf_detr.train --data datasets/bdd_coco --epochs 1 --batch-size 4 --grad-accum-steps 4

Validation runs each epoch (COCO mAP printed in the log). TensorBoard logs loss (pip install "rfdetr[metrics]").
After training, static plots are saved in output_dir: train_loss.png, val_map.png, train_loss_vs_val_ap.png.
  tensorboard --logdir <output-dir>
  python -m training_rf_detr.plot_training --logdir <output-dir> --epochs 5  # if not run automatically
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RF-DETR on BDD100K (COCO format)")
    parser.add_argument(
        "--data", type=Path,
        default=REPO_ROOT / "datasets" / "bdd_coco",
        help="Path to COCO-format dataset (train/ and valid/ with _annotations.coco.json)",
    )
    parser.add_argument(
        "--model", type=str, default="nano",
        choices=("nano", "small", "medium", "base", "large"),
        help="Model size (nano fits 24GB well)",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Effective batch = batch_size * grad_accum_steps (recommend 16)")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "rf_detr" / "bdd_1ep")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-plots", action="store_true", help="Skip generating loss/AP plots after training")
    args = parser.parse_args()

    try:
        from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge
    except ImportError as e:
        print(
            f"Error importing rfdetr: {e}\n"
            "Tip: unset PYTHONPATH and use your conda env python.\n"
            "Example: unset PYTHONPATH && /home/rahul/anaconda3/envs/bdd-rfdetr/bin/python -m pip install -U rfdetr",
            file=sys.stderr,
        )
        return 1

    dataset_dir = args.data.resolve()
    if not dataset_dir.exists():
        print(f"Error: dataset not found: {dataset_dir}", file=sys.stderr)
        print("Run: python -m training_rf_detr.prepare_dataset --bdd-root /path/to/BDD", file=sys.stderr)
        return 1

    train_json = dataset_dir / "train" / "_annotations.coco.json"
    valid_dir = dataset_dir / "valid"
    if not train_json.exists():
        print(f"Error: COCO annotations not found: {train_json}", file=sys.stderr)
        return 1
    if not valid_dir.exists():
        print(f"Error: valid/ not found in {dataset_dir}", file=sys.stderr)
        return 1

    model_map = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "base": "RFDETRBase",
        "large": "RFDETRLarge",
    }
    model_class_name = model_map[args.model]

    print(f"Training RF-DETR {args.model}: dataset={dataset_dir}, epochs={args.epochs}, batch_size={args.batch_size}, grad_accum_steps={args.grad_accum_steps}")
    print(f"Output: {args.output_dir}")

    model_class = {
        "RFDETRNano": RFDETRNano,
        "RFDETRSmall": RFDETRSmall,
        "RFDETRMedium": RFDETRMedium,
        "RFDETRBase": RFDETRBase,
        "RFDETRLarge": RFDETRLarge,
    }[model_class_name]

    model = model_class()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model.train(
        dataset_dir=str(dataset_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        output_dir=str(args.output_dir),
        run_test=False,
    )

    print(f"\nTraining complete. Best checkpoint: {args.output_dir}/checkpoint_best_total.pth")
    print(f"TensorBoard: tensorboard --logdir {args.output_dir}")

    if not args.no_plots:
        try:
            from training_rf_detr.plot_training import run_plots
            if run_plots(args.output_dir, epochs=args.epochs) == 0:
                print("Saved plots: train_loss.png, val_map.png, train_loss_vs_val_ap.png")
        except Exception as e:
            print(f"Could not generate plots: {e}. Run: python -m training_rf_detr.plot_training --logdir {args.output_dir} --epochs {args.epochs}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
