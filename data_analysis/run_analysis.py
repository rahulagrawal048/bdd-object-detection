#!/usr/bin/env python3
"""
CLI entrypoint for BDD object detection dataset analysis.

Usage:
    python -m data_analysis.run_analysis [--bdd-root /path/to/BDD] [--plots] [--out-dir ./output]
"""
import argparse
from pathlib import Path

from data.config import DEFAULT_BDD_ROOT
from data_analysis.analysis import run_full_analysis
from data_analysis.visualize import generate_all_plots


def main() -> int:
    """
    Run full analysis and optionally generate plots; print summary to stdout.

    Returns:
        0 on success, 1 if BDD root does not exist.
    """
    p = argparse.ArgumentParser(description="BDD object detection dataset analysis")
    p.add_argument(
        "--bdd-root", type=Path, default=DEFAULT_BDD_ROOT, help="Path to BDD dataset root"
    )
    p.add_argument("--plots", action="store_true", help="Generate static plots")
    p.add_argument(
        "--out-dir", type=Path, default=Path("output"), help="Output directory for plots"
    )
    args = p.parse_args()

    if not args.bdd_root.exists():
        print(f"Error: BDD root not found: {args.bdd_root}")
        return 1

    print("Running full analysis...")
    results = run_full_analysis(args.bdd_root)
    tv = results["train_val_analysis"]
    a = results["anomalies"]

    print("\n--- Overview ---")
    print(f"Train images: {tv['n_train_images']}, instances: {tv['n_train_instances']}")
    print(f"Val images:   {tv['n_val_images']}, instances: {tv['n_val_instances']}")
    print(f"Images with zero detections: {a.get('n_empty_images', 0)}")

    print("\n--- Class distribution (instances) ---")
    print(results["class_distribution"].to_string(index=False))

    print("\n--- Size stats (area px² by class) ---")
    print(results["size_stats"].to_string(index=False))

    print("\n--- Aspect ratio by class ---")
    print(results["aspect_ratio_stats"].to_string(index=False))

    print("\n--- Anomalies ---")
    print(f"Tiny boxes (area<100): {a.get('n_tiny_boxes', 0)}")
    print(f"Huge boxes (>50% frame): {a.get('n_huge_boxes', 0)}")

    if args.plots:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating plots in {args.out_dir}...")
        generate_all_plots(results, args.out_dir)
        print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
