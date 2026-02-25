"""
Visualization helpers for BDD analysis: static plots for class distribution,
train/val comparison, anomalies, and box area by class.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

# Optional matplotlib/seaborn for static plots
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

from data.config import DETECTION_CLASSES


def plot_class_distribution(class_dist: pd.DataFrame, save_path: Path | None = None) -> None:
    """
    Bar plot: instance count and image count per class, grouped by train/val.

    Args:
        class_dist: DataFrame with category, split, instance_count, image_count.
        save_path: If set, save figure to this path.
    """
    if not HAS_PLOT or class_dist.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ["instance_count", "image_count"]):
        pivot = (
            class_dist.pivot(index="category", columns="split", values=metric)
            .reindex(DETECTION_CLASSES)
            .fillna(0)
        )
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.legend(title="Split")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_train_val_comparison(train_val: dict, save_path: Path | None = None) -> None:
    """
    Bar chart comparing train vs val image counts and instance counts.

    Args:
        train_val: Dict with n_train_images, n_val_images, n_train_instances, n_val_instances.
        save_path: If set, save figure to this path.
    """
    if not HAS_PLOT:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    splits = ["train", "val"]
    images = [train_val["n_train_images"], train_val["n_val_images"]]
    instances = [train_val["n_train_instances"], train_val["n_val_instances"]]
    x = np.arange(len(splits))
    w = 0.35
    ax.bar(x - w / 2, images, w, label="Images")
    ax.bar(x + w / 2, instances, w, label="Detection instances")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Train vs Val: images and instances")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_anomaly_summary(
    anomalies: dict, save_path: Path | None = None
) -> None:
    """
    Bar plot of occlusion and truncation rate (%) by class.

    Args:
        anomalies: Dict with key occlusion_truncation_rates (class -> {occluded_pct, truncated_pct}).
        save_path: If set, save figure to this path.
    """
    if not HAS_PLOT:
        return
    occ = anomalies.get("occlusion_truncation_rates", {})
    if not occ:
        return
    df = pd.DataFrame(occ).T
    fig, ax = plt.subplots(figsize=(12, 5))
    df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Occlusion and truncation rate by class (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_box_area_distribution(
    records: pd.DataFrame, save_path: Path | None = None
) -> None:
    """
    Box plot of log10(area+1) per detection class.

    Args:
        records: Record-level DataFrame with category and area columns.
        save_path: If set, save figure to this path.
    """
    if not HAS_PLOT or records.empty:
        return
    df = records[records["area"] > 0].copy()
    df["area_log"] = np.log10(df["area"] + 1)
    order = [c for c in DETECTION_CLASSES if c in df["category"].values]
    if not order:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="category", y="area_log", order=order, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("log10(area + 1)")
    ax.set_title("Bounding box area distribution by class")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_size_stats(size_stats: pd.DataFrame, save_path: Path | None = None) -> None:
    """
    Bar chart of mean and median bbox area (px²) per class.

    Args:
        size_stats: DataFrame with category, area_mean, area_median (from analysis).
        save_path: If set, save figure to this path.
    """
    if not HAS_PLOT or size_stats.empty or "area_mean" not in size_stats.columns:
        return
    df = size_stats[size_stats["count"] > 0].copy()
    df = df.set_index("category").reindex([c for c in DETECTION_CLASSES if c in df.index]).dropna(how="all")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["area_mean"], w, label="Mean area")
    ax.bar(x + w / 2, df["area_median"], w, label="Median area")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha="right")
    ax.set_ylabel("Area (px²)")
    ax.set_title("Bounding box size by class (mean & median)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_aspect_ratio_distribution(
    records: pd.DataFrame, save_path: Path | None = None
) -> None:
    """
    Box plot of aspect ratio (width/height) per detection class.

    Args:
        records: Record-level DataFrame with category and aspect_ratio columns.
        save_path: If set, save figure to this path.
    """
    if not HAS_PLOT or records.empty:
        return
    df = records.copy()
    if "aspect_ratio" not in df.columns and "width" in df.columns and "height" in df.columns:
        df["aspect_ratio"] = np.where(df["height"] > 0, df["width"] / df["height"], np.nan)
    df = df[df["aspect_ratio"].notna() & (df["aspect_ratio"] > 0)]
    order = [c for c in DETECTION_CLASSES if c in df["category"].values]
    if not order:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="category", y="aspect_ratio", order=order, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Aspect ratio (width / height)")
    ax.set_title("Bounding box aspect ratio by class")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_plots(results: dict, output_dir: Path) -> None:
    """
    Generate all static analysis plots into the given directory.

    Creates: class_distribution.png, train_val_comparison.png,
    occlusion_truncation.png, box_area_by_class.png, size_stats.png,
    aspect_ratio_by_class.png.

    Args:
        results: Dict from run_full_analysis (class_distribution, train_val_analysis, etc.).
        output_dir: Directory to write PNG files into (created if needed).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(results["class_distribution"], output_dir / "class_distribution.png")
    plot_train_val_comparison(
        results["train_val_analysis"], output_dir / "train_val_comparison.png"
    )
    plot_anomaly_summary(results["anomalies"], output_dir / "occlusion_truncation.png")
    plot_box_area_distribution(results["records"], output_dir / "box_area_by_class.png")
    plot_size_stats(results["size_stats"], output_dir / "size_stats.png")
    plot_aspect_ratio_distribution(results["records"], output_dir / "aspect_ratio_by_class.png")
