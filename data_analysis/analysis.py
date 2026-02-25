"""
BDD object detection dataset analysis pipeline.

Builds record-level and image-level tables, class distribution, train/val summary,
anomalies, interesting samples, and size/aspect-ratio stats.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

from data.config import DETECTION_CLASSES, SPLITS
from data.parser import load_all, image_path_for, FrameLabels

# Parallel label loading is opt-in. On external drives with many small files,
# multiple workers can be slower due to disk seek thrashing.
DEFAULT_LOAD_WORKERS = int(os.environ.get("BDD_LOAD_WORKERS", "0"))

CACHE_FILENAME = ".bdd_analysis_cache.pkl"

# Reference frame size for "huge box" threshold (50% area)
REF_FRAME_W = 1280
REF_FRAME_H = 720
REF_FRAME_AREA = REF_FRAME_W * REF_FRAME_H
HUGE_AREA_THRESH = REF_FRAME_AREA * 0.5
TINY_AREA_THRESH = 100


RECORD_COLUMNS = [
    "split", "image_name", "category", "x1", "y1", "x2", "y2",
    "area", "width", "height", "aspect_ratio", "occluded", "truncated",
    "traffic_light_color",
]


def build_records(bdd_root: Path, frames: list[FrameLabels]) -> pd.DataFrame:
    """
    Build record-level DataFrame: one row per detection.

    Columns: split, image_name, category, x1, y1, x2, y2, area, width, height,
    aspect_ratio, occluded, truncated, traffic_light_color.
    """
    rows: list[dict] = []
    for fl in frames:
        for d in fl.detections:
            w = d.box.width
            h = d.box.height
            area = d.box.area
            aspect = w / h if h > 0 else 0.0
            rows.append({
                "split": fl.split,
                "image_name": fl.name,
                "category": d.category,
                "x1": d.box.x1,
                "y1": d.box.y1,
                "x2": d.box.x2,
                "y2": d.box.y2,
                "area": area,
                "width": w,
                "height": h,
                "aspect_ratio": aspect,
                "occluded": d.occluded,
                "truncated": d.truncated,
                "traffic_light_color": d.traffic_light_color,
            })
    if not rows:
        return pd.DataFrame(columns=RECORD_COLUMNS)
    return pd.DataFrame(rows)


IMAGE_STATS_COLUMNS = ["split", "image_name", "num_objects", "categories"]


def build_image_level_stats(frames: list[FrameLabels]) -> pd.DataFrame:
    """Build image-level DataFrame: one row per image."""
    rows = []
    for fl in frames:
        cats = list(fl.category_counts().keys())
        rows.append({
            "split": fl.split,
            "image_name": fl.name,
            "num_objects": fl.num_detections,
            "categories": ",".join(sorted(cats)) if cats else "",
        })
    if not rows:
        return pd.DataFrame(columns=IMAGE_STATS_COLUMNS)
    return pd.DataFrame(rows)


def class_distribution(records: pd.DataFrame) -> pd.DataFrame:
    """Per-class per-split instance count and image count."""
    if records.empty:
        return pd.DataFrame(columns=["category", "split", "instance_count", "image_count"])
    inst = records.groupby(["category", "split"]).size().reset_index(name="instance_count")
    images = records.groupby(["category", "split"])["image_name"].nunique().reset_index(name="image_count")
    merged = inst.merge(images, on=["category", "split"])
    return merged


def train_val_split_analysis(records: pd.DataFrame, image_stats: pd.DataFrame) -> dict:
    """Train vs val image and instance counts."""
    train_rec = records[records["split"] == "train"]
    val_rec = records[records["split"] == "val"]
    train_im = image_stats[image_stats["split"] == "train"]
    val_im = image_stats[image_stats["split"] == "val"]
    return {
        "n_train_images": len(train_im),
        "n_val_images": len(val_im),
        "n_train_instances": len(train_rec),
        "n_val_instances": len(val_rec),
    }


def anomaly_detection(records: pd.DataFrame, image_stats: pd.DataFrame) -> dict:
    """Empty images, tiny/huge boxes, occlusion/truncation rates."""
    anomalies: dict = {}
    # Empty images
    empty = image_stats[image_stats["num_objects"] == 0]
    anomalies["n_empty_images"] = len(empty)
    anomalies["empty_images"] = empty[["split", "image_name"]].to_dict("records")
    # Tiny / huge boxes
    tiny = records[records["area"] < TINY_AREA_THRESH]
    huge = records[records["area"] > HUGE_AREA_THRESH]
    anomalies["n_tiny_boxes"] = int(tiny["area"].count())
    anomalies["n_huge_boxes"] = int(huge["area"].count())
    anomalies["tiny_boxes"] = tiny.groupby("category").size().to_dict()
    anomalies["huge_boxes"] = huge.groupby("category").size().to_dict()
    # Occlusion / truncation by class
    if not records.empty:
        by_cat = records.groupby("category").agg(
            total=("category", "count"),
            occluded=("occluded", "sum"),
            truncated=("truncated", "sum"),
        )
        rates = {}
        for cat in by_cat.index:
            row = by_cat.loc[cat]
            total = int(row["total"])
            rates[cat] = {
                "occluded_pct": round(100 * row["occluded"] / total, 2),
                "truncated_pct": round(100 * row["truncated"] / total, 2),
            }
        anomalies["occlusion_truncation_rates"] = rates
    else:
        anomalies["occlusion_truncation_rates"] = {}
    return anomalies


def _all_boxes_for_image(records: pd.DataFrame, image_name: str) -> list[dict]:
    """Return all detection boxes for a given image, across all classes."""
    img_rows = records[records["image_name"] == image_name]
    return [
        {
            "x1": float(row["x1"]), "y1": float(row["y1"]),
            "x2": float(row["x2"]), "y2": float(row["y2"]),
            "category": row["category"],
        }
        for _, row in img_rows.iterrows()
    ]


def find_interesting_samples(
    frames: list[FrameLabels],
    records: pd.DataFrame,
    bdd_root: Path,
) -> dict[str, list[dict]]:
    """Per-class samples: most instances, smallest box, largest box.

    Each sample includes ALL boxes for the image (all classes) so the
    dashboard can draw them with labels for full context.
    """
    interesting: dict[str, list[dict]] = {c: [] for c in DETECTION_CLASSES}
    if records.empty:
        return interesting
    for cat in DETECTION_CLASSES:
        sub = records[records["category"] == cat]
        if sub.empty:
            continue
        # Most instances in one image
        by_img = sub.groupby("image_name").size()
        if not by_img.empty:
            top_img = by_img.idxmax()
            count = int(by_img.max())
            split = sub[sub["image_name"] == top_img].iloc[0]["split"]
            img_path = image_path_for(bdd_root, top_img, split)
            interesting[cat].append({
                "image_name": top_img,
                "split": split,
                "reason": f"Most instances ({count})",
                "image_path": str(img_path),
                "count": count,
                "boxes": _all_boxes_for_image(records, top_img),
            })
        # Smallest / largest by area – skip degenerate tiny boxes (label noise)
        min_area = max(sub["area"].quantile(0.05), TINY_AREA_THRESH)
        sub_valid = sub[sub["area"] >= min_area]
        if not sub_valid.empty:
            for label, idx in enumerate([sub_valid["area"].idxmin(), sub_valid["area"].idxmax()]):
                r = sub.loc[idx]
                reason = "Smallest box (by area)" if label == 0 else "Largest box (by area)"
                interesting[cat].append({
                    "image_name": r["image_name"],
                    "split": r["split"],
                    "reason": reason,
                    "image_path": str(image_path_for(bdd_root, r["image_name"], r["split"])),
                    "area": float(r["area"]),
                    "boxes": _all_boxes_for_image(records, r["image_name"]),
                })
    return interesting


def size_and_aspect_stats(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-class size (area) and aspect-ratio summary stats.

    Returns:
        size_stats: category, count, area_min, area_max, area_mean, area_median, area_p25, area_p75.
        aspect_stats: category, count, aspect_min, aspect_max, aspect_mean, aspect_median, aspect_p25, aspect_p75.
    """
    if records.empty or "area" not in records.columns:
        return (
            pd.DataFrame(columns=["category", "count", "area_min", "area_max", "area_mean", "area_median", "area_p25", "area_p75"]),
            pd.DataFrame(columns=["category", "count", "aspect_min", "aspect_max", "aspect_mean", "aspect_median", "aspect_p25", "aspect_p75"]),
        )
    valid = records[records["area"] > 0].copy()
    if "aspect_ratio" not in valid.columns and "width" in valid.columns and "height" in valid.columns:
        valid["aspect_ratio"] = np.where(valid["height"] > 0, valid["width"] / valid["height"], 0.0)
    valid = valid[valid["aspect_ratio"] > 0]  # exclude degenerate for aspect stats

    def quantiles(s: pd.Series) -> dict:
        return {
            "min": s.min(),
            "max": s.max(),
            "mean": s.mean(),
            "median": s.median(),
            "p25": s.quantile(0.25),
            "p75": s.quantile(0.75),
        }

    size_rows = []
    aspect_rows = []
    for cat in DETECTION_CLASSES:
        sub = valid[valid["category"] == cat]
        if sub.empty:
            size_rows.append({"category": cat, "count": 0, "area_min": np.nan, "area_max": np.nan, "area_mean": np.nan, "area_median": np.nan, "area_p25": np.nan, "area_p75": np.nan})
            aspect_rows.append({"category": cat, "count": 0, "aspect_min": np.nan, "aspect_max": np.nan, "aspect_mean": np.nan, "aspect_median": np.nan, "aspect_p25": np.nan, "aspect_p75": np.nan})
            continue
        q_a = quantiles(sub["area"])
        size_rows.append({
            "category": cat,
            "count": len(sub),
            "area_min": round(q_a["min"], 2),
            "area_max": round(q_a["max"], 2),
            "area_mean": round(q_a["mean"], 2),
            "area_median": round(q_a["median"], 2),
            "area_p25": round(q_a["p25"], 2),
            "area_p75": round(q_a["p75"], 2),
        })
        q_r = quantiles(sub["aspect_ratio"])
        aspect_rows.append({
            "category": cat,
            "count": len(sub),
            "aspect_min": round(q_r["min"], 3),
            "aspect_max": round(q_r["max"], 3),
            "aspect_mean": round(q_r["mean"], 3),
            "aspect_median": round(q_r["median"], 3),
            "aspect_p25": round(q_r["p25"], 3),
            "aspect_p75": round(q_r["p75"], 3),
        })
    return pd.DataFrame(size_rows), pd.DataFrame(aspect_rows)


def _cache_path(bdd_root: Path, max_frames_per_split: int | None) -> Path:
    suffix = f"_sample{max_frames_per_split}" if max_frames_per_split else ""
    return bdd_root / f".bdd_analysis_cache{suffix}.pkl"


def run_full_analysis(
    bdd_root: Path,
    max_frames_per_split: int | None = None,
    use_cache: bool = True,
) -> dict:
    """
    Run full analysis pipeline and return all results.

    On first run, reads all label JSONs (slow on external drives) and caches the
    parsed results to a pickle file in bdd_root. Subsequent runs load the cache
    in seconds. Pass use_cache=False to force a fresh read.

    Args:
        bdd_root: Path to BDD dataset root.
        max_frames_per_split: If set, only load this many frames per split (faster preview).
            None = load full dataset.
        use_cache: If True, save/load parsed results to/from a pickle cache file.

    Returns:
        Dict with keys: bdd_root, records, image_stats, class_distribution,
        train_val_analysis, anomalies, interesting_samples, size_stats, aspect_ratio_stats.
    """
    bdd_root = Path(bdd_root)
    if not bdd_root.exists():
        raise FileNotFoundError(f"BDD root does not exist: {bdd_root}")
    labels_dir = bdd_root / "labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(
            f"No labels/ directory under BDD root: {bdd_root}\n"
            f"Expected {labels_dir} with train/ and val/ subdirectories."
        )

    cache_file = _cache_path(bdd_root, max_frames_per_split)
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass  # stale or corrupt cache; re-run

    pbar = None
    try:
        from tqdm import tqdm
        pbar = tqdm(total=9, desc="Analysis", unit="step", leave=True)
        def step(name: str) -> None:
            pbar.set_description_str(f"Analysis: {name}")
            pbar.update(1)
    except ImportError:
        def step(name: str) -> None:
            print(f"  {name}...", flush=True)

    workers = DEFAULT_LOAD_WORKERS if max_frames_per_split is None else 0
    step("Loading labels")
    frames = load_all(
        bdd_root, SPLITS,
        max_frames_per_split=max_frames_per_split,
        workers=workers,
    )
    step("Building records")
    records = build_records(bdd_root, frames)
    step("Image stats")
    image_stats = build_image_level_stats(frames)
    step("Class distribution")
    class_dist = class_distribution(records)
    step("Train/val")
    train_val = train_val_split_analysis(records, image_stats)
    step("Anomalies")
    anomalies = anomaly_detection(records, image_stats)
    step("Interesting samples")
    interesting = find_interesting_samples(frames, records, bdd_root)
    step("Size/aspect stats")
    size_stats, aspect_stats = size_and_aspect_stats(records)

    result = {
        "bdd_root": str(bdd_root),
        "records": records,
        "image_stats": image_stats,
        "class_distribution": class_dist,
        "train_val_analysis": train_val,
        "anomalies": anomalies,
        "interesting_samples": interesting,
        "size_stats": size_stats,
        "aspect_ratio_stats": aspect_stats,
    }

    step("Saving cache")
    if use_cache:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            pass  # read-only filesystem; skip caching

    if pbar is not None:
        pbar.close()
    return result
