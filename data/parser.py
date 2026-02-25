"""
Parser and data structures for BDD100K object detection labels.
Reads JSON labels and exposes only detection annotations (bounding boxes);
ignores drivable area, lane marking, and other non-detection annotations.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator

from data.config import DEFAULT_BDD_ROOT, DETECTION_CLASSES, CATEGORY_ALIASES, SPLITS


@dataclass
class BoundingBox:
    """
    2D bounding box in image coordinates.

    Attributes:
        x1: Left coordinate.
        y1: Top coordinate.
        x2: Right coordinate.
        y2: Bottom coordinate.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Return box width (x2 - x1), at least 0."""
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Return box height (y2 - y1), at least 0."""
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Return box area (width * height)."""
        return self.width * self.height


@dataclass
class Detection:
    """
    Single object detection with bounding box and optional attributes.

    Attributes:
        category: Canonical class name from DETECTION_CLASSES.
        box: 2D bounding box.
        occluded: Whether the object is occluded.
        truncated: Whether the box is truncated at image boundary.
        traffic_light_color: For traffic lights: red, yellow, green, or None.
    """

    category: str  # canonical name from DETECTION_CLASSES
    box: BoundingBox
    occluded: bool = False
    truncated: bool = False
    traffic_light_color: str | None = None  # red, yellow, green, none


@dataclass
class FrameLabels:
    """
    Labels for one image (one frame).

    Attributes:
        name: Image base name without extension.
        split: Dataset split ('train' or 'val').
        detections: List of Detection objects for this frame.
        attributes: Frame-level attributes (e.g. weather, scene, timeofday).
    """

    name: str
    split: str
    detections: list[Detection] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)

    @property
    def num_detections(self) -> int:
        """Return the number of detection instances in this frame."""
        return len(self.detections)

    def category_counts(self) -> dict[str, int]:
        """Return per-category count of detections in this frame."""
        counts: dict[str, int] = {}
        for d in self.detections:
            counts[d.category] = counts.get(d.category, 0) + 1
        return counts


def _normalize_category(raw: str) -> str | None:
    """
    Map raw JSON category to canonical detection class.

    Args:
        raw: Category string from BDD JSON (e.g. 'person', 'car', 'area/drivable').

    Returns:
        Canonical class name from DETECTION_CLASSES, or None if not a detection class
        (e.g. area/, lane/ prefixes).
    """
    raw_lower = raw.strip().lower()
    if raw_lower.startswith("area/") or raw_lower.startswith("lane/"):
        return None
    return CATEGORY_ALIASES.get(raw_lower)


def parse_label_file(path: Path, split: str) -> FrameLabels | None:
    """
    Parse a single BDD label JSON file.

    Only objects with 'box2d' and a category in DETECTION_CLASSES are included.
    Objects with poly2d or categories under area/ or lane/ are ignored.

    Args:
        path: Path to the JSON label file.
        split: Split name ('train' or 'val') for the returned FrameLabels.

    Returns:
        FrameLabels for the frame, or None on read/parse error.
    """
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    name = data.get("name", path.stem)
    frames = data.get("frames", [])
    if not frames:
        return FrameLabels(name=name, split=split)

    frame = frames[0]
    objects = frame.get("objects", [])
    attributes = frame.get("attributes", {})

    detections: list[Detection] = []
    for obj in objects:
        if "box2d" not in obj:
            continue
        raw_cat = obj.get("category", "")
        cat = _normalize_category(raw_cat)
        if cat is None:
            continue
        box2d = obj["box2d"]
        attrs = obj.get("attributes", {})
        box = BoundingBox(
            x1=float(box2d["x1"]),
            y1=float(box2d["y1"]),
            x2=float(box2d["x2"]),
            y2=float(box2d["y2"]),
        )
        detections.append(
            Detection(
                category=cat,
                box=box,
                occluded=attrs.get("occluded", False),
                truncated=attrs.get("truncated", False),
                traffic_light_color=attrs.get("trafficLightColor") or None,
            )
        )

    return FrameLabels(name=name, split=split, detections=detections, attributes=attributes)


def iter_label_files(
    bdd_root: Path, splits: tuple[str, ...] = SPLITS
) -> Iterator[tuple[str, Path]]:
    """
    Yield (split, label_path) for each label file in the given splits.

    Args:
        bdd_root: Root directory containing labels/<split>/.
        splits: Tuple of split names (e.g. ('train', 'val')).

    Yields:
        (split_name, path_to_json) for each JSON file.
    """
    for split in splits:
        labels_dir = bdd_root / "labels" / split
        if not labels_dir.is_dir():
            continue
        for path in sorted(labels_dir.glob("*.json")):
            yield split, path


def _parse_one(path_and_split: tuple[Path, str]) -> FrameLabels | None:
    """Worker for parallel load: parse one file. Must be top-level for pickling."""
    path, split = path_and_split
    return parse_label_file(path, split)


def load_split(
    bdd_root: Path,
    split: str,
    max_frames: int | None = None,
    workers: int = 0,
) -> list[FrameLabels]:
    """
    Load frame labels for one split.

    Args:
        bdd_root: BDD dataset root.
        split: Split name ('train' or 'val').
        max_frames: If set, load at most this many frames (for quick preview).
        workers: If > 0, use this many processes to load in parallel (faster for large splits).

    Returns:
        List of FrameLabels, one per JSON file in labels/<split>/.
    """
    labels_dir = bdd_root / "labels" / split
    if not labels_dir.is_dir():
        return []
    paths = sorted(labels_dir.glob("*.json"))
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        return []

    if workers <= 0:
        try:
            from tqdm import tqdm
            path_iter = tqdm(paths, desc=f"Labels {split}", unit="file")
        except ImportError:
            path_iter = paths
        out: list[FrameLabels] = []
        for path in path_iter:
            fl = parse_label_file(path, split)
            if fl is not None:
                out.append(fl)
        return out

    tasks = [(p, split) for p in paths]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        chunksize = min(500, max(1, len(tasks) // (workers * 4)))
        results = list(executor.map(_parse_one, tasks, chunksize=chunksize))
    return [fl for fl in results if fl is not None]


def load_all(
    bdd_root: Path,
    splits: tuple[str, ...] = SPLITS,
    max_frames_per_split: int | None = None,
    workers: int = 0,
) -> list[FrameLabels]:
    """
    Load frame labels for the given splits.

    Args:
        bdd_root: BDD dataset root.
        splits: Tuple of split names (default: train, val).
        max_frames_per_split: If set, load at most this many frames per split (for quick preview).
        workers: If > 0, use this many processes per split for parallel load (speeds up full dataset).

    Returns:
        List of FrameLabels from all splits.
    """
    out: list[FrameLabels] = []
    try:
        from tqdm import tqdm
        split_iter = tqdm(splits, desc="Loading labels", unit="split")
    except ImportError:
        split_iter = splits
    for split in split_iter:
        out.extend(
            load_split(
                bdd_root, split,
                max_frames=max_frames_per_split,
                workers=workers,
            )
        )
    return out


def image_path_for(bdd_root: Path, name: str, split: str, ext: str = "jpg") -> Path:
    """
    Return path to image file for a given label name and split.

    Args:
        bdd_root: BDD dataset root.
        name: Image base name (no extension).
        split: Split name ('train' or 'val').
        ext: Image file extension (default 'jpg').

    Returns:
        Path to the image file (may not exist).
    """
    return bdd_root / "images" / split / f"{name}.{ext}"
