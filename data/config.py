"""
Configuration for BDD dataset paths and detection class definitions.

Constants:
    DEFAULT_BDD_ROOT: Default path to BDD dataset root (images/ and labels/).
    DETECTION_CLASSES: List of 10 canonical detection class names.
    CATEGORY_ALIASES: Mapping from raw JSON category strings to canonical names.
    SPLITS: Tuple of splits to analyze (train, val).
"""
import os
from pathlib import Path

# Default BDD root: use BDD_ROOT env if set (e.g. /data in Docker), else fallback
DEFAULT_BDD_ROOT = Path(os.environ.get("BDD_ROOT", "/media/rahul/Expansion/BDD"))

# 10 object detection classes (BDD100K detection task)
# Official order: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign
DETECTION_CLASSES = [
    "pedestrian",  # 0 - mapped from "person" in JSON
    "rider",  # 1
    "car",  # 2
    "truck",  # 3
    "bus",  # 4
    "train",  # 5
    "motorcycle",  # 6
    "bicycle",  # 7 - mapped from "bike" in JSON
    "traffic light",  # 8
    "traffic sign",  # 9
]

# Normalize raw category names from JSON to canonical class names
CATEGORY_ALIASES = {
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign",
}

# Splits to analyze
SPLITS = ("train", "val")
