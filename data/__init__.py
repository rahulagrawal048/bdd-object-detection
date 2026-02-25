"""
BDD data loading, parsing, analysis, and dataset utilities.
"""
from data.config import (
    DEFAULT_BDD_ROOT,
    DETECTION_CLASSES,
    CATEGORY_ALIASES,
    SPLITS,
)
from data.parser import (
    BoundingBox,
    Detection,
    FrameLabels,
    parse_label_file,
    load_split,
    load_all,
    image_path_for,
)

try:
    from data.dataset import BDDDataset, collate_fn, get_default_transform
except ImportError:
    BDDDataset = None  # type: ignore[misc, assignment]
    collate_fn = None  # type: ignore[misc, assignment]
    get_default_transform = None  # type: ignore[misc, assignment]

__all__ = [
    "DEFAULT_BDD_ROOT",
    "DETECTION_CLASSES",
    "CATEGORY_ALIASES",
    "SPLITS",
    "BoundingBox",
    "Detection",
    "FrameLabels",
    "parse_label_file",
    "load_split",
    "load_all",
    "image_path_for",
    "BDDDataset",
    "collate_fn",
    "get_default_transform",
]
