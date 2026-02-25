"""
PyTorch Dataset and DataLoader utilities for BDD100K object detection.

Loads BDD images and labels using the existing parser; returns tensors and targets
in torchvision detection format (boxes in xyxy, labels 1-indexed for 10 classes).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.utils.data import Dataset

from data.config import DETECTION_CLASSES
from data.parser import load_split, image_path_for, FrameLabels

# Model uses 1-indexed labels (0 = background). Map DETECTION_CLASSES[i] -> i+1.
CLASS_NAME_TO_IDX = {name: i + 1 for i, name in enumerate(DETECTION_CLASSES)}


def _pil_loader(path: Path) -> Any:
    """Load image with PIL; used by default if no transform provided."""
    from PIL import Image

    with open(path, "rb") as f:
        img = Image.open(f).convert("RGB")
    return img


def collate_fn(batch: list[tuple[Any, dict]]) -> tuple[list[Any], list[dict]]:
    """
    Collate batch so that images stay as a list and targets stay as a list of dicts.

    Torchvision detection models expect (list of image tensors, list of target dicts)
    rather than batched tensors.

    Args:
        batch: List of (image_tensor, target_dict) from BDDDataset.

    Returns:
        (list of image tensors, list of target dicts).
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


class BDDDataset(Dataset):
    """
    PyTorch Dataset for BDD100K object detection.

    Uses the existing BDD parser to load labels; loads images from images/<split>/.
    Returns images as tensors (C, H, W) and targets as dicts with "boxes" (N, 4) xyxy
    and "labels" (N,) long, 1-indexed (1-10 for BDD classes).
    """

    def __init__(
        self,
        bdd_root: Path | str,
        split: str = "train",
        subset: Optional[int] = None,
        transform: Optional[Callable[..., tuple[Any, dict]]] = None,
        image_loader: Optional[Callable[[Path], Any]] = None,
        require_detections: bool = True,
    ):
        """
        Args:
            bdd_root: Path to BDD dataset root (images/ and labels/).
            split: 'train' or 'val'.
            subset: If set, use only the first subset samples (for quick 1-epoch runs).
            transform: Callable (image, target) -> (tensor, target). If None, only to_tensor.
            image_loader: Callable that takes path and returns image. Default: PIL RGB.
            require_detections: If True, skip frames with zero detection instances.
        """
        self.bdd_root = Path(bdd_root)
        self.split = split
        self.subset = subset
        self.transform = transform
        self.image_loader = image_loader or _pil_loader
        self.require_detections = require_detections

        frame_labels = load_split(self.bdd_root, split)
        if require_detections:
            frame_labels = [fl for fl in frame_labels if fl.num_detections > 0]
        if subset is not None:
            frame_labels = frame_labels[:subset]
        self.frame_labels = frame_labels

    def __len__(self) -> int:
        return len(self.frame_labels)

    def __getitem__(self, idx: int) -> tuple[Any, dict]:
        fl = self.frame_labels[idx]
        img_path = image_path_for(self.bdd_root, fl.name, self.split)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = self.image_loader(img_path)
        boxes_list = []
        labels_list = []

        for det in fl.detections:
            cat_idx = CLASS_NAME_TO_IDX.get(det.category)
            if cat_idx is None:
                continue
            box = det.box
            boxes_list.append([box.x1, box.y1, box.x2, box.y2])
            labels_list.append(cat_idx)

        if not boxes_list:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
            labels = torch.as_tensor(labels_list, dtype=torch.int64)
            # Filter degenerate boxes (zero or negative width/height)
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            valid = (widths > 1) & (heights > 1)
            boxes = boxes[valid]
            labels = labels[valid]

        target = {"boxes": boxes, "labels": labels}

        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            import torchvision.transforms.functional as F

            image = F.to_tensor(image)

        return image, target


def get_default_transform(max_size: int = 1333, min_size: int = 800) -> Callable:
    """
    Return a transform that resizes image (and scales boxes) then converts to tensor.

    Resizing: shorter side >= min_size, longer side <= max_size; boxes scaled accordingly.

    Args:
        max_size: Maximum length of the longer side after resize.
        min_size: Minimum length of the shorter side after resize.

    Returns:
        A callable: (PIL Image, target dict) -> (tensor, target dict with scaled boxes).
    """
    from PIL import Image
    import torchvision.transforms.functional as F

    def transform(image: Any, target: dict) -> tuple[torch.Tensor, dict]:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB") if hasattr(image, "read") else image
        w, h = image.size
        min_side = min(w, h)
        max_side = max(w, h)
        scale = min_size / min_side if min_side > 0 else 1.0
        if max_side * scale > max_size:
            scale = max_size / max_side
        nw, nh = int(w * scale), int(h * scale)
        image = image.resize((nw, nh), Image.BILINEAR)
        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= scale
            boxes[:, [1, 3]] *= scale
        target = {"boxes": boxes, "labels": target["labels"]}
        return F.to_tensor(image), target

    return transform
