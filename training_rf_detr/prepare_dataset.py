#!/usr/bin/env python3
"""
Convert BDD100K labels to COCO format for RF-DETR training.

Creates:
  datasets/bdd_coco/
    train/
      _annotations.coco.json
      *.jpg -> symlinks to BDD images
    valid/
      _annotations.coco.json
      *.jpg -> symlinks to BDD images

RF-DETR expects train/ and valid/ (or test/) with _annotations.coco.json in each.
COCO bbox: [x, y, width, height] (top-left corner), category_id 1-based.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from data.config import DETECTION_CLASSES, CATEGORY_ALIASES

IMG_W, IMG_H = 1280, 720


def convert_split(bdd_root: Path, out_root: Path, split: str) -> int:
    """Write COCO JSON and symlink images for one split. Returns number of images."""
    labels_dir = bdd_root / "labels" / split
    images_dir = bdd_root / "images" / split
    out_split = out_root / ("valid" if split == "val" else split)
    out_split.mkdir(parents=True, exist_ok=True)

    categories = [{"id": i + 1, "name": name, "supercategory": "object"} for i, name in enumerate(DETECTION_CLASSES)]
    images_list = []
    annotations_list = []
    image_id = 0
    ann_id = 0

    for label_file in sorted(labels_dir.glob("*.json")):
        with open(label_file) as f:
            data = json.load(f)

        name = data.get("name", label_file.stem)
        img_src = images_dir / f"{name}.jpg"
        if not img_src.exists():
            continue

        image_id += 1
        file_name = f"{name}.jpg"
        img_dst = out_split / file_name
        if not img_dst.exists():
            os.symlink(img_src.resolve(), img_dst)

        images_list.append({
            "id": image_id,
            "file_name": file_name,
            "width": IMG_W,
            "height": IMG_H,
        })

        frames = data.get("frames", [])
        if not frames:
            continue
        for obj in frames[0].get("objects", []):
            if "box2d" not in obj:
                continue
            raw_cat = obj.get("category", "").strip().lower()
            if raw_cat.startswith("area/") or raw_cat.startswith("lane/"):
                continue
            canon = CATEGORY_ALIASES.get(raw_cat)
            if canon is None:
                continue
            category_id = DETECTION_CLASSES.index(canon) + 1  # 1-based for COCO

            b = obj["box2d"]
            x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
            w = x2 - x1
            h = y2 - y1
            if w <= 1 or h <= 1:
                continue

            ann_id += 1
            area = w * h
            annotations_list.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, w, h],
                "area": area,
                "iscrowd": 0,
            })

    coco = {
        "info": {"description": "BDD100K detection", "version": "1.0"},
        "licenses": [],
        "images": images_list,
        "categories": categories,
        "annotations": annotations_list,
    }
    json_path = out_split / "_annotations.coco.json"
    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)

    return len(images_list)


def main():
    parser = argparse.ArgumentParser(description="Convert BDD100K to COCO format for RF-DETR")
    parser.add_argument("--bdd-root", type=Path, default=Path("/media/rahul/Expansion/BDD"))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "datasets" / "bdd_coco")
    args = parser.parse_args()

    print(f"BDD root: {args.bdd_root}")
    print(f"Output:   {args.out}")

    for split in ("train", "val"):
        n = convert_split(args.bdd_root, args.out, split)
        out_split = "valid" if split == "val" else split
        print(f"  {out_split}: {n} images")

    print("\nDone! Train with:")
    print(f"  python -m training_rf_detr.train --data {args.out} --epochs 1")


if __name__ == "__main__":
    main()
