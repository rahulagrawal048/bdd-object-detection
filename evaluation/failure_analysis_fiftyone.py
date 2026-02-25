#!/usr/bin/env python3
"""
Failure analysis with FiftyOne: load BDD valid + RF-DETR predictions, run evaluation,
then open the App to inspect false positives, false negatives, and grouped failures.

Usage:
  python -m evaluation.failure_analysis_fiftyone --checkpoint runs/rf_detr/bdd_1ep/checkpoint_best_total.pth
  python -m evaluation.failure_analysis_fiftyone --checkpoint ... --subset 500 --view fp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.config import DETECTION_CLASSES


def _xyxy_to_rel_bbox(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> list[float]:
    """Convert absolute xyxy to FiftyOne normalized [x_min, y_min, width, height] in [0,1]."""
    return [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]


def load_gt_and_image_paths(valid_dir: Path, subset: int | None) -> list[tuple[Path, list, int, int]]:
    """Load COCO valid: list of (image_path, list of (x1,y1,x2,y2, category_id), width, height)."""
    json_path = valid_dir / "_annotations.coco.json"
    with open(json_path) as f:
        coco = json.load(f)
    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        if iid not in anns_by_img:
            anns_by_img[iid] = []
        x, y, w, h = ann["bbox"]
        anns_by_img[iid].append((x, y, x + w, y + h, ann["category_id"]))

    out = []
    for im in coco["images"]:
        path = valid_dir / im["file_name"]
        if not path.exists():
            continue
        w = im.get("width")
        h = im.get("height")
        if w is None or h is None:
            try:
                from PIL import Image
                with Image.open(path) as img:
                    w, h = img.size
            except Exception:
                continue
        anns = anns_by_img.get(im["id"], [])
        out.append((path, anns, w, h))
    if subset is not None:
        out = out[:subset]
    return out


def run_rfdetr_predictions(checkpoint: Path, image_paths: list[Path], threshold: float = 0.25) -> list[list[tuple]]:
    """Run RF-DETR on each image. Returns list of list of (x1,y1,x2,y2, class_id_1based, confidence)."""
    from PIL import Image
    from rfdetr import RFDETRNano
    model = RFDETRNano(pretrain_weights=str(checkpoint))
    model.optimize_for_inference()
    results_list = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        detections = model.predict(image, threshold=threshold)
        _xyxy = getattr(detections, "xyxy", None)
        _boxes = getattr(detections, "boxes", None)
        xyxy = _xyxy if _xyxy is not None else _boxes
        if xyxy is None:
            results_list.append([])
            continue
        xyxy = np.asarray(xyxy)
        class_id = getattr(detections, "class_id", None)
        if class_id is None:
            class_id = np.zeros(len(xyxy), dtype=np.int64)
        else:
            class_id = np.asarray(class_id)
            if class_id.size and class_id.max() <= 9:
                class_id = class_id + 1
        _conf = getattr(detections, "confidence", None)
        _scores = getattr(detections, "scores", None)
        confidence = _conf if _conf is not None else _scores
        if confidence is None:
            confidence = np.ones(len(class_id))
        else:
            confidence = np.asarray(confidence)
        preds = [
            (float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3]), int(class_id[i]), float(confidence[i]))
            for i in range(len(xyxy))
        ]
        results_list.append(preds)
    return results_list


def main() -> int:
    parser = argparse.ArgumentParser(description="Failure analysis with FiftyOne (BDD valid + RF-DETR)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="RF-DETR checkpoint (.pth)")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "datasets" / "bdd_coco", help="COCO dataset root (valid/ inside)")
    parser.add_argument("--subset", type=int, default=None, help="Use first N images (faster)")
    parser.add_argument("--view", type=str, default=None, choices=("fp", "fn", "all"),
                        help="Open App view: fp=most FPs first, fn=most FNs first, all=grouped by failure type")
    parser.add_argument("--dataset-name", type=str, default="bdd-failure-analysis", help="FiftyOne dataset name")
    parser.add_argument("--no-launch", action="store_true", help="Build dataset and run eval but do not launch App")
    parser.add_argument("--threshold", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    valid_dir = args.data / "valid"
    if not valid_dir.exists():
        print(f"Error: valid dir not found: {valid_dir}", file=sys.stderr)
        return 1

    try:
        import fiftyone as fo
        from fiftyone import ViewField as F
    except ImportError:
        print("Error: fiftyone not installed. Run: pip install fiftyone", file=sys.stderr)
        return 1

    print("Loading ground truth...")
    gt_list = load_gt_and_image_paths(valid_dir, args.subset)
    print(f"  {len(gt_list)} images")

    print("Running RF-DETR predictions...")
    preds_per_image = run_rfdetr_predictions(args.checkpoint, [p for p, _, _, _ in gt_list], args.threshold)

    print("Building FiftyOne dataset...")
    # Reuse or create dataset
    if args.dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(args.dataset_name)
        dataset.delete_samples(dataset)
    else:
        dataset = fo.Dataset(args.dataset_name, overwrite=True)

    samples = []
    for (path, anns, w, h), preds in zip(gt_list, preds_per_image):
        gt_dets = []
        for (x1, y1, x2, y2, cat_id) in anns:
            if cat_id < 1 or cat_id > len(DETECTION_CLASSES):
                continue
            label = DETECTION_CLASSES[cat_id - 1]
            bbox = _xyxy_to_rel_bbox(x1, y1, x2, y2, w, h)
            gt_dets.append(fo.Detection(label=label, bounding_box=bbox))

        pred_dets = []
        for (x1, y1, x2, y2, cat_id, conf) in preds:
            if cat_id < 1 or cat_id > len(DETECTION_CLASSES):
                continue
            label = DETECTION_CLASSES[cat_id - 1]
            bbox = _xyxy_to_rel_bbox(x1, y1, x2, y2, w, h)
            pred_dets.append(fo.Detection(label=label, bounding_box=bbox, confidence=conf))

        samples.append(
            fo.Sample(
                filepath=str(path.resolve()),
                ground_truth=fo.Detections(detections=gt_dets),
                predictions=fo.Detections(detections=pred_dets),
            )
        )

    dataset.add_samples(samples)
    print(f"  Added {len(samples)} samples")

    print("Running evaluate_detections (IoU 0.5)...")
    results = dataset.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key="eval",
        iou_thresh=0.5,
        classwise=False,
    )
    results.print_report(classes=DETECTION_CLASSES)

    # Tag each sample with a failure group for grouping in the App
    print("Tagging failure groups...")
    for sample in dataset:
        fp = getattr(sample, "eval_fp", None) or 0
        fn = getattr(sample, "eval_fn", None) or 0
        if fp > 0 and fn > 0:
            sample["failure_group"] = "both"
        elif fp > 0:
            sample["failure_group"] = "fp_only"
        elif fn > 0:
            sample["failure_group"] = "fn_only"
        else:
            sample["failure_group"] = "clean"
        sample.save()

    if args.no_launch:
        print("Dataset and evaluation ready. Run without --no-launch to open the App.")
        return 0

    # View sorted by eval_fp (most false positives first) for inspection
    high_conf_view = dataset
    if args.view == "fp":
        view = dataset.sort_by("eval_fp", reverse=True).filter_labels("predictions", F("eval") == "fp")
        print("Opening App: view sorted by most false positives (only FP boxes shown)")
        session = fo.launch_app(view=view)
    elif args.view == "fn":
        view = dataset.sort_by("eval_fn", reverse=True).filter_labels("ground_truth", F("eval") == "fn")
        print("Opening App: view sorted by most false negatives (only FN boxes shown)")
        session = fo.launch_app(view=view)
    else:
        print("Opening App: grouped by failure type (clean / fp_only / fn_only / both)")
        grouped_view = dataset.group_by("failure_group")
        session = fo.launch_app(view=grouped_view)
    # Apply sort by eval_fp so samples with most FPs appear first
    session.view = high_conf_view.sort_by("eval_fp", reverse=True)
    print("Close the App window or press Ctrl+C when done.")
    session.wait()

    return 0


if __name__ == "__main__":
    sys.exit(main())
