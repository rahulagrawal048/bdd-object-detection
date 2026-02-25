#!/usr/bin/env python3
"""
Evaluate a trained RF-DETR checkpoint on BDD100K valid set.
Reports COCO metrics (match training), custom mAP, and inference timing.

Usage:
  python -m evaluation.eval_rf_detr --checkpoint runs/rf_detr/bdd_1ep/checkpoint_best_total.pth
  python -m evaluation.eval_rf_detr --checkpoint ... --subset 1000 --save-viz --threshold 0.25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from data.config import DETECTION_CLASSES
from evaluation.evaluate import compute_metrics
from evaluation.visualize import plot_metrics, plot_pr_curves

NUM_CLASSES = 10


def load_coco_gt(valid_dir: Path) -> list:
    """Load valid set from _annotations.coco.json. Returns list of (path, boxes_xyxy, labels, image_id)."""
    with open(valid_dir / "_annotations.coco.json") as f:
        coco = json.load(f)
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
        anns = anns_by_img.get(im["id"], [])
        boxes = np.array([a[:4] for a in anns], dtype=np.float32) if anns else np.zeros((0, 4), dtype=np.float32)
        labels = np.array([a[4] for a in anns], dtype=np.int64) if anns else np.zeros(0, dtype=np.int64)
        out.append((path, torch.from_numpy(boxes), torch.from_numpy(labels), im["id"]))
    return out


def _parse_detections(detections) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (xyxy, class_id_1based, confidence) from rfdetr predict() output."""
    xyxy = getattr(detections, "xyxy", None)
    if xyxy is None:
        xyxy = getattr(detections, "boxes", None)
    xyxy = np.asarray(xyxy) if xyxy is not None else np.zeros((0, 4))
    class_id = getattr(detections, "class_id", None)
    if class_id is None:
        class_id = np.zeros(0, dtype=np.int64)
    else:
        class_id = np.asarray(class_id)
        if class_id.size and class_id.max() <= 9:
            class_id = class_id + 1
    conf = getattr(detections, "confidence", None)
    if conf is None:
        conf = getattr(detections, "scores", None)
    conf = np.asarray(conf) if conf is not None else np.ones(len(class_id))
    return xyxy, class_id, conf


def build_by_class_from_preds_gt(preds_per_image, gt_per_image):
    """preds_per_image: list of (boxes_xyxy, labels_1based, scores). gt_per_image: list of (boxes, labels)."""
    pred_boxes_by_class = [[] for _ in range(NUM_CLASSES)]
    pred_labels_by_class = [[] for _ in range(NUM_CLASSES)]
    pred_scores_by_class = [[] for _ in range(NUM_CLASSES)]
    gt_boxes_by_class = [[] for _ in range(NUM_CLASSES)]
    gt_labels_by_class = [[] for _ in range(NUM_CLASSES)]

    for (pboxes, plabs, pscores), (gboxes, glabs) in zip(preds_per_image, gt_per_image):
        pboxes = torch.as_tensor(pboxes, dtype=torch.float32)
        plabs = torch.as_tensor(plabs, dtype=torch.int64)
        pscores = torch.as_tensor(pscores, dtype=torch.float32)
        for c in range(NUM_CLASSES):
            c1 = c + 1
            pmask = plabs == c1
            gmask = glabs == c1
            pred_boxes_by_class[c].append(pboxes[pmask] if pmask.any() else torch.zeros(0, 4))
            pred_labels_by_class[c].append(plabs[pmask] if pmask.any() else torch.zeros(0, dtype=torch.long))
            pred_scores_by_class[c].append(pscores[pmask] if pmask.any() else torch.zeros(0))
            gt_boxes_by_class[c].append(gboxes[gmask] if gmask.any() else torch.zeros(0, 4))
            gt_labels_by_class[c].append(glabs[gmask] if gmask.any() else torch.zeros(0, dtype=torch.long))

    return (pred_boxes_by_class, pred_labels_by_class, pred_scores_by_class,
            gt_boxes_by_class, gt_labels_by_class)


def run_eval_return_metrics(
    checkpoint: Path,
    data_dir: Path,
    subset: int | None = None,
    threshold: float = 0.01,
    model_size: str = "nano",
) -> dict:
    """Run RF-DETR inference on valid set and return metrics (mAP_50, per_class_AP50, etc.)."""
    valid_dir = data_dir / "valid"
    if not valid_dir.exists():
        raise FileNotFoundError(f"valid dir not found: {valid_dir}")
    from PIL import Image
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge
    model_map = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium, "base": RFDETRBase, "large": RFDETRLarge}
    gt_list = load_coco_gt(valid_dir)
    if subset is not None:
        gt_list = gt_list[:subset]
    model = model_map[model_size](pretrain_weights=str(checkpoint))
    model.optimize_for_inference()
    preds_per_image = []
    for path, *_ in gt_list:
        detections = model.predict(Image.open(path).convert("RGB"), threshold=threshold)
        preds_per_image.append(_parse_detections(detections))
    gt_for_build = [(gboxes, glabs) for _, gboxes, glabs, *_ in gt_list]
    (pred_boxes_by_class, pred_labels_by_class, pred_scores_by_class,
     gt_boxes_by_class, gt_labels_by_class) = build_by_class_from_preds_gt(preds_per_image, gt_for_build)
    return compute_metrics(
        pred_boxes_by_class, pred_labels_by_class, pred_scores_by_class,
        gt_boxes_by_class, gt_labels_by_class, num_classes=NUM_CLASSES,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RF-DETR on BDD100K valid set")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint_best_total.pth")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "datasets" / "bdd_coco")
    parser.add_argument("--model-size", type=str, default="nano", choices=("nano", "small", "medium", "base", "large"))
    parser.add_argument("--subset", type=int, default=None, help="Use first N images for eval")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Confidence threshold (use low e.g. 0.01 to match training COCO mAP; 0.25 for fewer boxes)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--save-viz", action="store_true", help="Save images with predicted boxes")
    parser.add_argument("--viz-dir", type=Path, default=REPO_ROOT / "eval_viz" / "rf_detr", help="Output dir for visualization images")
    parser.add_argument("--max-viz", type=int, default=50, help="Max number of images to save when --save-viz")
    parser.add_argument("--show-gt", action="store_true", help="Draw GT boxes (green) in addition to predictions")
    parser.add_argument("--coco-metrics", action="store_true", default=True, help="Print COCO AP/AR (match training)")
    parser.add_argument("--no-coco-metrics", action="store_false", dest="coco_metrics")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "eval_output" / "rf_detr", help="Save metrics_plot.png here")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    valid_dir = args.data / "valid"
    if not valid_dir.exists():
        print(f"Error: valid dir not found: {valid_dir}", file=sys.stderr)
        return 1

    print("Loading GT...")
    gt_list = load_coco_gt(valid_dir)
    if args.subset:
        gt_list = gt_list[: args.subset]
    print(f"Images: {len(gt_list)}")

    print("Loading RF-DETR model...")
    try:
        from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge
        from PIL import Image
    except ImportError as e:
        print(f"Error: {e}. Install: pip install rfdetr>=1.4.0 Pillow", file=sys.stderr)
        return 1

    model_map = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium, "base": RFDETRBase, "large": RFDETRLarge}
    model = model_map[args.model_size](pretrain_weights=str(args.checkpoint))
    model.optimize_for_inference()

    preds_per_image = []
    for path, *_ in gt_list:
        detections = model.predict(Image.open(path).convert("RGB"), threshold=args.threshold)
        preds_per_image.append(_parse_detections(detections))

    if args.save_viz:
        import cv2
        import supervision as sv
        args.viz_dir.mkdir(parents=True, exist_ok=True)
        n_viz = min(args.max_viz, len(gt_list))
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT, text_scale=0.5)
        for idx in range(n_viz):
            path, gboxes, glabs, *_ = gt_list[idx]
            xyxy, class_id, confidence = preds_per_image[idx]
            xyxy = np.asarray(xyxy)
            if xyxy.size == 0:
                xyxy = np.zeros((0, 4))
            class_id = np.asarray(class_id)
            confidence = np.asarray(confidence)
            frame = np.array(Image.open(path).convert("RGB"))
            dets = sv.Detections(
                xyxy=xyxy,
                class_id=(class_id - 1).astype(np.int32) if class_id.size else np.zeros(0, dtype=np.int32),
                confidence=confidence,
            )
            labels = [f"{DETECTION_CLASSES[int(c) - 1]} {conf:.2f}" for c, conf in zip(class_id, confidence)] if class_id.size else []
            annotated = box_annotator.annotate(scene=frame.copy(), detections=dets)
            if labels:
                annotated = label_annotator.annotate(scene=annotated, detections=dets, labels=labels)
            if args.show_gt and gboxes is not None and len(gboxes) > 0:
                gb = gboxes.numpy() if hasattr(gboxes, "numpy") else np.asarray(gboxes)
                for box in gb:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.imwrite(str(args.viz_dir / path.name), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"Saved {n_viz} viz to {args.viz_dir}")

    gt_for_build = [(gboxes, glabs) for _, gboxes, glabs, *_ in gt_list]
    (pred_boxes_by_class, pred_labels_by_class, pred_scores_by_class,
     gt_boxes_by_class, gt_labels_by_class) = build_by_class_from_preds_gt(preds_per_image, gt_for_build)

    if args.coco_metrics:
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError:
            print("Error: pip install pycocotools", file=sys.stderr)
            return 1
        coco_dt_list = []
        for (_, __, ___, image_id), (xyxy, cid, conf) in zip(gt_list, preds_per_image):
            xyxy, cid, conf = np.asarray(xyxy), np.asarray(cid), np.asarray(conf)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                coco_dt_list.append({"image_id": image_id, "category_id": int(cid[i]), "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(conf[i])})
        coco_gt = COCO(str(valid_dir / "_annotations.coco.json"))
        coco_eval = COCOeval(coco_gt, coco_gt.loadRes(coco_dt_list), "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        print("\n" + "=" * 50 + "\nCOCO metrics (match training — mAP @ 0.50:0.95)\n" + "=" * 50)
        coco_eval.summarize()

    metrics = compute_metrics(
        pred_boxes_by_class, pred_labels_by_class, pred_scores_by_class,
        gt_boxes_by_class, gt_labels_by_class, num_classes=NUM_CLASSES,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_metrics(metrics, save_path=args.out_dir / "metrics_plot.png")
    plot_pr_curves(metrics, save_path=args.out_dir / "pr_curve.png")
    print(f"Saved mAP@0.5 plot to {args.out_dir / 'metrics_plot.png'}")
    print(f"Saved PR curve to {args.out_dir / 'pr_curve.png'}")
    print("\n" + "=" * 50 + "\nCustom mAP (PR-curve)\n" + "=" * 50)
    print(f"  mAP@0.5: {metrics['mAP_50']:.4f}  mAP@0.75: {metrics['mAP_75']:.4f}  mAP@0.5:0.95: {metrics['mAP_50_95']:.4f}")
    for name, ap in metrics["per_class_AP50"].items():
        print(f"    {name}: {ap:.4f}")

    print("\n" + "=" * 50 + "\nInference timing (single-image)\n" + "=" * 50)
    n_timing = min(200, len(gt_list))
    paths = [p for p, *_ in gt_list[:n_timing]]
    for _ in range(args.warmup):
        model.predict(Image.open(paths[0]).convert("RGB"), threshold=args.threshold)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latencies = []
    for path in paths:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(Image.open(path).convert("RGB"), threshold=args.threshold)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    mean_ms = sum(latencies) / len(latencies)
    print(f"  n={n_timing}  mean={mean_ms:.2f} ms  median={latencies[len(latencies) // 2]:.2f} ms  p95={latencies[int(len(latencies) * 0.95)]:.2f} ms  FPS={1000.0 / mean_ms:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
