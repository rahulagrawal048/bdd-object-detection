"""
Model evaluation on BDD validation set: inference and metric computation.

Metrics: mAP@0.5, mAP@0.75, mAP@0.5:0.95 (COCO-style), per-class AP@0.5.
Uses IoU-based matching and precision-recall curves (no pycocotools required).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
import numpy as np

from data.config import DETECTION_CLASSES
from data.dataset import BDDDataset, collate_fn, get_default_transform
from training_faster_rcnn.model import get_model

# Model labels are 1-indexed (1-10). Reverse mapping for display.
IDX_TO_CLASS = {i + 1: name for i, name in enumerate(DETECTION_CLASSES)}


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU between two sets of boxes (N,4) and (M,4). Returns (N, M)."""
    from torchvision.ops import box_iou
    return box_iou(boxes1, boxes2)


def _compute_ap(
    pred_boxes: list[torch.Tensor],
    pred_labels: list[torch.Tensor],
    pred_scores: list[torch.Tensor],
    gt_boxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    class_id: int,
    iou_threshold: float = 0.5,
    num_recall_points: int = 101,
    return_curve: bool = False,
) -> float | tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Average Precision for one class across all images.
    pred_* and gt_* are lists of tensors per image. class_id is 1-indexed.
    If return_curve=True, returns (ap, recall_101, precision_101) for PR plot.
    """
    all_scores = []
    all_matched = []
    num_gt = 0

    for pboxes, plabs, pscores, gboxes, glabs in zip(
        pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels
    ):
        gt_mask = glabs == class_id
        pred_mask = plabs == class_id
        gt_c = gboxes[gt_mask]
        p_c = pboxes[pred_mask]
        s_c = pscores[pred_mask]
        num_gt += len(gt_c)
        if len(p_c) == 0:
            continue
        if len(gt_c) == 0:
            all_scores.extend(s_c.tolist())
            all_matched.extend([0] * len(s_c))
            continue
        iou = _box_iou(p_c, gt_c)
        matched_gt = set()
        for idx in torch.argsort(s_c, descending=True):
            idx = idx.item()
            iou_row = iou[idx]
            best_j = iou_row.argmax().item()
            if iou_row[best_j] >= iou_threshold and best_j not in matched_gt:
                matched_gt.add(best_j)
                all_matched.append(1)
            else:
                all_matched.append(0)
            all_scores.append(s_c[idx].item())

    if num_gt == 0:
        return (float("nan"), np.linspace(0, 1, num_recall_points), np.zeros(num_recall_points)) if return_curve else float("nan")
    if not all_scores:
        return (0.0, np.linspace(0, 1, num_recall_points), np.zeros(num_recall_points)) if return_curve else 0.0

    all_scores = np.array(all_scores)
    all_matched = np.array(all_matched)
    order = np.argsort(-all_scores)
    all_matched = all_matched[order]
    tp = np.cumsum(all_matched)
    fp = np.cumsum(1 - all_matched)
    recall = tp / num_gt
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall_points = np.linspace(0, 1, num_recall_points)
    # p(r) = max precision at any recall >= r (VOC/COCO style). Do NOT use maximum.accumulate.
    prec_interp = np.array(
        [precision[recall >= r].max() if np.any(recall >= r) else 0.0 for r in recall_points]
    )
    ap = float(np.mean(prec_interp))
    if return_curve:
        return (ap, recall_points, prec_interp)
    return ap


def compute_metrics(
    pred_boxes_by_class: list[list[torch.Tensor]],
    pred_labels_by_class: list[list[torch.Tensor]],
    pred_scores_by_class: list[list[torch.Tensor]],
    gt_boxes_by_class: list[list[torch.Tensor]],
    gt_labels_by_class: list[list[torch.Tensor]],
    num_classes: int = 10,
) -> dict[str, Any]:
    """
    Compute mAP@0.5, mAP@0.75, mAP@0.5:0.95 and per-class AP@0.5.
    *_by_class: list of length num_classes; each element is list of tensors (one per image).
    """
    iou_thresholds = [0.5, 0.75] + list(np.arange(0.5, 1.0, 0.05))
    per_class_ap50: dict[int, float] = {}
    aps_at_50 = []
    aps_at_75 = []
    aps_5095 = []

    per_class_pr_curves: dict[str, dict[str, np.ndarray]] = {}

    for c in range(1, num_classes + 1):
        idx = c - 1
        ap50, recall_101, prec_101 = _compute_ap(
            pred_boxes_by_class[idx],
            pred_labels_by_class[idx],
            pred_scores_by_class[idx],
            gt_boxes_by_class[idx],
            gt_labels_by_class[idx],
            class_id=c,
            iou_threshold=0.5,
            return_curve=True,
        )
        per_class_ap50[c] = ap50
        per_class_pr_curves[IDX_TO_CLASS.get(c, str(c))] = {"recall": recall_101, "precision": prec_101}
        if not np.isnan(ap50):
            aps_at_50.append(ap50)
        ap75 = _compute_ap(
            pred_boxes_by_class[idx],
            pred_labels_by_class[idx],
            pred_scores_by_class[idx],
            gt_boxes_by_class[idx],
            gt_labels_by_class[idx],
            class_id=c,
            iou_threshold=0.75,
        )
        if not np.isnan(ap75):
            aps_at_75.append(ap75)
        aps_this_class = []
        for iou in iou_thresholds:
            ap = _compute_ap(
                pred_boxes_by_class[idx],
                pred_labels_by_class[idx],
                pred_scores_by_class[idx],
                gt_boxes_by_class[idx],
                gt_labels_by_class[idx],
                class_id=c,
                iou_threshold=iou,
            )
            if not np.isnan(ap):
                aps_this_class.append(ap)
        if aps_this_class:
            aps_5095.append(np.mean(aps_this_class))

    return {
        "mAP_50": float(np.mean(aps_at_50)) if aps_at_50 else 0.0,
        "mAP_75": float(np.mean(aps_at_75)) if aps_at_75 else 0.0,
        "mAP_50_95": float(np.mean(aps_5095)) if aps_5095 else 0.0,
        "per_class_AP50": {
            IDX_TO_CLASS.get(c, str(c)): ap for c, ap in per_class_ap50.items()
        },
        "per_class_AP50_list": [
            per_class_ap50.get(c, float("nan")) for c in range(1, num_classes + 1)
        ],
        "per_class_pr_curves": per_class_pr_curves,
    }


def _predict_batch(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    device: torch.device,
    score_thresh: float = 0.3,
) -> list[dict]:
    """Run model on a batch; return list of {boxes, labels, scores} per image."""
    model.eval()
    images_gpu = [img.to(device) for img in images]
    with torch.no_grad():
        preds = model(images_gpu)
    out = []
    for p in preds:
        keep = p["scores"] >= score_thresh
        out.append({
            "boxes": p["boxes"][keep].cpu(),
            "labels": p["labels"][keep].cpu(),
            "scores": p["scores"][keep].cpu(),
        })
    return out


def run_evaluation(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    score_thresh: float = 0.3,
    num_classes: int = 10,
) -> tuple[dict, list[dict], list[dict]]:
    """
    Run inference on the loader and compute metrics.
    Returns: metrics_dict, all_preds (list of pred dicts), all_targets.
    """
    all_pred_boxes_by_class = [[] for _ in range(num_classes)]
    all_pred_labels_by_class = [[] for _ in range(num_classes)]
    all_pred_scores_by_class = [[] for _ in range(num_classes)]
    all_gt_boxes_by_class = [[] for _ in range(num_classes)]
    all_gt_labels_by_class = [[] for _ in range(num_classes)]
    all_preds = []
    all_targets = []

    for images, targets in data_loader:
        preds = _predict_batch(model, images, device, score_thresh)
        for pred, gt in zip(preds, targets):
            all_preds.append(pred)
            all_targets.append(gt)
            pboxes, plabs, pscores = pred["boxes"], pred["labels"], pred["scores"]
            gboxes, glabs = gt["boxes"], gt["labels"]
            for c in range(num_classes):
                pc = c + 1
                pmask = plabs == pc
                gmask = glabs == pc
                all_pred_boxes_by_class[c].append(
                    pboxes[pmask] if pmask.any() else torch.zeros(0, 4)
                )
                all_pred_labels_by_class[c].append(
                    plabs[pmask] if pmask.any() else torch.zeros(0, dtype=torch.long)
                )
                all_pred_scores_by_class[c].append(
                    pscores[pmask] if pmask.any() else torch.zeros(0)
                )
                all_gt_boxes_by_class[c].append(
                    gboxes[gmask] if gmask.any() else torch.zeros(0, 4)
                )
                all_gt_labels_by_class[c].append(
                    glabs[gmask] if gmask.any() else torch.zeros(0, dtype=torch.long)
                )

    metrics = compute_metrics(
        all_pred_boxes_by_class,
        all_pred_labels_by_class,
        all_pred_scores_by_class,
        all_gt_boxes_by_class,
        all_gt_labels_by_class,
        num_classes=num_classes,
    )
    return metrics, all_preds, all_targets
