"""
Evaluation visualization: quantitative (metrics charts) and qualitative (GT vs pred on images).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from data.config import DETECTION_CLASSES

IDX_TO_CLASS = {i + 1: name for i, name in enumerate(DETECTION_CLASSES)}


def plot_metrics(metrics: dict[str, Any], save_path: Optional[Path] = None) -> None:
    """
    Plot quantitative metrics: per-class AP@0.5 bar chart and summary (mAP) text.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Per-class AP@0.5
    ap50 = metrics.get("per_class_AP50_list", [])
    if ap50:
        ap50 = [float(x) if not np.isnan(x) else 0.0 for x in ap50]
        x = np.arange(len(DETECTION_CLASSES))
        axes[0].bar(x, ap50, color="steelblue", edgecolor="black")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(DETECTION_CLASSES, rotation=45, ha="right")
        axes[0].set_ylabel("AP@0.5")
        axes[0].set_title("Per-class AP (IoU=0.5)")
        axes[0].set_ylim(0, 1.05)
    # Summary metrics as text
    axes[1].axis("off")
    summary = (
        f"mAP@0.5:    {metrics.get('mAP_50', 0):.3f}\n"
        f"mAP@0.75:   {metrics.get('mAP_75', 0):.3f}\n"
        f"mAP@0.5:0.95: {metrics.get('mAP_50_95', 0):.3f}"
    )
    axes[1].text(0.1, 0.5, summary, fontsize=14, verticalalignment="center")
    axes[1].set_title("Summary metrics")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pr_curves(metrics: dict[str, Any], save_path: Optional[Path] = None) -> None:
    """
    Plot Precision-Recall curves (one per class) at IoU=0.5.
    metrics must contain "per_class_pr_curves": {class_name: {"recall": array, "precision": array}}.
    """
    curves = metrics.get("per_class_pr_curves") or {}
    if not curves:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(curves), 1)))
    for i, (class_name, data) in enumerate(curves.items()):
        recall = np.asarray(data["recall"])
        precision = np.asarray(data["precision"])
        ax.plot(recall, precision, color=colors[i % len(colors)], label=class_name, linewidth=1.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (IoU=0.5)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def draw_boxes_on_image(
    image: Any,
    boxes: Any,
    labels: Any,
    scores: Optional[Any] = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label_map: Optional[dict[int, str]] = None,
) -> Any:
    """
    Draw bounding boxes and optional labels on an image (numpy HWC RGB or PIL).
    boxes: (N,4) xyxy; labels: (N,) 1-indexed; scores: (N,) optional.
    Returns numpy array HWC RGB.
    """
    import numpy as np
    from PIL import Image

    if hasattr(image, "numpy"):
        img = image.permute(1, 2, 0).numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
    elif isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    if hasattr(boxes, "numpy"):
        boxes = boxes.numpy()
    if hasattr(labels, "numpy"):
        labels = labels.numpy()
    if scores is not None and hasattr(scores, "numpy"):
        scores = scores.numpy()
    boxes = np.asarray(boxes)
    labels = np.asarray(labels)
    if boxes.ndim == 1:
        boxes = np.expand_dims(boxes, 0)
    if labels.ndim == 0:
        labels = np.expand_dims(labels, 0)
    if boxes.shape[0] == 0:
        return img

    try:
        import cv2
    except ImportError:
        return img
    label_map = label_map or IDX_TO_CLASS
    for i, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        lab = int(labels[i]) if i < len(labels) else 0
        text = label_map.get(lab, str(lab))
        if scores is not None and i < len(scores):
            text += f" {float(scores[i]):.2f}"
        cv2.putText(
            img, text, (int(x1), max(int(y1) - 4, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )
    return img


def save_qualitative_viz(
    image_paths: list[Path],
    all_preds: list[dict],
    all_targets: list[dict],
    output_dir: Path,
    max_images: int = 50,
    gt_color: tuple[int, int, int] = (0, 255, 0),
    pred_color: tuple[int, int, int] = (255, 0, 0),
) -> list[Path]:
    """
    Save side-by-side or overlaid visualizations: GT (green) vs predictions (red).
    image_paths[i] corresponds to all_preds[i] and all_targets[i].
    Returns list of saved paths.
    """
    from PIL import Image
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    try:
        import cv2
    except ImportError:
        return saved
    n = min(max_images, len(image_paths), len(all_preds), len(all_targets))
    for i in range(n):
        path = image_paths[i]
        if not path.exists():
            continue
        img = np.array(Image.open(path).convert("RGB"))
        pred = all_preds[i]
        gt = all_targets[i]
        img_gt = draw_boxes_on_image(
            img.copy(), gt["boxes"], gt["labels"], None, color=gt_color
        )
        img_pred = draw_boxes_on_image(
            img.copy(),
            pred["boxes"],
            pred["labels"],
            pred.get("scores"),
            color=pred_color,
        )
        # Side by side: GT | Pred
        h, w = img_gt.shape[:2]
        combined = np.hstack([img_gt, img_pred])
        cv2.putText(
            combined, "GT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gt_color, 2
        )
        cv2.putText(
            combined, "Pred", (w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2
        )
        out_path = output_dir / f"{path.stem}_viz.jpg"
        Image.fromarray(combined).save(out_path)
        saved.append(out_path)
    return saved


def cluster_failures(
    metrics: dict[str, Any],
    per_image_ap: Optional[list[float]] = None,
    image_attrs: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """
    Cluster where the model fails: by class (worst AP), and optionally by image/attribute.
    Returns a dict with worst_classes, worst_images (if per_image_ap given), etc.
    """
    per_class = metrics.get("per_class_AP50", {})
    sorted_classes = sorted(
        per_class.items(),
        key=lambda x: (float(x[1]) if not np.isnan(x[1]) else -1),
    )
    worst_classes = [c for c, ap in sorted_classes if (np.isnan(ap) or ap < 0.3)][:5]
    best_classes = [c for c, ap in sorted_classes if not np.isnan(ap) and ap >= 0.5][-5:]
    out = {
        "worst_AP_classes": sorted_classes[:5],
        "best_AP_classes": sorted_classes[-5:] if len(sorted_classes) >= 5 else sorted_classes,
        "low_performers": worst_classes,
        "high_performers": best_classes,
    }
    if per_image_ap is not None and len(per_image_ap) > 0:
        per_image_ap = np.array(per_image_ap)
        worst_idx = np.argsort(per_image_ap)[:20]
        out["worst_image_indices"] = worst_idx.tolist()
        out["worst_image_aps"] = per_image_ap[worst_idx].tolist()
    if image_attrs is not None and per_image_ap is not None:
        attrs_by_failure = {"weather": {}, "scene": {}, "timeofday": {}}
        worst_idx = np.argsort(per_image_ap)[:100]
        for idx in worst_idx:
            if idx < len(image_attrs):
                a = image_attrs[idx]
                for k in attrs_by_failure:
                    v = a.get(k, "unknown")
                    attrs_by_failure[k][v] = attrs_by_failure[k].get(v, 0) + 1
        out["failure_attribute_distribution"] = attrs_by_failure
    return out
