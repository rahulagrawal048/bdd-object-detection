"""
Faster R-CNN model builder for BDD object detection.
"""
from __future__ import annotations

import torch


def get_model(num_classes: int = 11, pretrained: bool = True) -> torch.nn.Module:
    """
    Build Faster R-CNN with ResNet-50 FPN backbone.

    num_classes=11: 10 BDD classes + 1 background (torchvision convention).
    If pretrained, load COCO weights then replace the box predictor for num_classes.

    Args:
        num_classes: Number of output classes (including background).
        pretrained: If True, load COCO-pretrained model and replace head for num_classes.

    Returns:
        torchvision.models.detection.FasterRCNN model.
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if pretrained:
        try:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

            model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            )
        except (ImportError, AttributeError):
            from torchvision.models import ResNet50_Weights

            model = fasterrcnn_resnet50_fpn(
                weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
                num_classes=num_classes,
            )
            return model
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        from torchvision.models import ResNet50_Weights

        model = fasterrcnn_resnet50_fpn(
            weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
            num_classes=num_classes,
        )

    return model
