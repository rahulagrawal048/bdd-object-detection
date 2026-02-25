"""
PyTorch Faster R-CNN training for BDD object detection.
"""
from training_faster_rcnn.model import get_model
from training_faster_rcnn.train import train_one_epoch, main

__all__ = ["get_model", "train_one_epoch", "main"]
