# Model: RF-DETR

We use **RF-DETR** as the main detection model: a **state-of-the-art real-time** detector that balances accuracy and speed for driving datasets like BDD100K.

---

## Why RF-DETR?

- **Real-time**: Fast inference with strong mAP; suitable for latency-sensitive applications.
- **Apache 2.0**: Permissive license, supports commercial use.
- **DINOv2 backbone**: Pre-trained vision backbone that generalizes well to datasets outside COCO (e.g. BDD100K driving scenes).
- **Simple pipeline**: Single library, standard COCO format; multiple sizes (nano fits ~24GB GPUs).

---

## Architecture (summary)

RF-DETR uses a **pre-trained ViT backbone** for multiscale feature extraction. It mixes **windowed and non-windowed attention** to trade off accuracy and latency. Deformable cross-attention and the segmentation head **bilinearly interpolate** the projector output so features stay spatially aligned. **Detection and segmentation losses** are applied at every decoder layer, which allows dropping decoder layers at inference for faster speed.

---

## How we use it

1. **Data**: BDD labels are converted to COCO format (train/valid + `_annotations.coco.json`) via `training_rf_detr.prepare_dataset`. Images are symlinked; no copy.
2. **Training**: `training_rf_detr.train` runs RF-DETR on that COCO dataset. Checkpoints and TensorBoard logs go to an output dir (e.g. `runs/rf_detr/bdd_1ep/`).
3. **Evaluation**: `evaluation.eval_rf_detr` loads a checkpoint, runs on the COCO valid set, and reports **COCO metrics** (mAP @ 0.50:0.95, etc.) so numbers are comparable to standard benchmarks.

---

## Training details

We trained **RF-DETR nano** for **5 epochs** starting from **COCO-pretrained weights**, due to limited time and resources. For production or higher mAP, longer training (e.g. 12–24 epochs) and optional larger variants (small/medium) are recommended.

**Example: Faster R-CNN (dataloader and training loop)** — To illustrate how BDD is used with a standard PyTorch detection setup, the repo includes a minimal Faster R-CNN training script that shows the dataloader and one-epoch loop. Full script: `training_faster_rcnn/train.py`.


## BDD classes

RF-DETR is trained and evaluated on the same **10 BDD detection classes** (pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign). Category IDs in COCO JSON are 1–10.

---

## References

- **RF-DETR: Neural Architecture Search for Real-Time Detection Transformers** — [arXiv:2511.09554](https://arxiv.org/abs/2511.09554), ICLR 2026.
- RF-DETR (Roboflow): [https://rfdetr.roboflow.com/](https://rfdetr.roboflow.com/)
