# BDD Object Detection

[BDD100K](https://www.bdd100k.com/) object detection (10 classes). This repo covers data analysis, training (Faster R-CNN and RF-DETR), and evaluation.

---

## 1. Dataset structure

Download **100K Images** and **detection Labels** from BDD100K. After unpacking you should have:

```
/path/to/BDD/
  images/
    train/   # *.jpg
    val/     # *.jpg
  labels/
    train/   # *.json (one per image, same base name as image)
    val/     # *.json
```

Scripts default to `BDD_ROOT` env or `--bdd-root`; override as needed.

---

## 2. Data analysis (Docker)

Build and run analysis or the dashboard with Docker (no local Python needed):

```bash
docker build -t bdd-analysis:latest .
```

**CLI analysis (with plots):**

```bash
docker run --rm -v /path/to/BDD:/data -v $(pwd)/output:/output bdd-analysis:latest \
  python -m data_analysis.run_analysis --bdd-root /data --plots --out-dir /output
```

**Streamlit dashboard:**

```bash
docker run --rm -p 8501:8501 -v /path/to/BDD:/data bdd-analysis:latest \
  streamlit run data_analysis/dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Then open http://localhost:8501.

---

## 3. RF-DETR: train and inference

Uses COCO-format data (train/valid with `_annotations.coco.json`). One-time prep, then train and evaluate.

**Setup (recommended):** Use the conda setup script to create an environment with Python 3.10, PyTorch (CUDA), and RF-DETR:

```bash
./training_rf_detr/setup_conda.sh              # creates env "bdd-rfdetr"
# optional: ./training_rf_detr/setup_conda.sh myenv 3.11   # custom env name and Python version
conda activate bdd-rfdetr
export PYTHONNOUSERSITE=1
```

Then run prep, training, and eval:

```bash
python -m training_rf_detr.prepare_dataset --bdd-root /path/to/BDD
python -m training_rf_detr.train --data datasets/bdd_coco --epochs 1
python -m evaluation.eval_rf_detr --checkpoint runs/rf_detr/bdd_1ep/checkpoint_best_total.pth
```

Alternatively, install RF-DETR in an existing environment: `pip install rfdetr>=1.4.0` and run the same commands.

---

## 4. Faster R-CNN: train and inference

Uses BDD images/labels directly (torchvision Faster R-CNN).

```bash
pip install -r requirements.txt
python -m training_faster_rcnn.train --bdd-root /path/to/BDD --epochs 1 --save checkpoints/frcnn.pt
python -m evaluation.run_eval --checkpoint checkpoints/frcnn.pt --bdd-root /path/to/BDD --out-dir eval_output
```

---

## 5. Failure analysis (FiftyOne)

Inspect false positives/negatives with FiftyOne (RF-DETR checkpoint on COCO valid):

```bash
pip install fiftyone
python -m evaluation.failure_analysis_fiftyone --checkpoint runs/rf_detr/bdd_1ep/checkpoint_best_total.pth
```

Opens the FiftyOne App; use `--view fp` or `--view fn` to sort by failure type.

---

**Docs:** [DATA_ANALYSIS.md](docs/DATA_ANALYSIS.md) · [MODEL.md](docs/MODEL.md) · [EVALUATION.md](docs/EVALUATION.md)
