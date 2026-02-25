#!/usr/bin/env bash
#
# Create a conda environment for RF-DETR (and this repo's BDD training/eval).
#
# Usage:
#   ./training_rf_detr/setup_conda.sh              # create env "bdd-rfdetr", Python 3.10
#   ./training_rf_detr/setup_conda.sh myenv        # custom env name
#   ./training_rf_detr/setup_conda.sh myenv 3.11   # custom env + Python 3.11
#
# Then:
#   conda activate bdd-rfdetr
#   python -m training_rf_detr.train --data datasets/bdd_coco --epochs 1 --batch-size 32
#
set -e

ENV_NAME="${1:-bdd-rfdetr}"
PYTHON_VERSION="${2:-3.10}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Conda env: $ENV_NAME"
echo "Python: $PYTHON_VERSION"
echo "Repo root: $REPO_ROOT"

if conda env list | grep -q "^${ENV_NAME} "; then
  echo "Environment '$ENV_NAME' already exists. To recreate: conda env remove -n $ENV_NAME"
else
  echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

CONDA_BASE="$(conda info --base)"
ENV_PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"

if [ ! -x "$ENV_PYTHON" ]; then
  echo "Error: expected python not found at $ENV_PYTHON"
  echo "Check env name/path and rerun."
  exit 1
fi

# Avoid mixing ~/.local packages with this conda env.
conda env config vars set -n "$ENV_NAME" PYTHONNOUSERSITE=1

echo "Installing PyTorch (CUDA 12.6 pip wheels) in env..."
PYTHONNOUSERSITE=1 "$ENV_PYTHON" -m pip install --no-user --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1 torchvision==0.22.1

# Project + RF-DETR via pip (rfdetr pulls transformers, pycocotools, etc.)
echo "Installing project requirements and rfdetr..."
PYTHONNOUSERSITE=1 "$ENV_PYTHON" -m pip install --no-user -r "$REPO_ROOT/requirements.txt"

echo "Verifying imports..."
PYTHONNOUSERSITE=1 "$ENV_PYTHON" -c "import importlib.metadata as m, torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('rfdetr', m.version('rfdetr'))"

echo ""
echo "Done. Activate with:"
echo "  conda activate $ENV_NAME"
echo "  export PYTHONNOUSERSITE=1"
echo ""
echo "Then run RF-DETR data prep, training, and eval:"
echo "  python -m training_rf_detr.prepare_dataset --bdd-root /path/to/BDD"
echo "  python -m training_rf_detr.train --data datasets/bdd_coco --epochs 1 --batch-size 32"
echo "  python -m evaluation.eval_rf_detr --checkpoint runs/rf_detr/bdd_1ep/checkpoint_best_total.pth"
