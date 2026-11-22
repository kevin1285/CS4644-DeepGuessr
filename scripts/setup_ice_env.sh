#!/bin/bash
# setup_ice_env.sh — one-time environment setup on the GT ICE cluster.
# Run this ONCE on a login node after cloning the repo.
# Usage:  bash scripts/setup_ice_env.sh
set -euo pipefail

echo "==> Loading Anaconda module (adjust name if this fails; see runbook step 3)"
module load anaconda3 2>/dev/null || module load anaconda3/2023.03 2>/dev/null || {
    echo "!! Could not load an anaconda module."
    echo "   Run 'module avail anaconda' and edit this line with the exact name."
    exit 1
}

ENV_NAME=deepguessr
echo "==> Creating conda env '$ENV_NAME' (python 3.12)"
conda create -y -n "$ENV_NAME" python=3.12

# shellcheck disable=SC1091
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

echo "==> Installing PyTorch with CUDA (cu121 wheels) + project deps"
# ICE GPUs support CUDA 12.x. If this errors, see runbook step 3 for the CUDA module.
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install open_clip_torch huggingface_hub datasets pandas numpy scikit-learn \
    matplotlib tqdm reverse_geocoder pycountry pycountry-convert pyyaml \
    peft transformers

echo "==> Verifying CUDA is visible to torch"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "==> Done. Activate later with:  conda activate $ENV_NAME"
