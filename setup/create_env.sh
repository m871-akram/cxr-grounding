#!/usr/bin/env bash
# Create the Python environment once, on nash:  bash setup/create_env.sh
# Takes ~6-8 GB (PyTorch + CUDA libraries), so run it where you have space.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/cxr-grounding}"
cd "$PROJECT_DIR"

if ! python3 -m venv .venv 2>/dev/null; then
  echo "python3 -m venv failed. Alternative: install uv (pip install --user uv) and run: uv venv .venv"
  exit 1
fi
source .venv/bin/activate

pip install --upgrade pip
# CUDA 12.4 build, matching the cluster driver (550.54.15 / CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install bitsandbytes   # optional (4-bit loading); does not install on macOS

mkdir -p logs outputs data .cache
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda)"
echo "Done. Next: hf auth login  (or huggingface-cli login on older versions; needed for RadEdit and MedGemma)"
