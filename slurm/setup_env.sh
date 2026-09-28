#!/usr/bin/env bash
# Create the Python environment once, on nash:  bash slurm/setup_env.sh
# Takes ~6-8 GB (PyTorch + CUDA libraries). Temporary files go to .cache/tmp because nash's /tmp is small.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/cxr-grounding}"
cd "$PROJECT_DIR"
mkdir -p logs outputs results data .cache/tmp
export TMPDIR="$PROJECT_DIR/.cache/tmp" PIP_NO_CACHE_DIR=1

if ! python3 -m venv .venv 2>/dev/null; then
  echo "python3 -m venv failed. Alternative: install uv (pip install --user uv) and run: uv venv .venv"
  exit 1
fi
source .venv/bin/activate

pip install --upgrade pip
# CUDA 12.4 build, matching the cluster driver (550 / CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install bitsandbytes   # optional (4-bit loading); does not install on macOS

python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda)"
echo "Done. Next: hf auth login  (needed for RadEdit and MedGemma)"
