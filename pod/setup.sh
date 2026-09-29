#!/bin/bash
# Prepare a pod, once the code is on it (bash pod/sync.sh push, on the Mac):
#   ssh cxr-pod bash /workspace/cxr-grounding/pod/setup.sh
# Run it on every new pod. The first run on a new network volume creates the Python environment
# (a few minutes); later runs only redo what a new pod loses (apt packages, ~/.bashrc).
set -euo pipefail
REPO=/workspace/cxr-grounding
VENV=/workspace/venv

# 1. apt packages live on the container disk, so every new pod installs them again.
if ! command -v tmux >/dev/null; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux python3-venv >/dev/null
fi

# 2. Everything that must outlive the pod goes on the network volume.
mkdir -p /workspace/hf /workspace/logs "$REPO/data" "$REPO/outputs" "$REPO/results"

# 3. Python environment on the volume. --system-site-packages lets it see the template's
#    PyTorch 2.8 (CUDA 12.8), so torch is not downloaded a second time.
if [ ! -x "$VENV/bin/pip" ]; then
  python3 -m venv --system-site-packages "$VENV"
fi
source "$REPO/pod/env.sh"
# Constraints file: pip must keep the template's torch and torchvision, or fail; never replace them.
python -c "import torch, torchvision; print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}')" \
  > /tmp/torch-pins.txt
echo "Installing requirements.txt (first run on a new volume: a few minutes)"
pip install -q -r "$REPO/requirements.txt" -c /tmp/torch-pins.txt

# 4. Interactive shells (ssh cxr-pod) start with the environment, in the repo.
grep -qs 'pod/env.sh' ~/.bashrc || echo "source $REPO/pod/env.sh && cd $REPO" >> ~/.bashrc

# 5. Summary
python - <<'EOF'
import importlib.metadata as md
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU VISIBLE"
print(f"torch {torch.__version__} from {torch.__file__.rsplit('/torch/', 1)[0]} | {gpu}")
if torch.cuda.is_available():  # a real kernel: proves this torch build supports the GPU architecture
    x = torch.ones(256, 256, device="cuda", dtype=torch.bfloat16)
    print("bf16 matmul on the GPU:", "ok" if (x @ x)[0, 0].item() == 256 else "WRONG RESULT")
print(" | ".join(f"{p} {md.version(p)}" for p in
                 ["transformers", "diffusers", "peft", "accelerate", "torchxrayvision", "huggingface_hub"]))
EOF
df -h /workspace /
[ -f /workspace/hf/token ] || echo "Next, once: hf auth login  (token kept in /workspace/hf; needed for RadEdit and MedGemma)"
