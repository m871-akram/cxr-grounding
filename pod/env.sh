# Environment for every command on the pod:  source /workspace/cxr-grounding/pod/env.sh
# Sourced by pod/run.sh, pod/data.sh and ~/.bashrc; pod/setup.sh creates what it points to.
export HF_HOME=/workspace/hf     # Hugging Face models, datasets and token, on the network volume
export PYTHONUNBUFFERED=1        # print() reaches the log file at once
source /workspace/venv/bin/activate
