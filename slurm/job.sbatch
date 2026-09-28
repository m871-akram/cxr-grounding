#!/bin/bash
# Generic GPU job. Usage (from the project folder on nash):
#   sbatch slurm/job.sbatch scripts/smoke_test.py
#   sbatch --partition=<a40-partition> --time=12:00:00 slurm/job.sbatch scripts/audit.py --finding cardiomegaly
# Command-line --options override the #SBATCH lines below.
#
#SBATCH --job-name=cxr
#SBATCH --partition=rtx6000        # check the real names with: sinfo
#SBATCH --gres=shard:1             # as in the Ensicompute guide; ask support how to get a full GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G                  # system RAM, not GPU memory
#SBATCH --time=04:00:00            # check the partition limit with: sinfo
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/cxr-grounding}"
cd "$PROJECT_DIR"
source .venv/bin/activate

# Keep caches inside the project folder (not in a small home quota).
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export WANDB_DIR="$PROJECT_DIR/outputs"
export WANDB_MODE="${WANDB_MODE:-offline}"   # sync later with: wandb sync outputs/wandb/offline-run-*

# Node-local scratch for big temporary files; deleted when the job ends.
export JOB_TMP="/tmp/${USER:-$(id -un)}/${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMP"
trap 'rm -rf "$JOB_TMP"' EXIT

echo "job=${SLURM_JOB_ID:-manual} node=$(hostname) start=$(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun python "$@"

echo "end=$(date '+%F %T')"
