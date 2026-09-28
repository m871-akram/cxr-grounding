#!/bin/bash
# Any GPU run of a stage, from the project folder on nash:
#   sbatch slurm/job.sbatch -m cxr.audit --finding cardiomegaly
#   sbatch --partition=a40 slurm/job.sbatch -m cxr.audit ...      # MedGemma needs the A40 (bf16)
#   sbatch --gres=shard:1 --time=00:15:00 slurm/job.sbatch -m ...  # quick test on 1/6 of a GPU
# Options given to sbatch override the #SBATCH lines below. Every partition stops jobs after 4 h,
# so long runs save checkpoints and resume when resubmitted.
#
#SBATCH --job-name=cxr
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1               # a full GPU; shards share one GPU's memory with other jobs
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G                  # system RAM, not GPU memory
#SBATCH --time=04:00:00            # partition limit
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/cxr-grounding}"
cd "$PROJECT_DIR"
source .venv/bin/activate

export WANDB_DIR="$PROJECT_DIR/outputs"
export WANDB_MODE="${WANDB_MODE:-offline}"   # sync later with: wandb sync outputs/wandb/offline-run-*

# Node-local scratch for big temporary files; deleted when the job ends.
export JOB_TMP="/tmp/${USER:-$(id -un)}/${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMP"
trap 'rm -rf "$JOB_TMP"' EXIT

echo "job=${SLURM_JOB_ID:-manual} node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} start=$(date '+%F %T')"
# Only the GPU(s) given to this job (plain nvidia-smi lists every GPU of the node).
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader ${CUDA_VISIBLE_DEVICES:+-i "$CUDA_VISIBLE_DEVICES"} || true

srun python "$@"

echo "end=$(date '+%F %T')"
