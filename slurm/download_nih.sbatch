#!/bin/bash
# Download NIH ChestX-ray14 from Kaggle into node-local /tmp and keep only the 512-px subset.
# No GPU needed. Needs ~45 GB free in the node's /tmp (check with scripts/check_cluster.sh)
# and a Kaggle API token in ~/.kaggle/kaggle.json (chmod 600).
#   sbatch slurm/download_nih.sbatch
#
#SBATCH --job-name=nih-download
#SBATCH --partition=rtx6000        # any partition allowed for CPU jobs; check with: sinfo
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/cxr-grounding}"
cd "$PROJECT_DIR"
source .venv/bin/activate

JOB_TMP="/tmp/${USER:-$(id -un)}/${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMP"
trap 'rm -rf "$JOB_TMP"' EXIT
df -h /tmp

echo "Downloading (about 45 GB)..."
kaggle datasets download -d nih-chest-xrays/data -p "$JOB_TMP"
ls -lh "$JOB_TMP"

python scripts/prepare_nih_subset.py --nih-zip "$JOB_TMP"/*.zip --out-dir data/nih512

# One tarball of the subset for the Google Drive archive (optional, needs rclone configured):
tar -cf data/nih512.tar -C data nih512
# rclone copy data/nih512.tar gdrive:cxr-grounding/
du -sh data/nih512 data/nih512.tar
