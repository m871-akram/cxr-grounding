#!/bin/bash
# Stage 1 in one job, from the project folder on nash:  sbatch slurm/data.sbatch
#   1. NIH labels (Kaggle)                     -> data/Data_Entry_2017.csv
#   2. CheXmask heart/lung masks for NIH       -> data/chexmask_nih.csv (2.2 GB)
#   3. CTR per image + first figure (~1 min)   -> outputs/ctr_nih.csv, results/
#   4. NIH images, ~45 GB zip (Kaggle)         -> 512-px subset in data/nih512, then the zip is deleted
# Finished steps are skipped, so if the 4 h limit stops the job, resubmit it: the NIH download
# resumes where it stopped. Needs a Kaggle API token in ~/.kaggle/kaggle.json.
#
#SBATCH --job-name=data
#SBATCH --partition=rtx6000
#SBATCH --gres=shard:1             # no GPU used; the smallest request, in case CPU-only jobs are refused
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00            # partition limit
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/cxr-grounding}"
cd "$PROJECT_DIR"
source .venv/bin/activate
mkdir -p data/raw outputs results
echo "node=$(hostname) start=$(date '+%F %T')"

# 1. Labels
if [ ! -f data/Data_Entry_2017.csv ]; then
  kaggle datasets download nih-chest-xrays/data -f Data_Entry_2017.csv -p data
  if [ -f data/Data_Entry_2017.csv.zip ]; then  # Kaggle sometimes zips single files
    python -m zipfile -e data/Data_Entry_2017.csv.zip data && rm data/Data_Entry_2017.csv.zip
  fi
fi

# 2. CheXmask. PhysioNet is slow, so the file comes from a Hugging Face mirror and must match
#    PhysioNet's official SHA-256 checksum.
if [ ! -f data/chexmask_nih.csv ]; then
  curl -fL -C - -o data/chexmask_nih.csv.part \
    https://huggingface.co/datasets/ming0100/chexmask/resolve/main/OriginalResolution/ChestX-Ray8.csv \
    || true  # a complete .part makes curl fail on resume; the checksum decides
  sum=$(curl -fsSL https://physionet.org/files/chexmask-cxr-segmentation-data/1.0.0/SHA256SUMS.txt \
        | awk '$2 == "OriginalResolution/ChestX-Ray8.csv" {print $1}')
  if ! echo "$sum  data/chexmask_nih.csv.part" | sha256sum -c -; then
    echo "CheXmask checksum mismatch: delete data/chexmask_nih.csv.part and resubmit." >&2
    exit 1
  fi
  mv data/chexmask_nih.csv.part data/chexmask_nih.csv
fi

# 3. CTR + figure
python -m cxr.data ctr

# 4. NIH images: the zip goes to data/raw (a node's /tmp is too small) and is read without unzipping.
ZIP=data/raw/data.zip
if [ -f data/nih512/metadata.csv ] && [ ! -f "$ZIP" ]; then
  echo "NIH subset already built."
else
  if ! python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1])" "$ZIP" 2>/dev/null; then
    echo "Downloading NIH ChestX-ray14 (about 45 GB)..."
    kaggle datasets download nih-chest-xrays/data -p data/raw
  fi
  python -m cxr.data prepare --nih-zip "$ZIP"
  du -sh data/nih512
  if [ "${KEEP_ZIP:-0}" != 1 ]; then
    rm -f "$ZIP"
    echo "Deleted $ZIP (KEEP_ZIP=1 sbatch ... keeps it)"
  fi
fi
echo "end=$(date '+%F %T')"
