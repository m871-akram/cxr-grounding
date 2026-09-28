#!/bin/bash
# Stage 1 in one job, from the project folder on nash:  sbatch slurm/data.sbatch   (only one at a time)
#   1. NIH labels (Kaggle)                   -> data/Data_Entry_2017.csv
#   2. CheXmask for NIH (2.2 GB, in /tmp)    -> data/chexmask_nih_landmarks.csv (~115 MB)
#   3. CTR per image + first figure          -> outputs/ctr_nih.csv, results/
#   4. NIH images: 12 zips of 2-4 GB, one at a time in /tmp -> 512-px subset in data/nih512 (~2 GB)
# Big downloads go to the node's /tmp, because the home folder has a quota. Finished steps are
# skipped, so if the 4 h limit stops the job, resubmit it. Needs a Kaggle API token for step 1.
#   IMAGES=0 sbatch slurm/data.sbatch    # steps 1-3 only (~130 MB kept in home)
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
echo "node=$(hostname) start=$(date '+%F %T')"

# Two data jobs would write the same files: the older one keeps going, this one stops.
older=$(squeue -h -u "${USER:-$(id -un)}" -n data -o %i 2>/dev/null \
        | awk -v me="${SLURM_JOB_ID:-0}" '$1 < me {print; exit}') || older=""
if [ -n "$older" ]; then
  echo "Data job $older is already queued or running: this one stops."
  exit 1
fi

JOB_TMP="/tmp/${USER:-$(id -un)}/${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMP" data outputs results
trap 'rm -rf "$JOB_TMP"' EXIT
rm -rf data/raw data/chexmask_nih.csv data/chexmask_nih.csv.part  # leftovers of the first version
df -h /tmp | tail -1

# 1. Labels
if [ ! -f data/Data_Entry_2017.csv ]; then
  kaggle datasets download nih-chest-xrays/data -f Data_Entry_2017.csv -p data
  if [ -f data/Data_Entry_2017.csv.zip ]; then  # Kaggle sometimes zips single files
    python -m zipfile -e data/Data_Entry_2017.csv.zip data && rm data/Data_Entry_2017.csv.zip
  fi
fi

# 2. CheXmask. PhysioNet is slow, so the file comes from a Hugging Face mirror and must match
#    PhysioNet's official SHA-256 checksum. Only the landmark columns are kept.
if [ ! -f data/chexmask_nih_landmarks.csv ]; then
  curl -fsSL -o "$JOB_TMP/chexmask.csv" \
    https://huggingface.co/datasets/ming0100/chexmask/resolve/main/OriginalResolution/ChestX-Ray8.csv
  sum=$(curl -fsSL https://physionet.org/files/chexmask-cxr-segmentation-data/1.0.0/SHA256SUMS.txt \
        | awk '$2 == "OriginalResolution/ChestX-Ray8.csv" {print $1}')
  echo "$sum  $JOB_TMP/chexmask.csv" | sha256sum -c -
  python -m cxr.data chexmask --src "$JOB_TMP/chexmask.csv"
  rm "$JOB_TMP/chexmask.csv"
fi

# 3. CTR + figure
python -m cxr.data ctr

# 4. NIH images from a Hugging Face mirror of the NIH release, split in 12 zips. Each zip is read
#    in /tmp, only the selected images are kept (512 px), then the zip is deleted.
if [ "${IMAGES:-1}" = 0 ]; then
  echo "Skipping the NIH images (IMAGES=0)."
else
  BASE=https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/images
  mkdir -p data/nih512
  for i in $(seq -f "%03g" 1 12); do
    part="images_$i.zip"
    if [ -f "data/nih512/.done_$part" ]; then continue; fi
    echo "== $part ($(date '+%T'))"
    curl -fsSL -o "$JOB_TMP/$part" "$BASE/$part"
    python -m cxr.data prepare --nih-zip "$JOB_TMP/$part" --csv data/Data_Entry_2017.csv
    rm "$JOB_TMP/$part"
    touch "data/nih512/.done_$part"
  done
  du -sh data/nih512
fi
echo "end=$(date '+%F %T')"
