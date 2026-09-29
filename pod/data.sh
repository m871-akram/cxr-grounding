#!/bin/bash
# Stage 1 on the pod, in tmux:  bash pod/run.sh data "bash pod/data.sh"
#   1. NIH labels (Hugging Face mirror, 9 MB)             -> data/Data_Entry_2017_v2020.csv
#   2. CheXmask for NIH (2.2 GB, on the container disk)   -> data/chexmask_nih_landmarks.csv (~115 MB)
#   3. CTR per image + first figure                       -> outputs/ctr_nih.csv, results/
#   4. NIH images: 12 zips of 2-3 GB, one at a time on the container disk
#                                                         -> 512-px subset in data/nih512 (~2 GB)
# Big downloads stay on the container disk (/tmp): fast, and deleted with the pod. Only the small
# outputs go to the network volume. Finished steps are skipped: after a crash, start it again.
#   IMAGES=0 bash pod/data.sh     # steps 1-3 only
set -euo pipefail
cd /workspace/cxr-grounding
source pod/env.sh
TMP=/tmp/cxr-data
mkdir -p "$TMP" data/nih512 outputs results
trap 'rm -rf "$TMP"' EXIT
df -h /tmp | tail -1

NIH=https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data
CHEXMASK=https://huggingface.co/datasets/ming0100/chexmask/resolve/main/OriginalResolution/ChestX-Ray8.csv
CHEXMASK_SHA256=48766ab0268235d63666bb2bacbd9f642b33fce7c1be40b9e1ecb381605545fa  # PhysioNet v1.0
LABELS=data/Data_Entry_2017_v2020.csv

# 1. Labels
if [ ! -f "$LABELS" ]; then
  curl -fsSL --retry 3 -o "$LABELS.part" "$NIH/Data_Entry_2017_v2020.csv"
  mv "$LABELS.part" "$LABELS"
fi

# 2. CheXmask. PhysioNet is slow, so the file comes from a Hugging Face mirror and must match
#    PhysioNet's official SHA-256. Only the landmark columns are kept.
if [ ! -f data/chexmask_nih_landmarks.csv ]; then
  curl -fsSL --retry 3 -o "$TMP/chexmask.csv" "$CHEXMASK"
  echo "$CHEXMASK_SHA256  $TMP/chexmask.csv" | sha256sum -c -
  python -m cxr.data chexmask --src "$TMP/chexmask.csv"
  rm "$TMP/chexmask.csv"
fi

# 3. CTR + figure (seconds, always redone)
python -m cxr.data ctr --labels "$LABELS"

# 4. NIH images: each zip is read on the container disk, only the selected images are kept
#    (512 px), then the zip is deleted.
if [ "${IMAGES:-1}" = 0 ]; then
  echo "Skipping the NIH images (IMAGES=0)."
  exit 0
fi
for i in $(seq -f "%03g" 1 12); do
  part="images_$i.zip"
  if [ -f "data/nih512/.done_$part" ]; then continue; fi
  echo "== $part ($(date '+%T'))"
  curl -fsSL --retry 3 -o "$TMP/$part" "$NIH/images/$part"
  python -m cxr.data prepare --nih-zip "$TMP/$part" --csv "$LABELS"
  rm "$TMP/$part"
  touch "data/nih512/.done_$part"
done
du -sh data/nih512
