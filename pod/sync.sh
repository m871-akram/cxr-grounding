#!/bin/bash
# On the Mac. Code goes Mac -> pod, results come back pod -> Mac; nothing else moves.
#   bash pod/sync.sh push   # code -> /workspace/cxr-grounding (the pod's data/, outputs/, results/ untouched)
#   bash pod/sync.sh pull   # the pod's results/ -> ./results (check with git diff, commit from the Mac)
# The pod is host "cxr-pod" in ~/.ssh/config; its IP and port change with every new pod.
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=cxr-pod:/workspace/cxr-grounding

case "${1:-}" in
  push)
    # rsync must exist on both ends, and apt packages do not survive a new pod.
    ssh cxr-pod 'command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null)'
    # --delete mirrors deleted code files; excluded folders on the pod are never touched.
    rsync -az --delete \
      --exclude=.git/ --exclude=/data/ --exclude=/outputs/ --exclude=/results/ --exclude=/logs/ \
      --exclude=.venv/ --exclude=__pycache__/ --exclude=.ipynb_checkpoints/ --exclude=.DS_Store \
      --exclude='CLAUDE*.md' \
      ./ "$REMOTE/"
    ;;
  pull)
    rsync -az "$REMOTE/results/" results/
    ;;
  *)
    echo "usage: bash pod/sync.sh push|pull" >&2
    exit 1
    ;;
esac
