#!/bin/bash
# Start a long run on the pod in a tmux session; its output is appended to /workspace/logs/<name>.log.
#   bash pod/run.sh data  "bash pod/data.sh"
#   bash pod/run.sh audit "python -m cxr.audit --finding cardiomegaly"
# Follow it: tail -f /workspace/logs/<name>.log   (or tmux attach -t <name>, and Ctrl-b d to leave)
set -euo pipefail
name=$1
cmd=$2
log=/workspace/logs/$name.log

if tmux has-session -t "=$name" 2>/dev/null; then
  echo "A run named '$name' is still going (tmux attach -t $name)." >&2
  exit 1
fi
{ echo "== $(date '+%F %T') start: $cmd"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
} >> "$log"
# The session ends with the command; ${PIPESTATUS[0]} is the command's exit code, not tee's.
tmux new-session -d -s "$name" -c /workspace/cxr-grounding bash -c \
  "source pod/env.sh; ($cmd) 2>&1 | tee -a $log; echo \"== \$(date '+%F %T') end, exit code \${PIPESTATUS[0]}\" >> $log"
echo "Started '$name' in tmux. Log: $log"
