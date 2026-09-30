#!/bin/bash
# Start a long run on the pod in a tmux session; its output is appended to /workspace/logs/<name>.log.
#   bash pod/run.sh data  "bash pod/data.sh"
#   bash pod/run.sh audit "python -m cxr.audit --finding cardiomegaly"
#   bash pod/run.sh generate "python -m cxr.edit generate" terminate   # removes the pod at the end
# Follow it: tail -f /workspace/logs/<name>.log   (or tmux attach -t <name>, and Ctrl-b d to leave)
set -euo pipefail
name=$1
cmd=$2
log=/workspace/logs/$name.log
# terminate: when the command ends (success or not), the pod removes itself, so a long run never
# leaves a GPU idle. RunPod gives the pod id and key to the container's first process, not to SSH
# shells, so they are read from /proc/1/environ. The 2-minute wait leaves time to pull results/
# (everything else stays on the network volume anyway).
finish=""
if [ "${3:-}" = terminate ]; then
  finish="; sleep 120; export \$(tr '\\0' '\\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)='); \
echo \"== removing pod \$RUNPOD_POD_ID\" >> $log; \
runpodctl remove pod \$RUNPOD_POD_ID >> $log 2>&1 || runpodctl pod remove \$RUNPOD_POD_ID >> $log 2>&1"
fi

if tmux has-session -t "=$name" 2>/dev/null; then
  echo "A run named '$name' is still going (tmux attach -t $name)." >&2
  exit 1
fi
{ echo "== $(date '+%F %T') start: $cmd"
  echo "== code commit: $(cat /workspace/cxr-grounding/COMMIT 2>/dev/null || echo unknown)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
} >> "$log"
# The session ends with the command; ${PIPESTATUS[0]} is the command's exit code, not tee's.
tmux new-session -d -s "$name" -c /workspace/cxr-grounding bash -c \
  "source pod/env.sh; ($cmd) 2>&1 | tee -a $log; echo \"== \$(date '+%F %T') end, exit code \${PIPESTATUS[0]}\" >> $log$finish"
echo "Started '$name' in tmux. Log: $log"
