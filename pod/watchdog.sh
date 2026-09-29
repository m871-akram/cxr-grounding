#!/bin/bash
# Hard limit: remove this pod after N hours, whatever is still running. Start it right after setup:
#   bash pod/run.sh watchdog "bash pod/watchdog.sh 3"
# RunPod gives the pod id and key to the container's first process, hence /proc/1/environ.
hours=${1:-3}
sleep $((hours * 3600))
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=')
echo "== $(date '+%F %T') watchdog: $hours h reached, removing pod $RUNPOD_POD_ID"
runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl pod remove "$RUNPOD_POD_ID"
