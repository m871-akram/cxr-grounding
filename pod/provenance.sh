#!/bin/bash
# Were the audited images made by one run with the final settings? For every image folder under
# outputs/pairs and outputs/dose: the number of PNGs and the first and last modification times (UTC),
# next to the start and end lines of the run logs. A file older than its run's start line would come
# from an earlier run. Writes results/provenance_check.txt.
#   bash pod/provenance.sh
set -euo pipefail
cd /workspace/cxr-grounding
out=results/provenance_check.txt
{
  echo "== image folders: count, first and last modification (UTC)"
  find outputs/pairs outputs/dose -mindepth 2 -type d | sort | while read -r d; do
    times=$(find "$d" -maxdepth 1 -name '*.png' -printf '%T@\n' | sort -n)
    [ -z "$times" ] && continue
    first=$(echo "$times" | head -1); last=$(echo "$times" | tail -1)
    echo "$d $(echo "$times" | wc -l) $(date -u -d "@${first%.*}" '+%F %T') $(date -u -d "@${last%.*}" '+%F %T')"
  done
  echo "== run logs: start and end lines"
  for f in /workspace/logs/*.log; do
    echo "-- $f"
    grep -a -E "^== .*(start|end)" "$f" || true
  done
} > "$out"
cat "$out"
