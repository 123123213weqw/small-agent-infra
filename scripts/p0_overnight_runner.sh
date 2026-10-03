#!/bin/bash
# Retry an interrupted Mac->V100 download/relay pass until all files verify or nine hours elapse.
set -u
repo="$(cd "$(dirname "$0")/.." && pwd)"
cache="$HOME/.cache/small-agent-p0-overnight-v1"
mkdir -p "$cache"
deadline=$(($(date +%s) + 9 * 3600))
while [ "$(date +%s)" -lt "$deadline" ]; do
  remaining=$((deadline - $(date +%s)))
  hours="$(awk -v seconds="$remaining" 'BEGIN { printf "%.4f", seconds / 3600 }')"
  /opt/homebrew/opt/python@3.11/libexec/bin/python3 \
    "$repo/scripts/p0_fetch_overnight_upstream.py" \
    --plan "$repo/configs/data/p0_overnight_upstream_v1.json" \
    --cache "$cache" --remote CPU_NODE \
    --remote-root /srv/small-agent/datasets/small-agent-p0/p0_500m_v1/upstream_v2 \
    --hours "$hours"
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "$(date '+%F %T %z') ALL_FILES_VERIFIED"
    exit 0
  fi
  echo "$(date '+%F %T %z') pass_exit=$code; retrying after 300s if time remains"
  sleep 300
done
echo "$(date '+%F %T %z') NINE_HOUR_WINDOW_ENDED"
exit 2
