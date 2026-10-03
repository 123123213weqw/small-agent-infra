#!/usr/bin/env bash
# Run pinned non-Wikipedia Qwen inputs on six L40 replicas; stop only replicas started here.
set -euo pipefail

: "${DATA:?set DATA to the versioned nonwiki review directory}"
ROOT=${ROOT:-/srv/small-agent/experiments/qwen38-fp8-serving-v1}
CODE=${CODE:-$ROOT/code}
RUNNER=${RUNNER:-$CODE/scripts/p0_run_nonwiki_qwen6.sh}
units=(qwen38-fp8-optimized@0.service qwen38-fp8-optimized@2.service \
       qwen38-fp8-optimized@3.service qwen38-fp8-optimized@4.service \
       qwen38-fp8-optimized@5.service qwen38-fp8-optimized@6.service)
started=()
cleanup() {
  if ((${#started[@]})); then
    systemctl --user stop "${started[@]}" || true
  fi
}
trap cleanup EXIT
for unit in "${units[@]}"; do
  if ! systemctl --user is-active --quiet "$unit"; then
    systemctl --user start "$unit"
    started+=("$unit")
  fi
done
DATA="$DATA" CODE="$CODE" ROOT="$ROOT" bash "$RUNNER"
