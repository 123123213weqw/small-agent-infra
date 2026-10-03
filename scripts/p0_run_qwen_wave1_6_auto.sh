#!/usr/bin/env bash
# Start only replicas this run owns, then score and stop only those replicas.
set -euo pipefail
units=(qwen38-fp8-optimized@0.service qwen38-fp8-optimized@2.service \
       qwen38-fp8-optimized@3.service qwen38-fp8-optimized@4.service \
       qwen38-fp8-optimized@5.service qwen38-fp8-optimized@6.service)
started=()
cleanup() {
  if ((${#started[@]})); then systemctl --user stop "${started[@]}" || true; fi
}
trap cleanup EXIT
for unit in "${units[@]}"; do
  if ! systemctl --user is-active --quiet "$unit"; then
    systemctl --user start "$unit"
    started+=("$unit")
  fi
done
bash "${CODE:-${ROOT:-/srv/small-agent/experiments/qwen38-fp8-serving-v1}/code}/scripts/p0_run_qwen_wave1_6.sh"
