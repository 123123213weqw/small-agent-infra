#!/usr/bin/env bash
# Run one versioned non-Wikipedia Qwen batch at a time on L40.
# The lock FD is inherited by scorer children: a V100/SSH restart cannot start
# a second writer while orphaned scorers are still appending results.
set -euo pipefail

: "${DATA:?set DATA to the versioned nonwiki review directory}"
test -d "$DATA" || { echo "missing batch directory: $DATA" >&2; exit 2; }
ROOT=${ROOT:-/srv/small-agent/experiments/qwen38-fp8-serving-v1}
CODE=${CODE:-$ROOT/code}
RUNNER=${RUNNER:-$CODE/scripts/p0_run_nonwiki_qwen8.sh}
PY=${PY:-/srv/small-agent/venvs/vllm-qwen38-fp8-v1/bin/python}
LOCK_WAIT_SECONDS=${LOCK_WAIT_SECONDS:-86400}
OWNED_UNITS_FILE="$DATA/qwen8_owned_units.list"
units=(qwen38-fp8-optimized@0.service qwen38-fp8-optimized@1.service qwen38-fp8-optimized@2.service \
       qwen38-fp8-optimized@3.service qwen38-fp8-optimized@4.service \
       qwen38-fp8-optimized@5.service qwen38-fp8-optimized@6.service qwen38-fp8-optimized@7.service)

exec 9>>"$DATA/qwen8_scoring.lock"
echo "waiting for exclusive L40 batch lock: $DATA"
flock -x -w "$LOCK_WAIT_SECONDS" 9 || { echo "timed out waiting for L40 batch lock" >&2; exit 75; }
echo "acquired exclusive L40 batch lock: $DATA"

# The other owner may have finished while this process waited for the lock.
# Verify the marker before starting any vLLM replica or scoring again.
if [[ -f "$DATA/FINISHED8.json" ]]; then
  "$PY" - "$DATA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = root / "input_bands8_v1/manifest.json"
finished = json.loads((root / "FINISHED8.json").read_text())
assert finished["input_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
assert finished["unattempted"] == 0
PY
  echo "batch already verified; skipping duplicate Qwen scoring"
  exit 0
fi

started=()
cleanup() {
  local to_stop=()
  if [[ -f "$DATA/FINISHED8.json" && -f "$OWNED_UNITS_FILE" ]]; then
    # Persist ownership across a V100 restart. Never trust arbitrary unit names
    # from the file: only the eight fixed services above can be stopped.
    for unit in "${units[@]}"; do
      if grep -Fxq -- "$unit" "$OWNED_UNITS_FILE"; then
        to_stop+=("$unit")
      fi
    done
  else
    to_stop=("${started[@]}")
  fi
  if ((${#to_stop[@]})); then
    systemctl --user stop "${to_stop[@]}" || true
  fi
}
trap cleanup EXIT
for unit in "${units[@]}"; do
  if ! systemctl --user is-active --quiet "$unit"; then
    systemctl --user start "$unit"
    printf '%s\n' "$unit" >>"$OWNED_UNITS_FILE"
    started+=("$unit")
  fi
done
DATA="$DATA" CODE="$CODE" ROOT="$ROOT" bash "$RUNNER"
