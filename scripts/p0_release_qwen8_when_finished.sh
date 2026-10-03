#!/usr/bin/env bash
# Release only the eight Qwen replicas used by this non-Wikipedia review after
# the batch verifier has written its final success marker. Survives V100 restarts.
set -euo pipefail

: "${DATA:?set DATA to the versioned nonwiki review directory}"
PY=${PY:-/srv/small-agent/venvs/vllm-qwen38-fp8-v1/bin/python}
units=(qwen38-fp8-optimized@{0..7}.service)

while [[ ! -f "$DATA/FINISHED8.json" ]]; do
  sleep 30
done

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

systemctl --user stop "${units[@]}"
for unit in "${units[@]}"; do
  if systemctl --user is-active --quiet "$unit"; then
    echo "replica still active: $unit" >&2
    exit 1
  fi
done
echo "verified FINISHED8.json; all eight Qwen vLLM replicas stopped"
# This one-shot watchdog is no longer needed after successful release.
systemctl --user disable p0-nonwiki-qwen8-autorelease.service || true
