#!/usr/bin/env bash
# Execute a GDN test with only the version-pinned, project-local L40 packages.
set -euo pipefail

ROOT="${SMALL_AGENT_GDN_ROOT:-/srv/small-agent/experiments/small-agent-p0-gdn-hybrid-smoke-v1}"
PYTHON="/opt/small-agent/venvs/small-agent-l40/bin/python"
if [[ ! -x "$PYTHON" || ! -d "$ROOT/deps" || ! -d "$ROOT/code" ]]; then
  echo "Missing pinned L40 GDN environment under $ROOT" >&2
  exit 2
fi

export PYTHONPATH="$ROOT/deps:$ROOT/code/src:$ROOT/code"
export PYTHONNOUSERSITE=1
export TRITON_CACHE_DIR="$ROOT/triton-cache"
export TMPDIR="$ROOT/tmp"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"

"$PYTHON" - <<'PY'
from importlib.metadata import version
from pathlib import Path
import json
import os

root = Path(os.environ["TRITON_CACHE_DIR"]).parent
lock = json.loads((root / "code" / "p0_l40_fast_env_v1.json").read_text())
for name, expected in lock["packages"].items():
    actual = version(name)
    if actual != expected:
        raise SystemExit(f"Environment mismatch: {name} expected {expected}, got {actual}")
PY

exec "$PYTHON" -u "$@"
