#!/usr/bin/env bash
set -euo pipefail

LOGDIR="${1:-runs}"
PORT="${2:-6006}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "TensorBoard: http://127.0.0.1:${PORT}"
echo "日志目录: ${LOGDIR}"
exec "${PYTHON_BIN}" -m tensorboard.main \
  --logdir "${LOGDIR}" \
  --host 127.0.0.1 \
  --port "${PORT}"
