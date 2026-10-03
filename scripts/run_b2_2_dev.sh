#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/small-agent/venvs/small-agent-l40/bin/python}"
GPU="${GPU:-2}"
DATA_DIR="${ROOT}/data/b2_2_robust"
RUN_DIR="${ROOT}/runs/b2_2_robust/C_robust_dev_seed1"
CHECKPOINT_DIR="${ROOT}/checkpoints/b2_2_robust"

if [[ ! -f "${DATA_DIR}/manifest.json" ]]; then
  echo "缺少 ${DATA_DIR}/manifest.json，请先生成冻结数据。" >&2
  exit 1
fi
if [[ -e "${RUN_DIR}/summary.json" ]]; then
  echo "正式结果已经存在，拒绝覆盖：${RUN_DIR}" >&2
  exit 1
fi
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')"
if (( used > 1024 )); then
  echo "GPU ${GPU} 当前已使用 ${used} MiB，停止启动。" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}"
nohup env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" "${ROOT}/experiments/phase0_b21_train.py" \
    --config "${ROOT}/configs/b2_2_robust/C_robust_dev_seed1.json" \
    --data-dir "${DATA_DIR}" \
    --output "${RUN_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    > "${RUN_DIR}/train.log" 2>&1 < /dev/null &
echo "$!" > "${RUN_DIR}/train.pid"
echo "B2.2-Dev Seed 1 已启动：GPU ${GPU}，PID $(cat "${RUN_DIR}/train.pid")"
