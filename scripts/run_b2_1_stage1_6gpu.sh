#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/small-agent/venvs/small-agent-l40/bin/python}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/b2_1_stage1}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/b2_1_stage1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/checkpoints/b2_1_stage1}"
MODE="${1:-full}"

names=(A_pointwise B_joint_mse C_joint_rank D_joint_evict E_joint_full F_joint_partial)
gpus=(2 3 4 5 6 7)

mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}"

for gpu in "${gpus[@]}"; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
  if (( used > 1024 )); then
    echo "GPU ${gpu} 当前已使用 ${used} MiB，停止启动。" >&2
    exit 1
  fi
done

for i in "${!names[@]}"; do
  name="${names[$i]}"
  gpu="${gpus[$i]}"
  output="${RUN_DIR}/${name}"
  mkdir -p "${output}"
  extra=()
  if [[ "${MODE}" == "smoke" ]]; then
    extra+=(--smoke)
  elif [[ "${MODE}" != "full" ]]; then
    echo "模式必须是 full 或 smoke" >&2
    exit 2
  fi
  echo "启动 ${name} -> 物理 GPU ${gpu}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" "${ROOT}/experiments/phase0_b21_train.py" \
      --config "${ROOT}/configs/b2_1_stage1/${name}.json" \
      --data-dir "${DATA_DIR}" \
      --output "${output}" \
      --checkpoint-dir "${CHECKPOINT_DIR}" \
      "${extra[@]}" \
      > "${output}/train.log" 2>&1 < /dev/null &
  echo "$!" > "${output}/train.pid"
done

echo "六个实验已经启动。使用 scripts/status_b2_1_stage1.sh 查看。"
