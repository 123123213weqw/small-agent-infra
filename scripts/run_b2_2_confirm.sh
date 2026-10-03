#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/small-agent/venvs/small-agent-l40/bin/python}"
DATA_DIR="${ROOT}/data/b2_2_robust"
RUN_DIR="${ROOT}/runs/b2_2_robust"
CHECKPOINT_DIR="${ROOT}/checkpoints/b2_2_robust"

names=(C_robust_confirm_seed3 C_robust_confirm_seed4 C_robust_confirm_seed5)
gpus=(2 3 4)

[[ -f "${DATA_DIR}/manifest.json" ]] || { echo "缺少冻结数据集" >&2; exit 1; }
for gpu in "${gpus[@]}"; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
  (( used <= 1024 )) || { echo "GPU ${gpu} 已使用 ${used} MiB" >&2; exit 1; }
done
for name in "${names[@]}"; do
  [[ ! -e "${RUN_DIR}/${name}/summary.json" ]] || { echo "拒绝覆盖 ${name}" >&2; exit 1; }
done

mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}"
for i in "${!names[@]}"; do
  name="${names[$i]}"
  gpu="${gpus[$i]}"
  output="${RUN_DIR}/${name}"
  mkdir -p "${output}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" "${ROOT}/experiments/phase0_b21_train.py" \
      --config "${ROOT}/configs/b2_2_robust/${name}.json" \
      --data-dir "${DATA_DIR}" \
      --output "${output}" \
      --checkpoint-dir "${CHECKPOINT_DIR}" \
      > "${output}/train.log" 2>&1 < /dev/null &
  echo "$!" > "${output}/train.pid"
  echo "启动 ${name} -> GPU ${gpu}, PID $(cat "${output}/train.pid")"
done
