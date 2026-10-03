#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/b2_1_relation_data}"
names=(C_seed0 C_seed1 C_seed2)

printf '%-10s %-9s %-9s %s\n' "实验" "PID" "状态" "最新输出"
for name in "${names[@]}"; do
  directory="${RUN_DIR}/${name}"
  pid="-"
  status="未启动"
  if [[ -f "${directory}/train.pid" ]]; then
    pid="$(cat "${directory}/train.pid")"
    if kill -0 "${pid}" 2>/dev/null; then
      status="运行中"
    elif [[ -f "${directory}/summary.json" ]]; then
      status="已完成"
    else
      status="失败"
    fi
  fi
  latest="$(tail -n 1 "${directory}/train.log" 2>/dev/null || true)"
  printf '%-10s %-9s %-9s %s\n' "${name}" "${pid}" "${status}" "${latest:0:100}"
done

echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
