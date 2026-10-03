#!/usr/bin/env bash
set -euo pipefail
ROOT=/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1
OUT="$ROOT/benchmarks/fusion8_v1"
ENV=/srv/small-agent/experiments/small-agent-base-1b-v1-engineering
PY=/opt/small-agent/venvs/small-agent-l40/bin/python
exec 9>"$OUT/.runner.lock"
flock -n 9 || exit 1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$OUT/deps/vendor:$ENV/deps:$ROOT/code/src:$ROOT/code"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export TRITON_CACHE_DIR="$ENV/triton-cache" TMPDIR="$ENV/tmp"
monitor_pid=
cleanup() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  rm -f "$OUT/tmp/baseline_gradients.pt"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
cd "$OUT"
ready=0
for attempt in $(seq 1 30); do
  prior_active=0
  for unit in perf4-g237.service p0-wave2-perf8-v1.service p0-wave2-perf8-v1-retry1.service p0-wave2-loss8-v1.service p0-wave2-base1b-full8-v1.service; do
    if systemctl --user is-active --quiet "$unit"; then prior_active=1; fi
  done
  if [[ "$prior_active" == 0 ]] && "$PY" benchmark_fusion8_v1.py preflight >queue_gpu_snapshot.json 2>queue_wait_reason.txt; then
    ready=1; break
  fi
  printf '%s waiting for eight free cards, %s/30\n' "$(date -Is)" "$attempt"
  sleep 30
done
[[ "$ready" == 1 ]] || exit 75
"$PY" benchmark_fusion8_gpu_guard_v1.py --parent-pid "$$" >guard.log 2>&1 &
monitor_pid=$!
"$PY" benchmark_fusion8_v1.py kernel >kernel_preflight.log 2>&1
tail -n 1 kernel_preflight.log
"$PY" benchmark_fusion8_v1.py prepare
for variant in torch_full rmsnorm_only swiglu_only norm_swi torch_full_repeat norm_swi_repeat; do
  kill -0 "$monitor_pid" || exit 76
  mkdir -p "$OUT/$variant"
  "$PY" benchmark_fusion8_v1.py preflight >"$OUT/$variant/gpu_before.json"
  set +e
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$OUT/benchmark_fusion8_v1.py" worker --variant "$variant" >"$OUT/$variant/worker.log" 2>&1
  rc=$?
  set -e
  echo "$rc" >"$OUT/$variant/exit_code.txt"
  tail -n 2 "$OUT/$variant/worker.log"
  if [[ "$variant" == torch_full && "$rc" != 0 ]]; then exit "$rc"; fi
done
"$PY" benchmark_fusion8_v1.py summarize
