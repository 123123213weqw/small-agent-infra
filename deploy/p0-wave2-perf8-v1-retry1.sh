#!/usr/bin/env bash
set -euo pipefail
ROOT=/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1
OUT="$ROOT/benchmarks/perf8_v1_retry1"
ENV=/srv/small-agent/experiments/small-agent-base-1b-v1-engineering
PY=/opt/small-agent/venvs/small-agent-l40/bin/python
exec 9>"$OUT/.runner.lock"
flock -n 9 || { echo "Eight-card benchmark already running"; exit 1; }
export SMALL_AGENT_PERF8_TRIAL=perf8_v1_retry1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$ENV/deps:$ROOT/code/src:$ROOT/code"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export TRITON_CACHE_DIR="$ENV/triton-cache" TMPDIR="$ENV/tmp"
monitor_pid=
cleanup_scratch() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  rm -f "$OUT/tmp/baseline_gradients.pt"
}
trap cleanup_scratch EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
cd "$OUT"
# Bounded queue only: do not terminate another job or use occupied cards.
ready=0
for attempt in $(seq 1 30); do
  # An active multi-variant benchmark can briefly have no CUDA processes between groups.
  # Wait for its entire service, not merely a momentarily empty GPU snapshot.
  prior_active=0
  if systemctl --user is-active --quiet perf4-g237.service; then
    prior_active=1
    echo "Existing four-card campaign is still active" >"$OUT/queue_wait_reason.txt"
  fi
  if [[ "$prior_active" == 0 ]] && "$PY" benchmark_perf8_v1.py preflight >"$OUT/queue_gpu_snapshot.json" 2>"$OUT/queue_wait_reason.txt"; then
    ready=1
    break
  fi
  printf '%s waiting for all eight cards, attempt %s/30\n' "$(date -Is)" "$attempt"
  sleep 30
done
if [[ "$ready" != 1 ]]; then
  echo "Eight cards did not become free within 15 minutes; no jobs were stopped."
  exit 75
fi
"$PY" benchmark_perf8_gpu_guard_v1.py --parent-pid "$$" >"$OUT/guard.log" 2>&1 &
monitor_pid=$!
"$PY" benchmark_perf8_v1.py prepare
run_variant() {
  local variant="$1" rc
  kill -0 "$monitor_pid" || { echo "GPU guard exited; abort own experiment"; exit 76; }
  mkdir -p "$OUT/$variant"
  "$PY" benchmark_perf8_v1.py preflight >"$OUT/$variant/gpu_before.json"
  set +e
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$OUT/benchmark_perf8_v1.py" worker --variant "$variant" \
    >"$OUT/$variant/worker.log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" >"$OUT/$variant/exit_code.txt"
  tail -n 5 "$OUT/$variant/worker.log"
  if [[ "$variant" == fp32_b25_fullgc && "$rc" != 0 ]]; then
    echo "Baseline failed; stop without touching other jobs"
    exit "$rc"
  fi
}
for variant in fp32_b25_fullgc bf16_b25_fullgc bf16_b100_fullgc \
               bf16_b25_halfgc fp32_b25_fullgc_repeat; do
  run_variant "$variant"
done
"$PY" benchmark_perf8_v1.py select
run_variant winner_confirmation
"$PY" benchmark_perf8_v1.py summarize
