#!/usr/bin/env bash
set -euo pipefail
ROOT=/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1
OUT="$ROOT/benchmarks/fusion8_tune_v1_retry1"
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
for attempt in $(seq 1 720); do
  state=$(systemctl --user show p0-wave2-fusion8-confirm1-v1.service --property=ActiveState --value)
  case "$state" in
    active|activating|reloading|deactivating)
      printf '%s waiting for prerequisite campaign, %s/720\n' "$(date -Is)" "$attempt"
      sleep 5; continue ;;
  esac
  if ! "$PY" benchmark_fusion8_tune_retry_v1.py prerequisite >prerequisite_check.log 2>&1; then
    printf '{"status":"prerequisite_failed_or_unverified"}\n' >"$OUT/PREREQUISITE_FAILED.json"
    exit 77
  fi
  if "$PY" benchmark_fusion8_tune_retry_v1.py preflight >queue_gpu_snapshot.json 2>queue_wait_reason.txt; then
    ready=1; break
  fi
  printf '%s waiting for eight free cards, %s/720\n' "$(date -Is)" "$attempt"
  sleep 5
done
if [[ "$ready" != 1 ]]; then
  printf '{"status":"waiting_expired","maximum_queue_seconds":3600}\n' >"$OUT/WAIT_EXPIRED.json"
  exit 75
fi
"$PY" benchmark_fusion8_tune_retry_gpu_guard_v1.py --parent-pid "$$" >guard.log 2>&1 &
monitor_pid=$!
"$PY" benchmark_fusion8_tune_retry_v1.py kernel >kernel_preflight.log 2>&1
tail -n 1 kernel_preflight.log
"$PY" benchmark_fusion8_tune_retry_v1.py prepare
run_variant() {
  variant="$1"
  kill -0 "$monitor_pid" || exit 76
  mkdir -p "$OUT/$variant"
  "$PY" benchmark_fusion8_tune_retry_v1.py preflight >"$OUT/$variant/gpu_before.json"
  set +e
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$OUT/benchmark_fusion8_tune_retry_v1.py" worker --variant "$variant" >"$OUT/$variant/worker.log" 2>&1
  rc=$?
  set -e
  echo "$rc" >"$OUT/$variant/exit_code.txt"
  tail -n 2 "$OUT/$variant/worker.log"
  if [[ "$rc" != 0 ]]; then exit "$rc"; fi
  "$PY" - "$OUT/$variant/result.json" <<'CHECK'
import json,sys
r=json.load(open(sys.argv[1]))
if not (r['status']=='measured' and r['gradient_check']['passed'] and r['validation_screen']['passed']):
    raise SystemExit('diagnostic gate failed; no subsequent variant will start')
CHECK
}
for variant in norm_full25 norm_half25 norm_full100 norm_half100 norm_full25_repeat; do
  run_variant "$variant"
done
"$PY" benchmark_fusion8_tune_retry_v1.py select
run_variant winner_confirmation
"$PY" benchmark_fusion8_tune_retry_v1.py summarize
