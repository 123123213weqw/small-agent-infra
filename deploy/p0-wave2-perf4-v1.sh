#!/usr/bin/env bash
set -euo pipefail
ROOT=/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1
OUT="$ROOT/benchmarks/perf4_v1"
ENV=/srv/small-agent/experiments/small-agent-base-1b-v1-engineering
PY=/opt/small-agent/venvs/small-agent-l40/bin/python
exec 9>"$OUT/.runner.lock"
flock -n 9 || { echo "Four-card benchmark already running"; exit 1; }
export CUDA_VISIBLE_DEVICES=0,1,5,6
export PYTHONPATH="$ENV/deps:$ROOT/code/src:$ROOT/code"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export TRITON_CACHE_DIR="$ENV/triton-cache" TMPDIR="$ENV/tmp"
cleanup_scratch() { rm -f "$OUT/tmp/baseline_gradients.pt"; }
trap cleanup_scratch EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
cd "$OUT"
"$PY" benchmark_perf4_v1.py prepare
run_variant() {
  local variant="$1" rc
  mkdir -p "$OUT/$variant"
  "$PY" benchmark_perf4_v1.py preflight >"$OUT/$variant/gpu_before.json"
  set +e
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$OUT/benchmark_perf4_v1.py" worker --variant "$variant" \
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
for variant in fp32_b25_fullgc fp32_b100_fullgc fp32_b200_fullgc \
               bf16_b25_fullgc bf16_b100_fullgc \
               fp32_b25_halfgc fp32_b25_nogc fp32_b25_fullgc_repeat; do
  run_variant "$variant"
done
"$PY" benchmark_perf4_v1.py select
run_variant winner_confirmation
"$PY" benchmark_perf4_v1.py summarize
