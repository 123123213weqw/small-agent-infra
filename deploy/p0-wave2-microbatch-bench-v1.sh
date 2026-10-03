#!/usr/bin/env bash
set -euo pipefail
ROOT=/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1
OUT="$ROOT/benchmarks/microbatch_v1"
ENV=/srv/small-agent/experiments/small-agent-base-1b-v1-engineering
PY=/opt/small-agent/venvs/small-agent-l40/bin/python
exec 9>"$OUT/.runner.lock"
flock -n 9 || { echo "This benchmark is already running"; exit 1; }
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$ENV/deps:$ROOT/code/src:$ROOT/code"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export TRITON_CACHE_DIR="$ENV/triton-cache" TMPDIR="$ENV/tmp"
cleanup_scratch() { rm -f "$OUT/tmp/baseline_gradients.pt"; }
trap cleanup_scratch EXIT
cd "$OUT"
"$PY" "$OUT/benchmark_microbatch_v1.py" prepare
for variant in micro1_acc4 micro2_acc2 micro4_acc1; do
  mkdir -p "$OUT/$variant"
  "$PY" -c "import sys;sys.path.insert(0,'$OUT');from benchmark_microbatch_v1 import gpu_is_free;gpu_is_free()"
  set +e
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$OUT/benchmark_microbatch_v1.py" worker --variant "$variant" \
    >"$OUT/$variant/worker.log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" >"$OUT/$variant/exit_code.txt"
  tail -n 8 "$OUT/$variant/worker.log"
  if [[ "$variant" == micro1_acc4 && "$rc" != 0 ]]; then
    echo "Baseline failed; not launching other variants"
    exit "$rc"
  fi
done
"$PY" "$OUT/benchmark_microbatch_v1.py" summarize
