#!/usr/bin/env bash
set -euo pipefail
ROOT=/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1
OUT="$ROOT/benchmarks/fusion8_tune_v1"
ENV=/srv/small-agent/experiments/small-agent-base-1b-v1-engineering
PY=/opt/small-agent/venvs/small-agent-l40/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$OUT/deps/vendor:$ENV/deps:$ROOT/code/src:$ROOT/code"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export TRITON_CACHE_DIR="$ENV/triton-cache" TMPDIR="$ENV/tmp"
rm -f "$OUT/tmp/baseline_gradients.pt"
cd "$OUT"
exec "$PY" benchmark_fusion8_tune_v1.py finish
