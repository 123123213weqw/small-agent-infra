#!/usr/bin/env bash
# Trusted launcher for the v3 owned-service wrapper. No corpus commands execute.
set -euo pipefail
gpu=${1:?GPU index required}; port=${2:?loopback port required}
[[ "$gpu" =~ ^[0-7]$ && "$port" =~ ^[0-9]+$ ]] || exit 2
venv=/srv/small-agent/venvs/vllm-qwen38-fp8-v1
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
export CUDA_HOME="$venv/lib/python3.10/site-packages/nvidia/cu13"
export PATH="$venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
exec "$venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --model /srv/small-agent/models/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b-fp8 --host 127.0.0.1 --port "$port" \
  --tensor-parallel-size 1 --language-model-only --enforce-eager \
  --max-model-len 16384 --max-num-seqs 4 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.80
