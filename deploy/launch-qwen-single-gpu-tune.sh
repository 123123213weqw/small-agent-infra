#!/usr/bin/env bash
# Isolated single-GPU vLLM launcher; does not alter the eight production units.
set -euo pipefail
gpu="${1:?GPU_INDEX}"
port="${2:?PORT}"
sequences="${3:?MAX_NUM_SEQS}"
mode="${4:?eager|graphs}"
memory_util="${5:-0.85}"
[[ "$gpu" =~ ^[0-7]$ && "$port" =~ ^[0-9]+$ && "$sequences" =~ ^[0-9]+$ ]] || exit 2
[[ "$mode" == eager || "$mode" == graphs ]] || exit 2
venv=/srv/small-agent/venvs/vllm-qwen38-fp8-v1
model=/srv/small-agent/models/Qwen3.8-27B-FP8
cuda="$venv/lib/python3.10/site-packages/nvidia/cu13"
flags=(--served-model-name qwen3.8-27b-fp8
  --host 127.0.0.1 --port "$port"
  --language-model-only --max-model-len 4096 --max-num-seqs "$sequences"
  --gpu-memory-utilization "$memory_util")
if [[ "$mode" == eager ]]; then flags+=(--enforce-eager); fi
exec env CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_HOME="$cuda" PATH="$venv/bin:$cuda/bin:$PATH" \
  LD_LIBRARY_PATH="$cuda/lib:${LD_LIBRARY_PATH:-}" \
  "$venv/bin/vllm" serve "$model" "${flags[@]}"
