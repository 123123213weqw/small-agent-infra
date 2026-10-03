#!/usr/bin/env bash
# Persistent one-L40-GPU audit of complete <6000-character documents.
set -euo pipefail
ROOT=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/finewiki_zh_quality_compare_v2
VENV=/srv/small-agent/venvs/vllm-qwen38-fp8-v1
MODEL=/srv/small-agent/models/Qwen3.8-27B-FP8
CUDA="$VENV/lib/python3.10/site-packages/nvidia/cu13"
UNIT=p0-quality-whole-short-qwen
PORT=18021
export QWEN_LOCAL_API_KEY=EMPTY

cleanup() { systemctl --user stop "$UNIT.service" >/dev/null 2>&1 || true; }
trap cleanup EXIT
if nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F, '$1==0 && $2+0>1024 {found=1} END {exit !found}'; then
  echo 'GPU 0 is already in use; refusing to start scorer' >&2
  exit 3
fi
systemd-run --user --unit="$UNIT" --collect \
  /usr/bin/env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_HOME="$CUDA" PATH="$VENV/bin:$CUDA/bin:$PATH" \
  LD_LIBRARY_PATH="$CUDA/lib:${LD_LIBRARY_PATH:-}" \
  "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name qwen3.8-27b-fp8 --host 127.0.0.1 --port "$PORT" \
  --language-model-only --max-model-len 8192 --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 --enforce-eager
ready=0
for ((i=0; i<100; i++)); do
  if curl --fail --silent --max-time 4 "http://127.0.0.1:$PORT/health" >/dev/null; then
    ready=1; break
  fi
  systemctl --user is-active --quiet "$UNIT.service" || { echo 'Qwen service exited' >&2; exit 4; }
  sleep 6
done
[[ "$ready" == 1 ]] || { echo 'Qwen service health timeout' >&2; exit 5; }
complete=0
for pass in 1 2 3; do
  echo "whole-document quality pass=$pass" >&2
  if "$VENV/bin/python" "$ROOT/code/p0_annotate_quality_chunks_v2.py" \
    --input "$ROOT/input/whole_short_v1/whole_short.input.jsonl" \
    --output "$ROOT/results/whole_short.qwen.results.jsonl" \
    --prompt-file "$ROOT/code/p0_teacher_quality_prompt_v2.txt" \
    --schema-file "$ROOT/code/p0_teacher_quality_v2.json" \
    --text-field review_payload --id-fields document_id \
    --carry-fields source,stratum,chars,population_in_stratum,sample_size_in_stratum,prompt_tokens_est \
    --api-base "http://127.0.0.1:$PORT/v1" --api-key-env QWEN_LOCAL_API_KEY \
    --model qwen3.8-27b-fp8 --thinking omit \
    --chat-template-kwargs '{"enable_thinking":false}' \
    --temperature 0 --response-format json_schema --require-document-id-match \
    --workers 8 --max-tokens 1024 --timeout 360 --max-retries 2; then
    complete=1; break
  fi
done
[[ "$complete" == 1 ]] || { echo 'whole short-document scoring incomplete after 3 passes' >&2; exit 6; }
printf '%s\n' "$(date -u +%FT%TZ)" > "$ROOT/results/whole_short_finished_utc.txt"
