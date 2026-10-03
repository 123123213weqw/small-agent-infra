#!/usr/bin/env bash
# One-GPU, resumable, local-only Qwen audit of complete-coverage quality chunks.
set -euo pipefail
ROOT=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/finewiki_zh_quality_compare_v2
PY=/srv/small-agent/venvs/vllm-qwen38-fp8-v1/bin/python
LAUNCH=/srv/small-agent/experiments/qwen38-fp8-serving-v1/launch-qwen-single-gpu-tune.sh
UNIT=p0-quality-compare-v2-qwen-b
PORT=18020
export QWEN_LOCAL_API_KEY=EMPTY

cleanup() { systemctl --user stop "$UNIT.service" >/dev/null 2>&1 || true; }
trap cleanup EXIT
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q .; then
  echo 'GPU compute apps already active; refusing to start audit scorer' >&2
  exit 3
fi
systemd-run --user --unit="$UNIT" --collect \
  /bin/bash "$LAUNCH" 0 "$PORT" 32 graphs 0.85
ready=0
for ((i=0; i<80; i++)); do
  if curl --fail --silent --max-time 4 "http://127.0.0.1:$PORT/health" >/dev/null; then
    ready=1; break
  fi
  if ! systemctl --user is-active --quiet "$UNIT.service"; then
    echo 'Qwen service exited before healthy' >&2
    exit 4
  fi
  sleep 6
done
[[ "$ready" == 1 ]] || { echo 'Qwen service health timeout' >&2; exit 5; }
complete=0
for pass in 1 2 3 4 5; do
  echo "quality comparison scoring pass=$pass" >&2
  if "$PY" "$ROOT/code/p0_annotate_quality_chunks_v2.py" \
    --input "$ROOT/input/chunks_v2_640/chunks.input.jsonl" \
    --output "$ROOT/results/chunks.qwen640.results.jsonl" \
    --prompt-file "$ROOT/code/p0_quality_chunk_prompt_v2.txt" \
    --schema-file "$ROOT/code/p0_teacher_quality_v2.json" \
    --text-field review_payload --id-fields document_id \
    --carry-fields parent_document_id,source,stratum,chunk_index,chunk_count,char_start,char_end,prompt_tokens_est \
    --api-base "http://127.0.0.1:$PORT/v1" --api-key-env QWEN_LOCAL_API_KEY \
    --model qwen3.8-27b-fp8 --thinking omit \
    --chat-template-kwargs '{"enable_thinking":false}' \
    --temperature 0 --response-format json_schema --require-document-id-match \
    --workers 12 --max-tokens 640 --timeout 360 --max-retries 2; then
    complete=1
    break
  fi
done
[[ "$complete" == 1 ]] || { echo 'quality scoring incomplete after 5 passes' >&2; exit 6; }
"$PY" "$ROOT/code/p0_summarize_quality_compare_v2.py" \
  --input-dir "$ROOT/input/chunks_v2_640" \
  --results "$ROOT/results/chunks.qwen640.results.jsonl" \
  --output "$ROOT/results/summary_v2.json"
printf '%s\n' "$(date -u +%FT%TZ)" > "$ROOT/results/scoring_process_finished_utc.txt"
