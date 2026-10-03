#!/usr/bin/env bash
# Run eight independent, resumable rating workers. Output is not a training set.
set -uo pipefail

ROOT=/srv/small-agent/experiments/qwen38-fp8-serving-v1
DATA=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/qwen_official_fp8_wave1_v1
PY=/srv/small-agent/venvs/vllm-qwen38-fp8-v1/bin/python
SCRIPT="$ROOT/code/scripts/annotate_jsonl.py"
PROMPT=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/code/configs/data/p0_teacher_quality_prompt_v2.txt
SCHEMA=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/code/schemas/p0_teacher_quality_v2.json

test -f "$DATA/input_shards/manifest.json" || { echo 'missing input manifest' >&2; exit 2; }
test -f "$PROMPT" && test -f "$SCHEMA" || { echo 'missing prompt/schema' >&2; exit 2; }
mkdir -p "$DATA/results"
export QWEN_LOCAL_API_KEY=EMPTY
ready=0
for attempt in {1..60}; do
  ready=1
  for i in {0..7}; do
    curl --fail --silent --max-time 5 "http://127.0.0.1:1800${i}/health" >/dev/null || ready=0
  done
  [[ "$ready" -eq 1 ]] && break
  sleep 10
done
[[ "$ready" -eq 1 ]] || { echo 'Qwen replicas did not become healthy in 10 minutes' >&2; exit 3; }
pids=()
for i in {0..7}; do
  "$PY" "$SCRIPT" \
    --input "$DATA/input_shards/shard-${i}.input.jsonl" \
    --output "$DATA/results/shard-${i}.results.jsonl" \
    --prompt-file "$PROMPT" --schema-file "$SCHEMA" \
    --text-field review_payload --id-fields document_id \
    --carry-fields current_bucket,token_count,prompt_tokens_est,candidate_relpath,row_in_shard \
    --api-base "http://127.0.0.1:1800${i}/v1" \
    --api-key-env QWEN_LOCAL_API_KEY --model qwen3.8-27b-fp8 \
    --thinking omit --chat-template-kwargs '{"enable_thinking":false}' \
    --temperature 0 --response-format json_schema --require-document-id-match \
    --workers 8 --max-tokens 384 --timeout 180 --max-retries 1 \
    >"$DATA/results/shard-${i}.log" 2>&1 &
  pids+=("$!")
done

status=0
for i in {0..7}; do
  if wait "${pids[i]}"; then
    echo "shard ${i} done"
  else
    echo "shard ${i} exited nonzero" >&2
    status=1
  fi
done
if [[ "$status" -eq 0 ]]; then
  "$PY" "$ROOT/code/scripts/p0_verify_qwen_wave1.py" \
    --data-dir "$DATA" --write-complete || status=1
fi
exit "$status"
