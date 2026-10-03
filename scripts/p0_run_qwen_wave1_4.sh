#!/usr/bin/env bash
# Four-card, append-only Wave1 rating. Labels are NOT training-approved.
set -uo pipefail

ROOT=/srv/small-agent/experiments/qwen38-fp8-serving-v1
DATA=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/qwen_official_fp8_wave1_v1
PY=/srv/small-agent/venvs/vllm-qwen38-fp8-v1/bin/python
PROMPT=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/code/configs/data/p0_teacher_quality_prompt_v2.txt
SCHEMA=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/code/schemas/p0_teacher_quality_v2.json
GPUS=(0 2 3 4)

test -f "$DATA/input_shards/manifest.json" || { echo 'missing input manifest' >&2; exit 2; }
"$PY" - "$DATA/input_shards/manifest.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert len(x['shards'])==4 and x['scoreable_documents']>0
PY
mkdir -p "$DATA/results"
export QWEN_LOCAL_API_KEY=EMPTY
ready=0
for attempt in {1..60}; do
  ready=1
  for gpu in "${GPUS[@]}"; do
    curl --fail --silent --max-time 5 "http://127.0.0.1:1800${gpu}/health" >/dev/null || ready=0
  done
  [[ "$ready" -eq 1 ]] && break
  sleep 10
done
[[ "$ready" -eq 1 ]] || { echo 'four Qwen replicas did not become healthy' >&2; exit 3; }

pids=()
for index in {0..3}; do
  gpu="${GPUS[index]}"
  "$PY" "$ROOT/code/scripts/annotate_jsonl.py" \
    --input "$DATA/input_shards/shard-${index}.input.jsonl" \
    --output "$DATA/results/shard-${index}.results.jsonl" \
    --prompt-file "$PROMPT" --schema-file "$SCHEMA" \
    --text-field review_payload --id-fields document_id \
    --carry-fields current_bucket,token_count,prompt_tokens_est,candidate_relpath,row_in_shard \
    --api-base "http://127.0.0.1:1800${gpu}/v1" \
    --api-key-env QWEN_LOCAL_API_KEY --model qwen3.8-27b-fp8 \
    --thinking omit --chat-template-kwargs '{"enable_thinking":false}' \
    --temperature 0 --response-format json_schema --require-document-id-match \
    --workers 32 --max-tokens 384 --timeout 180 --max-retries 1 \
    >"$DATA/results/shard-${index}.log" 2>&1 &
  pids+=("$!")
done

for index in {0..3}; do
  if wait "${pids[index]}"; then
    echo "shard ${index} scorer exited 0"
  else
    echo "shard ${index} scorer exited nonzero; checking whether invalid documents can be quarantined" >&2
  fi
done
"$PY" "$ROOT/code/scripts/p0_verify_qwen_wave1.py" \
  --data-dir "$DATA" --write-finished --max-missing-ratio 0.01
