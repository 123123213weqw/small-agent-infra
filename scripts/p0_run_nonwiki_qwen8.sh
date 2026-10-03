#!/usr/bin/env bash
# Eight-card Wave1 rating with length-aware concurrency. NOT training-approved.
set -uo pipefail

ROOT=${ROOT:-/srv/small-agent/experiments/qwen38-fp8-serving-v1}
CODE=${CODE:-$ROOT/code}
DATA=${DATA:-/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/qwen_official_fp8_wave1_v1}
PY=${PY:-/srv/small-agent/venvs/vllm-qwen38-fp8-v1/bin/python}
PROMPT=${PROMPT:-/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/code/configs/data/p0_teacher_quality_prompt_v2.txt}
SCHEMA=${SCHEMA:-/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/code/schemas/p0_teacher_quality_v2.json}
GPUS=(0 1 2 3 4 5 6 7)

test -f "$DATA/input_bands8_v1/manifest.json" || { echo 'missing eight-card input manifest' >&2; exit 2; }
"$PY" - "$DATA/input_bands8_v1/manifest.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert list(x['gpu_indices'])==[0,1,2,3,4,5,6,7]
assert [b['name'] for b in x['bands']]==['heavy','medium','light']
assert [b['workers_per_gpu'] for b in x['bands']]==[12,20,32]
assert sum(b['documents'] for b in x['bands'])==x['total_documents']>0
PY
export QWEN_LOCAL_API_KEY=EMPTY

for band in heavy medium light; do
  case "$band" in
    heavy) workers=12; timeout=600 ;;
    medium) workers=20; timeout=360 ;;
    light) workers=32; timeout=180 ;;
  esac
  ready=0
  for attempt in {1..60}; do
    ready=1
    for gpu in "${GPUS[@]}"; do
      curl --fail --silent --max-time 5 "http://127.0.0.1:1800${gpu}/health" >/dev/null || ready=0
    done
    [[ "$ready" -eq 1 ]] && break
    sleep 10
  done
  [[ "$ready" -eq 1 ]] || { echo "eight Qwen replicas unhealthy before $band" >&2; exit 3; }

  mkdir -p "$DATA/results8/$band"
  pids=()
  for index in {0..7}; do
    gpu="${GPUS[index]}"
    "$PY" "$CODE/scripts/annotate_jsonl.py" \
      --input "$DATA/input_bands8_v1/$band/shard-${index}.input.jsonl" \
      --output "$DATA/results8/$band/shard-${index}.results.jsonl" \
      --prompt-file "$PROMPT" --schema-file "$SCHEMA" \
      --text-field review_payload --id-fields document_id \
      --carry-fields current_bucket,token_count,prompt_tokens_est,candidate_relpath,row_in_shard \
      --api-base "http://127.0.0.1:1800${gpu}/v1" \
      --api-key-env QWEN_LOCAL_API_KEY --model qwen3.8-27b-fp8 \
      --thinking omit --chat-template-kwargs '{"enable_thinking":false}' \
      --temperature 0 --response-format json_schema --require-document-id-match \
      --workers "$workers" --max-tokens 384 --timeout "$timeout" --max-retries 1 \
      >"$DATA/results8/$band/shard-${index}.log" 2>&1 &
    pids+=("$!")
  done
  for index in {0..7}; do
    if wait "${pids[index]}"; then
      echo "$band shard $index scorer exited 0"
    else
      echo "$band shard $index scorer exited nonzero; verifying quarantined failures" >&2
    fi
  done
  "$PY" "$CODE/scripts/p0_verify_qwen_eight.py" \
    --data-dir "$DATA" --band "$band" --max-missing-ratio 0.01 || exit 4
done
"$PY" "$CODE/scripts/p0_verify_qwen_eight.py" \
  --data-dir "$DATA" --final --max-missing-ratio 0.01
