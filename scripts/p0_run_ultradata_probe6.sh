#!/usr/bin/env bash
# Six-card source-quality probe. Results are audit labels, not training approval.
set -uo pipefail

ROOT=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/qwen_ultradata_probe6_v1
CODE=/srv/small-agent/experiments/qwen38-fp8-serving-v1/code/scripts
PY=/srv/small-agent/venvs/vllm-qwen38-fp8-v1/bin/python
PROMPT=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/code/configs/data/p0_teacher_quality_prompt_v2.txt
SCHEMA=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/code/schemas/p0_teacher_quality_v2.json
GPUS=(0 2 3 4 5 6)
export QWEN_LOCAL_API_KEY=EMPTY

test -f "$ROOT/inputs/manifest.json" || exit 2
for band in heavy medium light; do
  case "$band" in
    heavy) workers=12; timeout=600 ;;
    medium) workers=20; timeout=360 ;;
    light) workers=32; timeout=180 ;;
  esac
  for gpu in "${GPUS[@]}"; do
    curl -fsS --max-time 10 "http://127.0.0.1:1800${gpu}/health" >/dev/null || {
      echo "GPU $gpu Qwen endpoint unavailable before $band" >&2; exit 3;
    }
  done
  mkdir -p "$ROOT/results/$band"
  pids=()
  for shard in {0..5}; do
    gpu=${GPUS[shard]}
    "$PY" "$CODE/annotate_jsonl.py" \
      --input "$ROOT/inputs/$band/shard-${shard}.input.jsonl" \
      --output "$ROOT/results/$band/shard-${shard}.results.jsonl" \
      --prompt-file "$PROMPT" --schema-file "$SCHEMA" \
      --text-field review_payload --id-fields document_id \
      --carry-fields source_id,current_bucket,raw_row,source_path,upstream_score,text_chars,prompt_tokens_est \
      --api-base "http://127.0.0.1:1800${gpu}/v1" \
      --api-key-env QWEN_LOCAL_API_KEY --model qwen3.8-27b-fp8 \
      --thinking omit --chat-template-kwargs '{"enable_thinking":false}' \
      --temperature 0 --response-format json_schema --require-document-id-match \
      --workers "$workers" --max-tokens 384 --timeout "$timeout" --max-retries 1 \
      >"$ROOT/results/$band/shard-${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || echo "one scorer exited with failures; checking quarantine" >&2
  done
  "$PY" "$CODE/p0_verify_ultradata_probe6.py" --root "$ROOT" --band "$band" --max-missing-ratio 0.01 || exit 4
done
"$PY" "$CODE/p0_verify_ultradata_probe6.py" --root "$ROOT" --final --max-missing-ratio 0.01
