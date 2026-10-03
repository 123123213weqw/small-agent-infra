#!/usr/bin/env bash
set -uo pipefail
ROOT=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/finewiki_zh_quality_probe_v1
BASE=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/code
CODE=/srv/small-agent/experiments/qwen38-fp8-serving-v1/code/scripts/annotate_jsonl.py
PY=/srv/small-agent/venvs/vllm-qwen38-fp8-v1/bin/python
export QWEN_LOCAL_API_KEY=EMPTY

if ! curl --fail --silent --max-time 5 http://127.0.0.1:18000/health >/dev/null; then
  echo 'Qwen single-GPU service is not healthy' >&2
  printf '%s\n' 3 > "$ROOT/logs/qwen.exit"
  exit 3
fi
"$PY" "$CODE" \
  --input "$ROOT/input/context_split_v1/scoreable.input.jsonl" \
  --output "$ROOT/results/qwen_v1.results.jsonl" \
  --prompt-file "$BASE/configs/data/p0_teacher_quality_prompt_v2.txt" \
  --schema-file "$BASE/schemas/p0_teacher_quality_v2.json" \
  --text-field review_payload --id-fields document_id \
  --carry-fields stratum,chars,has_math,prompt_tokens_est,title \
  --api-base http://127.0.0.1:18000/v1 --api-key-env QWEN_LOCAL_API_KEY \
  --model qwen3.8-27b-fp8 --thinking omit \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --temperature 0 --response-format json_schema --require-document-id-match \
  --workers 12 --max-tokens 384 --timeout 360 --max-retries 1
status=$?
printf '%s\n' "$status" > "$ROOT/logs/qwen.exit"
exit "$status"
