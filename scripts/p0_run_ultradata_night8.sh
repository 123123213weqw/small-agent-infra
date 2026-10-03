#!/usr/bin/env bash
# Resumable, eight-endpoint quality audit. No training data is approved here.
set -euo pipefail

BASE=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1
RAW=$BASE/upstream_ultradata_probe_v1
PRIOR=$BASE/qwen_ultradata_probe6_v1
ROOT=$BASE/qwen_ultradata_night8_v1
CODE=/srv/small-agent/experiments/qwen38-fp8-serving-v1/code/scripts
PY=/srv/small-agent/venvs/vllm-qwen38-fp8-v1/bin/python
PROMPT=$BASE/code/configs/data/p0_teacher_quality_prompt_v2.txt
SCHEMA=$BASE/code/schemas/p0_teacher_quality_v2.json
MODEL=/srv/small-agent/models/Qwen3.8-27B-FP8
export QWEN_LOCAL_API_KEY=EMPTY

ready=0
for attempt in {1..90}; do
  ready=1
  for gpu in {0..7}; do
    curl -fsS --max-time 5 "http://127.0.0.1:1800${gpu}/health" >/dev/null || ready=0
  done
  [[ "$ready" -eq 1 ]] && break
  sleep 10
done
[[ "$ready" -eq 1 ]] || { echo "eight endpoints not healthy" >&2; exit 3; }
echo "all eight Qwen endpoints healthy"

if [[ ! -f "$ROOT/inputs/manifest.json" ]]; then
  [[ ! -e "$ROOT/inputs" ]] || { echo "partial input directory exists; refusing to overwrite" >&2; exit 2; }
  mkdir -p "$ROOT"
  "$PY" "$CODE/p0_prepare_ultradata_night8.py" \
    --raw-dir "$RAW" --prior-probe-dir "$PRIOR" \
    --output-dir "$ROOT/inputs" --prompt "$PROMPT" --model-path "$MODEL"
fi
"$PY" "$CODE/p0_verify_ultradata_night8.py" --root "$ROOT" --status

if [[ ! -f "$ROOT/STARTED.json" ]]; then
  "$PY" - "$ROOT" <<'PY'
import datetime, hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = root / "inputs" / "manifest.json"
payload = {"stage": "night8_quality_audit_not_training_approved",
           "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "input_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
           "gpu_indices": list(range(8)), "max_runtime_hours": 9}
(root / "STARTED.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
fi

for band in heavy medium light; do
  case "$band" in
    heavy) normal_workers=12; shared_workers=6; timeout=600 ;;
    medium) normal_workers=20; shared_workers=8; timeout=360 ;;
    light) normal_workers=32; shared_workers=12; timeout=180 ;;
  esac
  for gpu in {0..7}; do
    curl -fsS --max-time 10 "http://127.0.0.1:1800${gpu}/health" >/dev/null || {
      echo "GPU $gpu endpoint unavailable before $band" >&2; exit 3;
    }
  done
  mkdir -p "$ROOT/results/$band"
  pids=()
  for gpu in {0..7}; do
    workers=$normal_workers
    [[ "$gpu" == 1 || "$gpu" == 7 ]] && workers=$shared_workers
    "$PY" "$CODE/annotate_jsonl.py" \
      --input "$ROOT/inputs/$band/shard-${gpu}.input.jsonl" \
      --output "$ROOT/results/$band/shard-${gpu}.results.jsonl" \
      --prompt-file "$PROMPT" --schema-file "$SCHEMA" \
      --text-field review_payload --id-fields document_id \
      --carry-fields source_id,current_bucket,raw_row,source_path,upstream_score,text_chars,prompt_tokens_est \
      --api-base "http://127.0.0.1:1800${gpu}/v1" \
      --api-key-env QWEN_LOCAL_API_KEY --model qwen3.8-27b-fp8 \
      --thinking omit --chat-template-kwargs '{"enable_thinking":false}' \
      --temperature 0 --response-format json_schema --require-document-id-match \
      --workers "$workers" --max-tokens 384 --timeout "$timeout" --max-retries 1 \
      >"$ROOT/results/$band/shard-${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for gpu in {0..7}; do
    if wait "${pids[gpu]}"; then
      echo "$band GPU $gpu scorer exited successfully"
    else
      echo "$band GPU $gpu scorer returned nonzero; checking quarantined failures" >&2
    fi
  done
  "$PY" "$CODE/p0_verify_ultradata_night8.py" --root "$ROOT" --band "$band" --max-missing-ratio 0.01
done
"$PY" "$CODE/p0_verify_ultradata_night8.py" --root "$ROOT" --final --max-missing-ratio 0.01
