#!/usr/bin/env bash
# Benchmark one Qwen replica on GPU 0 with the same 96 real documents each run.
set -uo pipefail
tag="${1:?usage: p0_benchmark_qwen_single_gpu.sh TAG WORKERS [INPUT_JSONL]}"
workers="${2:?usage: p0_benchmark_qwen_single_gpu.sh TAG WORKERS [INPUT_JSONL]}"
[[ "$tag" =~ ^[a-z0-9_-]+$ && "$workers" =~ ^[0-9]+$ ]] || exit 2
ROOT=/srv/small-agent/experiments/qwen38-fp8-serving-v1
BASE="$ROOT/single-gpu-tune-v1"
DIR="$BASE/$tag"
input="${3:-$BASE/input96.jsonl}"
PY=/srv/small-agent/venvs/vllm-qwen38-fp8-v1/bin/python
PROMPT=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/code/configs/data/p0_teacher_quality_prompt_v2.txt
SCHEMA=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/code/schemas/p0_teacher_quality_v2.json
test -f "$input" || exit 2
test ! -e "$DIR" || { echo "result directory already exists: $DIR" >&2; exit 2; }
mkdir -p "$DIR"
ready=0
for attempt in {1..60}; do
  if curl --fail --silent --max-time 5 http://127.0.0.1:18000/health >/dev/null; then ready=1; break; fi
  sleep 10
done
[[ "$ready" -eq 1 ]] || { echo 'GPU 0 replica unhealthy' >&2; exit 3; }
export QWEN_LOCAL_API_KEY=EMPTY
(
  while true; do
    printf '%s,' "$(date +%s)"
    nvidia-smi -i 0 --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits
    sleep 1
  done
) >"$DIR/gpu.csv" &
sampler=$!
started=$(date +%s)
"$PY" "$ROOT/code/scripts/annotate_jsonl.py" \
  --input "$input" --output "$DIR/results.jsonl" \
  --prompt-file "$PROMPT" --schema-file "$SCHEMA" \
  --text-field review_payload --id-fields document_id \
  --carry-fields current_bucket,token_count,prompt_tokens_est \
  --api-base http://127.0.0.1:18000/v1 --api-key-env QWEN_LOCAL_API_KEY \
  --model qwen3.8-27b-fp8 --thinking omit \
  --chat-template-kwargs '{"enable_thinking":false}' --temperature 0 \
  --response-format json_schema --require-document-id-match \
  --workers "$workers" --max-tokens 384 --timeout 180 --max-retries 0 \
  >"$DIR/scorer.log" 2>&1
code=$?
ended=$(date +%s)
kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true
"$PY" - "$DIR" "$tag" "$workers" "$started" "$ended" "$code" "$input" <<'PY'
import csv,hashlib,json,statistics,sys
from pathlib import Path
folder=Path(sys.argv[1]); tag=sys.argv[2]; workers=int(sys.argv[3]); elapsed=max(1,int(sys.argv[5])-int(sys.argv[4])); code=int(sys.argv[6]); input_path=Path(sys.argv[7])
requested=sum(1 for x in input_path.open() if x.strip())
results=[json.loads(x) for x in (folder/'results.jsonl').read_text().splitlines() if x.strip()] if (folder/'results.jsonl').exists() else []
ok=[r for r in results if r.get('status')=='ok']
gpu=[]
for row in csv.reader((folder/'gpu.csv').open()):
    try: gpu.append({'util':float(row[1]),'mem_util':float(row[2]),'memory_mib':float(row[3]),'power_w':float(row[4])})
    except (ValueError,IndexError): pass
report={'tag':tag,'workers':workers,'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
        'requested':requested,'returned':len(results),'valid':len(ok),'exit_code':code,'wall_s':elapsed,
        'valid_docs_per_s':round(len(ok)/elapsed,3),
        'prompt_tokens_per_s':round(sum(r.get('usage',{}).get('prompt_tokens',0) for r in ok)/elapsed,1),
        'completion_tokens_per_s':round(sum(r.get('usage',{}).get('completion_tokens',0) for r in ok)/elapsed,1),
        'mean_gpu_util_pct':round(statistics.mean(x['util'] for x in gpu),1) if gpu else None,
        'mean_power_w':round(statistics.mean(x['power_w'] for x in gpu),1) if gpu else None,
        'mean_memory_mib':round(statistics.mean(x['memory_mib'] for x in gpu),1) if gpu else None,
        'gpu_samples':len(gpu)}
(folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
PY
exit "$code"
