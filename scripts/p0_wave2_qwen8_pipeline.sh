#!/usr/bin/env bash
# Resumable V100 -> L40 eight-GPU Wave2 review. Qwen labels are NOT training approval.
set -euo pipefail

R=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/fineweb_edu_wave2_v1
V=$R/audit/qwen8_review_v1
L=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/fineweb_edu_wave2_v1/qwen8_review_v1
HOST=L40_Data_Pipeline
PY=/srv/small-agent/venvs/vllm-qwen38-fp8-v1/bin/python
MODEL=/srv/small-agent/models/Qwen3.8-27B-FP8
PROMPT=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/code/configs/data/p0_teacher_quality_prompt_v2.txt
SRC=$V/source.input.jsonl
META=$V/source.input.manifest.json

mkdir -p "$V"
exec 9>>"$V/pipeline.lock"
flock -n 9 || { echo 'another Wave2 pipeline owns V100 lock' >&2; exit 75; }

for attempt in {1..120}; do
  [[ -f "$META" && -f "$SRC" ]] && break
  echo "waiting for V100 export: attempt $attempt/120"
  sleep 30
done
[[ -f "$META" && -f "$SRC" ]] || { echo 'V100 export did not complete in one hour' >&2; exit 2; }
read -r expected documents < <(python3 - "$META" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m['stage']=='wave2_qwen_review_export_not_training_approved'
assert m['limit']==0 and m['documents']==1116227
print(m['input_sha256'],m['documents'])
PY
)
[[ "$(sha256sum "$SRC" | cut -d ' ' -f 1)" == "$expected" ]] || {
  echo 'V100 source checksum mismatch' >&2; exit 3;
}
echo "V100 export verified: $documents docs, SHA $expected"

ssh -o BatchMode=yes "$HOST" "mkdir -p '$L'"
if ssh -o BatchMode=yes "$HOST" "test -f '$L/source.input.jsonl'"; then
  remote=$(ssh -o BatchMode=yes "$HOST" "sha256sum '$L/source.input.jsonl'" | cut -d ' ' -f 1)
  [[ "$remote" == "$expected" ]] || { echo 'L40 final source checksum mismatch' >&2; exit 4; }
else
  rsync -az --partial --append-verify -e 'ssh -o BatchMode=yes' \
    "$SRC" "$HOST:$L/source.input.jsonl.pipeline-part"
  remote=$(ssh -o BatchMode=yes "$HOST" "sha256sum '$L/source.input.jsonl.pipeline-part'" | cut -d ' ' -f 1)
  [[ "$remote" == "$expected" ]] || { echo 'L40 partial source checksum mismatch' >&2; exit 5; }
  ssh -o BatchMode=yes "$HOST" "mv '$L/source.input.jsonl.pipeline-part' '$L/source.input.jsonl'"
fi
rsync -az -e 'ssh -o BatchMode=yes' "$META" "$HOST:$L/source.input.manifest.json"
echo 'L40 source transfer verified'

if ssh -o BatchMode=yes "$HOST" "test -f '$L/input_bands8_v1/manifest.json'"; then
  ssh -o BatchMode=yes "$HOST" "$PY - '$L/input_bands8_v1/manifest.json' '$expected' <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m['source_input_sha256']==sys.argv[2]
assert m['all_documents']==1116227
assert m['total_documents']+m['overlong_documents']==m['all_documents']
PY"
  echo 'L40 Qwen tokenizer preflight already verified'
else
  ssh -o BatchMode=yes "$HOST" \
    "$PY '$L/code/scripts/p0_nonwiki_review_batch_v2.py' prepare --input '$L/source.input.jsonl' --output-dir '$L/input_bands8_v1' --model-path '$MODEL' --prompt '$PROMPT'"
fi

# The fixed per-GPU systemd units must not be shared with the smoke batch.
for attempt in {1..60}; do
  if ! ssh -o BatchMode=yes "$HOST" 'systemctl --user is-active --quiet p0-wave2-qwen8-smoke64-v1.service'; then
    break
  fi
  echo "waiting for the smoke scorer to release eight Qwen units: $attempt/60"
  sleep 30
done
ssh -o BatchMode=yes "$HOST" 'if systemctl --user is-active --quiet p0-wave2-qwen8-smoke64-v1.service; then exit 6; fi'

ssh -o BatchMode=yes "$HOST" \
  "DATA='$L' CODE='$L/code' bash '$L/code/scripts/p0_run_nonwiki_qwen8_auto.sh'"
ssh -o BatchMode=yes "$HOST" "$PY - '$L' <<'PY'
import hashlib,json,sys
from pathlib import Path
d=Path(sys.argv[1]);m=d/'input_bands8_v1/manifest.json'
f=json.loads((d/'FINISHED8.json').read_text())
assert f['input_manifest_sha256']==hashlib.sha256(m.read_bytes()).hexdigest()
assert f['unattempted']==0
PY"

if ! ssh -o BatchMode=yes "$HOST" "test -f '$L/gate8_v1/manifest.json'"; then
  ssh -o BatchMode=yes "$HOST" \
    "$PY '$L/code/scripts/p0_nonwiki_review_batch_v2.py' gate --data-dir '$L' --output-dir '$L/gate8_v1'"
fi
mkdir -p "$V/gate8_v1"
for name in manifest.json strict_keep_candidates.jsonl rejected_or_unscored.jsonl; do
  rsync -az -e 'ssh -o BatchMode=yes' "$HOST:$L/gate8_v1/$name" "$V/gate8_v1/$name"
done
rsync -az -e 'ssh -o BatchMode=yes' "$HOST:$L/input_bands8_v1/overlong.jsonl" "$V/overlong_quarantine.jsonl"
echo 'Wave2 eight-GPU Qwen review and strict candidate gate complete; no training approval'
