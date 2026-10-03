#!/usr/bin/env bash
# Run the pinned full non-Wikipedia Qwen review from V100 over direct Tailnet SSH.
# No Mac relay, paid API, or training-pool approval.
set -euo pipefail

VROOT=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1
LROOT=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1
VDIR=$VROOT/upstream_nonwiki_v1/audit/auto_review_full_v1
LDIR=$LROOT/nonwiki_new_quality_probe_v1/auto_review_full_v1
LHOST=L40_Data_Pipeline
SRC=$VDIR/source.input.jsonl
SRC_MANIFEST=$VDIR/source.input.manifest.json
DEST=$LDIR/source.input.jsonl
PART=$LDIR/source.input.jsonl.pipeline-part
CODE=$LDIR/code
LPY=/srv/small-agent/venvs/vllm-qwen38-fp8-v1/bin/python

exec 9>"$VDIR/direct_run.lock"
flock -n 9 || { echo 'another direct run owns the lock' >&2; exit 2; }

expected=$(python3 - "$SRC_MANIFEST" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['input_sha256'])
PY
)
actual=$(sha256sum "$SRC" | cut -d ' ' -f 1)
test "$actual" = "$expected" || { echo 'V100 source hash mismatch' >&2; exit 3; }
echo "source verified: $expected"

ssh -o BatchMode=yes "$LHOST" "mkdir -p '$LDIR'"
if ssh -o BatchMode=yes "$LHOST" "test -f '$DEST'"; then
  remote=$(ssh -o BatchMode=yes "$LHOST" "sha256sum '$DEST'" | cut -d ' ' -f 1)
  test "$remote" = "$expected" || { echo 'L40 final source exists with wrong hash' >&2; exit 4; }
  echo 'source already present on L40; checksum matched'
else
  echo 'resuming direct V100 -> L40 rsync over Tailnet'
  rsync -az --partial --append-verify -e 'ssh -o BatchMode=yes' "$SRC" "$LHOST:$PART"
  remote=$(ssh -o BatchMode=yes "$LHOST" "sha256sum '$PART'" | cut -d ' ' -f 1)
  test "$remote" = "$expected" || { echo 'L40 partial source hash mismatch' >&2; exit 5; }
  ssh -o BatchMode=yes "$LHOST" "mv '$PART' '$DEST'"
  echo 'L40 source checksum matched; atomic rename complete'
fi
rsync -az -e 'ssh -o BatchMode=yes' "$SRC_MANIFEST" "$LHOST:$LDIR/source.input.manifest.json"

if ssh -o BatchMode=yes "$LHOST" "test -f '$LDIR/input_bands8_v1/manifest.json'"; then
  ssh -o BatchMode=yes "$LHOST" "$LPY -c 'import json; from pathlib import Path; m=json.loads(Path(\"$LDIR/input_bands8_v1/manifest.json\").read_text()); assert m[\"source_input_sha256\"]==\"$expected\"'"
  echo 'exact-token preflight already verified'
else
  ssh -o BatchMode=yes "$LHOST" \
    "$LPY '$CODE/scripts/p0_nonwiki_review_batch_v2.py' prepare --input '$DEST' --output-dir '$LDIR/input_bands8_v1' --model-path /srv/small-agent/models/Qwen3.8-27B-FP8 --prompt '$LROOT/code/configs/data/p0_teacher_quality_prompt_v2.txt'"
fi

if ssh -o BatchMode=yes "$LHOST" "test -f '$LDIR/FINISHED8.json'"; then
  ssh -o BatchMode=yes "$LHOST" "$LPY -c 'import json,hashlib; from pathlib import Path; d=Path(\"$LDIR\"); m=d/\"input_bands8_v1/manifest.json\"; f=json.loads((d/\"FINISHED8.json\").read_text()); assert f[\"input_manifest_sha256\"]==hashlib.sha256(m.read_bytes()).hexdigest() and f[\"unattempted\"]==0'"
  echo 'eight-GPU Qwen result already verified'
else
  ssh -o BatchMode=yes "$LHOST" \
    "DATA='$LDIR' CODE='$CODE' bash '$CODE/scripts/p0_run_nonwiki_qwen8_auto.sh'"
fi

if ssh -o BatchMode=yes "$LHOST" "test -f '$LDIR/gate8_v1/manifest.json'"; then
  echo 'strict candidate gate already present'
else
  ssh -o BatchMode=yes "$LHOST" \
    "$LPY '$CODE/scripts/p0_nonwiki_review_batch_v2.py' gate --data-dir '$LDIR' --output-dir '$LDIR/gate8_v1'"
fi

mkdir -p "$VDIR/gate8_v1"
for name in manifest.json strict_keep_candidates.jsonl rejected_or_unscored.jsonl; do
  rsync -az -e 'ssh -o BatchMode=yes' "$LHOST:$LDIR/gate8_v1/$name" "$VDIR/gate8_v1/$name"
done
rsync -az -e 'ssh -o BatchMode=yes' "$LHOST:$LDIR/input_bands8_v1/overlong.jsonl" "$VDIR/overlong_quarantine.jsonl"
echo 'direct full pipeline complete: Qwen keep list remains unapproved candidate data'
