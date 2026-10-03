#!/usr/bin/env bash
set -euo pipefail

ROOT=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1
CODE="$ROOT/code"
SOURCE="$ROOT/wave1_consolidated_100m_v1"
OUTPUT="$ROOT/batches/unreviewed_candidate_100m_sharded_v1"
LOGS="$ROOT/logs"
EXIT_FILE="$LOGS/unreviewed_candidate_100m_pack.exit"

mkdir -p "$LOGS" "$ROOT/batches"
if [[ -e "$OUTPUT" ]]; then
  echo "output already exists; refusing to overwrite: $OUTPUT" >&2
  exit 2
fi
rm -f "$EXIT_FILE"
trap 'printf "%s\n" "$?" > "$EXIT_FILE"' EXIT

cd "$CODE"
export PYTHONPATH="$CODE/src:$CODE"
exec_python=/home/user/anaconda3/bin/python
nice -n 10 "$exec_python" scripts/p0_pack_sharded_v1.py \
  --source "$SOURCE" \
  --tokenizer "$ROOT/tokenizer_v1" \
  --internal-candidate-smoke \
  --mixture-json configs/data/unreviewed_candidate_100m_diagnostic_mixture_v1.json \
  --output "$OUTPUT" \
  --sequence-length 4096 \
  --target-shard-tokens 32000000
