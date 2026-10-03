#!/usr/bin/env bash
set -uo pipefail
ROOT=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/finewiki_zh_2025_candidate_v1
PY=/srv/small-agent/venvs/deepspec_gemma4/bin/python

for ((i=0; i<720; i++)); do
  if [[ -f "$ROOT/logs/clean.exit" ]]; then
    clean_status=$(cat "$ROOT/logs/clean.exit")
    if [[ "$clean_status" != 0 ]]; then
      echo "Cleaning failed with exit $clean_status; token count not started" >&2
      printf '%s\n' 2 > "$ROOT/logs/token_count.exit"
      exit 2
    fi
    break
  fi
  sleep 30
done

"$PY" "$ROOT/code/scripts/p0_count_finewiki_tokens.py" \
  --root "$ROOT" \
  --tokenizer /srv/small-agent/datasets/small-agent-p0/p0_1m_smoke_v1/tokenizer_v1/tokenizer.json
status=$?
printf '%s\n' "$status" > "$ROOT/logs/token_count.exit"
exit "$status"
