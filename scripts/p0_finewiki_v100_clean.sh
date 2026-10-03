#!/usr/bin/env bash
set -uo pipefail
ROOT=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/finewiki_zh_2025_candidate_v1
PY=/srv/small-agent/venvs/deepspec_gemma4/bin/python

for ((i=0; i<720; i++)); do
  if [[ -f "$ROOT/logs/download.exit" ]]; then
    download_status=$(cat "$ROOT/logs/download.exit")
    if [[ "$download_status" != 0 ]]; then
      echo "Download failed with exit $download_status; cleaning not started" >&2
      printf '%s\n' 2 > "$ROOT/logs/clean.exit"
      exit 2
    fi
    break
  fi
  sleep 30
done

if [[ ! -f "$ROOT/raw_manifest.json" ]]; then
  echo "Verified raw manifest is missing; cleaning not started" >&2
  printf '%s\n' 3 > "$ROOT/logs/clean.exit"
  exit 3
fi

"$PY" "$ROOT/code/scripts/p0_clean_finewiki_zh.py" \
  --spec "$ROOT/code/configs/finewiki_zh_2025_candidate_v1.json" \
  --root "$ROOT" --batch-size 256
status=$?
printf '%s\n' "$status" > "$ROOT/logs/clean.exit"
exit "$status"
