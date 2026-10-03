#!/usr/bin/env bash
set -uo pipefail
ROOT=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/finewiki_zh_2025_candidate_v1
PY=/srv/small-agent/venvs/deepspec_gemma4/bin/python
"$PY" "$ROOT/code/scripts/p0_fetch_finewiki_zh.py" \
  --spec "$ROOT/code/configs/finewiki_zh_2025_candidate_v1.json" \
  --root "$ROOT" --endpoint https://hf-mirror.com --workers 3
status=$?
printf '%s\n' "$status" > "$ROOT/logs/download.exit"
exit "$status"
