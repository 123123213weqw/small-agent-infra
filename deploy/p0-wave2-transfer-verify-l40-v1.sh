#!/usr/bin/env bash
# Wait for the final transfer receipt, verify, publish, and inventory. No training.
set -euo pipefail
umask 077
ROOT=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/pretrain_reuse_wave2_v1
BASE=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1
CODE=/srv/small-agent/experiments/small-agent-base-1b-v1-engineering/code
PY=/opt/small-agent/venvs/small-agent-l40/bin/python
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec >>"$ROOT/reports/verify_l40_v1.log" 2>&1
exec 9>"$ROOT/reports/verify_l40_v1.lock"
flock -n 9
date -Is
echo WAITING_FOR_V100_TRANSFER_RECEIPT
while [[ ! -f "$ROOT/reports/TRANSFER_INPUTS.json" ]]; do
  sleep 15
done
"$PY" "$ROOT/code/scripts/p0_wave2_transfer_stats_v1.py" \
  --root "$ROOT" \
  --prior-pack "$BASE/approved_plus_finewiki_short_tokens_internal_fullpass_v1" \
  --prior-ready "$BASE/pretrain_ready_v1/audit/READY.json" \
  --runtime-code "$CODE" \
  --tokenizer /srv/small-agent/experiments/small-agent-base-1b-v1-engineering/data/tokenizer_v1/tokenizer.json
date -Is
