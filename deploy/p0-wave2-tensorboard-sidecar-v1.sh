#!/usr/bin/env bash
set -euo pipefail
ROOT=/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1
MON="$ROOT/monitoring/tensorboard_v1"
ENV=/srv/small-agent/experiments/small-agent-base-1b-v1-engineering
PY=/opt/small-agent/venvs/small-agent-l40/bin/python
export CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1
export PYTHONPATH="$ENV/deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$MON"
exec "$PY" "$MON/tensorboard_training_sidecar_v1.py" \
  --output "$MON/session_0001" \
  --training-run "$ROOT/runs/prior-wave2-full-1040m-8l40-v1" \
  --campaign "$ROOT/benchmarks/fusion8_v1_confirm1" \
  --campaign "$ROOT/benchmarks/fusion8_tune_v1" \
  --campaign "$ROOT/benchmarks/fusion8_tune_v1_retry1" \
  --quality-policy "$ROOT/benchmarks/QUALITY_ADOPTION_HOLD_v1.json" \
  --follow-until-postcheck "$ROOT/benchmarks/fusion8_tune_v1_retry1/POSTCHECK.json" \
  --interval 10 --max-seconds 2700 --hardware
