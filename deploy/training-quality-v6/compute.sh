#!/bin/bash
set -euo pipefail
V=/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1/evaluation/training_quality_v6
P=/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1/evaluation/training_quality_v5
B=/srv/small-agent/experiments/small-agent-base-1b-v1-engineering
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1
export PYTHONPATH="$V/deps:$B/deps:$V/code/src:$V/code/scripts"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
trap '/bin/bash "$V/deploy/stop-scopes.sh"' EXIT
/opt/small-agent/venvs/small-agent-l40/bin/python -u "$V/code/scripts/finalize_training_quality_recovery_v6.py" \
 --root "$V" --parent "$P" --base-root "$B" > "$V/logs/recovery.log" 2>&1
