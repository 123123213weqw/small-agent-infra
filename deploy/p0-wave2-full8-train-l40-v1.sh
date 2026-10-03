#!/usr/bin/env bash
# Full authorized train pass. Foreground trainctl inherits controller NCCL vars.
set -euo pipefail
umask 077
ROOT=/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1
ENV=/srv/small-agent/experiments/small-agent-base-1b-v1-engineering
CODE="$ROOT/code"
PY=/opt/small-agent/venvs/small-agent-l40/bin/python
CONFIG="$CODE/configs/runs/base_1b_prior_wave2_internal_full_8l40_v1.json"
RUN_ID=prior-wave2-full-1040m-8l40-v1
GPUS=0,1,2,3,4,5,6,7
export PYTHONPATH="$ENV/deps:$CODE/src:$CODE" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=
exec >>"$ROOT/reports/full8.log" 2>&1
exec 9>"$ROOT/reports/full8.lock"
flock -n 9
trap 'echo "FULL8_EXIT_STATUS=$?"; date -Is' EXIT
date -Is
if [[ ! -e "$ROOT/reports/PREPARED.json" ]]; then
  echo VERIFYING_SOURCE_AND_EXPLICIT_6_TO_8_STATE_TRANSITION
  # A free-card gate before the two short RNG initialization CUDA contexts.
  "$PY" - <<'PY'
import subprocess
processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
if processes or len(used) != 8 or any(int(x.strip()) > 1024 for x in used):
    raise SystemExit('Eight-card transition requires all GPUs free; no jobs are killed.')
PY
  CUDA_VISIBLE_DEVICES=6,7 "$PY" "$CODE/scripts/p0_wave2_full8_bundle_v1.py" prepare
else
  "$PY" "$CODE/scripts/p0_wave2_full8_bundle_v1.py" verify-prepared
fi
echo LIVE_8GPU_RESUME_PREFLIGHT
"$PY" "$CODE/scripts/trainctl.py" preflight "$CONFIG" --run-id "$RUN_ID" \
  --nproc-per-node 8 --cuda-visible-devices "$GPUS" --environment-root "$ENV" \
  --resume --save-final-checkpoint >"$ROOT/reports/TRAINCTL_PREFLIGHT.json"
echo FULL_8GPU_TRAINING_FROM_PRESERVED_PROGRESS_TO_END_OF_TRAIN_PASS
"$PY" "$CODE/scripts/trainctl.py" launch "$CONFIG" --run-id "$RUN_ID" \
  --nproc-per-node 8 --cuda-visible-devices "$GPUS" --environment-root "$ENV" \
  --resume --save-final-checkpoint
echo VERIFYING_FINAL_CHECKPOINT_AND_ACCOUNTING
"$PY" "$CODE/scripts/p0_wave2_full8_bundle_v1.py" complete
echo FULL8_FINISHED_RELEASE_ALL_GPUS_NO_SECOND_EPOCH
