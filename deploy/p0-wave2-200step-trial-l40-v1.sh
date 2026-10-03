#!/usr/bin/env bash
# User-authorized 200 total steps: 100, checkpoint, actual resume to 200, stop.
set -euo pipefail
umask 077
ROOT=/srv/small-agent/experiments/small-agent-base-1b-wave2-reuse-v1
DATA=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1
ENV=/srv/small-agent/experiments/small-agent-base-1b-v1-engineering
CODE="$ROOT/code"
PY=/opt/small-agent/venvs/small-agent-l40/bin/python
CONFIG="$CODE/configs/runs/base_1b_prior_wave2_internal_1040m_6l40_v1.json"
RUN_ID=prior-wave2-reuse-1040m-6l40-v1
GPUS=0,1,3,4,5,6
export PYTHONPATH="$ENV/deps:$CODE/src:$CODE" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=
exec >>"$ROOT/reports/trial200.log" 2>&1
exec 9>"$ROOT/reports/trial200.lock"
flock -n 9
date -Is
trap 'echo "TRIAL_EXIT_STATUS=$?"; date -Is' EXIT
echo PREPARING_HARDLINK_PACK_AND_CPU_SCHEDULE
"$PY" "$CODE/scripts/p0_wave2_trial_bundle_v1.py" prepare --root "$ROOT" --data-root "$DATA"
echo LIVE_6GPU_PREFLIGHT
"$PY" "$CODE/scripts/trainctl.py" preflight "$CONFIG" --run-id "$RUN_ID" \
  --nproc-per-node 6 --cuda-visible-devices "$GPUS" --environment-root "$ENV" \
  --max-steps 100 --save-final-checkpoint >"$ROOT/reports/TRAINCTL_PREFLIGHT.json"
echo TRAIN_FROM_SCRATCH_TO_STEP_100
"$PY" "$CODE/scripts/trainctl.py" launch "$CONFIG" --run-id "$RUN_ID" \
  --nproc-per-node 6 --cuda-visible-devices "$GPUS" --environment-root "$ENV" \
  --max-steps 100 --save-final-checkpoint
echo ACTUAL_CHECKPOINT_RESUME_100_TO_200
"$PY" "$CODE/scripts/trainctl.py" launch "$CONFIG" --run-id "$RUN_ID" \
  --nproc-per-node 6 --cuda-visible-devices "$GPUS" --environment-root "$ENV" \
  --resume --max-steps 200 --save-final-checkpoint
export CUDA_VISIBLE_DEVICES=
echo VERIFYING_TWO_FULL_CHECKPOINTS_AND_METRICS
"$PY" "$CODE/scripts/p0_wave2_trial_bundle_v1.py" report --root "$ROOT" --data-root "$DATA"
echo TRIAL200_FINISHED_NO_AUTOMATIC_FULL_TRAINING
