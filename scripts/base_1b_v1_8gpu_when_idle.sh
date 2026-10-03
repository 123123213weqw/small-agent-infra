#!/usr/bin/env bash
# One-shot 8-L40 engineering run, without interrupting existing GPU jobs.
set -euo pipefail

ROOT=/srv/small-agent/experiments/small-agent-base-1b-v1-engineering
RUN="$ROOT/run/ddp8_4096_accum4"
mkdir -p "$RUN"
exec 9>"$RUN/queue.lock"
flock -n 9 || { echo 'An 8-GPU queue is already running'; exit 2; }
echo waiting >"$RUN/status"
deadline=$(( $(date +%s) + 21600 ))
while true; do
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if (( ${#used[@]} == 8 )); then
    idle=1
    for mib in "${used[@]}"; do
      if (( mib >= 512 )); then idle=0; break; fi
    done
    if (( idle )); then break; fi
  fi
  if (( $(date +%s) >= deadline )); then
    echo timed_out >"$RUN/status"
    echo 'Timed out waiting for all 8 GPUs to become idle' >&2
    exit 124
  fi
  sleep 60
done

echo running >"$RUN/status"
export SMALL_AGENT_GDN_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
if bash "$ROOT/code/l40_gdn_env.sh" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$ROOT/code/base_1b_v1_engineering.py" \
  --config "$ROOT/code/base_1b_v1.json" \
  --data /srv/small-agent/experiments/small-agent-p0-engineering-v1/data \
  --run-dir "$RUN" --context 4096 --batch-size 1 --accumulation-steps 4 \
  --steps 3 --gradient-checkpointing; then
  echo complete >"$RUN/status"
else
  echo failed >"$RUN/status"
  exit 1
fi
