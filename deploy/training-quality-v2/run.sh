#!/bin/bash
set -euo pipefail
Q=/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1/evaluation/training_quality_v2
B=/srv/small-agent/experiments/small-agent-base-1b-v1-engineering
R=/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1
PY=/opt/small-agent/venvs/small-agent-l40/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH="$R/benchmarks/fusion8_v1/deps/vendor:$B/deps:$Q/code/src:$Q/code/scripts"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TRITON_CACHE_DIR="$B/triton-cache" TMPDIR="$B/tmp"
exec 9>"$Q/.campaign.lock"; flock -n 9
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then echo 'GPUs occupied; no work started';exit 75;fi
"$PY" "$Q/deploy/gpu_guard.py" --root "$Q" --parent $$ & guard=$!
trap 'kill "$guard" 2>/dev/null || true' EXIT
if ! "$PY" -c 'import json,sys;sys.exit(json.load(open(sys.argv[1])).get("state")!="completed")' "$Q/runs/native_m8_smoke_v2/status.json" 2>/dev/null;then
 "$PY" -u -m torch.distributed.run --standalone --nproc_per_node=8 -m small_agent.training.trainer --run-config "$Q/code/configs/runs/quality_smoke_v2.json" --run-id native_m8_smoke_v2 --save-final-checkpoint > "$Q/logs/native_smoke.log" 2>&1
fi
for case in resume_continuous resume_first resume_second; do
 "$PY" -u -m torch.distributed.run --standalone --nproc_per_node=8 "$Q/code/scripts/training_quality_campaign_v2.py" worker --root "$Q" --case "$case" > "$Q/logs/$case.log" 2>&1
done
"$PY" -u "$Q/code/scripts/training_quality_campaign_v2.py" compare-resume --root "$Q" > "$Q/logs/resume_compare.log" 2>&1
for seed in 20261003 20261004;do
 for variant in baseline candidate;do
  case=seed${seed}_${variant}
  "$PY" -u -m torch.distributed.run --standalone --nproc_per_node=8 "$Q/code/scripts/training_quality_campaign_v2.py" worker --root "$Q" --case "$case" > "$Q/logs/$case.log" 2>&1
 done
done
"$PY" -u "$Q/code/scripts/training_quality_campaign_v2.py" summarize --root "$Q" > "$Q/logs/paired_compare.log" 2>&1
