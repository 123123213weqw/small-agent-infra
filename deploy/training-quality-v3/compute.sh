#!/bin/bash
set -euo pipefail
V=/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1/evaluation/training_quality_v3
B=/srv/small-agent/experiments/small-agent-base-1b-v1-engineering
R=/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1
PY=/opt/small-agent/venvs/small-agent-l40/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH="$R/benchmarks/fusion8_v1/deps/vendor:$B/deps:$V/code/src:$V/code/scripts"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TRITON_CACHE_DIR="$B/triton-cache" TMPDIR="$B/tmp"
# Queue/timeout pass fd 9 with the SAME open-file description. Reopening the
# file here would deadlock against the lock still held by the timeout parent.
flock -n 9
phase=compute_initializing;guard=''
status() {
 "$PY" - "$V" "$phase" "$1" <<'PY'
import json,sys,time,os
from pathlib import Path
root=Path(sys.argv[1]);p=root/'reports/CAMPAIGN_STATUS.json';t=p.with_suffix('.tmp')
t.write_text(json.dumps({'phase':sys.argv[2],'exit_status':sys.argv[3],'updated_unix':time.time(),'formal_training_started':False,'automatic_adoption_allowed':False},indent=2)+'\n');os.replace(t,p)
PY
}
finish() { rc=$?; test -z "$guard" || kill "$guard" 2>/dev/null || true; status "$rc"; }
trap finish EXIT
trap 'exit 143' TERM INT
status running
"$PY" "$V/deploy/gpu_guard.py" --root "$V" --parent $$ --unit p0-quality-v3-campaign.service & guard=$!
phase=convolution_numeric_preflight;status running
"$PY" -u "$V/code/scripts/training_conv_reduction_diagnostic_v3.py" --root "$V" > "$V/logs/conv_preflight.log" 2>&1
for case in resume_continuous resume_first resume_second;do
 phase=$case;status running
 "$PY" -u -m torch.distributed.run --standalone --nproc_per_node=8 "$V/code/scripts/training_quality_campaign_v2.py" worker --root "$V" --case "$case" > "$V/logs/$case.log" 2>&1
done
phase=deterministic_reference_resume_comparison;status running
"$PY" -u "$V/code/scripts/training_quality_campaign_v2.py" compare-resume --root "$V" > "$V/logs/resume_compare.log" 2>&1
# These exploratory long runs use the original fast convolution on BOTH sides.
# The reference resume receipt is explicitly NOT a native bitwise certificate.
for seed in 20261003 20261004;do
 for variant in baseline candidate;do
  case=seed${seed}_${variant};phase=$case;status running
  "$PY" -u -m torch.distributed.run --standalone --nproc_per_node=8 "$V/code/scripts/training_quality_campaign_v2.py" worker --root "$V" --case "$case" > "$V/logs/$case.log" 2>&1
 done
done
phase=paired_numeric_comparison;status running
"$PY" -u "$V/code/scripts/training_quality_campaign_v2.py" summarize --root "$V" > "$V/logs/paired_compare.log" 2>&1
phase=primary_completed
