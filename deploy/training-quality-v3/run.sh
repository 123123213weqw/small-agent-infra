#!/bin/bash
set -euo pipefail
V=/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1/evaluation/training_quality_v3
B=/srv/small-agent/experiments/small-agent-base-1b-v1-engineering
R=/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1
PY=/opt/small-agent/venvs/small-agent-l40/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH="$R/benchmarks/fusion8_v1/deps/vendor:$B/deps:$V/code/src:$V/code/scripts"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TRITON_CACHE_DIR="$B/triton-cache" TMPDIR="$B/tmp"
exec 9>"$V/.campaign.lock"; flock -n 9
phase=waiting_for_all_eight_idle;guard=''
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
# Require three consecutive idle observations; this is not a card reservation.
stable=0
while [ "$stable" -lt 3 ];do
 if "$PY" - <<'PY'
import subprocess,sys
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
rows=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
free=len(rows)==8 and not apps and all('L40' in r and int(r.split(',')[1])<100 and int(r.split(',')[2])<5 for r in rows)
sys.exit(not free)
PY
 then stable=$((stable+1));else stable=0;fi
 if [ "$stable" -lt 3 ];then sleep 10;fi
done
exec /usr/bin/timeout --signal=TERM --kill-after=30 28800 /bin/bash "$V/deploy/compute.sh"
