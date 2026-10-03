#!/bin/bash
set -euo pipefail
V=/srv/small-agent/experiments/small-agent-base-1b-wave2-full8-v1/evaluation/training_quality_v6
PY=/opt/small-agent/venvs/small-agent-l40/bin/python
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1
exec 9>"$V/.campaign.lock"; flock -n 9
phase=waiting_for_all_eight_idle
status() {
 "$PY" - "$V" "$phase" "$1" <<'PY'
import json,sys,time,os
from pathlib import Path
p=Path(sys.argv[1])/'reports/CAMPAIGN_STATUS.json';t=p.with_suffix('.tmp')
t.write_text(json.dumps({'phase':sys.argv[2],'exit_status':sys.argv[3],'updated_unix':time.time(),'gpu_used_by_recovery':False,'formal_training_started':False,'automatic_adoption_allowed':False},indent=2)+'\n');os.replace(t,p)
PY
}
finish() { rc=$?; status "$rc"; }
trap finish EXIT
trap 'exit 143' TERM INT
status running
deadline=$((SECONDS+172800));stable=0
while [ "$stable" -lt 3 ];do
 if [ "$SECONDS" -ge "$deadline" ];then phase=queue_timeout;exit 124;fi
 if "$PY" - <<'PY'
import subprocess,sys
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=15).strip()
rows=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()
free=len(rows)==8 and not apps and all('L40' in r and int(r.split(',')[1])<100 and int(r.split(',')[2])<5 for r in rows)
sys.exit(not free)
PY
 then stable=$((stable+1));else stable=0;fi
 if [ "$stable" -lt 3 ];then sleep 10;fi
done
# CPU-only repair; gate preserved, but no model/GPU work or reservation follows.
phase=cpu_only_code_scoring;status running
set +e
/usr/bin/timeout --signal=TERM --kill-after=30 28800 /bin/bash "$V/deploy/compute.sh"
rc=$?
set -e
phase=cpu_recovery_finished
exit "$rc"
