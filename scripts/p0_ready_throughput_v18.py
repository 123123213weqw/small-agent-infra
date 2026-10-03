"""Whole-pipeline goal accounting: new READY only, measured windows >= one hour."""
import json,time
from pathlib import Path
from p0_curation_worker_v3 import atomic_status

def record(root,total,now=None):
 now=time.time() if now is None else now
 root=Path(root)/'throughput_v18';root.mkdir(parents=True,exist_ok=True);p=root/'HISTORY.json'
 rows=json.loads(p.read_text()) if p.exists() else []
 if rows and (now<rows[-1]['unix'] or total<rows[-1]['ready_train_body_tokens']):raise ValueError('Nonmonotonic throughput accounting')
 if not rows or now>rows[-1]['unix']:
  rows.append(dict(unix=now,ready_train_body_tokens=total))
 if len(rows)>50000:raise ValueError('Throughput history bound exceeded; inspect retention')
 atomic_status(p,rows)
 baseline=rows[0];delta=total-baseline['ready_train_body_tokens'];elapsed=now-baseline['unix']
 older=[x for x in rows if x['unix']<=now-3600];hour=None
 if older:
  start=older[-1];seconds=now-start['unix'];added=total-start['ready_train_body_tokens']
  hour=dict(seconds=seconds,new_ready_train_body_tokens=added,tokens_per_hour=added/seconds*3600)
 result=dict(goal_tokens_per_hour=200000000,minimum_window_seconds=3600,total_new_gate_ready_train_body_tokens=total,
  baseline=baseline,since_baseline_seconds=elapsed,since_baseline_added_train_tokens=delta,
  since_baseline_tokens_per_hour=delta/elapsed*3600 if elapsed else None,
  measured_hour_window=hour,goal_verified=bool(hour and hour['tokens_per_hour']>=200000000),
  legacy_and_candidates_counted=False,stage_peak_counted=False)
 atomic_status(root/'LATEST.json',result);return result
