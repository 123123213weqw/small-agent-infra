#!/usr/bin/env bash
# One-shot transfer only. Verification/publishing is a separate explicit step.
set -euo pipefail
umask 077
ROOT=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/pretrain_reuse_wave2_v1
DEST=/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/pretrain_reuse_wave2_v1
HOST=L40_Data_Pipeline
PY=/home/user/anaconda3/bin/python
mkdir -p "$ROOT/reports"
exec >>"$ROOT/reports/transfer_l40_v1.log" 2>&1
exec 9>"$ROOT/reports/transfer_l40_v1.lock"
flock -n 9
date -Is
"$PY" - "$ROOT" <<'PY'
import hashlib, json, pathlib
import os
import tempfile
import sys
root = pathlib.Path(sys.argv[1])
def sha(p):
    d = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            d.update(b)
    return d.hexdigest()
ready = json.loads((root/'reports/READY.json').read_text())
assert ready['status'] == 'data_ready_for_internal_pretraining_experiment'
assert ready['all_shard_hashes_and_indexes_verified'] is True
assert ready['all_token_ranges_and_eos_counts_verified'] is True
assert sha(root/'token_pack/manifest.json') == ready['pack_manifest_sha256']
assert sha(root/'frozen_source/manifest.json') == ready['source_manifest_sha256']
manifest = json.loads((root/'token_pack/manifest.json').read_text())
files = []
for s in manifest['shards']:
    for kind in ('bin', 'idx'):
        name = s[kind]
        assert pathlib.Path(name).name == name
        p = root/'token_pack'/name
        assert p.stat().st_size == s[kind+'_bytes']
        assert sha(p) == s[kind+'_sha256'], str(p)
        files.append({'relative_path':'token_pack/'+name, 'bytes':p.stat().st_size, 'sha256':s[kind+'_sha256']})
for source, destination in (
    ('token_pack/manifest.json','token_pack/manifest.json'),
    ('reports/READY.json','reports/V100_READY.json'),
    ('frozen_source/manifest.json','reports/SOURCE_MANIFEST.json'),
    ('code/configs/curation/p0_wave2_pretrain_reuse_v1.json','reports/REUSE_POLICY.json'),
):
    p=root/source
    files.append({'relative_path':destination,'bytes':p.stat().st_size,'sha256':sha(p)})
value = {'version':'wave2_l40_transfer_inputs_v1', 'files':files,
         'source_root':str(root), 'destination_host':'L40_Data_Pipeline',
         'destination_root':'/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/pretrain_reuse_wave2_v1',
         'training_started':False, 'corpus_downloaded_to_mac':False}
p=root/'reports/TRANSFER_INPUTS.json'
if p.exists():
    assert json.loads(p.read_text()) == value, 'existing transfer receipt has different inputs'
else:
    fd, tmp = tempfile.mkstemp(prefix='.transfer-inputs-',dir=p.parent)
    try:
        with os.fdopen(fd,'w') as f: json.dump(value,f,indent=2);f.write('\n')
        os.replace(tmp,p)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
print(json.dumps({'status':'source_files_verified','files':len(files),'bytes':sum(x['bytes'] for x in files)}),flush=True)
PY
SSH='ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=10 -o ServerAliveCountMax=2'
$SSH "$HOST" "test ! -e '$DEST/token_pack' && mkdir -p '$DEST/reports'"
done_copy=false
for attempt in 1 2 3 4 5 6; do
  if rsync -a --partial --append-verify --stats --timeout=120 -e "$SSH" \
      "$ROOT/token_pack/" "$HOST:$DEST/.token_pack.incoming/"; then
    done_copy=true
    break
  fi
  echo "Transfer attempt $attempt failed; retrying after 20 seconds."
  sleep 20
done
[[ "$done_copy" == true ]]
rsync -a -e "$SSH" "$ROOT/reports/READY.json" "$HOST:$DEST/reports/V100_READY.json"
rsync -a -e "$SSH" "$ROOT/frozen_source/manifest.json" "$HOST:$DEST/reports/SOURCE_MANIFEST.json"
rsync -a -e "$SSH" "$ROOT/code/configs/curation/p0_wave2_pretrain_reuse_v1.json" "$HOST:$DEST/reports/REUSE_POLICY.json"
rsync -a -e "$SSH" "$ROOT/reports/TRANSFER_INPUTS.json" "$HOST:$DEST/reports/TRANSFER_INPUTS.json"
echo TRANSFER_FINISHED_AWAITING_L40_VERIFICATION
date -Is
