#!/usr/bin/env bash
# Persistent-worker wrapper: stop only the uniquely named service started here.
# Launcher and its args are trusted operator configuration, not corpus content.
set -euo pipefail
if (($# < 4)); then
  echo 'usage: wrapper GPU PORT TRUSTED_LAUNCHER command [args...]' >&2
  exit 2
fi
gpu=$1; port=$2; launcher=$3; shift 3
[[ "$gpu" =~ ^[0-7]$ && "$port" =~ ^[0-9]+$ && -f "$launcher" ]] || exit 2
((port >= 1024 && port <= 65535)) || exit 2
lease="${XDG_RUNTIME_DIR:?}/p0-curation-v3-gpu-$gpu.lock"
exec 9>"$lease"
flock -n 9 || { echo 'GPU lease held by another cooperating curation worker' >&2; exit 3; }
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
if grep -Fxq "$uuid" <<<"$apps"; then
  echo 'GPU has compute jobs; refusing to share/kill existing work' >&2; exit 3
fi
command -v ss >/dev/null
if ss -H -ltn | awk '{print $4}' | grep -Eq ":${port}$"; then
  echo 'port already in use; refusing to attach to an unrelated service' >&2; exit 3
fi
parent=${P0_PARENT_UNIT:?Run this wrapper from a named systemd user coordinator}
[[ "$parent" =~ ^[a-zA-Z0-9_.@-]+\.service$ ]] || exit 2
systemctl --user is-active --quiet "$parent" || { echo 'parent coordinator is not active' >&2; exit 3; }
unit="p0-curation-v3-owned-$(date +%s)-$$"
owned=0
cleanup() {
  if [[ "$owned" == 1 ]]; then systemctl --user stop "$unit.service" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
systemd-run --user --unit="$unit" --collect --property=KillMode=control-group \
  --property="BindsTo=$parent" --property="After=$parent" \
  --property=RuntimeMaxSec=12h /bin/bash "$launcher" "$gpu" "$port"
owned=1
healthy=0
for ((i=0; i<100; i++)); do
  systemctl --user is-active --quiet "$unit.service" || { echo 'teacher exited before healthy' >&2; exit 4; }
  if curl --fail --silent --max-time 3 "http://127.0.0.1:$port/health" >/dev/null; then healthy=1; break; fi
  sleep 6
done
[[ "$healthy" == 1 ]] || { echo 'teacher health timeout' >&2; exit 5; }
# Foreground command must not background itself; completion/error triggers cleanup.
"$@"
