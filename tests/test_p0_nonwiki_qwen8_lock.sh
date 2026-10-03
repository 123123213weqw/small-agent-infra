#!/usr/bin/env bash
# Linux integration test: no GPU/systemd access required. Run on V100 or L40.
set -euo pipefail

SCRIPT=${1:?pass absolute path to p0_run_nonwiki_qwen8_auto.sh}
PY=$(command -v python3)
tmp=$(mktemp -d)
cleanup() {
  pkill -P "$$" sleep 2>/dev/null || true
  rm -rf "$tmp"
}
trap cleanup EXIT
mkdir -p "$tmp/bin" "$tmp/state" "$tmp/data/input_bands8_v1"
printf '{"batch":"lock-test"}\n' >"$tmp/data/input_bands8_v1/manifest.json"

cat >"$tmp/bin/systemctl" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
test "$1" = --user
case "$2" in
  is-active) test -f "$MOCK_STATE/$4" ;;
  start) touch "$MOCK_STATE/$3" ;;
  stop) shift 2; for unit in "$@"; do rm -f "$MOCK_STATE/$unit"; done ;;
  *) echo "unexpected systemctl call: $*" >&2; exit 2 ;;
esac
SH
cat >"$tmp/runner.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'run\n' >>"$MOCK_RUNS"
printf '%s\n' "$$" >"$MOCK_RUNNER_PID"
sleep "$MOCK_SLEEP" &
child=$!
printf '%s\n' "$child" >"$MOCK_CHILD_PID"
wait "$child"
if test "${MOCK_FINISH:-1}" = 1; then
  python3 - "$DATA" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); manifest=root/'input_bands8_v1/manifest.json'
(root/'FINISHED8.json').write_text(json.dumps({
    'input_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
    'unattempted':0})+'\n')
PY
fi
SH
chmod +x "$tmp/bin/systemctl" "$tmp/runner.sh"
export PATH="$tmp/bin:$PATH" MOCK_STATE="$tmp/state" MOCK_RUNS="$tmp/runs"
export MOCK_RUNNER_PID="$tmp/runner.pid" MOCK_CHILD_PID="$tmp/child.pid"
export DATA="$tmp/data" RUNNER="$tmp/runner.sh" PY ROOT="$tmp" CODE="$tmp"

# Two simultaneous orchestrators must serialize. The second observes the
# first one's valid FINISHED marker and must not invoke the scorer again.
export MOCK_SLEEP=2 MOCK_FINISH=1
bash "$SCRIPT" >"$tmp/first.log" 2>&1 & first=$!
sleep 0.3
bash "$SCRIPT" >"$tmp/second.log" 2>&1 & second=$!
wait "$first"
wait "$second"
test "$(wc -l <"$tmp/runs")" -eq 1
grep -q 'batch already verified; skipping duplicate Qwen scoring' "$tmp/second.log"
test "$(find "$tmp/state" -type f | wc -l)" -eq 0

# Simulate the V100/SSH owner and L40 runner both dying while an inherited
# scorer process remains. The lock must stay held until that child exits.
rm -f "$tmp/data/FINISHED8.json" "$tmp/runner.pid" "$tmp/child.pid"
export MOCK_SLEEP=4 MOCK_FINISH=0
bash "$SCRIPT" >"$tmp/orphan.log" 2>&1 & owner=$!
for _ in {1..50}; do test -s "$tmp/child.pid" && break; sleep 0.1; done
test -s "$tmp/child.pid"
child=$(cat "$tmp/child.pid")
runner=$(cat "$tmp/runner.pid")
test -e "/proc/$child/fd/9"
kill -KILL "$owner" "$runner" 2>/dev/null || true
wait "$owner" 2>/dev/null || true
test -e "/proc/$child/fd/9"
if LOCK_WAIT_SECONDS=1 bash "$SCRIPT" >"$tmp/blocked.log" 2>&1; then
  echo 'second scorer acquired an orphan-held lock' >&2
  exit 1
fi
grep -q 'timed out waiting for L40 batch lock' "$tmp/blocked.log"
while kill -0 "$child" 2>/dev/null; do
  if test "$(ps -o stat= -p "$child" 2>/dev/null | head -c1)" = Z; then break; fi
  sleep 0.1
done
export MOCK_SLEEP=0.1 MOCK_FINISH=1
LOCK_WAIT_SECONDS=5 bash "$SCRIPT" >"$tmp/resumed.log" 2>&1
test "$(find "$tmp/state" -type f | wc -l)" -eq 0
test -f "$tmp/data/FINISHED8.json"
echo 'PASS: parallel starts serialize, orphan scorer retains lock, resumed batch releases owned units'
