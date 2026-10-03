#!/bin/bash
# Exact v6 CPU scope names only. No GPU/user processes are enumerated or killed.
for c in seed20261003_baseline seed20261003_candidate seed20261004_baseline seed20261004_candidate;do
 systemctl --user stop "p0-quality-v6-code-$c.scope" 2>/dev/null || true
done
