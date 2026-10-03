#!/usr/bin/env bash
set -euo pipefail
ROOT=/srv/small-agent/datasets/small-agent-p0/curation_v4
PY=/srv/small-agent/venvs/deepspec_gemma4/bin/python
cd "$ROOT/code"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
"$PY" scripts/p0_rebind_math_protocol_v7.py --source "$ROOT/references/math_quality_v6_r3" --output "$ROOT/references/math_quality_v7_r4" --prompt configs/curation/p0_math_audit_prompt_v7_r4.txt --kind reference
"$PY" scripts/p0_rebind_math_protocol_v7.py --source "$ROOT/source_audits/math_calibration_20261002_v6_r3" --output "$ROOT/source_audits/math_calibration_20261002_v7_r4" --prompt configs/curation/p0_math_audit_prompt_v7_r4.txt --kind source
"$PY" scripts/p0_deepseek_worker_v7.py --profile /home/user/.config/small-agent/curation-deepseek-v4.json --batch "$ROOT/references/math_quality_v7_r4/batch" --allow-api --initial-concurrency 32 --maximum-concurrency 2500 --limit 10000
"$PY" scripts/p0_deepseek_worker_v7.py --profile /home/user/.config/small-agent/curation-deepseek-v4.json --batch "$ROOT/source_audits/math_calibration_20261002_v7_r4/batch" --allow-api --initial-concurrency 64 --maximum-concurrency 2500 --limit 10000
