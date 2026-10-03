#!/usr/bin/env bash
set -euo pipefail
R=/srv/small-agent/datasets/small-agent-p0/curation_v4
PY=/srv/small-agent/venvs/deepspec_gemma4/bin/python
cd "$R/code"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
"$PY" scripts/p0_rebind_math_protocol_v8.py --source "$R/references/math_quality_v7_r4" --output "$R/references/math_quality_v8_r5" --prompt configs/curation/p0_math_audit_prompt_v8_r5.txt --kind reference
"$PY" scripts/p0_deepseek_worker_v8.py --profile /home/user/.config/small-agent/curation-deepseek-v4.json --batch "$R/references/math_quality_v8_r5/batch" --allow-api --initial-concurrency 32 --maximum-concurrency 2500 --limit 10000
"$PY" scripts/p0_math_reference_gate_v8.py --reference "$R/references/math_quality_v8_r5" --policy configs/curation/p0_release_gates_v5.json
"$PY" scripts/p0_math_clean_stock_v7.py --stock "$R/stock_117b_v6" --output "$R/stock_117b_v6/math_clean_v7" --config configs/curation/p0_math_structural_filter_v7.json --baseline "$R/source_audits/math_calibration_20261002_v6_r3" --reference "$R/references/math_quality_v8_r5" --audit-output "$R/source_audits/math_clean_calibration_20261002_v8_r5"
"$PY" scripts/p0_deepseek_worker_v8.py --profile /home/user/.config/small-agent/curation-deepseek-v4.json --batch "$R/source_audits/math_clean_calibration_20261002_v8_r5/batch" --allow-api --initial-concurrency 64 --maximum-concurrency 2500 --limit 10000
"$PY" scripts/p0_math_clean_scope_v8.py
