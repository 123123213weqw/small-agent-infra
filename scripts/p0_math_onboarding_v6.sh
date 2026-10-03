#!/usr/bin/env bash
set -euo pipefail
ROOT=/srv/small-agent/datasets/small-agent-p0/curation_v4
PY=/srv/small-agent/venvs/deepspec_gemma4/bin/python
cd "$ROOT/code"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
"$PY" scripts/p0_bind_finemath_terms_v6.py --output "$ROOT/references/finemath_terms_v6" --authorize-source-level-internal-use
"$PY" scripts/p0_math_reference_v6.py --registry "$ROOT/references/eval_v5/registry.json" --profile /home/user/.config/small-agent/curation-deepseek-v4.json --output "$ROOT/references/math_quality_v6"
"$PY" scripts/p0_deepseek_adaptive_worker_v5.py --profile /home/user/.config/small-agent/curation-deepseek-v4.json --batch "$ROOT/references/math_quality_v6/batch" --allow-api --initial-concurrency 32 --maximum-concurrency 2500 --limit 2000
"$PY" scripts/p0_source_audit_v4.py prepare --census-root /srv/small-agent/datasets/small-agent-p0/p0_500m_v1/upstream_supply_v4/candidate_token_census_v1 --download-plan configs/curation/p0_supply_hfmirror_download_20261002_v1.json --profile /home/user/.config/small-agent/curation-deepseek-v4.json --output "$ROOT/source_audits/math_calibration_20261002_v6" --source finemath_4plus_supply_v4 --mode calibration --round-id math_stock_v6_20261002 --sampling configs/curation/p0_source_audit_sampling_parallel_v5.json --policy configs/curation/p0_source_audit_policy_parallel_v5.json --prompt configs/curation/p0_source_audit_prompt_v4_r2.txt
"$PY" scripts/p0_deepseek_adaptive_worker_v5.py --profile /home/user/.config/small-agent/curation-deepseek-v4.json --batch "$ROOT/source_audits/math_calibration_20261002_v6/batch" --allow-api --initial-concurrency 64 --maximum-concurrency 2500 --limit 2000
