#!/usr/bin/env bash
# Explicit allowlist bootstrap for the V100 FineWiki CPU audit pipeline.
# No raw corpus, credentials, .env files, Git metadata, or build outputs move.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="${1:-/srv/small-agent/datasets/small-agent-p0/p0_500m_v1/finewiki_zh_2025_candidate_v1}"
[[ "$REMOTE_ROOT" == /* ]] || { echo "Remote root must be absolute" >&2; exit 2; }

ssh -o BatchMode=yes CPU_NODE "mkdir -p '$REMOTE_ROOT/code/scripts' '$REMOTE_ROOT/code/configs'"

scripts=(
  p0_data_pipeline.py
  p0_fetch_finewiki_zh.py p0_clean_finewiki_zh.py p0_count_finewiki_tokens.py
  p0_sample_finewiki_quality.py p0_audit_finewiki_extraction.py
  p0_count_finewiki_artifact_tokens.py p0_overlap_finewiki_wave1.py
  p0_index_finewiki_unflagged.py p0_audit_finewiki_residual.py
  p0_index_finewiki_residual_flags.py p0_near_dedup_ultradata_night8.py
  p0_probe_finewiki_near_overlap.py p0_fetch_ceval_validation.py
  p0_screen_finewiki_ceval.py p0_screen_finewiki_ceval_short.py
  p0_near_dedup_finewiki_priority.py p0_triage_finewiki_near_hits.py
  p0_finalize_finewiki_near_policy.py p0_verify_finewiki_stage.py
)
sources=()
for script in "${scripts[@]}"; do sources+=("$REPO_ROOT/scripts/$script"); done
rsync -azc --no-perms --no-owner --no-group \
  "${sources[@]}" "CPU_NODE:$REMOTE_ROOT/code/scripts/"
rsync -azc --no-perms --no-owner --no-group \
  "$REPO_ROOT/configs/data/finewiki_zh_2025_candidate_v1.json" \
  "CPU_NODE:$REMOTE_ROOT/code/configs/"

# A second checksum dry-run must have nothing left to transfer.
pending="$(rsync -aznci --no-perms --no-owner --no-group \
  "${sources[@]}" "CPU_NODE:$REMOTE_ROOT/code/scripts/")"
[[ -z "$pending" ]] || { printf 'Remote code drift:\n%s\n' "$pending" >&2; exit 1; }
pending="$(rsync -aznci --no-perms --no-owner --no-group \
  "$REPO_ROOT/configs/data/finewiki_zh_2025_candidate_v1.json" \
  "CPU_NODE:$REMOTE_ROOT/code/configs/")"
[[ -z "$pending" ]] || { printf 'Remote spec drift:\n%s\n' "$pending" >&2; exit 1; }
printf 'FineWiki code snapshot verified on CPU_NODE: %s/code\n' "$REMOTE_ROOT"
