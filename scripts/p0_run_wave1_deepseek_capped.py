#!/usr/bin/env python3
"""Run Wave1 second-judge review in cost-capped, resumable batches."""

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path


MISS_USD_PER_M = 0.30
HIT_USD_PER_M = 0.006
OUTPUT_USD_PER_M = 1.20


def usage_cost(path: Path, *, start_row: int = 0, price_factor: float = 1.0) -> tuple[int, int, float]:
    if not path.exists():
        return 0, 0, 0.0
    rows = valid = 0
    cost = 0.0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            if rows <= start_row:
                continue
            row = json.loads(line)
            usage = row.get("usage") or {}
            if usage:
                valid += 1
            miss = int(usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0)))
            hit = int(usage.get("prompt_cache_hit_tokens", 0))
            output = int(usage.get("completion_tokens", 0))
            cost += price_factor * (miss * MISS_USD_PER_M + hit * HIT_USD_PER_M + output * OUTPUT_USD_PER_M) / 1_000_000
    return rows, valid, cost


def new_http_errors(path: Path, previous_rows: int) -> tuple[int, int]:
    """Count only errors added by the just-finished batch."""
    rate_limited = insufficient_balance = 0
    with path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle):
            if row_number < previous_rows or not line.strip():
                continue
            error = json.loads(line).get("error")
            rate_limited += error == "HTTP 429"
            insufficient_balance += error == "HTTP 402"
    return rate_limited, insufficient_balance


def valid_document_count(path: Path) -> int:
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            try:
                document_id, = json.loads(row["source_id"])
            except (KeyError, ValueError, TypeError):
                continue
            if row.get("annotation", {}).get("document_id") == document_id:
                ids.add(document_id)
    return len(ids)


def verified_key(path: Path, expected_sha256: str) -> str:
    matches = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip().removeprefix("export ")
        if line.startswith("DEEPSEEK_API_KEY="):
            matches.append(line.split("=", 1)[1].strip().strip("\"'"))
    if len(matches) != 1 or hashlib.sha256(matches[0].encode()).hexdigest() != expected_sha256:
        raise ValueError("explicit DeepSeek key fingerprint mismatch")
    return matches[0]


def balance_cny(key: str) -> Decimal:
    request = urllib.request.Request(
        "https://api.deepseek.com/user/balance", headers={"Authorization": "Bearer " + key}
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.load(response)
    rows = [x for x in data.get("balance_infos", []) if x.get("currency") == "CNY"]
    if len(rows) != 1 or not data.get("is_available"):
        raise RuntimeError("DeepSeek account is unavailable or CNY balance is missing")
    return Decimal(rows[0]["total_balance"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--review-dir", type=Path, required=True)
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--key-file", type=Path, required=True)
    p.add_argument("--expected-key-sha256", required=True)
    p.add_argument("--budget-usd", type=float, default=10.0)
    p.add_argument("--billing-start-row", type=int, default=0,
                   help="Existing result rows billed to a different account; exclude from this key's cap")
    p.add_argument("--price-factor", type=float, choices=(0.5, 1.0), default=1.0,
                   help="0.5 only for confirmed DeepSeek off-peak; peak is the default")
    p.add_argument("--reserve-multiplier", type=float, default=1.5)
    p.add_argument("--starting-balance-cny", type=Decimal,
                   help="Optional fixed baseline; otherwise persisted on the first invocation")
    p.add_argument("--max-balance-drop-cny", type=Decimal, default=Decimal("60"))
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cooldown-seconds", type=int, default=60)
    p.add_argument("--total-docs", type=int, required=True)
    p.add_argument("--min-valid-ratio", type=float, default=0.999,
                   help="Below this coverage the command fails even if the spending cap stops it")
    args = p.parse_args()
    if (args.batch < 1 or args.workers < 1 or args.cooldown_seconds < 0
            or args.total_docs < 1 or args.billing_start_row < 0
            or args.budget_usd <= 0 or args.reserve_multiplier < 1
            or args.max_balance_drop_cny <= 0 or not 0 < args.min_valid_ratio <= 1):
        p.error("batch, workers and total-docs must be positive; cooldown must be nonnegative")
    key = verified_key(args.key_file, args.expected_key_sha256)
    output = args.review_dir / "deepseek_full.results.jsonl"
    args.review_dir.mkdir(parents=True, exist_ok=True)
    output.touch(exist_ok=True)
    baseline_path = args.review_dir / "deepseek_full.billing_baseline.json"
    if args.starting_balance_cny is None:
        fingerprint = hashlib.sha256((args.review_dir / "deepseek_full.input.jsonl").read_bytes()).hexdigest()
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
            if baseline["key_sha256"] != args.expected_key_sha256 or baseline["input_sha256"] != fingerprint:
                raise ValueError("billing baseline belongs to a different key or input")
            args.starting_balance_cny = Decimal(baseline["starting_balance_cny"])
        else:
            if output.stat().st_size:
                raise ValueError("cannot infer a billing baseline after result rows already exist")
            args.starting_balance_cny = balance_cny(key)
            baseline_path.write_text(json.dumps({"key_sha256": args.expected_key_sha256,
                                                 "input_sha256": fingerprint,
                                                 "starting_balance_cny": str(args.starting_balance_cny)}) + "\n")
    existing_rows, _, _ = usage_cost(output)
    if existing_rows < args.billing_start_row:
        raise ValueError(f"billing boundary is beyond the output: {existing_rows} < {args.billing_start_row}")
    reserve_multiplier = args.reserve_multiplier  # cover retries and unrecorded schema-invalid responses
    stagnant_batches = 0
    previous_valid = -1
    workers = args.workers
    stop_reason = None
    while True:
        if args.price_factor == 0.5 and datetime.now(timezone.utc).weekday() < 5:
            print("OFF_PEAK_WEEKEND_WINDOW_ENDED", flush=True)
            stop_reason = "off_peak_ended"
            break
        result_rows, _, _ = usage_cost(output)
        result_valid = valid_document_count(output)
        _, new_account_valid, observed = usage_cost(
            output, start_row=args.billing_start_row, price_factor=args.price_factor
        )
        current_balance = balance_cny(key)
        balance_drop = args.starting_balance_cny - current_balance
        projected = observed * reserve_multiplier
        state = {
            "rows": result_rows,
            "valid_documents_total": result_valid,
            "new_account_rows_with_usage": new_account_valid,
            "billing_start_row": args.billing_start_row,
            "price_factor": args.price_factor,
            "new_account_logged_usd": round(observed, 4),
            "conservative_total_usd": round(projected, 4),
            "cap_usd": args.budget_usd,
            "balance_cny": str(current_balance),
            "balance_drop_cny": str(balance_drop),
            "balance_drop_cap_cny": str(args.max_balance_drop_cny),
        }
        print(json.dumps(state), flush=True)
        if result_valid <= previous_valid:
            stagnant_batches += 1
        else:
            stagnant_batches = 0
        previous_valid = result_valid
        if stagnant_batches >= 2:
            print("NO_PROGRESS_STOP", flush=True)
            stop_reason = "no_progress"
            break
        if result_valid >= args.total_docs:
            print("COMPLETE", flush=True)
            break
        if balance_drop >= args.max_balance_drop_cny:
            print("BALANCE_DROP_STOP", flush=True)
            stop_reason = "balance_drop"
            break
        if projected >= args.budget_usd:
            print("BUDGET_STOP", flush=True)
            stop_reason = "budget"
            break
        per_doc = observed / new_account_valid if new_account_valid else 0.0005
        remaining = args.budget_usd / reserve_multiplier - observed
        safe_batch = int(remaining / max(per_doc * 2, 0.0006))
        n = min(args.batch, safe_batch, args.total_docs - result_valid)
        if n < 1:
            print("BUDGET_STOP", flush=True)
            stop_reason = "budget"
            break
        cmd = [
            args.python, str(args.runner),
            "--review-dir", str(args.review_dir),
            "--key-file", str(args.key_file),
            "--expected-key-sha256", args.expected_key_sha256,
            "--input-name", "deepseek_full.input.jsonl",
            "--output-name", "deepseek_full.results.jsonl",
            "--workers", str(workers),
            "--max-tokens", "2048", "--limit", str(n),
        ]
        print(f"RUN_BATCH {n}", flush=True)
        completed = subprocess.run(cmd, check=False)
        if completed.returncode:
            print(f"BATCH_NONZERO_EXIT {completed.returncode}; checking appended results", flush=True)
        rate_limited, insufficient_balance = new_http_errors(output, result_rows)
        if insufficient_balance:
            print(f"INSUFFICIENT_BALANCE_STOP HTTP_402={insufficient_balance}", flush=True)
            stop_reason = "insufficient_balance"
            break
        if rate_limited:
            workers = max(1, workers // 2)
            print(f"RATE_LIMIT_BACKOFF HTTP_429={rate_limited} next_workers={workers}", flush=True)
            time.sleep(args.cooldown_seconds)
    final_valid = valid_document_count(output)
    if stop_reason == "insufficient_balance" or final_valid / args.total_docs < args.min_valid_ratio:
        raise RuntimeError(f"second-judge coverage {final_valid}/{args.total_docs} is below gate or account exhausted")


if __name__ == "__main__":
    main()
