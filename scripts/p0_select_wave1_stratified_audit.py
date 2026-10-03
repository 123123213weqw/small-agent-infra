#!/usr/bin/env python3
"""Select a fresh, domain/length-stratified audit sample from wave 1."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


DOMAINS = ("general_zh", "general_en", "math_en", "code_python", "code_shell", "code_other")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["document_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--tokenizer-json", type=Path, required=True)
    p.add_argument("--exclude-input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--per-stratum", type=int, default=5)
    p.add_argument("--max-chars", type=int, default=20000)
    p.add_argument("--seed", default="p0-wave1-stratified-audit90-v1")
    p.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    args = p.parse_args()
    if args.output.exists() or args.manifest.exists() or args.per_stratum < 1:
        raise ValueError("output exists or invalid per-stratum size")

    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    manifest_path = args.pool / "manifest.json"
    pool = json.loads(manifest_path.read_text(encoding="utf-8"))
    if pool.get("stage") not in {"consolidate_candidates", "build_candidates"}:
        raise ValueError("pool must contain built or consolidated candidates")
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    excluded = load_ids(args.exclude_input)
    groups: dict[str, list[dict]] = defaultdict(list)
    excluded_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"documents": 0, "own_tokens": 0})
    for shard in pool["shards"]:
        path = args.pool / shard["domain"] / shard["file"]
        if path.stat().st_size != shard["bytes"] or sha256_file(path) != shard["sha256"]:
            raise ValueError(f"shard checksum mismatch: {path}")
        for row in pq.read_table(path, columns=[
            "document_id", "domain", "split", "source_id", "source_locator", "language", "text", "token_count"
        ]).to_pylist():
            if row["split"] != "train" or row["document_id"] in excluded:
                continue
            own_tokens = len(tokenizer.encode(row["text"], add_special_tokens=False).ids)
            if len(row["text"]) > args.max_chars:
                excluded_stats[row["domain"]]["documents"] += 1
                excluded_stats[row["domain"]]["own_tokens"] += own_tokens
                continue
            row["own_token_count"] = own_tokens
            groups[row["domain"]].append(row)

    selected: list[dict] = []
    strata = {}
    for domain in args.domains:
        rows = sorted(groups[domain], key=lambda row: (row["own_token_count"], row["document_id"]))
        if len(rows) < args.per_stratum * 3:
            raise ValueError(f"too few rows in {domain}: {len(rows)}")
        cuts = [0, len(rows) // 3, 2 * len(rows) // 3, len(rows)]
        for bin_idx in range(3):
            bin_rows = rows[cuts[bin_idx]:cuts[bin_idx + 1]]
            ranked = sorted(bin_rows, key=lambda row: hashlib.sha256(
                f"{args.seed}\0{row['document_id']}".encode()
            ).hexdigest())
            chosen = ranked[:args.per_stratum]
            key = f"{domain}/length_{bin_idx + 1}"
            strata[key] = {
                "domain": domain,
                "length_bin": bin_idx + 1,
                "documents": len(bin_rows),
                "own_tokens": sum(row["own_token_count"] for row in bin_rows),
                "min_own_tokens_per_document": min(row["own_token_count"] for row in bin_rows),
                "max_own_tokens_per_document": max(row["own_token_count"] for row in bin_rows),
                "sampled_documents": len(chosen),
                "sampled_own_tokens": sum(row["own_token_count"] for row in chosen),
            }
            for row in chosen:
                envelope = {
                    "document_id": row["document_id"],
                    "source_id": row["source_id"],
                    "source_locator": row["source_locator"],
                    "current_bucket": domain,
                    "language": row["language"],
                    "token_count": row["token_count"],
                    "own_token_count": row["own_token_count"],
                    "document_text": row["text"],
                }
                selected.append({
                    "document_id": row["document_id"],
                    "current_bucket": domain,
                    "length_bin": bin_idx + 1,
                    "own_token_count": row["own_token_count"],
                    "text_chars": len(row["text"]),
                    "review_payload": json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                })
    selected.sort(key=lambda row: hashlib.sha256(f"{args.seed}:order\0{row['document_id']}".encode()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected)
    args.output.write_text(content, encoding="utf-8")
    report = {
        "stage": "stratified_quality_probe_not_training_approved",
        "domains": args.domains,
        "pool_manifest_sha256": sha256_file(manifest_path),
        "tokenizer_json_sha256": sha256_file(args.tokenizer_json),
        "seed": args.seed,
        "per_stratum": args.per_stratum,
        "max_chars": args.max_chars,
        "excluded_prior_sample_ids": len(excluded),
        "overlong_train_rows_by_domain": dict(excluded_stats),
        "strata": strata,
        "selected_documents": len(selected),
        "input_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
    args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "strata"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
