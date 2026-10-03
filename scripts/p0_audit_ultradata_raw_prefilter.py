#!/usr/bin/env python3
"""Count the four raw shards after conservative hard filters and exact dedup.

Writes only aggregate audit statistics; never writes training data or changes raw
Parquet. Token counts use tokenizer_v1 without chat formatting or special tokens.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from p0_replay_ultradata_prefilter import CODE_SOURCE, classify_hard


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan(raw_dir: Path, tokenizer_json: Path, batch_size: int) -> dict:
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    plan_path = raw_dir / "download_plan_v1.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    seen: set[bytes] = set()
    report = {
        "stage": "full_raw_prefilter_audit_not_training_approval",
        "plan_sha256": file_sha256(plan_path),
        "tokenizer_sha256": file_sha256(tokenizer_json),
        "dedup_method": "exact_utf8_sha256_across_four_sources_after_hard_path_filter",
        "source_order": [item["source_id"] for item in plan["items"]],
        "sources": {},
    }
    for item in plan["items"]:
        source = item["source_id"]
        path = raw_dir / "raw" / source / Path(item["path"]).name
        if path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise ValueError(f"raw shard integrity mismatch: {source}")
        pf = pq.ParquetFile(path)
        columns = ["content"]
        if source == CODE_SOURCE:
            columns += ["repo_name", "relative_path"]
        stats = Counter()
        started = time.monotonic()
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            data = batch.to_pydict()
            texts = [value if isinstance(value, str) else "" for value in data["content"]]
            encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
            for i, text in enumerate(texts):
                stats["input_rows"] += 1
                tokens = len(encodings[i].ids)
                stats["input_tokens"] += tokens
                if source == CODE_SOURCE:
                    locator = f"{data['repo_name'][i]}/{data['relative_path'][i]}"
                else:
                    locator = ""
                reasons = classify_hard({"source_id": source, "source_locator": locator}, text)
                if not isinstance(data["content"][i], str):
                    reasons.append("invalid_content_type")
                if reasons:
                    for reason in reasons:
                        stats["reject_" + reason] += 1
                    stats["rejected_rows"] += 1
                    stats["rejected_tokens"] += tokens
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                if digest in seen:
                    stats["reject_exact_duplicate_text"] += 1
                    stats["rejected_rows"] += 1
                    stats["rejected_tokens"] += tokens
                    continue
                seen.add(digest)
                stats["candidate_rows"] += 1
                stats["candidate_tokens"] += tokens
        if stats["input_rows"] != pf.metadata.num_rows:
            raise ValueError(f"row count mismatch: {source}")
        report["sources"][source] = {
            "dataset": item["dataset"], "revision": item["revision"],
            "license_status": item["license"], "parquet_sha256": item["sha256"],
            "elapsed_seconds": round(time.monotonic() - started, 1),
            **dict(sorted(stats.items())),
        }
        print(json.dumps({"source": source, **report["sources"][source]}, ensure_ascii=False), flush=True)
    totals = Counter()
    for source in report["sources"].values():
        totals.update({key: value for key, value in source.items() if isinstance(value, int)})
    report["totals"] = dict(sorted(totals.items()))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    report = scan(args.raw_dir, args.tokenizer_json, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "totals": report["totals"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
