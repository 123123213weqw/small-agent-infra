#!/usr/bin/env python3
"""Stream pinned FineWeb-Edu shards into pointer-only candidate indexes.

This is deterministic prefiltering and exact deduplication, not training-data
approval. Each output is committed atomically and can be rebuilt after a crash.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from p0_prefilter_nonwiki_new_v1 import classify, normalize


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hashes(index: Path, column: str, seen: set[str]) -> None:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(index)
    for group in range(pf.num_row_groups):
        seen.update(pf.read_row_group(group, columns=[column]).column(0).to_pylist())


def process_shard(item: dict, root: Path, tokenizer, seen: set[str], output: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    name = Path(item["path"]).name
    raw = root / "raw" / name
    if raw.stat().st_size != item["bytes"] or sha256_file(raw) != item["sha256"]:
        raise ValueError(f"raw integrity mismatch: {raw}")
    pf = pq.ParquetFile(raw)
    columns = {"text", "url", "language", "language_score", "score", "int_score", "token_count"}
    if not columns <= set(pf.schema_arrow.names):
        raise ValueError(f"schema mismatch: {raw}")
    schema = pa.schema([
        ("source_id", pa.string()), ("source_file", pa.string()),
        ("raw_row", pa.int64()), ("content_sha256", pa.string()),
        ("locator", pa.string()), ("chars", pa.int32()),
        ("upstream_score", pa.float64()), ("upstream_token_count", pa.int64()),
        ("tokenizer_v1_tokens", pa.int64()),
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".parquet.tmp")
    tmp.unlink(missing_ok=True)
    counts = Counter()
    new_hashes = set()
    row_offset = 0
    try:
        with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
            for group in range(pf.num_row_groups):
                rows = pf.read_row_group(group, columns=sorted(columns)).to_pylist()
                kept = []
                for local_row, row in enumerate(rows):
                    counts["raw_rows"] += 1
                    text = normalize(row.get("text")) if isinstance(row.get("text"), str) else ""
                    reasons = classify("fineweb_edu_new", row, text)
                    if reasons:
                        counts["rule_rejected"] += 1
                        for reason in reasons:
                            counts["reason:" + reason] += 1
                        continue
                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if digest in seen or digest in new_hashes:
                        counts["exact_duplicate"] += 1
                        continue
                    token_count = len(tokenizer.encode(text, add_special_tokens=False).ids)
                    if not token_count:
                        counts["zero_token"] += 1
                        continue
                    new_hashes.add(digest)
                    counts["candidate_rows"] += 1
                    counts["candidate_upstream_tokens"] += int(row.get("token_count") or 0)
                    counts["candidate_tokenizer_v1_tokens"] += token_count
                    kept.append({
                        "source_id": "fineweb_edu_wave2_v1",
                        "source_file": item["path"], "raw_row": row_offset + local_row,
                        "content_sha256": digest, "locator": str(row.get("url") or ""),
                        "chars": len(text), "upstream_score": float(row.get("score") or 0),
                        "upstream_token_count": int(row.get("token_count") or 0),
                        "tokenizer_v1_tokens": token_count,
                    })
                if kept:
                    writer.write_table(pa.Table.from_pylist(kept, schema=schema))
                row_offset += len(rows)
                print(json.dumps({"file": name, "row_group": group + 1,
                                  "row_groups": pf.num_row_groups,
                                  "raw_rows": counts["raw_rows"],
                                  "candidate_rows": counts["candidate_rows"]}), flush=True)
        report = {
            "status": "candidate_only_not_training_approved",
            "source_file": item["path"], "raw_bytes": item["bytes"],
            "raw_sha256": item["sha256"], "index_sha256": sha256_file(tmp),
            "counts": dict(counts),
        }
        tmp.replace(output)
        output.with_suffix(".done.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        seen.update(new_hashes)
        return report
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--old-candidate-index", type=Path, required=True)
    parser.add_argument("--old-wave1-index", type=Path, required=True)
    parser.add_argument("--wait-timeout", type=int, default=86400)
    args = parser.parse_args()
    from tokenizers import Tokenizer

    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["version"] != "p0_fineweb_edu_wave2_v1" or len(manifest["items"]) != 13:
        raise ValueError("unexpected manifest")
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    seen: set[str] = set()
    load_hashes(args.old_candidate_index, "content_sha256", seen)
    load_hashes(args.old_wave1_index, "normalized_sha256", seen)
    print(json.dumps({"baseline_hashes": len(seen), "shards": len(manifest["items"])}), flush=True)
    output_dir = root / "audit" / "prefilter_v1"
    reports = []
    for item in manifest["items"]:
        name = Path(item["path"]).name
        output = output_dir / f"{name}.candidate_index.parquet"
        done = output.with_suffix(".done.json")
        if done.exists():
            report = json.loads(done.read_text())
            if (not output.exists() or sha256_file(output) != report["index_sha256"]
                    or report["raw_sha256"] != item["sha256"]):
                raise ValueError(f"completed index integrity mismatch: {output}")
            load_hashes(output, "content_sha256", seen)
            reports.append(report)
            print(json.dumps({"file": name, "status": "already_done"}), flush=True)
            continue
        raw = root / "raw" / name
        deadline = time.monotonic() + args.wait_timeout
        while not raw.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"waiting for transferred shard: {raw}")
            time.sleep(30)
        report = process_shard(item, root, tokenizer, seen, output)
        reports.append(report)
        print(json.dumps({"file": name, "counts": report["counts"]}), flush=True)
    summary = {
        "status": "candidate_only_not_training_approved",
        "dataset": manifest["dataset"], "revision": manifest["revision"],
        "shards": len(reports),
        "candidate_rows": sum(x["counts"].get("candidate_rows", 0) for x in reports),
        "candidate_tokenizer_v1_tokens": sum(
            x["counts"].get("candidate_tokenizer_v1_tokens", 0) for x in reports),
        "pending": ["near_dedup", "evaluation_decontamination", "source_and_license_review",
                    "stratified_second_judge", "mixture_selection", "pack_and_split"],
    }
    (output_dir / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
