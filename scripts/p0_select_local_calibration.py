#!/usr/bin/env python3
"""Select a deterministic, label-balanced local-judge calibration set."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


DOMAINS = ("general_zh", "general_en", "math_en", "code_python", "code_shell", "code_other")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select(pool: Path, adjudicated: Path, per_class: int, max_chars: int, seed: str) -> list[dict]:
    import pyarrow.parquet as pq

    labels = {
        row["document_id"]: row
        for row in (json.loads(line) for line in adjudicated.open(encoding="utf-8") if line.strip())
    }
    manifest = json.loads((pool / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("stage") != "consolidate_candidates":
        raise ValueError("calibration pool must be consolidated")
    groups: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
    for shard in manifest["shards"]:
        path = pool / shard["domain"] / shard["file"]
        if file_sha256(path) != shard["sha256"]:
            raise ValueError(f"shard hash mismatch: {path}")
        columns = [
            "document_id", "domain", "source_id", "source_locator", "language", "text", "token_count"
        ]
        for row in pq.read_table(path, columns=columns).to_pylist():
            label = labels.get(row["document_id"])
            if not label or label.get("teacher_status") != "ok":
                continue
            route = label.get("route")
            if route not in {"keep", "drop"} or not 400 <= len(row["text"]) <= max_chars:
                continue
            rank = hashlib.sha256(f"{seed}\0{row['document_id']}".encode()).hexdigest()
            groups[(row["domain"], route)].append((rank, row))

    selected = []
    for domain in DOMAINS:
        for route in ("keep", "drop"):
            ranked = sorted(groups[(domain, route)], key=lambda item: item[0])
            if len(ranked) < per_class:
                raise ValueError(f"{domain}/{route}: only {len(ranked)} eligible rows")
            for _, row in ranked[:per_class]:
                envelope = {
                    "document_id": row["document_id"],
                    "source_id": row["source_id"],
                    "source_locator": row["source_locator"],
                    "current_bucket": row["domain"],
                    "language": row["language"],
                    "token_count": row["token_count"],
                    "document_text": row["text"],
                }
                selected.append(
                    {
                        "document_id": row["document_id"],
                        "current_bucket": domain,
                        "old_route": route,
                        "text_chars": len(row["text"]),
                        "review_payload": json.dumps(
                            envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                        ),
                    }
                )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--seed", default="p0b-local-cal12")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.per_class < 1 or args.max_chars < 400 or args.output.exists():
        raise ValueError("invalid selection arguments or existing output")
    rows = select(args.pool, args.adjudicated, args.per_class, args.max_chars, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "documents": len(rows), "sha256": file_sha256(args.output)}))


if __name__ == "__main__":
    main()
