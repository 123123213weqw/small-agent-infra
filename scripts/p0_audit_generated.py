#!/usr/bin/env python3
"""Read-only integrity and Python-syntax audit for a generated P0 corpus."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


PYTHON_FENCE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(root: Path) -> tuple[dict, list[dict]]:
    import pyarrow.parquet as pq

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for shard in manifest["shards"]:
        path = root / shard["domain"] / shard["file"]
        if path.stat().st_size != shard["bytes"] or sha256_file(path) != shard["sha256"]:
            raise ValueError(f"shard integrity failure: {path}")
        shard_rows = pq.read_table(path).to_pylist()
        if len(shard_rows) != shard["rows"]:
            raise ValueError(f"shard row count mismatch: {path}")
        rows.extend(shard_rows)
    return manifest, rows


def audit(root: Path, tokenizer_json: Path, compare_root: Path | None) -> dict:
    from tokenizers import Tokenizer

    manifest, rows = load_rows(root)
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    ids: set[str] = set()
    hashes: set[str] = set()
    family_splits: dict[str, set[str]] = defaultdict(set)
    syntax: dict[str, Counter] = defaultdict(Counter)
    own_tokens: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    syntax_fail_examples = []
    max_doc_tokens = 0
    for row in rows:
        document_id = row["document_id"]
        digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        if digest != row["normalized_sha256"]:
            raise ValueError(f"content hash mismatch: {document_id}")
        if document_id in ids or digest in hashes:
            raise ValueError(f"duplicate id or exact content: {document_id}")
        ids.add(document_id)
        hashes.add(digest)
        family_splits[row["family_id"]].add(row["split"])
        split_counts[row["split"]] += 1
        encoded = tokenizer.encode(row["text"], add_special_tokens=False).ids
        if tokenizer.decode(encoded, skip_special_tokens=False) != row["text"]:
            raise ValueError(f"tokenizer roundtrip failure: {document_id}")
        n_tokens = len(encoded)
        max_doc_tokens = max(max_doc_tokens, n_tokens)
        key = f"{row['source_id']}/{row['domain']}"
        own_tokens[key] += n_tokens
        blocks = PYTHON_FENCE.findall(row["text"])
        syntax[key]["documents"] += 1
        syntax[key]["documents_with_python_fence"] += bool(blocks)
        for block in blocks:
            try:
                ast.parse(block)
                syntax[key]["parse_ok"] += 1
            except SyntaxError as error:
                syntax[key]["parse_failed"] += 1
                if len(syntax_fail_examples) < 8:
                    syntax_fail_examples.append(
                        {"document_id": document_id, "error": str(error)}
                    )
    if len(rows) != manifest["counts"]["documents"]:
        raise ValueError("manifest document count mismatch")
    if dict(split_counts) != manifest["splits"]:
        raise ValueError("manifest split counts mismatch")
    if any(len(splits) > 1 for splits in family_splits.values()):
        raise ValueError("family split conflict within generated corpus")

    comparison = None
    if compare_root is not None:
        _, compare_rows = load_rows(compare_root)
        compare_ids = {row["document_id"] for row in compare_rows}
        compare_hashes = {row["normalized_sha256"] for row in compare_rows}
        cross_families: dict[str, set[str]] = defaultdict(set)
        for row in rows + compare_rows:
            cross_families[row["family_id"]].add(row["split"])
        comparison = {
            "root": str(compare_root),
            "document_id_overlap": len(ids & compare_ids),
            "exact_content_overlap": len(hashes & compare_hashes),
            "family_split_conflicts": sum(len(splits) > 1 for splits in cross_families.values()),
        }

    return {
        "dataset_root": str(root),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "tokenizer_json_sha256": sha256_file(tokenizer_json),
        "documents": len(rows),
        "own_tokenizer_tokens_excluding_eos": sum(own_tokens.values()),
        "own_tokens_by_source_domain": dict(sorted(own_tokens.items())),
        "max_document_tokens": max_doc_tokens,
        "splits": dict(sorted(split_counts.items())),
        "python_syntax_by_source_domain": {
            key: dict(counter) for key, counter in sorted(syntax.items())
        },
        "python_syntax_fail_examples": syntax_fail_examples,
        "comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--compare-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(
        args.dataset_root.resolve(),
        args.tokenizer_json.resolve(),
        args.compare_root.resolve() if args.compare_root else None,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
