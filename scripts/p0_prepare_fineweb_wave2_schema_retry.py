#!/usr/bin/env python3
"""Freeze records lacking valid Wave2 review for a schema-constrained retry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    review = args.review_dir.resolve()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    report = json.loads((review / "REPORT.json").read_text())
    source = review / "deepseek_review.input.jsonl"
    results = review / "deepseek_full.results.jsonl"
    if sha256(source) != report["outputs_sha256"][source.name]:
        raise ValueError("source sample hash mismatch")
    valid = set()
    for line in results.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        try:
            doc, = json.loads(row["source_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if row.get("annotation", {}).get("document_id") == doc:
            valid.add(doc)
    out.mkdir(parents=True)
    retry = out / "deepseek_full.input.jsonl"
    counts = 0
    with retry.open("w", encoding="utf-8") as dest:
        for line in source.open(encoding="utf-8"):
            if json.loads(line)["document_id"] in valid:
                continue
            dest.write(line)
            counts += 1
    manifest = {"status": "retry_only_not_training_approved", "retry_count": counts,
                "original_valid_count": len(valid), "source_input_sha256": sha256(source),
                "source_results_sha256": sha256(results), "retry_input_sha256": sha256(retry),
                "response_format": "json_schema", "model": "deepseek-flash"}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
