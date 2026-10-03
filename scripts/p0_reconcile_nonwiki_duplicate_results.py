#!/usr/bin/env python3
"""Reconcile duplicate Qwen JSONL writes after a host restart.

Run only when every scorer process is stopped. Originals are archived before
the per-shard JSONL files are atomically replaced. Disagreeing ratings are
recorded separately and the downstream gate drops those document IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile

from p0_nonwiki_review_batch_v2 import strict_keep


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verdict(record: dict) -> tuple[bool, str, str]:
    annotation = record["annotation"]
    return (
        strict_keep(annotation, record["source"]["current_bucket"]),
        annotation.get("decision", ""),
        annotation.get("recommended_bucket", ""),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_dir.resolve()
    light = root / "results8/light"
    manifest = json.loads((root / "input_bands8_v1/manifest.json").read_text())
    band = next(item for item in manifest["bands"] if item["name"] == "light")
    paths = [light / f"shard-{i}.results.jsonl" for i in range(8)]
    archive = root / "results8/light_pre_reconcile_v1.tar.gz"
    report_path = light / "reconciliation_v1.json"
    conflict_path = light / "reconcile_conflicts.jsonl"
    if archive.exists() or report_path.exists() or conflict_path.exists():
        raise FileExistsError("reconciliation artifacts already exist; refusing a second pass")
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("one or more light-band result files are absent")

    raw_hashes = {path.name: sha256(path) for path in paths}
    with tarfile.open(archive, "w:gz") as tar:
        for path in paths:
            tar.add(path, arcname=path.name)

    staged: list[tuple[Path, Path]] = []
    conflicts: dict[str, set[tuple[bool, str, str]]] = {}
    rows = []
    try:
        for i, path in enumerate(paths):
            input_path = root / "input_bands8_v1" / band["shards"][i]["file"]
            expected = [json.loads(line)["document_id"] for line in input_path.open()]
            expected_set = set(expected)
            if len(expected_set) != len(expected):
                raise ValueError(f"duplicate input document ID in shard {i}")
            chosen: dict[str, bytes] = {}
            good: set[str] = set()
            first_verdict: dict[str, tuple[bool, str, str]] = {}
            seen_lines = 0
            with path.open("rb") as stream:
                for line in stream:
                    seen_lines += 1
                    record = json.loads(line)
                    doc_id, = json.loads(record["source_id"])
                    if doc_id not in expected_set:
                        raise ValueError(f"unknown document ID {doc_id} in {path}")
                    if record["status"] == "ok":
                        value = verdict(record)
                        if doc_id in first_verdict and first_verdict[doc_id] != value:
                            conflicts.setdefault(doc_id, {first_verdict[doc_id]}).add(value)
                        if doc_id not in good:
                            chosen[doc_id] = line
                            good.add(doc_id)
                            first_verdict[doc_id] = value
                    elif doc_id not in chosen:
                        chosen[doc_id] = line
            fd, name = tempfile.mkstemp(prefix=f".shard-{i}.reconciled-", dir=light)
            with os.fdopen(fd, "wb") as stream:
                for doc_id in expected:
                    if doc_id in chosen:
                        stream.write(chosen[doc_id])
                stream.flush()
                os.fsync(stream.fileno())
            temp_path = Path(name)
            staged.append((temp_path, path))
            rows.append({"shard": i, "raw_rows": seen_lines, "canonical_rows": len(chosen),
                         "good_rows": len(good), "unattempted": len(expected_set - chosen.keys()),
                         "duplicate_rows_removed": seen_lines - len(chosen)})

        with conflict_path.open("w", encoding="utf-8") as stream:
            for doc_id, values in sorted(conflicts.items()):
                stream.write(json.dumps({"document_id": doc_id,
                                         "gate_verdicts": sorted([list(v) for v in values])}) + "\n")
        for temp_path, destination in staged:
            os.replace(temp_path, destination)
        report = {"stage": "reconcile_after_concurrent_qwen_resume_v1",
                  "raw_archive": archive.name, "raw_archive_sha256": sha256(archive),
                  "raw_result_sha256": raw_hashes, "shards": rows,
                  "conflict_documents": len(conflicts),
                  "conflicts_sha256": sha256(conflict_path)}
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"canonical_rows": sum(row["canonical_rows"] for row in rows),
                          "duplicate_rows_removed": sum(row["duplicate_rows_removed"] for row in rows),
                          "unattempted": sum(row["unattempted"] for row in rows),
                          "conflict_documents": len(conflicts)}), flush=True)
    finally:
        for temp_path, _ in staged:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
