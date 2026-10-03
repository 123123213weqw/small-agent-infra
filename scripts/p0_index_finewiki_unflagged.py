#!/usr/bin/env python3
"""Freeze a text-free index of FineWiki records lacking one known extraction artifact.

The output is an audit candidate index, not a training manifest. Source text is
referenced by clean Parquet filename and zero-based row, never copied.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

import p0_data_pipeline as core
from p0_audit_finewiki_extraction import EMPTY_LANGUAGE


def index_schema():
    import pyarrow as pa
    return pa.schema([
        ("document_id", pa.string()), ("normalized_sha256", pa.string()),
        ("family_id", pa.string()), ("split", pa.string()),
        ("url", pa.string()), ("chars", pa.int64()), ("has_math", pa.bool_()),
        ("token_count", pa.int64()), ("clean_file", pa.string()),
        ("row_in_clean_file", pa.int64()), ("wave1_url_family_overlap", pa.bool_()),
    ])


def load_wave1_overlap(path: Path) -> set[str]:
    report = json.loads((path / "manifest.json").read_text())
    hits_path = path / "hits.jsonl"
    if (report["stage"] != "finewiki_zh_vs_wave1_exact_and_family_overlap_audit_only" or
            core.sha256_file(hits_path) != report["hits_sha256"]):
        raise ValueError("Wave1 overlap audit integrity mismatch")
    ids = set()
    for line in hits_path.read_text().splitlines():
        row = json.loads(line)
        if row["exact_content"] or not row["same_url_family"]:
            raise ValueError("unexpected overlap hit type")
        ids.add(row["candidate_document_id"])
    if len(ids) != report["counts"]["either_hit"]:
        raise ValueError("Wave1 overlap hit count mismatch")
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    clean_root = args.root / "clean_rule_v1"
    clean_manifest_path = clean_root / "manifest.json"
    clean_manifest = json.loads(clean_manifest_path.read_text())
    token_manifest_path = clean_root / "token_count_manifest.json"
    token_manifest = json.loads(token_manifest_path.read_text())
    artifact_count_path = args.root / "audit/empty_language_token_count_v1.json"
    artifact_count = json.loads(artifact_count_path.read_text())
    overlap_root = args.root / "audit/wave1_overlap_v1"
    overlap_ids = load_wave1_overlap(overlap_root)
    if (clean_manifest["stage"] != "finewiki_zh_rule_clean_audit_only" or
            clean_manifest["training_eligible"] or
            token_manifest["clean_manifest_sha256"] != core.sha256_file(clean_manifest_path) or
            artifact_count["clean_manifest_sha256"] != core.sha256_file(clean_manifest_path) or
            artifact_count["tokenizer_sha256"] != core.sha256_file(args.tokenizer) or
            artifact_count["artifact_regex"] != EMPTY_LANGUAGE.pattern):
        raise ValueError("input manifest or selection rule mismatch")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    schema = index_schema()
    counts = Counter()
    family_splits = {}
    selected_ids = set()
    file_reports = []
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finewiki-index-", dir=args.output_dir.parent) as tmp:
        stage = Path(tmp) / args.output_dir.name
        stage.mkdir()
        for item in clean_manifest["files"]:
            source = clean_root / item["output_file"]
            if core.sha256_file(source) != item["output_sha256"]:
                raise ValueError(f"clean shard checksum mismatch: {source}")
            output = stage / item["output_file"]
            row_index = 0
            file_counts = Counter()
            with pq.ParquetWriter(output, schema=schema, compression="zstd", compression_level=3) as writer:
                for batch in pq.ParquetFile(source).iter_batches(batch_size=256, columns=[
                    "document_id", "normalized_sha256", "family_id", "split", "url",
                    "chars", "has_math", "text",
                ]):
                    rows = batch.to_pylist()
                    encodings = tokenizer.encode_batch([r["text"] for r in rows], add_special_tokens=False)
                    chosen = []
                    for row, encoded in zip(rows, encodings):
                        counts["input_documents"] += 1
                        n = len(encoded.ids)
                        flagged = bool(EMPTY_LANGUAGE.search(row["text"]))
                        group = "flagged" if flagged else "unflagged"
                        counts[f"documents:{group}"] += 1
                        counts[f"tokens:{group}"] += n
                        previous_split = family_splits.setdefault(row["family_id"], row["split"])
                        if previous_split != row["split"]:
                            raise ValueError("same URL family assigned to multiple splits")
                        if not flagged:
                            if row["document_id"] in selected_ids:
                                raise ValueError("duplicate selected document ID")
                            selected_ids.add(row["document_id"])
                            overlap = row["document_id"] in overlap_ids
                            chosen.append({
                                "document_id": row["document_id"],
                                "normalized_sha256": row["normalized_sha256"],
                                "family_id": row["family_id"], "split": row["split"],
                                "url": row["url"], "chars": row["chars"],
                                "has_math": row["has_math"], "token_count": n,
                                "clean_file": source.name, "row_in_clean_file": row_index,
                                "wave1_url_family_overlap": overlap,
                            })
                            counts[f"unflagged_tokens:{row['split']}"] += n
                            file_counts["selected"] += 1
                            file_counts["selected_tokens"] += n
                            if overlap:
                                counts["unflagged_wave1_url_family_overlap"] += 1
                        row_index += 1
                    if chosen:
                        writer.write_table(pa.Table.from_pylist(chosen, schema=schema))
            if row_index != item["counts"]["kept"]:
                raise ValueError(f"clean shard row count mismatch: {source.name}")
            file_reports.append({
                "file": output.name, "rows_in_clean": row_index,
                "selected": file_counts["selected"], "selected_tokens": file_counts["selected_tokens"],
                "bytes": output.stat().st_size, "sha256": core.sha256_file(output),
            })
            print(f"indexed {source.name}: {dict(file_counts)}", flush=True)
        expected = artifact_count["counts"]
        for name in ("flagged", "unflagged"):
            if (counts[f"documents:{name}"] != expected[f"documents:{name}"] or
                    counts[f"tokens:{name}"] != expected[f"tokens:{name}"]):
                raise ValueError(f"{name} count mismatch against prior audit")
        if counts["input_documents"] != clean_manifest["counts"]["kept"]:
            raise ValueError("overall document count mismatch")
        report = {
            "stage": "finewiki_zh_unflagged_index_audit_only",
            "selection": "no EMPTY_LANGUAGE regex hit; Wave1 URL overlaps are tagged, not silently removed",
            "clean_manifest_sha256": core.sha256_file(clean_manifest_path),
            "artifact_count_sha256": core.sha256_file(artifact_count_path),
            "wave1_overlap_manifest_sha256": core.sha256_file(overlap_root / "manifest.json"),
            "tokenizer_sha256": core.sha256_file(args.tokenizer),
            "script_sha256": core.sha256_file(Path(__file__)),
            "counts": dict(counts), "files": file_reports,
            "license_status": "pending", "training_eligible": False,
            "remaining_gates": ["quality_review", "near_dedup", "heldout_decontamination", "license_review"],
        }
        (stage / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        stage.rename(args.output_dir)
    print(json.dumps({"complete": True, "counts": dict(counts)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
