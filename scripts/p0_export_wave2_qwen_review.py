#!/usr/bin/env python3
"""Export deduplicated, eval-quarantined Wave2 pointers for local Qwen review.

Only reads frozen candidates and raw text. The output is an unapproved review
queue; the original Parquet files and audit indexes are never modified.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def export(root: Path, output: Path, limit: int) -> dict:
    import pyarrow.parquet as pq

    root, output = root.resolve(), output.resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("review export already exists; verify or use another version")
    wave = json.loads((root / "manifest.json").read_text())
    near_path = root / "audit/near_dedup_v1/verified_v1/SUMMARY.json"
    near = json.loads(near_path.read_text())
    evaluation_path = root / "audit/eval_screen_v1/SUMMARY.json"
    evaluation = json.loads(evaluation_path.read_text())
    if len(wave["items"]) != 13 or evaluation["documents_scanned"] != 1_116_353:
        raise ValueError("unexpected Wave2 source population")
    near_index = root / "audit/near_dedup_v1/verified_v1/wave2_exclusion_index.parquet"
    eval_index = root / "audit/eval_screen_v1/quarantine_index.parquet"
    if sha256(near_index) != near["outputs_sha256"][near_index.name]:
        raise ValueError("near-dedup exclusion index changed")
    if sha256(eval_index) != evaluation["quarantine_index_sha256"]:
        raise ValueError("eval quarantine index changed")
    exclusions = Counter()
    for reason, path in (("near_duplicate", near_index), ("eval_phrase_quarantine", eval_index)):
        for row in pq.read_table(path, columns=["source_file", "raw_row"]).to_pylist():
            key = (Path(row["source_file"]).name, row["raw_row"])
            exclusions[key] += 1
    if max(exclusions.values(), default=0) != 1:
        raise ValueError("overlapping exclusion indexes")
    if len(exclusions) != near["wave2_excluded_documents"] + evaluation["hit_documents_quarantined"]:
        raise ValueError("exclusion count mismatch")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    if temp.exists():
        raise FileExistsError(temp)
    count = tokens = 0
    by_shard = {}
    try:
        with temp.open("w", encoding="utf-8") as dest:
            for item in wave["items"]:
                filename = Path(item["path"]).name
                sketch = root / "audit/near_dedup_v1/sketch_wave2" / f"{filename}.sketch.parquet"
                receipt = json.loads(sketch.with_suffix(".done.json").read_text())
                if sha256(sketch) != receipt["output_sha256"]:
                    raise ValueError(f"sketch receipt mismatch: {filename}")
                pointers = {}
                for batch in pq.ParquetFile(sketch).iter_batches(
                    batch_size=8192,
                    columns=["raw_row", "content_sha256", "tokenizer_v1_tokens"],
                ):
                    for row in batch.to_pylist():
                        if (filename, row["raw_row"]) not in exclusions:
                            pointers[row["raw_row"]] = row
                shard_count = shard_tokens = raw_row = 0
                raw = pq.ParquetFile(root / "raw" / filename)
                for batch in raw.iter_batches(batch_size=2048, columns=["text", "url"]):
                    for text, url in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                        pointer = pointers.get(raw_row)
                        if pointer is not None and (limit == 0 or count < limit):
                            if not isinstance(text, str):
                                raise ValueError(f"missing raw text: {filename}/{raw_row}")
                            text = normalized(text)
                            if hashlib.sha256(text.encode()).hexdigest() != pointer["content_sha256"]:
                                raise ValueError(f"raw text hash mismatch: {filename}/{raw_row}")
                            doc_id = pointer["content_sha256"]
                            body = {"document_id": doc_id, "source_id": "fineweb_edu_new",
                                    "source_locator": url or "", "current_bucket": "general_en",
                                    "language": "en", "document_text": text}
                            row = {"document_id": doc_id, "source_id": "fineweb_edu_new",
                                   "source_file": filename, "raw_row": raw_row,
                                   "current_bucket": "general_en",
                                   "token_count": pointer["tokenizer_v1_tokens"],
                                   "candidate_relpath": f"fineweb_edu_wave2_v1/{filename}",
                                   "row_in_shard": raw_row,
                                   "review_payload": json.dumps(body, ensure_ascii=False,
                                                                separators=(",", ":"))}
                            dest.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                            count += 1
                            tokens += pointer["tokenizer_v1_tokens"]
                            shard_count += 1
                            shard_tokens += pointer["tokenizer_v1_tokens"]
                        raw_row += 1
                if raw_row != receipt["raw_rows_scanned"]:
                    raise ValueError(f"raw row count changed: {filename}")
                by_shard[filename] = {"documents": shard_count, "tokenizer_v1_tokens": shard_tokens,
                                      "sketch_sha256": receipt["output_sha256"]}
                print(json.dumps({"shard": filename, "exported": count}), flush=True)
                if limit and count == limit:
                    break
            if limit == 0:
                expected = evaluation["documents_scanned"] - evaluation["hit_documents_quarantined"]
                expected_tokens = (evaluation["tokenizer_v1_tokens_scanned"]
                                   - evaluation["hit_document_tokens_quarantined"])
                if count != expected or tokens != expected_tokens:
                    raise ValueError(f"export mismatch: {count}/{tokens} != {expected}/{expected_tokens}")
            elif count != limit:
                raise ValueError(f"smoke requested {limit}, got {count}")
        temp.replace(output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    meta = {"stage": "wave2_qwen_review_export_not_training_approved",
            "limit": limit, "source_revision": wave["revision"],
            "near_summary_sha256": sha256(near_path),
            "eval_summary_sha256": sha256(evaluation_path),
            "near_exclusion_sha256": sha256(near_index),
            "eval_quarantine_sha256": sha256(eval_index),
            "documents": count, "tokenizer_v1_tokens": tokens,
            "input_sha256": sha256(output), "input_bytes": output.stat().st_size,
            "by_shard": by_shard}
    manifest_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    print(json.dumps(export(args.root, args.output, args.limit), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
