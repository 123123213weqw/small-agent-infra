#!/usr/bin/env python3
"""Build immutable mmap token shards from an approved Parquet source or token JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_agent.data.sharded_format import sha256_file  # noqa: E402
from small_agent.data.sharded_pack import TokenizedDocument, pack_sharded  # noqa: E402


def from_token_jsonl(path: Path) -> Iterator[TokenizedDocument]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                yield TokenizedDocument(
                    document_id=str(row["document_id"]),
                    split=str(row["split"]),
                    domain=str(row["domain"]),
                    token_ids=list(map(int, row["token_ids"])),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"bad token JSONL row {line_no}") from error


def from_frozen_parquet(
    source: Path, tokenizer_dir: Path, *, internal_candidate_smoke: bool = False
) -> tuple[Iterator[TokenizedDocument], dict, dict]:
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    tokenizer_manifest = json.loads((tokenizer_dir / "manifest.json").read_text(encoding="utf-8"))
    source_stage = source_manifest.get("stage")
    supported = {"accepted_real", "p0_1m_smoke_frozen"}
    if internal_candidate_smoke:
        supported.add("consolidate_candidates")
    if source_stage not in supported:
        raise ValueError("unsupported frozen source stage")
    if internal_candidate_smoke and source_stage != "consolidate_candidates":
        raise ValueError("internal candidate smoke requires consolidate_candidates source")
    if source_stage == "consolidate_candidates" and not source_manifest.get("license_gate", {}).get("training_eligible"):
        raise ValueError("candidate source license gate is not approved")
    if source_stage == "accepted_real" and not source_manifest.get("license_gate", {}).get("training_eligible"):
        raise ValueError("approved source is not training eligible")
    if tokenizer_manifest.get("stage") != "frozen_tokenizer":
        raise ValueError("tokenizer is not frozen")
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if sha256_file(tokenizer_path) != tokenizer_manifest["artifacts"]["tokenizer.json"]["sha256"]:
        raise ValueError("tokenizer SHA-256 mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = int(tokenizer_manifest["special_token_ids"]["<eos>"])
    vocab_size = int(tokenizer_manifest["vocab_size"])

    def rows() -> Iterator[TokenizedDocument]:
        seen = 0
        splits = {"train": 0, "validation": 0, "test": 0}
        for shard in source_manifest["shards"]:
            path = source / shard["domain"] / shard["file"]
            if path.stat().st_size != shard["bytes"] or sha256_file(path) != shard["sha256"]:
                raise ValueError(f"source Parquet integrity failure: {path}")
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=256, columns=["document_id", "split", "text"]):
                for row in batch.to_pylist():
                    ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
                    if tokenizer.decode(ids, skip_special_tokens=False) != row["text"]:
                        raise ValueError(f"tokenizer roundtrip failed: {row['document_id']}")
                    seen += 1
                    splits[row["split"]] += 1
                    yield TokenizedDocument(
                        document_id=row["document_id"],
                        split=row["split"],
                        domain=shard["domain"],
                        token_ids=ids,
                    )
        counts = source_manifest.get("counts", {})
        count_field = {
            "accepted_real": "accepted_documents",
            "p0_1m_smoke_frozen": "documents",
            "consolidate_candidates": "retained_documents",
        }[source_stage]
        expected_documents = counts.get(count_field)
        if expected_documents is not None and seen != expected_documents:
            raise ValueError(f"source document count changed: {seen} != {expected_documents}")
        if source_stage == "p0_1m_smoke_frozen" and splits != source_manifest["splits"]:
            raise ValueError("source split counts changed")

    return rows(), source_manifest, {
        "vocab_size": vocab_size,
        "eos_token_id": eos,
        "tokenizer_sha256": sha256_file(tokenizer_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--source", type=Path, help="frozen accepted_real/P0 Parquet directory")
    sources.add_argument("--tokenized-jsonl", type=Path, help="already-tokenized documents")
    parser.add_argument("--tokenizer", type=Path, help="required with --source")
    parser.add_argument(
        "--internal-candidate-smoke", action="store_true",
        help="explicitly pack license-approved but quality-unreviewed candidates for non-distributable diagnostic runs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mixture-json", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--target-shard-tokens", type=int, default=32_000_000)
    parser.add_argument("--vocab-size", type=int, help="required with --tokenized-jsonl")
    parser.add_argument("--eos-token-id", type=int, help="required with --tokenized-jsonl")
    parser.add_argument("--tokenizer-sha256", help="required with --tokenized-jsonl")
    args = parser.parse_args()
    mixture = json.loads(args.mixture_json.read_text(encoding="utf-8"))
    if args.source:
        if not args.tokenizer:
            parser.error("--tokenizer is required with --source")
        documents, source_manifest, tokenizer = from_frozen_parquet(
            args.source, args.tokenizer, internal_candidate_smoke=args.internal_candidate_smoke
        )
        license_gate = source_manifest["license_gate"]
        if args.internal_candidate_smoke:
            license_gate = {
                "training_eligible": False,
                "source_license_eligible": True,
                "quality_approved": False,
                "source_stage": "consolidate_candidates",
                "restriction": "internal_non_distributable_diagnostic_only",
            }
        source_hash = sha256_file(args.source / "manifest.json")
    else:
        if args.internal_candidate_smoke:
            parser.error("--internal-candidate-smoke requires --source")
        if args.vocab_size is None or args.eos_token_id is None or not args.tokenizer_sha256:
            parser.error("token JSONL requires --vocab-size, --eos-token-id, --tokenizer-sha256")
        documents = from_token_jsonl(args.tokenized_jsonl)
        tokenizer = {
            "vocab_size": args.vocab_size,
            "eos_token_id": args.eos_token_id,
            "tokenizer_sha256": args.tokenizer_sha256,
        }
        license_gate = {"training_eligible": False, "source": "token_jsonl_engineering"}
        source_hash = sha256_file(args.tokenized_jsonl)
    manifest = pack_sharded(
        documents,
        args.output,
        sequence_length=args.sequence_length,
        target_shard_tokens=args.target_shard_tokens,
        mixture=mixture,
        license_gate=license_gate,
        source_manifest_sha256=source_hash,
        **tokenizer,
    )
    print(json.dumps({"version": manifest["version"], "shards": len(manifest["shards"]), "samples": sum(s["samples"] for s in manifest["shards"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
