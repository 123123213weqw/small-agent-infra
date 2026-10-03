#!/usr/bin/env python3
"""Merge prior accepted real data with dual-judge Wave1 additions, no synthetic mixing."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


SCHEMA = pa.schema([
    ("document_id", pa.string()), ("source_id", pa.string()),
    ("source_revision", pa.string()), ("source_locator", pa.string()),
    ("family_id", pa.string()), ("domain", pa.string()), ("language", pa.string()),
    ("text", pa.large_string()), ("raw_sha256", pa.string()),
    ("normalized_sha256", pa.string()), ("license_status", pa.string()),
    ("split", pa.string()), ("own_text_tokens", pa.int64()),
    ("provenance_group", pa.string()),
])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--wave1", type=Path, required=True)
    p.add_argument("--tokenizer-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifests = [json.loads((root / "manifest.json").read_text()) for root in (args.prior, args.wave1)]
    if any(m.get("stage") != "accepted_real" or not m["license_gate"]["training_eligible"] for m in manifests):
        raise ValueError("input source is not approved real")
    tokenizer_hash = sha256(args.tokenizer_json)
    if tokenizer_hash != manifests[1]["tokenizer"]["tokenizer_json_sha256"]:
        raise ValueError("Wave1 tokenizer mismatch")
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    rows_by_domain = defaultdict(list)
    ids, hashes = set(), set()
    family_split = {}
    counts = Counter()
    token_counts = Counter()
    source_counts = Counter()
    for label, root, manifest in zip(("prior_approved_real", "new_dual_judge"),
                                     (args.prior, args.wave1), manifests):
        count = 0
        for shard in manifest["shards"]:
            path = root / shard["domain"] / shard["file"]
            if path.stat().st_size != shard["bytes"] or sha256(path) != shard["sha256"]:
                raise ValueError(f"source shard checksum mismatch: {path}")
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    count += 1
                    if row["document_id"] in ids or row["normalized_sha256"] in hashes:
                        raise ValueError("cross-source exact duplicate")
                    if row["license_status"] != "approved" or row["split"] not in ("train", "validation", "test"):
                        raise ValueError("source license/split mismatch")
                    if row["domain"] != shard["domain"]:
                        raise ValueError("source shard/domain mismatch")
                    old = family_split.setdefault(row["family_id"], row["split"])
                    if old != row["split"]:
                        raise ValueError("family split conflict")
                    n = len(tokenizer.encode(row["text"], add_special_tokens=False).ids)
                    if label == "new_dual_judge" and n != row["own_text_tokens"]:
                        raise ValueError("Wave1 token count mismatch")
                    ids.add(row["document_id"])
                    hashes.add(row["normalized_sha256"])
                    data = {name: row[name] for name in (
                        "document_id", "source_id", "source_revision", "source_locator",
                        "family_id", "domain", "language", "text", "raw_sha256",
                        "normalized_sha256", "license_status", "split",
                    )}
                    data["own_text_tokens"] = n
                    data["provenance_group"] = label
                    rows_by_domain[row["domain"]].append(data)
                    counts[(row["split"], row["domain"])] += 1
                    token_counts[(row["split"], row["domain"])] += n
                    source_counts[label] += 1
        if count != manifest["counts"]["accepted_documents"]:
            raise ValueError("source manifest document count mismatch")
    if source_counts != {"prior_approved_real": manifests[0]["counts"]["accepted_documents"],
                         "new_dual_judge": manifests[1]["counts"]["accepted_documents"]}:
        raise ValueError("input source counts changed")
    temp = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.tmp-", dir=args.output.parent))
    try:
        shards = []
        with (temp / "combined_index.jsonl").open("w", encoding="utf-8") as index:
            for domain in sorted(rows_by_domain):
                target = temp / domain
                target.mkdir()
                rows = rows_by_domain[domain]
                for row in rows:
                    index.write(json.dumps({k: row[k] for k in (
                        "document_id", "family_id", "domain", "split", "normalized_sha256",
                        "own_text_tokens", "provenance_group",
                    )}, sort_keys=True) + "\n")
                for offset in range(0, len(rows), 1000):
                    chunk = rows[offset:offset + 1000]
                    name = f"part-{offset // 1000:05d}.parquet"
                    path = target / name
                    pq.write_table(pa.Table.from_pylist(chunk, schema=SCHEMA), path, compression="zstd")
                    shards.append({"domain": domain, "file": name, "rows": len(chunk),
                                   "bytes": path.stat().st_size, "sha256": sha256(path)})
        total_docs = sum(counts.values())
        total_text_tokens = sum(token_counts.values())
        manifest = {
            "stage": "accepted_real", "version": args.output.name,
            "license_gate": {"training_eligible": True, "statuses": {"approved": total_docs}},
            "quality_gate": (
                f"prior teacher-approved {manifests[0]['counts']['accepted_documents']} plus "
                "Qwen/DeepSeek strict-agreement "
                f"{manifests[1]['counts']['accepted_documents']}"
            ),
            "evaluation_gate": "partial benchmark phrase screen; internal training okay, benchmark-release not cleared",
            "synthetic_data_included": False,
            "inputs": {
                "prior_manifest_sha256": sha256(args.prior / "manifest.json"),
                "wave1_manifest_sha256": sha256(args.wave1 / "manifest.json"),
                "tokenizer_json_sha256": tokenizer_hash,
            },
            "tokenizer": {"vocab_size": 32768, "tokenizer_json_sha256": tokenizer_hash},
            "counts": {"accepted_documents": total_docs,
                       "accepted_text_tokens_excluding_eos": total_text_tokens,
                       "accepted_tokens_with_one_eos_each": total_text_tokens + total_docs,
                       "documents_by_source_group": dict(source_counts)},
            "splits": {s: {"documents": sum(v for (split, _), v in counts.items() if split == s),
                           "text_tokens": sum(v for (split, _), v in token_counts.items() if split == s)}
                       for s in ("train", "validation", "test")},
            "domains": {d: {"documents": sum(v for (s, domain), v in counts.items() if domain == d),
                            "text_tokens": sum(v for (s, domain), v in token_counts.items() if domain == d)}
                        for d in sorted(rows_by_domain)},
            "family_split_conflicts": 0, "shards": shards,
            "combined_index": {"file": "combined_index.jsonl", "rows": total_docs,
                               "sha256": sha256(temp / "combined_index.jsonl")},
        }
        (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        temp.rename(args.output)
        print(json.dumps(manifest, ensure_ascii=False))
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
