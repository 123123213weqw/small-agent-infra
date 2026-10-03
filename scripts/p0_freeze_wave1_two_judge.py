#!/usr/bin/env python3
"""Freeze license-approved, dual-judge Wave1 text with family-safe splits."""

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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def split_for_family(family_id: str) -> str:
    n = int.from_bytes(hashlib.sha256(family_id.encode()).digest()[:8], "big") % 10000
    return "train" if n < 9800 else "validation" if n < 9900 else "test"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--screen-dir", type=Path, required=True)
    p.add_argument("--gate-dir", type=Path, required=True)
    p.add_argument("--prior-index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pool_manifest = json.loads((args.pool / "manifest.json").read_text())
    screen_summary = json.loads((args.screen_dir / "screen_summary.json").read_text())
    gate_summary = json.loads((args.gate_dir / "gate_summary.json").read_text())
    if pool_manifest.get("stage") != "consolidate_candidates" or not pool_manifest["license_gate"]["training_eligible"]:
        raise ValueError("unapproved source pool")
    if sha256(args.pool / "manifest.json") != screen_summary["wave1_manifest_sha256"]:
        raise ValueError("source/screen manifest mismatch")
    provisional_path = args.screen_dir / "provisional_index.jsonl"
    if sha256(provisional_path) != screen_summary["index_sha256"]:
        raise ValueError("screen index checksum mismatch")
    input_path = args.screen_dir / "deepseek_full_v1/deepseek_full.input.jsonl"
    if sha256(input_path) != gate_summary["input_sha256"]:
        raise ValueError("second-judge input checksum mismatch")
    accepted_path = args.gate_dir / "two_judge_accepted_ids.jsonl"
    if sha256(accepted_path) != gate_summary["accepted_ids_sha256"]:
        raise ValueError("second-judge acceptance checksum mismatch")
    provisional = {x["document_id"]: x for x in jsonl(provisional_path)}
    accepted = {x["document_id"]: x for x in jsonl(accepted_path)}
    if len(provisional) != screen_summary["provisional_documents"] or len(accepted) != gate_summary["accepted_documents"]:
        raise ValueError("screen/gate count mismatch")
    if not set(accepted).issubset(provisional):
        raise ValueError("accepted document outside provisional pool")
    prior_rows = list(jsonl(args.prior_index))
    if not prior_rows or len({x["document_id"] for x in prior_rows}) != len(prior_rows):
        raise ValueError("prior approved index is empty or has duplicate IDs")
    prior_hashes = {x["normalized_sha256"] for x in prior_rows}
    prior_families = {x["family_id"] for x in prior_rows}

    rows_by_domain: dict[str, list[dict]] = defaultdict(list)
    seen_hashes = set(prior_hashes)
    seen_ids = set()
    retained_ids = set()
    reasons = Counter()
    docs = Counter()
    tokens = Counter()
    family_splits = {}
    source_schema = None
    scanned = 0
    for shard in pool_manifest["shards"]:
        path = args.pool / shard["domain"] / shard["file"]
        if path.stat().st_size != shard["bytes"] or sha256(path) != shard["sha256"]:
            raise ValueError(f"source shard mismatch: {path}")
        pf = pq.ParquetFile(path)
        if source_schema is None:
            source_schema = pf.schema_arrow
        elif pf.schema_arrow != source_schema:
            raise ValueError("inconsistent source schemas")
        for batch in pf.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                scanned += 1
                doc_id = row["document_id"]
                if doc_id not in accepted:
                    continue
                if doc_id in seen_ids:
                    raise ValueError("duplicate accepted source ID")
                seen_ids.add(doc_id)
                meta = provisional[doc_id]
                if row["domain"] != meta["domain"] or row["domain"] != accepted[doc_id]["domain"]:
                    raise ValueError("accepted domain mismatch")
                if row["normalized_sha256"] != meta["normalized_sha256"] or row["family_id"] != meta["family_id"]:
                    raise ValueError("accepted source metadata mismatch")
                if row["split"] != "train" or row["license_status"] != "approved":
                    raise ValueError("accepted source split/license failure")
                if row["family_id"] in prior_families:
                    reasons["overlap_with_prior_approved_family"] += 1
                    continue
                if row["normalized_sha256"] in seen_hashes:
                    reasons["exact_duplicate_vs_prior_or_wave1"] += 1
                    continue
                seen_hashes.add(row["normalized_sha256"])
                split = split_for_family(row["family_id"])
                previous_split = family_splits.setdefault(row["family_id"], split)
                if previous_split != split:
                    raise ValueError("family split conflict")
                row["split"] = split
                row["own_text_tokens"] = meta["own_text_tokens"]
                row["qwen_quality"] = meta["qwen_quality"]
                row["qwen_confidence"] = meta["qwen_confidence"]
                row["two_judge_gate"] = "strict_same_domain_keep_both"
                rows_by_domain[row["domain"]].append(row)
                retained_ids.add(doc_id)
                docs[(split, row["domain"])] += 1
                tokens[(split, row["domain"])] += meta["own_text_tokens"]
    if scanned != pool_manifest["counts"]["retained_documents"] or seen_ids != set(accepted):
        raise ValueError("source scan did not account for all accepted IDs")
    extra_schema = pa.schema([
        ("own_text_tokens", pa.int64()), ("qwen_quality", pa.int64()),
        ("qwen_confidence", pa.float64()), ("two_judge_gate", pa.string()),
    ])
    schema = pa.schema(list(source_schema) + list(extra_schema))
    temp = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.tmp-", dir=args.output.parent))
    try:
        shards = []
        for domain in sorted(rows_by_domain):
            items = rows_by_domain[domain]
            target = temp / domain
            target.mkdir()
            for offset in range(0, len(items), 1000):
                chunk = items[offset:offset + 1000]
                name = f"part-{offset // 1000:05d}.parquet"
                path = target / name
                pq.write_table(pa.Table.from_pylist(chunk, schema=schema), path, compression="zstd")
                shards.append({"domain": domain, "file": name, "rows": len(chunk),
                               "bytes": path.stat().st_size, "sha256": sha256(path)})
        with (temp / "accepted_index.jsonl").open("w", encoding="utf-8") as f:
            for domain in sorted(rows_by_domain):
                for row in rows_by_domain[domain]:
                    f.write(json.dumps({
                        "document_id": row["document_id"], "family_id": row["family_id"],
                        "domain": domain, "split": row["split"],
                        "normalized_sha256": row["normalized_sha256"],
                        "own_text_tokens": row["own_text_tokens"],
                    }, sort_keys=True) + "\n")
        manifest = {
            "stage": "accepted_real",
            "version": args.output.name,
            "license_gate": {"training_eligible": True, "statuses": {"approved": len(retained_ids)}},
            "quality_gate": "Qwen and DeepSeek both strict same-domain keep; disagreement/errors dropped",
            "evaluation_gate": "partial exact-phrase ARC-Easy/HellaSwag screen only; not benchmark-release cleared",
            "source_pool": str(args.pool),
            "inputs": {
                "pool_manifest_sha256": sha256(args.pool / "manifest.json"),
                "screen_summary_sha256": sha256(args.screen_dir / "screen_summary.json"),
                "gate_summary_sha256": sha256(args.gate_dir / "gate_summary.json"),
                "prior_approved_index_sha256": sha256(args.prior_index),
            },
            "tokenizer": {"vocab_size": 32768, "tokenizer_json_sha256": screen_summary["tokenizer_json_sha256"]},
            "counts": {
                "candidate_documents": len(accepted),
                "accepted_documents": len(retained_ids),
                "accepted_text_tokens_excluding_eos": sum(tokens.values()),
                "accepted_tokens_with_one_eos_each": sum(tokens.values()) + len(retained_ids),
                "removed_after_two_judge_gate": dict(sorted(reasons.items())),
            },
            "splits": {s: {"documents": sum(v for (split, _), v in docs.items() if split == s),
                           "text_tokens": sum(v for (split, _), v in tokens.items() if split == s)}
                       for s in ("train", "validation", "test")},
            "domains": {d: {"documents": sum(v for (s, domain), v in docs.items() if domain == d),
                            "text_tokens": sum(v for (s, domain), v in tokens.items() if domain == d)}
                        for d in sorted(rows_by_domain)},
            "family_split_conflicts": 0,
            "shards": shards,
            "accepted_index": {"file": "accepted_index.jsonl", "sha256": sha256(temp / "accepted_index.jsonl"),
                               "rows": len(retained_ids)},
        }
        (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.rename(args.output)
        print(json.dumps(manifest, ensure_ascii=False))
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
