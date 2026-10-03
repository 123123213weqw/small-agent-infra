#!/usr/bin/env python3
"""Two-pass approximate near-dedup audit for all FineWiki priority records.

Pass 1 builds only text-free MinHash signatures/pointers and LSH candidate
pairs. Pass 2 re-reads only pair members and computes exact shingle Jaccard.
Nothing is removed from the source corpus or approved for training.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import resource
import struct
import tempfile

import p0_data_pipeline as core
from p0_near_dedup_ultradata_night8 import BANDS, MAX_SHINGLES, NearIndex, NUM_PERM, SHINGLE_SIZE, THRESHOLD, shingles
from p0_screen_finewiki_ceval import load_priority


PAIR = struct.Struct("<II")
MAX_PAIRS = 5_000_000
MAX_CANDIDATES_PER_DOCUMENT = 5_000
ADDRESS_SPACE_LIMIT_GIB = 64


def metadata_schema():
    import pyarrow as pa
    return pa.schema([
        ("document_id", pa.string()), ("source", pa.string()), ("split", pa.string()),
        ("file", pa.string()), ("row", pa.int64()), ("token_count", pa.int64()),
        ("signature", pa.list_(pa.uint32(), NUM_PERM)),
    ])


def add_record(index: NearIndex, buckets: dict, records: list[dict], row: dict,
               text: str, pair_writer, counters: Counter) -> None:
    values = shingles(text)
    signature = index.signature(values)
    new_idx = len(records)
    if new_idx >= 2**32:
        raise ValueError("document index exceeds binary pair format")
    keys = tuple(index.keys(signature))
    if row["source"] == "finewiki":
        possible = set()
        for key in keys:
            possible.update(buckets.get(key, ()))
        if len(possible) > MAX_CANDIDATES_PER_DOCUMENT:
            raise RuntimeError(f"LSH candidate explosion at {row['document_id']}: {len(possible)}")
        for old_idx in sorted(possible):
            pair_writer.write(PAIR.pack(old_idx, new_idx))
        counters["candidate_pairs"] += len(possible)
        if counters["candidate_pairs"] > MAX_PAIRS:
            raise RuntimeError(f"LSH pair cap exceeded: {counters['candidate_pairs']}")
        counters["finewiki_documents"] += 1
        counters[f"finewiki:{row['split']}"] += 1
    else:
        counters["wave1_zh_documents"] += 1
        counters[f"wave1_zh:{row['split']}"] += 1
    row["signature"] = list(signature)
    records.append(row)
    for key in keys:
        buckets[key].append(new_idx)


def pass_one(candidate_root: Path, reference_root: Path, pair_path: Path):
    import pyarrow.parquet as pq

    selected, priority_provenance = load_priority(candidate_root)
    wave_path = reference_root / "manifest.json"
    wave = json.loads(wave_path.read_text())
    if wave["stage"] != "consolidate_candidates":
        raise ValueError("unexpected Wave1 reference stage")
    index = NearIndex()
    buckets = defaultdict(list)
    records = []
    counters = Counter()
    with pair_path.open("wb") as pairs:
        for item in wave["shards"]:
            if item["domain"] != "general_zh":
                continue
            path = reference_root / item["domain"] / item["file"]
            if path.stat().st_size != item["bytes"] or core.sha256_file(path) != item["sha256"]:
                raise ValueError(f"Wave1 shard checksum mismatch: {path}")
            offset = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["document_id", "split", "text"]):
                for source in batch.to_pylist():
                    add_record(index, buckets, records, {
                        "document_id": source["document_id"], "source": "wave1_zh",
                        "split": source["split"], "file": str(Path(item["domain"]) / item["file"]),
                        "row": offset, "token_count": 0,
                    }, source["text"], pairs, counters)
                    offset += 1
            if offset != item["rows"]:
                raise ValueError("Wave1 shard row count mismatch")
            print(f"indexed Wave1 zh {counters['wave1_zh_documents']}", flush=True)
        expected_wave = sum(x["rows"] for x in wave["shards"] if x["domain"] == "general_zh")
        if counters["wave1_zh_documents"] != expected_wave:
            raise ValueError("Wave1 zh count mismatch")
        clean_root = candidate_root / "clean_rule_v1"
        clean_path = clean_root / "manifest.json"
        clean = json.loads(clean_path.read_text())
        if clean["stage"] != "finewiki_zh_rule_clean_audit_only" or clean["training_eligible"]:
            raise ValueError("unexpected FineWiki clean stage")
        for item in clean["files"]:
            name = item["output_file"]
            wanted = selected.pop(name, {})
            path = clean_root / name
            if core.sha256_file(path) != item["output_sha256"]:
                raise ValueError(f"FineWiki clean checksum mismatch: {path}")
            offset = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["document_id", "text"]):
                for source in batch.to_pylist():
                    indexed = wanted.get(offset)
                    if indexed is not None:
                        if indexed["document_id"] != source["document_id"]:
                            raise ValueError("FineWiki source pointer mismatch")
                        add_record(index, buckets, records, {
                            "document_id": source["document_id"], "source": "finewiki",
                            "split": indexed["split"], "file": name, "row": offset,
                            "token_count": indexed["token_count"],
                        }, source["text"], pairs, counters)
                    offset += 1
            if offset != item["counts"]["kept"]:
                raise ValueError("FineWiki clean shard row count mismatch")
            print(f"indexed FineWiki {counters['finewiki_documents']}; pairs {counters['candidate_pairs']}",
                  flush=True)
        if selected or counters["finewiki_documents"] != priority_provenance["selection_counts"]["selected"]:
            raise ValueError("priority selection not fully indexed")
    provenance = {
        **priority_provenance,
        "wave1_manifest_sha256": core.sha256_file(wave_path),
        "clean_manifest_sha256": core.sha256_file(clean_path),
    }
    return records, dict(counters), provenance


def pair_members(pair_path: Path) -> set[int]:
    wanted = set()
    with pair_path.open("rb") as source:
        while block := source.read(PAIR.size * 65536):
            if len(block) % PAIR.size:
                raise ValueError("truncated binary candidate pairs")
            for old_idx, new_idx in PAIR.iter_unpack(block):
                wanted.add(old_idx)
                wanted.add(new_idx)
    return wanted


def load_pair_shingles(candidate_root: Path, reference_root: Path,
                       records: list[dict], wanted: set[int]):
    import pyarrow.parquet as pq

    by_file = defaultdict(dict)
    for idx in wanted:
        row = records[idx]
        key = row["source"], row["file"]
        if row["row"] in by_file[key]:
            raise ValueError("duplicate source pointer among pair members")
        by_file[key][row["row"]] = idx
    values = {}
    for (source_kind, name), positions in sorted(by_file.items()):
        path = (candidate_root / "clean_rule_v1" / name if source_kind == "finewiki"
                else reference_root / name)
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["document_id", "text"]):
            for row in batch.to_pylist():
                idx = positions.get(offset)
                if idx is not None:
                    if row["document_id"] != records[idx]["document_id"]:
                        raise ValueError("pair source pointer mismatch")
                    values[idx] = shingles(row["text"])
                offset += 1
    if len(values) != len(wanted):
        raise ValueError("missing pair-member source text")
    return values


def verify_pairs(pair_path: Path, records: list[dict], values: dict[int, set[int]],
                 hits_path: Path) -> dict:
    counts = Counter()
    with pair_path.open("rb") as source, hits_path.open("w") as output:
        while block := source.read(PAIR.size * 65536):
            if len(block) % PAIR.size:
                raise ValueError("truncated binary candidate pairs")
            for old_idx, new_idx in PAIR.iter_unpack(block):
                old, new = records[old_idx], records[new_idx]
                left, right = values[old_idx], values[new_idx]
                union = len(left | right)
                similarity = len(left & right) / union if union else 1.0
                group = ("wave1_to_finewiki" if old["source"] == "wave1_zh"
                         else "finewiki_internal")
                counts[f"retrieved_pairs:{group}"] += 1
                if similarity >= THRESHOLD:
                    counts[f"near_hits:{group}"] += 1
                    if old["split"] != new["split"]:
                        counts[f"cross_split_near_hits:{group}"] += 1
                    output.write(core.canonical_json({
                        "left_document_id": old["document_id"], "left_source": old["source"],
                        "left_split": old["split"], "right_document_id": new["document_id"],
                        "right_source": new["source"], "right_split": new["split"],
                        "right_token_count": new["token_count"],
                        "jaccard": round(similarity, 6), "group": group,
                    }) + "\n")
    return dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-root", type=Path, required=True)
    ap.add_argument("--reference-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    # Keep this CPU audit from exhausting a shared server if anomalous buckets
    # or unusually large documents appear.
    limit = ADDRESS_SPACE_LIMIT_GIB * 1024**3
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (min(limit, soft) if soft > 0 else limit, hard))
    import pyarrow as pa
    import pyarrow.parquet as pq

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finewiki-full-near-", dir=args.output_dir.parent) as tmp:
        stage = Path(tmp) / args.output_dir.name
        stage.mkdir()
        pair_path = stage / "candidate_pairs.bin"
        records, counts, provenance = pass_one(args.candidate_root, args.reference_root, pair_path)
        if pair_path.stat().st_size != PAIR.size * counts.get("candidate_pairs", 0):
            raise ValueError("pair count/byte count mismatch")
        metadata_path = stage / "signature_index.parquet"
        pq.write_table(pa.Table.from_pylist(records, schema=metadata_schema()), metadata_path,
                       compression="zstd", compression_level=3)
        for row in records:
            del row["signature"]
        wanted = pair_members(pair_path)
        print(f"exact checking {counts.get('candidate_pairs', 0)} candidate pairs; "
              f"{len(wanted)} member documents", flush=True)
        values = load_pair_shingles(args.candidate_root, args.reference_root, records, wanted) if wanted else {}
        hits_path = stage / "near_hits.jsonl"
        exact = verify_pairs(pair_path, records, values, hits_path)
        pair_path.unlink()
        report = {
            "stage": "finewiki_priority_full_near_dedup_audit_only",
            **provenance, "counts": counts, "exact_check": exact,
            "signature_index_sha256": core.sha256_file(metadata_path),
            "signature_index_bytes": metadata_path.stat().st_size,
            "near_hits_sha256": core.sha256_file(hits_path),
            "near_hits_bytes": hits_path.stat().st_size,
            "pair_member_documents": len(wanted),
            "algorithm": {"shingle_size": SHINGLE_SIZE, "max_shingles_per_document": MAX_SHINGLES,
                          "minhash_permutations": NUM_PERM, "lsh_bands": BANDS,
                          "jaccard_threshold": THRESHOLD,
                          "approximate_retrieval_not_exhaustive": True,
                          "pair_limit": MAX_PAIRS,
                          "address_space_limit_gib": ADDRESS_SPACE_LIMIT_GIB},
            "limitations": ["LSH can miss near pairs", "Wave1 comparison only covers general_zh",
                            "other external sources and benchmarks not screened"],
            "training_eligible": False,
        }
        (stage / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        stage.rename(args.output_dir)
    print(json.dumps({"complete": True, "counts": counts, "exact_check": exact}, ensure_ascii=False),
          flush=True)


if __name__ == "__main__":
    main()
