#!/usr/bin/env python3
"""Freeze a matched Chinese quality audit sample and FineWiki attribution index.

CPU-only. Keeps complete text only for sampled documents, never promotes data.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
from urllib.parse import urlparse


SEED = "finewiki-zh-strict-quality-compare-v2-20260926"
FINE_QUOTA = 38  # 8 strata = 304 documents
BASE_QUOTA = 75  # 4 length strata = 300 documents


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def length_bin(chars: int) -> str:
    if chars < 2000:
        return "500-1999"
    if chars < 6000:
        return "2000-5999"
    if chars < 12000:
        return "6000-11999"
    return "12000+"


def rank(group: str, doc_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED}\0{group}\0{doc_id}".encode()).digest(), "big")


def add_sample(heaps: dict, group: str, row: dict, quota: int) -> None:
    score = rank(group, row["document_id"])
    entry = (-score, row["document_id"], row)
    heap = heaps[group]
    if len(heap) < quota:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def jsonl(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fine-root", type=Path, required=True)
    ap.add_argument("--base-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import pyarrow as pa
    import pyarrow.parquet as pq

    fine = args.fine_root
    base = args.base_root
    index_manifest = fine / "audit/unflagged_index_v1/manifest.json"
    residual_manifest = fine / "audit/residual_flags_v1/manifest.json"
    near_manifest = fine / "audit/strict_near_policy_v1/manifest.json"
    clean_manifest = fine / "clean_rule_v1/manifest.json"
    base_manifest = base / "manifest.json"
    manifests = [index_manifest, residual_manifest, near_manifest, clean_manifest, base_manifest]
    parsed = [json.loads(p.read_text()) for p in manifests]
    idx_m, residual_m, near_m, clean_m, base_m = parsed
    if any(m.get("training_eligible") for m in (idx_m, residual_m, near_m, clean_m)):
        raise ValueError("FineWiki audit manifest unexpectedly promotes training")
    if near_m["counts"]["remaining_documents"] != 223252:
        raise ValueError("strict candidate count changed")

    residual = set()
    for item in residual_m["files"]:
        p = fine / "audit/residual_flags_v1" / item["file"]
        if digest(p) != item["sha256"]:
            raise ValueError(f"residual hash mismatch: {p}")
        residual.update(pq.read_table(p, columns=["document_id"]).column(0).to_pylist())
    exclusion_path = fine / "audit/strict_near_policy_v1/exclusion_overlay.jsonl"
    if digest(exclusion_path) != near_m["exclusion_overlay_sha256"]:
        raise ValueError("strict exclusion hash mismatch")
    excluded = {json.loads(s)["document_id"] for s in exclusion_path.read_text().splitlines()}
    if len(excluded) != near_m["counts"]["excluded_documents"]:
        raise ValueError("strict exclusion count mismatch")

    args.output.mkdir(parents=True)
    provenance_path = args.output / "finewiki_provenance.parquet"
    schema = pa.schema([
        ("document_id", pa.string()), ("source_dataset", pa.string()),
        ("source_revision", pa.string()), ("source_id", pa.string()),
        ("page_id", pa.int64()), ("version", pa.string()),
        ("date_modified", pa.string()), ("title", pa.string()),
        ("url", pa.string()), ("split", pa.string()),
        ("clean_file", pa.string()), ("row_in_clean_file", pa.int64()),
        ("token_count", pa.int64()), ("normalized_sha256", pa.string()),
    ])
    writer = pq.ParquetWriter(provenance_path, schema, compression="zstd")
    fine_heaps = defaultdict(list)
    base_heaps = defaultdict(list)
    fine_counts = Counter()
    base_counts = Counter()
    metadata_issues = Counter()
    unique_page = set()
    unique_url_version = set()
    total_tokens = 0
    try:
        for item in idx_m["files"]:
            name = item["file"]
            p = fine / "audit/unflagged_index_v1" / name
            if digest(p) != item["sha256"]:
                raise ValueError(f"index hash mismatch: {p}")
            selected = {}
            for batch in pq.ParquetFile(p).iter_batches(batch_size=2048):
                for row in batch.to_pylist():
                    did = row["document_id"]
                    if (did in residual or did in excluded or row["wave1_url_family_overlap"]):
                        continue
                    selected[row["row_in_clean_file"]] = row
            clean_path = fine / "clean_rule_v1" / name
            clean_item = next(x for x in clean_m["files"] if x["output_file"] == name)
            if digest(clean_path) != clean_item["output_sha256"]:
                raise ValueError(f"clean file hash mismatch: {clean_path}")
            prov_rows = []
            pos = 0
            cols = ["document_id", "id", "source_revision", "page_id", "version",
                    "date_modified", "title", "url", "split", "chars", "has_math",
                    "normalized_sha256", "text"]
            for batch in pq.ParquetFile(clean_path).iter_batches(batch_size=512, columns=cols):
                for row in batch.to_pylist():
                    pointer = selected.get(pos)
                    pos += 1
                    if pointer is None:
                        continue
                    did = row["document_id"]
                    if did != pointer["document_id"] or row["normalized_sha256"] != pointer["normalized_sha256"]:
                        raise ValueError(f"index/clean pointer mismatch: {name}:{pos-1}")
                    if row["chars"] != pointer["chars"] or len(row["text"]) != row["chars"]:
                        raise ValueError("text length mismatch")
                    for field in ("id", "page_id", "version", "date_modified", "title", "url"):
                        if not row[field]:
                            metadata_issues[f"missing_{field}"] += 1
                    if row["id"] != f"zhwiki/{row['page_id']}":
                        metadata_issues["id_page_mismatch"] += 1
                    if urlparse(row["url"]).netloc != "zh.wikipedia.org":
                        metadata_issues["unexpected_url_host"] += 1
                    if row["source_revision"] != clean_m["revision"]:
                        metadata_issues["revision_mismatch"] += 1
                    if row["page_id"] in unique_page:
                        metadata_issues["duplicate_page_id"] += 1
                    unique_page.add(row["page_id"])
                    uv = (row["url"], row["version"])
                    if uv in unique_url_version:
                        metadata_issues["duplicate_url_version"] += 1
                    unique_url_version.add(uv)
                    total_tokens += pointer["token_count"]
                    group = f"{length_bin(row['chars'])}/math={int(row['has_math'])}"
                    fine_counts[group] += 1
                    add_sample(fine_heaps, group, {
                        "document_id": did, "source": "finewiki_zh", "stratum": group,
                        "text": row["text"], "chars": row["chars"],
                        "split": row["split"], "title": row["title"],
                        "has_math": row["has_math"], "token_count": pointer["token_count"],
                        "url": row["url"], "page_id": row["page_id"],
                        "version": row["version"],
                    }, FINE_QUOTA)
                    prov_rows.append({
                        "document_id": did, "source_dataset": "HuggingFaceFW/finewiki",
                        "source_revision": row["source_revision"], "source_id": row["id"],
                        "page_id": row["page_id"], "version": row["version"],
                        "date_modified": row["date_modified"], "title": row["title"],
                        "url": row["url"], "split": row["split"],
                        "clean_file": name, "row_in_clean_file": pos - 1,
                        "token_count": pointer["token_count"],
                        "normalized_sha256": row["normalized_sha256"],
                    })
            if len(prov_rows) != len(selected):
                raise ValueError(f"index pointers not fully resolved: {name}")
            writer.write_table(pa.Table.from_pylist(prov_rows, schema=schema))
    finally:
        writer.close()
    if sum(fine_counts.values()) != near_m["counts"]["remaining_documents"] or total_tokens != near_m["counts"]["remaining_tokens"]:
        raise ValueError("strict candidate total mismatch")

    # The existing frozen Wave1 general_zh is the comparison population.
    for p in sorted((base / "general_zh").glob("*.parquet")):
        for batch in pq.ParquetFile(p).iter_batches(batch_size=512, columns=["document_id", "text", "split", "source_id"]):
            for row in batch.to_pylist():
                n = len(row["text"])
                group = length_bin(n)
                base_counts[group] += 1
                add_sample(base_heaps, group, {
                    "document_id": row["document_id"], "source": "wave1_general_zh",
                    "stratum": group, "text": row["text"], "chars": n,
                    "split": row["split"], "source_id": row["source_id"],
                }, BASE_QUOTA)
    expected_base = next(d["retained_documents"] for d in base_m["domains"] if d["domain"] == "general_zh")
    if sum(base_counts.values()) != expected_base:
        raise ValueError("Wave1 population count mismatch")
    if len(fine_heaps) != 8 or any(len(x) != FINE_QUOTA for x in fine_heaps.values()):
        raise ValueError("FineWiki sample stratum shortfall")
    if len(base_heaps) != 4 or any(len(x) != BASE_QUOTA for x in base_heaps.values()):
        raise ValueError("Wave1 sample stratum shortfall")
    sample = [entry[2] for heap in (fine_heaps, base_heaps) for entries in heap.values() for entry in entries]
    sample.sort(key=lambda r: rank("order", r["document_id"]))
    for row in sample:
        row["population_in_stratum"] = (fine_counts if row["source"] == "finewiki_zh" else base_counts)[row["stratum"]]
        row["sample_size_in_stratum"] = FINE_QUOTA if row["source"] == "finewiki_zh" else BASE_QUOTA
    sample_path = args.output / "sample_full_text.jsonl"
    jsonl(sample_path, sample)
    report = {
        "stage": "finewiki_strict_vs_wave1_zh_quality_sample_audit_only",
        "seed": SEED, "finewiki_population": dict(sorted(fine_counts.items())),
        "wave1_population": dict(sorted(base_counts.items())),
        "finewiki_sample": 8 * FINE_QUOTA, "wave1_sample": 4 * BASE_QUOTA,
        "finewiki_candidates": sum(fine_counts.values()), "finewiki_tokens": total_tokens,
        "provenance_issues": dict(metadata_issues),
        "sample_sha256": digest(sample_path), "provenance_sha256": digest(provenance_path),
        "input_manifest_sha256": {str(p): digest(p) for p in manifests},
        "dataset_card": "https://huggingface.co/datasets/HuggingFaceFW/finewiki",
        "source_license_claim": "CC BY-SA 4.0 / GFDL; processed release CC BY-SA 4.0",
        "license_review_status": "pending_release_review",
        "training_eligible": False,
        "note": "Sample is stratified, not simple random; model ratings are not ground truth.",
    }
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "input_manifest_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
