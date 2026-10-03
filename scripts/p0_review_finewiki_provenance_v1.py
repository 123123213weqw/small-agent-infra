#!/usr/bin/env python3
"""Audit page-level attribution metadata in the strict FineWiki candidate index."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--sample-manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import pyarrow.parquet as pq
    manifest = json.loads(args.sample_manifest.read_text())
    if digest(args.index) != manifest["provenance_sha256"]:
        raise ValueError("provenance index hash mismatch")
    issues = Counter()
    pages = defaultdict(list)
    count = 0
    token_total = 0
    revisions = set()
    doc_ids = set()
    for batch in pq.ParquetFile(args.index).iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            count += 1
            token_total += row["token_count"]
            revisions.add(row["source_revision"])
            if row["document_id"] in doc_ids:
                issues["duplicate_document_id"] += 1
            doc_ids.add(row["document_id"])
            for field in ("source_id", "page_id", "version", "date_modified", "title", "url"):
                if row[field] in (None, ""):
                    issues[f"missing_{field}"] += 1
            if row["source_id"] != f"zhwiki/{row['page_id']}":
                issues["source_id_page_mismatch"] += 1
            if urlparse(row["url"]).netloc != "zh.wikipedia.org":
                issues["unexpected_url_host"] += 1
            if not str(row["version"]).isdigit():
                issues["non_numeric_revision"] += 1
            try:
                datetime.fromisoformat(row["date_modified"].replace("Z", "+00:00"))
            except ValueError:
                issues["invalid_date_modified"] += 1
            pages[row["page_id"]].append({
                "document_id": row["document_id"], "version": row["version"],
                "date_modified": row["date_modified"], "title": row["title"],
                "url": row["url"], "split": row["split"],
            })
    if count != manifest["finewiki_candidates"] or token_total != manifest["finewiki_tokens"]:
        raise ValueError("provenance totals mismatch")
    multi = {k: v for k, v in pages.items() if len(v) > 1}
    multi_report = {
        "pages_with_multiple_revisions": len(multi),
        "extra_revision_documents": sum(len(v) - 1 for v in multi.values()),
        "split_conflict_pages": sum(len({r["split"] for r in v}) > 1 for v in multi.values()),
        "url_conflict_pages": sum(len({r["url"] for r in v}) > 1 for v in multi.values()),
        "title_conflict_pages": sum(len({r["title"] for r in v}) > 1 for v in multi.values()),
        "exact_duplicate_revision_pages": sum(len({r["version"] for r in v}) < len(v) for v in multi.values()),
    }
    args.output.mkdir(parents=True)
    mult_path = args.output / "multi_revision_pages.jsonl"
    with mult_path.open("x", encoding="utf-8") as f:
        for page_id in sorted(multi):
            f.write(json.dumps({"page_id": page_id, "records": multi[page_id]}, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "stage": "finewiki_strict_candidate_provenance_review_audit_only",
        "index_sha256": digest(args.index), "documents": count,
        "tokens": token_total, "unique_pages": len(pages),
        "source_revisions": sorted(revisions), "field_issues": dict(issues),
        "multiple_revisions": multi_report,
        "multi_revision_pages_sha256": digest(mult_path),
        "attribution": {
            "dataset": "HuggingFaceFW/finewiki",
            "dataset_card": "https://huggingface.co/datasets/HuggingFaceFW/finewiki",
            "wikipedia_contributors_route": "per-record url and page_id",
            "specific_revision_route": "https://zh.wikipedia.org/w/index.php?oldid=<version>",
            "dataset_card_claim": "Wikipedia text CC BY-SA 4.0 / GFDL; processed FineWiki release CC BY-SA 4.0",
            "model_release_obligations": "pending independent review; not inferred from metadata presence",
        },
        "license_review_status": "pending_release_review", "training_eligible": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
