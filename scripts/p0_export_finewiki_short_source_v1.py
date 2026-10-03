#!/usr/bin/env python3
"""Promote the audited FineWiki <6000-character source for internal training only.

Requires an explicit completed whole-document quality report. The strict
candidate's frozen provenance index is the inclusion list; no sample label is
misrepresented as a per-document approval.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile


ASCII_EMPTY_PARENS = re.compile(r"\(\s*\)")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fine-root", type=Path, required=True)
    p.add_argument("--quality-report", type=Path, required=True)
    p.add_argument("--quality-results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--approve-internal", action="store_true", required=True)
    a = p.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq

    if a.output.exists():
        raise FileExistsError(a.output)
    quality = json.loads(a.quality_report.read_text())
    if quality["stage"] != "finewiki_short_whole_document_quality_report_audit_only":
        raise ValueError("whole-document quality gate is missing")
    if quality["successful_documents"] != 302:
        raise ValueError("quality comparison was not complete")
    if sha(a.quality_results) != quality["results_sha256"]:
        raise ValueError("whole-document results changed after summary")
    known_drops = set()
    with a.quality_results.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["status"] != "ok":
                raise ValueError("quality results contain a failed inference")
            if r["source"]["source"] == "finewiki_zh" and r["annotation"]["decision"] == "drop":
                did = r["annotation"]["document_id"]
                if did != json.loads(r["source_id"])[0]:
                    raise ValueError("Qwen drop ID mismatch")
                known_drops.add(did)
    if len(known_drops) != 5:
        raise ValueError("frozen quality drop count changed")
    qa_root = a.fine_root / "audit/quality_compare_v2"
    qa_manifest_path = qa_root / "manifest.json"
    qa_manifest = json.loads(qa_manifest_path.read_text())
    provenance = qa_root / "finewiki_provenance.parquet"
    if sha(provenance) != qa_manifest["provenance_sha256"]:
        raise ValueError("provenance hash mismatch")
    clean_manifest_path = a.fine_root / "clean_rule_v1/manifest.json"
    clean_manifest = json.loads(clean_manifest_path.read_text())
    if sha(clean_manifest_path) != qa_manifest["input_manifest_sha256"][str(clean_manifest_path)]:
        raise ValueError("clean manifest changed after audit")

    by_file = defaultdict(dict)
    for batch in pq.ParquetFile(provenance).iter_batches(batch_size=2048):
        for r in batch.to_pylist():
            pointer = r["row_in_clean_file"]
            if pointer in by_file[r["clean_file"]]:
                raise ValueError("duplicate provenance pointer")
            by_file[r["clean_file"]][pointer] = r
    if sum(len(x) for x in by_file.values()) != qa_manifest["finewiki_candidates"]:
        raise ValueError("provenance row count mismatch")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="." + a.output.name + ".tmp-", dir=a.output.parent))
    domain_dir = tmp / "finewiki_zh"
    domain_dir.mkdir()
    counts = Counter()
    split_counts = defaultdict(Counter)
    shards = []
    seen_docs = set()
    family_splits = defaultdict(set)
    try:
        for i, item in enumerate(clean_manifest["files"]):
            name = item["output_file"]
            pointers = by_file[name]
            src = a.fine_root / "clean_rule_v1" / name
            if src.stat().st_size != item["output_bytes"] or sha(src) != item["output_sha256"]:
                raise ValueError(f"clean shard integrity failure: {src}")
            selected = []
            pos = 0
            resolved = 0
            for batch in pq.ParquetFile(src).iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    prov = pointers.get(pos)
                    pos += 1
                    if prov is None:
                        continue
                    resolved += 1
                    if any(row[k] != prov[k] for k in (
                        "document_id", "source_revision", "page_id", "version",
                        "date_modified", "title", "url", "split", "normalized_sha256",
                    )):
                        raise ValueError(f"provenance/clean mismatch: {name}:{pos-1}")
                    if row["chars"] != len(row["text"]) or row["chars"] < 500:
                        raise ValueError("invalid text length")
                    if row["chars"] >= 6000:
                        counts["long_excluded_docs"] += 1
                        counts["long_excluded_tokens"] += prov["token_count"]
                        continue
                    if ASCII_EMPTY_PARENS.search(row["text"]):
                        counts["ascii_empty_parens_excluded_docs"] += 1
                        counts["ascii_empty_parens_excluded_tokens"] += prov["token_count"]
                        continue
                    if row["document_id"] in known_drops:
                        counts["known_qwen_drop_excluded_docs"] += 1
                        counts["known_qwen_drop_excluded_tokens"] += prov["token_count"]
                        continue
                    did = row["document_id"]
                    if did in seen_docs:
                        raise ValueError(f"duplicate document: {did}")
                    seen_docs.add(did)
                    row["domain"] = "finewiki_zh"
                    row["license_status"] = "approved_internal_only"
                    row["text_tokens"] = prov["token_count"]
                    selected.append(row)
                    counts["accepted_documents"] += 1
                    counts["accepted_text_tokens_excluding_eos"] += prov["token_count"]
                    split_counts[row["split"]]["documents"] += 1
                    split_counts[row["split"]]["text_tokens"] += prov["token_count"]
                    family_splits[row["family_id"]].add(row["split"])
            if resolved != len(pointers):
                raise ValueError("pointer accounting mismatch")
            if selected:
                out = domain_dir / f"part-{i:05d}.parquet"
                pq.write_table(pa.Table.from_pylist(selected), out, compression="zstd")
                shards.append({
                    "domain": "finewiki_zh", "file": out.name, "rows": len(selected),
                    "bytes": out.stat().st_size, "sha256": sha(out),
                })
        if (counts["accepted_documents"] + counts["long_excluded_docs"]
                + counts["ascii_empty_parens_excluded_docs"]
                + counts["known_qwen_drop_excluded_docs"] != qa_manifest["finewiki_candidates"]):
            raise ValueError("candidate document accounting mismatch")
        if (counts["accepted_text_tokens_excluding_eos"] + counts["long_excluded_tokens"]
                + counts["ascii_empty_parens_excluded_tokens"]
                + counts["known_qwen_drop_excluded_tokens"]
                != qa_manifest["finewiki_tokens"]):
            raise ValueError("candidate token accounting mismatch")
        if (counts["accepted_documents"] + counts["ascii_empty_parens_excluded_docs"]
                + counts["known_qwen_drop_excluded_docs"] != 207013
                or counts["accepted_text_tokens_excluding_eos"] + counts["ascii_empty_parens_excluded_tokens"]
                + counts["known_qwen_drop_excluded_tokens"] != 212868209
                or counts["ascii_empty_parens_excluded_docs"] != 449):
            raise ValueError("frozen short candidate count changed")
        if counts["known_qwen_drop_excluded_docs"] != 5:
            raise ValueError("known Qwen rejects were not all excluded")
        if any(len(s) > 1 for s in family_splits.values()):
            raise ValueError("family crosses data splits")
        manifest = {
            "stage": "accepted_real", "version": "finewiki_zh_short_internal_v2",
            "source_dataset": "HuggingFaceFW/finewiki",
            "source_revision": clean_manifest["revision"],
            "license_gate": {
                "training_eligible": True, "distribution_eligible": False,
                "scope": "internal_only", "source_license": "CC BY-SA 4.0",
                "release_review_required": True,
            },
            "quality_gate": {
                "type": "strict_deterministic_filter_plus_stratified_whole_document_model_audit",
                "source_level_approval_only": True,
                "quality_report_sha256": sha(a.quality_report),
                "quality_results_sha256": sha(a.quality_results),
                "excluded_length_chars_gte": 6000,
                "excluded_ascii_empty_parentheses": True,
                "excluded_known_qwen_drops": len(known_drops),
            },
            "inputs": {
                "provenance_sha256": sha(provenance),
                "clean_manifest_sha256": sha(clean_manifest_path),
                "quality_report_sha256": sha(a.quality_report),
                "quality_results_sha256": sha(a.quality_results),
            },
            "counts": dict(counts), "splits": dict(split_counts),
            "family_split_conflicts": 0, "shards": shards,
            "notes": [
                "All included articles are in a strict candidate pool; the sample audit is not per-document approval.",
                "Internal pretraining diagnostics only. Do not distribute text, packed tokens, or resulting weights without review.",
            ],
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        tmp.rename(a.output)
        print(json.dumps({"output": str(a.output), "counts": manifest["counts"], "splits": manifest["splits"]}, ensure_ascii=False))
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
