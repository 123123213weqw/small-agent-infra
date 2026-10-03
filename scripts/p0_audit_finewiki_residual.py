#!/usr/bin/env python3
"""Read-only scan for obvious residual extraction artifacts after first flag."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import p0_data_pipeline as core
from p0_audit_finewiki_extraction import EMPTY_LANGUAGE
from p0_sample_finewiki_quality import stratum


EMPTY_PARENS = re.compile(r"（\s*）")
# Restricted to language-label syntax, not arbitrary field separators.
OTHER_EMPTY_LANGUAGE = re.compile(r"(?:[\u3400-\u9fff]{0,12}(?:語|语|文))[：:]\s*[；;，,、）)]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import pyarrow.parquet as pq
    clean_root = args.root / "clean_rule_v1"
    manifest_path = clean_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["stage"] != "finewiki_zh_rule_clean_audit_only":
        raise ValueError("unexpected clean manifest")
    counts = Counter()
    examples = []
    for item in manifest["files"]:
        path = clean_root / item["output_file"]
        if core.sha256_file(path) != item["output_sha256"]:
            raise ValueError(f"clean shard checksum mismatch: {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=[
            "document_id", "title", "text", "chars", "has_math",
        ]):
            for row in batch.to_pylist():
                counts["documents"] += 1
                if EMPTY_LANGUAGE.search(row["text"]):
                    continue
                counts["previously_unflagged"] += 1
                parens = EMPTY_PARENS.search(row["text"])
                language = OTHER_EMPTY_LANGUAGE.search(row["text"])
                key = stratum(row["chars"], row["has_math"])
                if parens:
                    counts["residual_empty_parentheses"] += 1
                    counts[f"residual_empty_parentheses:{key}"] += 1
                if language:
                    counts["residual_other_empty_language"] += 1
                    counts[f"residual_other_empty_language:{key}"] += 1
                if parens or language:
                    counts["residual_union"] += 1
                    counts[f"residual_union:{key}"] += 1
                    if len(examples) < 20:
                        match = parens or language
                        examples.append({
                            "document_id": row["document_id"], "title": row["title"],
                            "kind": "empty_parentheses" if parens else "other_empty_language",
                            "excerpt": row["text"][max(0, match.start()-45):match.end()+45].replace("\n", " "),
                        })
        print(f"audited {path.name}", flush=True)
    if counts["documents"] != manifest["counts"]["kept"]:
        raise ValueError("row count mismatch")
    prior = json.loads((args.root / "audit/empty_language_annotations_v1.json").read_text())
    if counts["previously_unflagged"] != counts["documents"] - prior["counts"]["flagged_documents"]:
        raise ValueError("prior extraction flag count mismatch")
    report = {
        "stage": "finewiki_zh_residual_extraction_audit_only",
        "clean_manifest_sha256": core.sha256_file(manifest_path),
        "prior_audit_sha256": core.sha256_file(args.root / "audit/empty_language_annotations_v1.json"),
        "patterns": {"empty_parentheses": EMPTY_PARENS.pattern,
                     "other_empty_language": OTHER_EMPTY_LANGUAGE.pattern},
        "counts": dict(counts), "examples": examples,
        "training_eligible": False,
        "note": "Flags indicate visible defects, not proof of overall document unusability.",
    }
    core.atomic_write_text(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"complete": True, "counts": dict(counts)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
