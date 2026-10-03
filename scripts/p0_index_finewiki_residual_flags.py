#!/usr/bin/env python3
"""Write a small overlay of residual extraction flags atop the unflagged index.

This avoids copying source text or another full candidate index. A stricter
priority audit subset is the unflagged index minus this overlay, not approved
training data.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

import p0_data_pipeline as core
from p0_audit_finewiki_extraction import EMPTY_LANGUAGE
from p0_audit_finewiki_residual import EMPTY_PARENS, OTHER_EMPTY_LANGUAGE


def output_schema():
    import pyarrow as pa
    return pa.schema([
        ("document_id", pa.string()), ("clean_file", pa.string()),
        ("row_in_clean_file", pa.int64()), ("token_count", pa.int64()),
        ("empty_parentheses", pa.bool_()), ("other_empty_language", pa.bool_()),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    import pyarrow as pa
    import pyarrow.parquet as pq
    clean_root = args.root / "clean_rule_v1"
    clean_manifest_path = clean_root / "manifest.json"
    clean_manifest = json.loads(clean_manifest_path.read_text())
    index_root = args.root / "audit/unflagged_index_v1"
    index_manifest_path = index_root / "manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text())
    residual_report_path = args.root / "audit/residual_extraction_v1.json"
    residual_report = json.loads(residual_report_path.read_text())
    if (clean_manifest["stage"] != "finewiki_zh_rule_clean_audit_only" or
            index_manifest["stage"] != "finewiki_zh_unflagged_index_audit_only" or
            index_manifest["clean_manifest_sha256"] != core.sha256_file(clean_manifest_path) or
            residual_report["clean_manifest_sha256"] != core.sha256_file(clean_manifest_path) or
            residual_report["patterns"]["empty_parentheses"] != EMPTY_PARENS.pattern or
            residual_report["patterns"]["other_empty_language"] != OTHER_EMPTY_LANGUAGE.pattern):
        raise ValueError("input manifest mismatch")
    index_files = {item["file"]: item for item in index_manifest["files"]}
    counts = Counter()
    files = []
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finewiki-residual-flags-", dir=args.output_dir.parent) as tmp:
        stage = Path(tmp) / args.output_dir.name
        stage.mkdir()
        for clean_item in clean_manifest["files"]:
            name = clean_item["output_file"]
            clean_path = clean_root / name
            index_path = index_root / name
            index_item = index_files[name]
            if (core.sha256_file(clean_path) != clean_item["output_sha256"] or
                    core.sha256_file(index_path) != index_item["sha256"]):
                raise ValueError(f"source checksum mismatch: {name}")
            selected = {}
            for batch in pq.ParquetFile(index_path).iter_batches(batch_size=4096, columns=[
                "document_id", "row_in_clean_file", "token_count", "wave1_url_family_overlap",
            ]):
                for row in batch.to_pylist():
                    key = row["row_in_clean_file"]
                    if key in selected:
                        raise ValueError("duplicate index row position")
                    selected[key] = row
            if len(selected) != index_item["selected"]:
                raise ValueError("index selected count mismatch")
            position = 0
            out_rows = []
            for batch in pq.ParquetFile(clean_path).iter_batches(batch_size=512, columns=["document_id", "text"]):
                for row in batch.to_pylist():
                    indexed = selected.get(position)
                    if indexed:
                        if indexed["document_id"] != row["document_id"]:
                            raise ValueError("index pointer does not match source document")
                        if EMPTY_LANGUAGE.search(row["text"]):
                            raise ValueError("unflagged index points to flagged source")
                        parens = bool(EMPTY_PARENS.search(row["text"]))
                        language = bool(OTHER_EMPTY_LANGUAGE.search(row["text"]))
                        if parens or language:
                            out_rows.append({
                                "document_id": indexed["document_id"], "clean_file": name,
                                "row_in_clean_file": position, "token_count": indexed["token_count"],
                                "empty_parentheses": parens, "other_empty_language": language,
                            })
                            counts["residual_documents"] += 1
                            counts["residual_tokens"] += indexed["token_count"]
                            counts["residual_empty_parentheses"] += parens
                            counts["residual_other_empty_language"] += language
                            if indexed["wave1_url_family_overlap"]:
                                counts["residual_wave1_overlap"] += 1
                        counts["checked_unflagged"] += 1
                    position += 1
            if position != clean_item["counts"]["kept"]:
                raise ValueError("clean row count mismatch")
            output = stage / name
            pq.write_table(pa.Table.from_pylist(out_rows, schema=output_schema()), output,
                           compression="zstd", compression_level=3)
            files.append({"file": name, "rows": len(out_rows), "bytes": output.stat().st_size,
                          "sha256": core.sha256_file(output)})
            print(f"indexed residual flags {name}: {len(out_rows)}", flush=True)
        if (counts["checked_unflagged"] != index_manifest["counts"]["documents:unflagged"] or
                counts["residual_documents"] != residual_report["counts"]["residual_union"] or
                counts["residual_empty_parentheses"] != residual_report["counts"]["residual_empty_parentheses"] or
                counts["residual_other_empty_language"] != residual_report["counts"]["residual_other_empty_language"]):
            raise ValueError("residual flag aggregate mismatch")
        counts["priority_documents"] = counts["checked_unflagged"] - counts["residual_documents"]
        counts["priority_tokens"] = (index_manifest["counts"]["tokens:unflagged"] -
                                     counts["residual_tokens"])
        counts["priority_wave1_overlap"] = (index_manifest["counts"].get("unflagged_wave1_url_family_overlap", 0) -
                                            counts["residual_wave1_overlap"])
        report = {
            "stage": "finewiki_zh_residual_flag_overlay_audit_only",
            "index_manifest_sha256": core.sha256_file(index_manifest_path),
            "residual_report_sha256": core.sha256_file(residual_report_path),
            "selection": "index_v1 minus residual flags; URL overlap remains tagged",
            "counts": dict(counts), "files": files,
            "training_eligible": False,
            "limitations": ["other extraction failures may remain", "quality and license gates pending",
                            "not a near-duplicate or benchmark decontamination pass"],
        }
        (stage / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        stage.rename(args.output_dir)
    print(json.dumps({"complete": True, "counts": dict(counts)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
