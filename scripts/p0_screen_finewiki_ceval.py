#!/usr/bin/env python3
"""Exact normalized C-Eval question screen over FineWiki priority candidates.

The C-Eval validation set is an audit reference, not training data. GNU grep
receives corpus text through a pipe and emits only line numbers and matched
question strings; no copy of candidate text is written to disk.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unicodedata

import p0_data_pipeline as core


MIN_QUESTION_CHARS = 20


def normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in folded if char.isalnum())


def load_questions(reference_root: Path) -> tuple[dict[str, list[str]], dict]:
    import pyarrow.parquet as pq

    manifest_path = reference_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["stage"] != "ceval_validation_frozen_for_decontamination_audit_only" or manifest["training_eligible"]:
        raise ValueError("unexpected C-Eval reference stage")
    phrases = defaultdict(list)
    counts = Counter()
    for item in manifest["files"]:
        path = reference_root / item["file"]
        if path.stat().st_size != item["bytes"] or core.sha256_file(path) != item["sha256"]:
            raise ValueError(f"C-Eval reference checksum mismatch: {path}")
        subject = path.parent.name
        table = pq.read_table(path, columns=["id", "question"])
        for row in table.to_pylist():
            counts["questions_total"] += 1
            phrase = normalize(row["question"])
            if len(phrase) < MIN_QUESTION_CHARS:
                counts["questions_short_skipped"] += 1
                continue
            phrases[phrase].append(f"{subject}:{row['id']}")
            counts["questions_screened"] += 1
    counts["unique_phrases"] = len(phrases)
    return dict(phrases), {"reference_manifest_sha256": core.sha256_file(manifest_path),
                           "reference_counts": dict(counts)}


def load_priority(root: Path) -> tuple[dict[str, dict[int, dict]], dict]:
    import pyarrow.parquet as pq

    index_root = root / "audit/unflagged_index_v1"
    overlay_root = root / "audit/residual_flags_v1"
    index_path = index_root / "manifest.json"
    overlay_path = overlay_root / "manifest.json"
    index = json.loads(index_path.read_text())
    overlay = json.loads(overlay_path.read_text())
    if (index["stage"] != "finewiki_zh_unflagged_index_audit_only" or
            overlay["stage"] != "finewiki_zh_residual_flag_overlay_audit_only" or
            index["training_eligible"] or overlay["training_eligible"] or
            overlay["index_manifest_sha256"] != core.sha256_file(index_path)):
        raise ValueError("priority audit index provenance mismatch")
    excluded = set()
    for item in overlay["files"]:
        path = overlay_root / item["file"]
        if core.sha256_file(path) != item["sha256"]:
            raise ValueError(f"residual flags checksum mismatch: {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["document_id"]):
            excluded.update(batch.column(0).to_pylist())
    if len(excluded) != overlay["counts"]["residual_documents"]:
        raise ValueError("residual flags count mismatch")
    selected = defaultdict(dict)
    counts = Counter()
    for item in index["files"]:
        path = index_root / item["file"]
        if core.sha256_file(path) != item["sha256"]:
            raise ValueError(f"unflagged index checksum mismatch: {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=[
            "document_id", "split", "token_count", "clean_file", "row_in_clean_file",
            "wave1_url_family_overlap",
        ]):
            for row in batch.to_pylist():
                counts["index_rows"] += 1
                if row["document_id"] in excluded:
                    counts["residual_excluded"] += 1
                    continue
                if row["wave1_url_family_overlap"]:
                    counts["wave1_url_excluded"] += 1
                    continue
                pos = row["row_in_clean_file"]
                target = selected[row["clean_file"]]
                if pos in target:
                    raise ValueError("duplicate selected source row")
                target[pos] = row
                counts["selected"] += 1
                counts["selected_tokens"] += row["token_count"]
    if (counts["index_rows"] != index["counts"]["documents:unflagged"] or
            counts["residual_excluded"] != overlay["counts"]["residual_documents"] or
            counts["selected"] != overlay["counts"]["priority_documents"] -
            overlay["counts"]["priority_wave1_overlap"]):
        raise ValueError("priority count mismatch")
    return dict(selected), {"index_manifest_sha256": core.sha256_file(index_path),
                            "overlay_manifest_sha256": core.sha256_file(overlay_path),
                            "selection_counts": dict(counts)}


def screen(root: Path, selected: dict[str, dict[int, dict]], pattern_path: Path,
           raw_hits_path: Path) -> tuple[list[dict], dict]:
    import pyarrow.parquet as pq

    clean_root = root / "clean_rule_v1"
    clean_path = clean_root / "manifest.json"
    clean = json.loads(clean_path.read_text())
    if clean["stage"] != "finewiki_zh_rule_clean_audit_only" or clean["training_eligible"]:
        raise ValueError("unexpected FineWiki clean stage")
    line_map = []
    counts = Counter()
    env = dict(os.environ, LC_ALL="C")
    with raw_hits_path.open("wb") as out:
        process = subprocess.Popen(["grep", "-aFon", "-f", str(pattern_path)],
                                   stdin=subprocess.PIPE, stdout=out, env=env)
        assert process.stdin is not None
        try:
            for item in clean["files"]:
                name = item["output_file"]
                targets = selected.pop(name, {})
                path = clean_root / name
                if core.sha256_file(path) != item["output_sha256"]:
                    raise ValueError(f"clean corpus checksum mismatch: {path}")
                offset = 0
                for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["document_id", "text"]):
                    for row in batch.to_pylist():
                        indexed = targets.get(offset)
                        if indexed is not None:
                            if indexed["document_id"] != row["document_id"]:
                                raise ValueError("index pointer does not match clean corpus")
                            line_map.append(indexed)
                            process.stdin.write((normalize(row["text"]) + "\n").encode("utf-8"))
                            counts["documents_scanned"] += 1
                            counts[f"split:{indexed['split']}"] += 1
                            counts["tokens_scanned"] += indexed["token_count"]
                        offset += 1
                if offset != item["counts"]["kept"]:
                    raise ValueError("clean corpus row count mismatch")
                print(f"screened {name}: {counts['documents_scanned']} documents", flush=True)
            if selected:
                raise ValueError("selected index points outside clean corpus")
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            process.stdin.close()
        if process.wait() not in (0, 1):
            raise RuntimeError("GNU grep screen failed")
    return line_map, {"clean_manifest_sha256": core.sha256_file(clean_path),
                      "screen_counts": dict(counts)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-root", type=Path, required=True)
    ap.add_argument("--reference-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    questions, ref_provenance = load_questions(args.reference_root)
    selected, index_provenance = load_priority(args.candidate_root)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finewiki-ceval-screen-", dir=args.output_dir.parent) as tmp:
        stage = Path(tmp) / args.output_dir.name
        stage.mkdir()
        patterns = stage / "patterns.txt"
        patterns.write_text("\n".join(sorted(questions)) + "\n")
        raw_hits = stage / "grep_hits.tsv"
        line_map, screen_provenance = screen(args.candidate_root, selected, patterns, raw_hits)
        if (screen_provenance["screen_counts"]["documents_scanned"] !=
                index_provenance["selection_counts"]["selected"]):
            raise ValueError("screen did not visit all selected documents")
        hits = []
        seen = set()
        with raw_hits.open("r", encoding="utf-8") as source:
            for line in source:
                line_number, phrase = line.rstrip("\n").split(":", 1)
                row = line_map[int(line_number) - 1]
                for item_id in questions[phrase]:
                    key = row["document_id"], item_id
                    if key in seen:
                        continue
                    seen.add(key)
                    hits.append({"document_id": row["document_id"], "split": row["split"],
                                 "token_count": row["token_count"], "ceval_item_id": item_id})
        output_hits = stage / "hits.jsonl"
        with output_hits.open("w") as out:
            for hit in hits:
                out.write(core.canonical_json(hit) + "\n")
        report = {
            "stage": "finewiki_priority_ceval_validation_exact_question_screen_only",
            **ref_provenance, **index_provenance, **screen_provenance,
            "minimum_normalized_question_chars": MIN_QUESTION_CHARS,
            "normalization": "NFKC casefold; retain Unicode alphanumeric characters only",
            "matched_documents": len({x["document_id"] for x in hits}),
            "matched_eval_items": len({x["ceval_item_id"] for x in hits}),
            "hits": len(hits),
            "hits_sha256": core.sha256_file(output_hits),
            "limitations": ["exact normalized question stems only", "short stems skipped",
                            "paraphrases, partial questions, answer-only contamination not detected",
                            "C-Eval validation only; other benchmarks not checked"],
            "training_eligible": False,
        }
        # Raw grep matches contain only C-Eval question strings, not corpus text;
        # remove this intermediate before publishing the audit directory.
        patterns.unlink()
        raw_hits.unlink()
        (stage / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        stage.rename(args.output_dir)
    print(json.dumps({"complete": True, "documents_scanned": len(line_map),
                      "matched_documents": report["matched_documents"], "hits": report["hits"]}),
          flush=True)


if __name__ == "__main__":
    main()
