#!/usr/bin/env python3
"""Screen short C-Eval stems, requiring two options near the stem in a document."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile

import p0_data_pipeline as core
from p0_probe_finewiki_near_overlap import load_sample_text
from p0_screen_finewiki_ceval import load_priority, normalize, screen


MIN_STEM_CHARS = 12
MAX_STEM_CHARS = 19
MIN_OPTION_CHARS = 5
WINDOW_CHARS = 700


def eligible_question(question: str, options: list[str]) -> bool:
    return (MIN_STEM_CHARS <= len(question) <= MAX_STEM_CHARS and
            sum(len(option) >= MIN_OPTION_CHARS for option in options) >= 2)


def matches_two_options(text: str, stem: str, options: list[str]) -> bool:
    start = 0
    while True:
        pos = text.find(stem, start)
        if pos < 0:
            return False
        window = text[pos + len(stem):pos + len(stem) + WINDOW_CHARS]
        if sum(option in window for option in set(options) if len(option) >= MIN_OPTION_CHARS) >= 2:
            return True
        start = pos + 1


def load_short_items(reference_root: Path) -> tuple[dict[str, list[dict]], dict]:
    import pyarrow.parquet as pq

    manifest_path = reference_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["stage"] != "ceval_validation_frozen_for_decontamination_audit_only" or manifest["training_eligible"]:
        raise ValueError("unexpected C-Eval reference stage")
    items = defaultdict(list)
    counts = Counter()
    for file in manifest["files"]:
        path = reference_root / file["file"]
        if path.stat().st_size != file["bytes"] or core.sha256_file(path) != file["sha256"]:
            raise ValueError(f"C-Eval checksum mismatch: {path}")
        subject = path.parent.name
        for row in pq.read_table(path, columns=["id", "question", "A", "B", "C", "D"]).to_pylist():
            counts["questions_total"] += 1
            stem = normalize(row["question"])
            if not MIN_STEM_CHARS <= len(stem) <= MAX_STEM_CHARS:
                continue
            counts["short_stem_range"] += 1
            options = [normalize(row[k] or "") for k in "ABCD"]
            if not eligible_question(stem, options):
                counts["short_insufficient_options"] += 1
                continue
            items[stem].append({"ceval_item_id": f"{subject}:{row['id']}", "options": options})
            counts["short_items_screened"] += 1
    counts["unique_short_stems"] = len(items)
    return dict(items), {"reference_manifest_sha256": core.sha256_file(manifest_path),
                         "reference_counts": dict(counts)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-root", type=Path, required=True)
    ap.add_argument("--reference-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    items, ref = load_short_items(args.reference_root)
    selected, index = load_priority(args.candidate_root)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finewiki-ceval-short-", dir=args.output_dir.parent) as tmp:
        stage = Path(tmp) / args.output_dir.name
        stage.mkdir()
        patterns = stage / "patterns.txt"
        patterns.write_text("\n".join(sorted(items)) + "\n")
        raw_hits = stage / "grep_hits.tsv"
        line_map, screened = screen(args.candidate_root, selected, patterns, raw_hits)
        if screened["screen_counts"]["documents_scanned"] != index["selection_counts"]["selected"]:
            raise ValueError("incomplete priority screen")
        anchor_pairs = set()
        with raw_hits.open("r") as source:
            for line in source:
                line_number, stem = line.rstrip("\n").split(":", 1)
                anchor_pairs.add((int(line_number) - 1, stem))
        candidate_docs = {line_map[line]["document_id"]: line_map[line]
                          for line, _ in anchor_pairs}
        texts = load_sample_text(args.candidate_root, list(candidate_docs.values())) if candidate_docs else {}
        normalized = {doc_id: normalize(text) for doc_id, text in texts.items()}
        findings = []
        for line, stem in sorted(anchor_pairs):
            row = line_map[line]
            document_id = row["document_id"]
            for item in items[stem]:
                if matches_two_options(normalized[document_id], stem, item["options"]):
                    findings.append({"document_id": document_id, "split": row["split"],
                                     "token_count": row["token_count"],
                                     "ceval_item_id": item["ceval_item_id"]})
        hits = stage / "hits.jsonl"
        with hits.open("w") as out:
            for hit in findings:
                out.write(core.canonical_json(hit) + "\n")
        report = {
            "stage": "finewiki_priority_ceval_short_stem_plus_options_screen_only",
            **ref, **index, **screened,
            "stem_chars": [MIN_STEM_CHARS, MAX_STEM_CHARS],
            "minimum_option_chars": MIN_OPTION_CHARS,
            "required_options_after_stem": 2,
            "option_window_normalized_chars": WINDOW_CHARS,
            "anchor_pairs": len(anchor_pairs),
            "anchor_documents": len(candidate_docs),
            "matched_documents": len({x["document_id"] for x in findings}),
            "matched_eval_items": len({x["ceval_item_id"] for x in findings}),
            "hits": len(findings), "hits_sha256": core.sha256_file(hits),
            "limitations": ["short stems under 12 normalized characters not screened",
                            "requires two options after an exact normalized stem within 700 characters",
                            "paraphrases, answer-only contamination and other benchmarks not checked"],
            "training_eligible": False,
        }
        patterns.unlink()
        raw_hits.unlink()
        (stage / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        stage.rename(args.output_dir)
    print(json.dumps({"complete": True, "anchor_documents": report["anchor_documents"],
                      "matched_documents": report["matched_documents"], "hits": report["hits"]}),
          flush=True)


if __name__ == "__main__":
    main()
