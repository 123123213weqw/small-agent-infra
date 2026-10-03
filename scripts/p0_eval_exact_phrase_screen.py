#!/usr/bin/env python3
"""Narrow exact token-phrase screen against frozen ARC-Easy/HellaSwag holdouts.

This does not prove absence of paraphrased or partial benchmark contamination.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq


WORD = re.compile(r"\w+", re.UNICODE)
ANCHOR_WORDS = 8
MIN_WORDS_BY_TASK = {"arc_easy_validation": 6, "arc_easy_test": 6, "hellaswag_validation": 12}


def tokens(text: str) -> tuple[str, ...]:
    return tuple(WORD.findall(text.casefold()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eval_phrases(eval_dir: Path):
    specs = [
        ("arc_easy_validation", eval_dir / "ai2_arc/ARC-Easy/validation-00000-of-00001.parquet"),
        ("arc_easy_test", eval_dir / "ai2_arc/ARC-Easy/test-00000-of-00001.parquet"),
        ("hellaswag_validation", eval_dir / "hellaswag/data/validation-00000-of-00001.parquet"),
    ]
    files = {}
    for name, path in specs:
        files[name] = {"path": str(path), "sha256": sha256(path)}
        table = pq.read_table(path)
        for row in table.to_pylist():
            if name.startswith("arc_easy"):
                yield name, str(row["id"]), tokens(row["question"]), files
            else:
                for choice, ending in enumerate(row["endings"]):
                    phrase = tokens(row["ctx"] + " " + ending)
                    yield name, f"{row['ind']}:{choice}", phrase, files


def corpus_rows(pool: Path, index: Path, inputs_dir: Path):
    manifest = json.loads((pool / "manifest.json").read_text(encoding="utf-8"))
    for shard in manifest["shards"]:
        path = pool / shard["domain"] / shard["file"]
        if path.stat().st_size != shard["bytes"] or sha256(path) != shard["sha256"]:
            raise ValueError(f"Wave1 shard integrity failure: {path}")
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["document_id", "split", "text"]):
            for row in batch.to_pylist():
                yield "wave1_" + row["split"], row["document_id"], row["text"]
    selected = set()
    with index.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["postfilter_status"] == "quality_candidate_license_pending":
                selected.add(row["document_id"])
    seen = set()
    for band in ("heavy", "medium", "light"):
        for shard in range(8):
            path = inputs_dir / band / f"shard-{shard}.input.jsonl"
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    document_id = row["document_id"]
                    if document_id not in selected:
                        continue
                    if document_id in seen:
                        raise ValueError(f"candidate repeated: {document_id}")
                    seen.add(document_id)
                    yield "new_candidate", document_id, json.loads(row["review_payload"])["document_text"]
    if seen != selected:
        raise ValueError(f"missing candidate rows: {len(selected - seen)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchors = defaultdict(list)
    files = {}
    phrase_count = Counter()
    skipped_short = Counter()
    for task, item_id, phrase, files in eval_phrases(args.eval_dir):
        if len(phrase) < MIN_WORDS_BY_TASK[task]:
            skipped_short[task] += 1
            continue
        anchors[(len(phrase[:ANCHOR_WORDS]), phrase[:ANCHOR_WORDS])].append((task, item_id, phrase))
        phrase_count[task] += 1
    scanned = Counter()
    hit_counts = Counter()
    out_path = args.output_dir / "eval_exact_phrase_hits.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for group, document_id, text in corpus_rows(args.pool, args.index, args.inputs_dir):
            scanned[group] += 1
            words = tokens(text)
            seen_hits = set()
            for anchor_len in (6, 7, 8):
                for pos in range(max(0, len(words) - anchor_len + 1)):
                    for task, item_id, phrase in anchors.get((anchor_len, words[pos : pos + anchor_len]), ()):
                        key = (task, item_id)
                        if key not in seen_hits and words[pos : pos + len(phrase)] == phrase:
                            seen_hits.add(key)
                            hit_counts[(group, task)] += 1
                            out.write(json.dumps({
                                "document_id": document_id,
                                "corpus_group": group,
                                "task": task,
                                "eval_item_id": item_id,
                                "matched_words": len(phrase),
                            }, sort_keys=True) + "\n")
    summary = {
        "stage": "narrow_exact_phrase_eval_screen_not_full_decontamination",
        "eval_files": files,
        "minimum_phrase_words_by_task": MIN_WORDS_BY_TASK,
        "short_eval_phrases_not_screened": dict(sorted(skipped_short.items())),
        "phrases_screened": dict(sorted(phrase_count.items())),
        "corpus_documents_scanned": dict(sorted(scanned.items())),
        "hits": [
            {"corpus_group": group, "task": task, "count": n}
            for (group, task), n in sorted(hit_counts.items())
        ],
        "hits_sha256": sha256(out_path),
        "limitations": "exact contiguous word sequence only; no paraphrase, answer-only, code benchmark, or shorter phrase coverage",
    }
    (args.output_dir / "eval_exact_phrase_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
