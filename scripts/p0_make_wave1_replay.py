#!/usr/bin/env python3
"""Build a deterministic 100-document offline replay of completed Wave1 judgments.

This copies prior model responses; it never calls an API or changes the approved corpus.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_rows(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def pick(values: list[dict], n: int) -> list[dict]:
    return sorted(values, key=lambda x: hashlib.sha256(x["document_id"].encode()).hexdigest())[:n]


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source
    accepted = list(rows(args.accepted / "accepted_index.jsonl"))
    selected = []
    for split, count in (("train", 70), ("validation", 5), ("test", 5)):
        candidates = [x for x in accepted if x["split"] == split]
        if len(candidates) < count:
            raise ValueError(f"not enough accepted {split} records for replay")
        selected.extend(pick(candidates, count))
    provisional = {x["document_id"]: x for x in rows(source / "provisional_index.jsonl")}
    gate_ids = {x["document_id"] for x in rows(source / "deepseek_full_v1/two_judge_gate_full_v2/two_judge_accepted_ids.jsonl")}
    rejected = [x for x in provisional.values() if x["document_id"] not in gate_ids]
    if len(rejected) < 20:
        raise ValueError("not enough rejected records for replay")
    selected.extend(pick(rejected, 20))
    selected_ids = {x["document_id"] for x in selected}
    if len(selected_ids) != 100:
        raise ValueError("replay selection has duplicate IDs")
    input_rows = [x for x in rows(source / "deepseek_full_v1/deepseek_full.input.jsonl")
                  if x["document_id"] in selected_ids]
    if len(input_rows) != 100:
        raise ValueError("input selection does not cover all IDs")
    args.output.mkdir(parents=True)
    review_dir = args.output / "deepseek_full_v1"
    review_dir.mkdir()
    index_path = args.output / "provisional_index.jsonl"
    selected_index = [provisional[x["document_id"]] for x in selected]
    write_rows(index_path, selected_index)
    input_path = review_dir / "deepseek_full.input.jsonl"
    write_rows(input_path, input_rows)
    source_summary = json.loads((source / "screen_summary.json").read_text())
    source_summary["stage"] = "wave1_offline_replay_screen_not_new_training_data"
    source_summary["provisional_documents"] = 100
    source_summary["provisional_own_text_tokens"] = sum(x["own_text_tokens"] for x in selected_index)
    source_summary["documents_by_domain"] = dict(sorted(Counter(x["domain"] for x in selected_index).items()))
    source_summary["tokens_by_domain"] = dict(sorted({
        domain: sum(x["own_text_tokens"] for x in selected_index if x["domain"] == domain)
        for domain in {x["domain"] for x in selected_index}
    }.items()))
    source_summary["index_sha256"] = sha(index_path)
    source_summary["replay_of_screen_summary_sha256"] = sha(source / "screen_summary.json")
    (args.output / "screen_summary.json").write_text(json.dumps(source_summary, indent=2, ensure_ascii=False) + "\n")
    input_manifest = {
        "stage": "wave1_offline_replay_second_judge_input",
        "screen_index_sha256": sha(index_path), "documents": 100,
        "input_sha256": sha(input_path),
    }
    (review_dir / "deepseek_full.input.manifest.json").write_text(json.dumps(input_manifest, indent=2) + "\n")
    selection = {
        "stage": "offline_replay_only_not_new_training_data", "documents": 100,
        "previously_accepted": 80, "previously_rejected": 20,
        "source_screen_sha256": sha(source / "screen_summary.json"),
        "source_frozen_sha256": sha(args.accepted / "manifest.json"),
        "input_sha256": sha(input_path),
        "selected_ids_sha256": hashlib.sha256("\n".join(sorted(selected_ids)).encode()).hexdigest(),
    }
    (args.output / "selection_manifest.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection))


def filter_results(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    expected = {x["document_id"] for x in rows(args.input)}
    if len(expected) != 100:
        raise ValueError("replay input must have 100 unique documents")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = []
    seen = set()
    for result in rows(args.source_results):
        document_id, = json.loads(result["source_id"])
        if document_id in expected:
            selected.append(result)
            seen.add(document_id)
    write_rows(args.output, selected)
    manifest = {"stage": "offline_replay_responses_no_api_call", "input_sha256": sha(args.input),
                "source_results_sha256": sha(args.source_results), "results_sha256": sha(args.output),
                "input_documents": 100, "documents_with_attempts": len(seen), "attempt_rows": len(selected)}
    (args.output.parent / "replay_results_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--source", type=Path, required=True)
    a.add_argument("--accepted", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    b = sub.add_parser("filter-results")
    b.add_argument("--input", type=Path, required=True)
    b.add_argument("--source-results", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    prepare(args) if args.action == "prepare" else filter_results(args)


if __name__ == "__main__":
    main()
