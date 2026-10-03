#!/usr/bin/env python3
"""Prepare existing Qwen-strict Wave1 candidates for same-rubric DeepSeek review."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-input", type=Path, required=True)
    p.add_argument("--screen-index", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    selected = {}
    with args.screen_index.open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            if x["status"] != "provisional_review_required":
                raise ValueError("non-provisional row in screen index")
            if x["document_id"] in selected:
                raise ValueError("screen index ID repeated")
            selected[x["document_id"]] = x
    seen = set()
    by_domain = Counter()
    out_path = args.output_dir / "deepseek_full.input.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for band_dir in sorted(x for x in args.qwen_input.iterdir() if x.is_dir()):
            for path in sorted(band_dir.glob("shard-*.input.jsonl")):
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        doc_id = row["document_id"]
                        if doc_id not in selected:
                            continue
                        if doc_id in seen:
                            raise ValueError("Qwen input ID repeated")
                        seen.add(doc_id)
                        metadata = selected[doc_id]
                        if row["current_bucket"] != metadata["domain"]:
                            raise ValueError("domain mismatch")
                        row.update({
                            "source_id": metadata["source_id"],
                            "raw_row": row["row_in_shard"],
                            "source_path": row["candidate_relpath"],
                            "upstream_score": metadata["source_score"],
                        })
                        out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        by_domain[metadata["domain"]] += 1
    if seen != set(selected):
        raise ValueError(f"missing Qwen input IDs: {len(set(selected) - seen)}")
    manifest = {
        "stage": "wave1_full_second_judge_input_not_training_approved",
        "screen_index_sha256": sha256(args.screen_index),
        "documents": len(seen),
        "by_domain": dict(sorted(by_domain.items())),
        "input_sha256": sha256(out_path),
    }
    (args.output_dir / "deepseek_full.input.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
