#!/usr/bin/env python3
"""Deterministic, source-balanced audit probe for four newly downloaded shards.

This creates scoring inputs only. It neither changes the raw Parquet files nor
approves any document for training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import tempfile


BUCKETS = {
    "ultrafineweb_zh": ("general_zh", "zh"),
    "ultrafineweb_en": ("general_en", "en"),
    "ultradata_code_l2_py": ("code_python", "en"),
    "ultradata_math_l2_preview": ("math_en", "en"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iter_selected_rows(parquet_file, selected: set[int], columns: list[str]):
    offset = 0
    for group in range(parquet_file.num_row_groups):
        count = parquet_file.metadata.row_group(group).num_rows
        positions = sorted(i - offset for i in selected if offset <= i < offset + count)
        if positions:
            table = parquet_file.read_row_group(group, columns=columns)
            for pos in positions:
                yield offset + pos, {name: table.column(name)[pos].as_py() for name in columns}
        offset += count


def document_id(source_id: str, revision: str, source_path: str, row: int, content: str) -> str:
    value = json.dumps([source_id, revision, source_path, row,
                        hashlib.sha256(content.encode("utf-8")).hexdigest()],
                       ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--candidates-per-source", type=int, default=2500)
    ap.add_argument("--target-per-source", type=int, default=1000)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-output-tokens", type=int, default=384)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise SystemExit("output directory already exists; refusing to overwrite")
    if args.target_per_source > args.candidates_per_source:
        raise SystemExit("target exceeds candidate sample")

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    plan_path = args.raw_dir / "download_plan_v1.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    bands = {name: [] for name in ("heavy", "medium", "light")}
    source_stats = {}
    overlong = []

    for source_number, item in enumerate(plan["items"]):
        source_id = item["source_id"]
        bucket, language = BUCKETS[source_id]
        path = args.raw_dir / "raw" / source_id / Path(item["path"]).name
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"raw shard integrity failure: {source_id}")
        parquet_file = pq.ParquetFile(path)
        population = parquet_file.metadata.num_rows
        rng = random.Random(args.seed + source_number)
        indices = rng.sample(range(population), min(population, args.candidates_per_source))
        order = {index: rank for rank, index in enumerate(indices)}
        columns = list(parquet_file.schema_arrow.names)
        rows = dict(iter_selected_rows(parquet_file, set(indices), columns))
        accepted = 0
        stats = Counter()
        for index in indices:
            row = rows[index]
            content = row.get("content")
            stats["sampled"] += 1
            if not isinstance(content, str) or not content.strip():
                stats["empty"] += 1
                continue
            doc_id = document_id(source_id, item["revision"], item["path"], index, content)
            upstream_score = (row.get("score") if "score" in row else
                              row.get("quality_score") if "quality_score" in row else
                              row.get("quality_label"))
            source_locator = (f"{row.get('repo_name')}/{row.get('relative_path')}"
                              if source_id == "ultradata_code_l2_py" else
                              f"{Path(item['path']).name}:row={index}")
            envelope = {
                "document_id": doc_id,
                "source_id": source_id,
                "source_locator": source_locator,
                "current_bucket": bucket,
                "language": language,
                "document_text": content,
            }
            payload = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            encoded = tokenizer.apply_chat_template([
                {"role": "system", "content": prompt},
                {"role": "user", "content": payload},
            ], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            prompt_tokens = len(encoded["input_ids"])
            minimal = {
                "document_id": doc_id, "source_id": source_id, "raw_row": index,
                "source_path": item["path"], "upstream_score": upstream_score,
                "text_chars": len(content), "prompt_tokens_est": prompt_tokens,
            }
            if prompt_tokens + args.max_output_tokens > args.max_model_len:
                overlong.append(minimal)
                stats["overlong"] += 1
                continue
            stats["scoreable_in_sample"] += 1
            if accepted >= args.target_per_source:
                continue
            band = "heavy" if prompt_tokens > 3000 else "medium" if prompt_tokens > 1800 else "light"
            bands[band].append({**minimal, "current_bucket": bucket, "review_payload": payload})
            accepted += 1
            stats["queued"] += 1
        source_stats[source_id] = {"population": population, **dict(stats)}
        print(json.dumps({"source": source_id, **source_stats[source_id]}, ensure_ascii=False), flush=True)

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ultradata-probe6-", dir=args.output_dir.parent) as tmp:
        directory = Path(tmp)
        band_stats = {}
        for band, rows in bands.items():
            shards = [[] for _ in range(6)]
            loads = [0] * 6
            for row in sorted(rows, key=lambda x: (-x["prompt_tokens_est"], x["document_id"])):
                slot = min(range(6), key=lambda i: (loads[i], len(shards[i]), i))
                shards[slot].append(row)
                loads[slot] += row["prompt_tokens_est"]
            (directory / band).mkdir()
            for i, shard in enumerate(shards):
                with (directory / band / f"shard-{i}.input.jsonl").open("w", encoding="utf-8") as handle:
                    for row in shard:
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            band_stats[band] = {"documents": len(rows), "estimated_prompt_tokens": sum(loads),
                                "shards": [len(s) for s in shards]}
        with (directory / "overlong_sample.jsonl").open("w", encoding="utf-8") as handle:
            for row in overlong:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        manifest = {
            "stage": "new_raw_source_audit_probe_not_training_approved",
            "seed": args.seed,
            "source_plan_sha256": sha256_file(plan_path),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "model_path": str(args.model_path),
            "target_per_source": args.target_per_source,
            "candidates_per_source": args.candidates_per_source,
            "max_model_len": args.max_model_len,
            "max_output_tokens": args.max_output_tokens,
            "source_stats": source_stats,
            "bands": band_stats,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        directory.rename(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "bands": band_stats}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
