#!/usr/bin/env python3
"""Freeze a balanced, resumable eight-GPU Qwen audit queue; not training data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import tempfile

from p0_prepare_ultradata_probe6 import BUCKETS, document_id, iter_selected_rows, sha256_file
from p0_replay_ultradata_prefilter import classify_hard


TARGETS = {
    "ultrafineweb_zh": 45_000,
    "ultrafineweb_en": 45_000,
    "ultradata_code_l2_py": 35_000,
    "ultradata_math_l2_preview": 25_000,
}
# GPUs 1 and 7 share their device with pre-existing jobs and use smaller KV
# budgets; assign them fewer prompt tokens rather than making them the barrier.
SHARD_WEIGHTS = [1.0, 0.55, 1.0, 1.0, 1.0, 1.0, 1.0, 0.55]
CANDIDATES = {
    "ultrafineweb_zh": 50_000,
    "ultrafineweb_en": 50_000,
    "ultradata_code_l2_py": 60_000,
    "ultradata_math_l2_preview": 80_000,
}


def load_prior_ids(probe_dir: Path) -> set[str]:
    ids = set()
    for path in sorted((probe_dir / "inputs").glob("*/*.input.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    ids.add(json.loads(line)["document_id"])
    if len(ids) != 4000:
        raise ValueError(f"expected 4000 prior probe IDs, got {len(ids)}")
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--prior-probe-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-output-tokens", type=int, default=384)
    ap.add_argument("--num-shards", type=int, default=8)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    if args.num_shards != 8:
        raise ValueError("this frozen plan uses eight shards")

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    plan_path = args.raw_dir / "download_plan_v1.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if set(TARGETS) != {item["source_id"] for item in plan["items"]}:
        raise ValueError("source plan differs from frozen quotas")
    prior_ids = load_prior_ids(args.prior_probe_dir)
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    bands = {name: [] for name in ("heavy", "medium", "light")}
    stats = {}
    overlong = []
    seen_content_hashes: set[bytes] = set()

    for source_number, item in enumerate(plan["items"]):
        source = item["source_id"]
        bucket, language = BUCKETS[source]
        path = args.raw_dir / "raw" / source / Path(item["path"]).name
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"raw shard integrity mismatch: {source}")
        pf = pq.ParquetFile(path)
        count = min(CANDIDATES[source], pf.metadata.num_rows)
        indices = random.Random(args.seed + source_number).sample(range(pf.metadata.num_rows), count)
        columns = list(pf.schema_arrow.names)
        rows = dict(iter_selected_rows(pf, set(indices), columns))
        counter = Counter()
        for index in indices:
            counter["sampled"] += 1
            row = rows[index]
            content = row.get("content")
            if not isinstance(content, str):
                counter["invalid_content"] += 1
                continue
            locator = (f"{row.get('repo_name')}/{row.get('relative_path')}"
                       if source == "ultradata_code_l2_py" else
                       f"{Path(item['path']).name}:row={index}")
            if classify_hard({"source_id": source, "source_locator": locator}, content):
                counter["hard_path_or_empty"] += 1
                continue
            digest = hashlib.sha256(content.encode("utf-8")).digest()
            if digest in seen_content_hashes:
                counter["exact_duplicate_in_selection"] += 1
                continue
            seen_content_hashes.add(digest)
            doc_id = document_id(source, item["revision"], item["path"], index, content)
            if doc_id in prior_ids:
                counter["already_scored_probe"] += 1
                continue
            upstream_score = (row.get("score") if "score" in row else
                              row.get("quality_score") if "quality_score" in row else
                              row.get("quality_label"))
            # In the prior 4,000-document scoreable probe, the largest input
            # was 15,225 characters. Quarantine much larger records rather
            # than asking the Qwen tokenizer to materialize 300k-token texts.
            # This is an explicit sampling exclusion, not an exact token count.
            if len(content) > 30_000:
                overlong.append({"document_id": doc_id, "source_id": source,
                                 "raw_row": index, "source_path": item["path"],
                                 "upstream_score": upstream_score, "text_chars": len(content),
                                 "prompt_tokens_est": None,
                                 "quarantine_reason": "oversize_char_precheck"})
                counter["oversize_char_precheck"] += 1
                continue
            envelope = {"document_id": doc_id, "source_id": source,
                        "source_locator": locator, "current_bucket": bucket,
                        "language": language, "document_text": content}
            payload = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            encoded = tokenizer.apply_chat_template([
                {"role": "system", "content": prompt},
                {"role": "user", "content": payload},
            ], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            prompt_tokens = len(encoded["input_ids"])
            minimal = {"document_id": doc_id, "source_id": source, "raw_row": index,
                       "source_path": item["path"], "upstream_score": upstream_score,
                       "text_chars": len(content), "prompt_tokens_est": prompt_tokens}
            if prompt_tokens + args.max_output_tokens > args.max_model_len:
                overlong.append(minimal)
                counter["overlong"] += 1
                continue
            band = "heavy" if prompt_tokens > 3000 else "medium" if prompt_tokens > 1800 else "light"
            bands[band].append({**minimal, "current_bucket": bucket, "review_payload": payload})
            counter["queued"] += 1
            if counter["queued"] >= TARGETS[source]:
                break
        if counter["queued"] < TARGETS[source]:
            raise ValueError(f"source {source} quota shortfall: {counter['queued']} / {TARGETS[source]}")
        stats[source] = {"population": pf.metadata.num_rows, "candidate_sample": count,
                         "target": TARGETS[source], **dict(counter)}
        print(json.dumps({"source": source, **stats[source]}, ensure_ascii=False), flush=True)
        del rows

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ultradata-night8-", dir=args.output_dir.parent) as tmp:
        out = Path(tmp)
        band_stats = {}
        for band, items in bands.items():
            shards = [[] for _ in range(args.num_shards)]
            loads = [0] * args.num_shards
            for row in sorted(items, key=lambda x: (-x["prompt_tokens_est"], x["document_id"])):
                slot = min(range(args.num_shards),
                           key=lambda i: (loads[i] / SHARD_WEIGHTS[i], len(shards[i]), i))
                shards[slot].append(row)
                loads[slot] += row["prompt_tokens_est"]
            (out / band).mkdir()
            paths = []
            for i, shard in enumerate(shards):
                path = out / band / f"shard-{i}.input.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    for row in shard:
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                paths.append({"shard": i, "documents": len(shard), "sha256": sha256_file(path)})
            band_stats[band] = {"documents": len(items), "estimated_prompt_tokens": sum(loads),
                                "shards": paths}
        overlong_path = out / "overlong_sample.jsonl"
        with overlong_path.open("w", encoding="utf-8") as handle:
            for row in overlong:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        manifest = {"stage": "eight_gpu_quality_audit_not_training_approved",
                    "seed": args.seed, "source_plan_sha256": sha256_file(plan_path),
                    "prior_probe_manifest_sha256": sha256_file(args.prior_probe_dir / "inputs" / "manifest.json"),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "model_path": str(args.model_path), "targets": TARGETS, "candidates": CANDIDATES,
                    "max_model_len": args.max_model_len, "max_output_tokens": args.max_output_tokens,
                    "num_shards": args.num_shards, "sources": stats, "bands": band_stats,
                    "shard_weights": SHARD_WEIGHTS,
                    "overlong_count": len(overlong), "overlong_sha256": sha256_file(overlong_path)}
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        out.rename(args.output_dir)
    print(json.dumps({"output": str(args.output_dir), "bands": band_stats}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
