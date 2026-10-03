#!/usr/bin/env python3
"""Export, preflight, and gate the pinned non-Wikipedia v2 candidate queue (eight GPUs).

Every output is an audit artifact. A Qwen keep is never training approval.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import shutil
import tempfile


SOURCES = ("fineweb_edu_new", "finemath_4plus_new", "github_code_clean_new")
GPUS = tuple(range(8))
BANDS = (("heavy", 3001, 3712, 12), ("medium", 1801, 3000, 20), ("light", 0, 1800, 32))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bucket(row: dict) -> str:
    if row["source_id"] == "fineweb_edu_new":
        return "general_en"
    if row["source_id"] == "finemath_4plus_new":
        return "math_en"
    return {"Python": "code_python", "Shell": "code_shell", "JavaScript": "code_other"}[row["language"]]


def export(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    index_dir = root / "upstream_nonwiki_v1/audit/high_precision_v2"
    raw_dir = root / "upstream_nonwiki_v1/raw"
    index_manifest = json.loads((index_dir / "report.json").read_text())
    if index_manifest["stage"] != "high_precision_filtered_candidates_not_training_approved":
        raise ValueError("unexpected candidate index status")
    selected = defaultdict(dict)
    source_counts = Counter()
    index_hashes = {}
    for source in SOURCES:
        path = index_dir / f"{source}.candidate_index.parquet"
        if sha256(path) != index_manifest["outputs"][path.name]["sha256"]:
            raise ValueError(f"candidate index hash mismatch: {source}")
        index_hashes[source] = sha256(path)
        table = pq.read_table(path)
        rows = table.to_pylist()
        if args.limit_per_source:
            rows = heapq.nsmallest(args.limit_per_source, rows,
                                   key=lambda x: x["content_sha256"])
        for row in rows:
            key = (row["source_id"], row["source_file"])
            if row["raw_row"] in selected[key]:
                raise ValueError(f"duplicate raw pointer: {key}/{row['raw_row']}")
            selected[key][row["raw_row"]] = row
            source_counts[source] += 1
    if not source_counts or set(source_counts) != set(SOURCES):
        raise ValueError("empty source selection")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name("." + output.name + ".part")
    if tmp.exists():
        raise FileExistsError(tmp)
    count = 0
    try:
        with tmp.open("w", encoding="utf-8") as dest:
            for source in SOURCES:
                for (src, filename), pointers in sorted(selected.items()):
                    if src != source:
                        continue
                    pf = pq.ParquetFile(raw_dir / source / filename)
                    offsets = [0]
                    for group in range(pf.num_row_groups):
                        offsets.append(offsets[-1] + pf.metadata.row_group(group).num_rows)
                    grouped = defaultdict(list)
                    for raw_row, row in pointers.items():
                        group = bisect_right(offsets, raw_row) - 1
                        if group < 0 or group >= pf.num_row_groups:
                            raise ValueError(f"raw pointer out of range: {source}/{filename}/{raw_row}")
                        grouped[group].append(row)
                    field = "code" if source == "github_code_clean_new" else "text"
                    for group in sorted(grouped):
                        values = pf.read_row_group(group, columns=[field]).column(0)
                        for row in sorted(grouped[group], key=lambda x: x["raw_row"]):
                            text = values[row["raw_row"] - offsets[group]].as_py()
                            if not isinstance(text, str):
                                raise ValueError("raw text missing")
                            text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
                            if hashlib.sha256(text.encode()).hexdigest() != row["content_sha256"]:
                                raise ValueError(f"raw hash mismatch: {source}/{filename}/{row['raw_row']}")
                            body = {"document_id": row["content_sha256"], "source_id": source,
                                    "source_locator": row["locator"], "current_bucket": bucket(row),
                                    "language": row["language"], "document_text": text}
                            out = {"document_id": row["content_sha256"], "source_id": source,
                                   "source_file": filename, "raw_row": row["raw_row"],
                                   "current_bucket": bucket(row), "token_count": row["upstream_token_count"],
                                   "candidate_relpath": f"{source}/{filename}",
                                   "row_in_shard": row["raw_row"],
                                   "review_payload": json.dumps(body, ensure_ascii=False, separators=(",", ":"))}
                            dest.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
                            count += 1
        if count != sum(source_counts.values()):
            raise ValueError(f"exported {count}, expected {sum(source_counts.values())}")
        tmp.rename(output)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    meta = {"stage": "nonwiki_review_export_not_training_approved",
            "source_index_sha256": index_hashes, "limit_per_source": args.limit_per_source,
            "by_source": dict(source_counts), "documents": count,
            "input_sha256": sha256(output), "input_bytes": output.stat().st_size}
    output.with_suffix(".manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(meta, ensure_ascii=False))


def prepare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    inp, output = args.input.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    tok = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".nonwiki-qwen6-", dir=output.parent))
    handles = {}
    costs = defaultdict(lambda: [0] * len(GPUS))
    counts = defaultdict(lambda: [0] * len(GPUS))
    long_count = 0
    seen = set()
    try:
        for name, *_ in BANDS:
            (tmp / name).mkdir()
            for index in range(len(GPUS)):
                handles[name, index] = (tmp / name / f"shard-{index}.input.jsonl").open("w", encoding="utf-8")
        with (tmp / "overlong.jsonl").open("w", encoding="utf-8") as long_file:
            for line in inp.open(encoding="utf-8"):
                row = json.loads(line)
                doc_id = row["document_id"]
                if doc_id in seen:
                    raise ValueError(f"duplicate document ID: {doc_id}")
                seen.add(doc_id)
                encoded = tok.apply_chat_template([
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": row["review_payload"]},
                ], tokenize=True, add_generation_prompt=True, enable_thinking=False)
                n = len(encoded["input_ids"])
                row["prompt_tokens_est"] = n
                if n + 384 > 4096:
                    long_count += 1
                    long_file.write(json.dumps({"document_id": doc_id, "prompt_tokens": n,
                                                "source_id": row["source_id"]}) + "\n")
                    continue
                name = next(name for name, low, high, _ in BANDS if low <= n <= high)
                index = min(range(len(GPUS)), key=lambda i: (costs[name][i], counts[name][i], i))
                handles[name, index].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[name][index] += 1
                costs[name][index] += n + 384
        for f in handles.values():
            f.close()
        bands = []
        for name, low, high, workers in BANDS:
            shards = []
            for i, gpu in enumerate(GPUS):
                path = tmp / name / f"shard-{i}.input.jsonl"
                shards.append({"file": f"{name}/{path.name}", "gpu": gpu,
                               "documents": counts[name][i], "prompt_tokens": costs[name][i],
                               "sha256": sha256(path)})
            bands.append({"name": name, "min_prompt_tokens": low, "max_prompt_tokens": high,
                          "workers_per_gpu": workers, "documents": sum(counts[name]),
                          "shards": shards})
        meta = {"stage": "nonwiki_qwen6_inputs_not_training_approved",
                "source_input_sha256": sha256(inp), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "total_documents": sum(b["documents"] for b in bands),
                "overlong_documents": long_count, "all_documents": len(seen),
                "overlong_sha256": sha256(tmp / "overlong.jsonl"),
                "gpu_indices": GPUS, "bands": bands}
        if meta["total_documents"] + long_count != len(seen):
            raise ValueError("coverage mismatch")
        (tmp / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        tmp.rename(output)
        print(json.dumps({"total_documents": meta["total_documents"],
                          "overlong_documents": long_count, "output": str(output)}))
    except BaseException:
        for f in handles.values():
            f.close()
        shutil.rmtree(tmp)
        raise


def strict_keep(annotation: dict, expected_bucket: str) -> bool:
    return (annotation.get("decision") == "keep"
            and annotation.get("confidence", 0) >= 0.85
            and annotation.get("quality", 0) >= 3
            and annotation.get("completeness", 0) >= 4
            and annotation.get("educational_value", 0) >= 3
            and annotation.get("format_integrity", 0) >= 4
            and annotation.get("recommended_bucket") == expected_bucket
            and not annotation.get("reason_codes"))


def gate(args: argparse.Namespace) -> None:
    data, output = args.data_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = data / "input_bands8_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    finished = json.loads((data / "FINISHED8.json").read_text())
    if finished["input_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Qwen completion/input mismatch")
    # A resumed batch may have been scored concurrently after an SSH host
    # restart. Reconciliation keeps one valid result per ID, but any document
    # whose duplicate ratings disagree on the strict gate must be dropped.
    conflicts_path = data / "results8/light/reconcile_conflicts.jsonl"
    conflicts = set()
    if conflicts_path.exists():
        for line in conflicts_path.open():
            item = json.loads(line)
            conflicts.add(item["document_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".nonwiki-qwen-gate-", dir=output.parent))
    counts = Counter()
    settings = set()
    try:
        with (tmp / "strict_keep_candidates.jsonl").open("w") as keeps, \
             (tmp / "rejected_or_unscored.jsonl").open("w") as rejects:
            for band in manifest["bands"]:
                for i, shard in enumerate(band["shards"]):
                    path = data / "input_bands8_v1" / shard["file"]
                    if sha256(path) != shard["sha256"]:
                        raise ValueError(f"input changed: {path}")
                    expected = {}
                    for line in path.open():
                        row = json.loads(line)
                        expected[row["document_id"]] = row
                    results = data / "results8" / band["name"] / f"shard-{i}.results.jsonl"
                    if expected and not results.exists():
                        raise ValueError(f"missing Qwen result file: {results}")
                    valid = {}
                    for line in (results.open() if results.exists() else ()):
                        x = json.loads(line)
                        doc_id, = json.loads(x["source_id"])
                        if doc_id not in expected:
                            raise ValueError(f"unknown Qwen result ID: {doc_id}")
                        if x["text_sha256"] != hashlib.sha256(expected[doc_id]["review_payload"].encode()).hexdigest():
                            raise ValueError(f"payload mismatch: {doc_id}")
                        if x["status"] == "ok":
                            if x["annotation"].get("document_id") != doc_id:
                                raise ValueError(f"annotation ID mismatch: {doc_id}")
                            if doc_id in valid:
                                raise ValueError(f"duplicate valid result: {doc_id}")
                            valid[doc_id] = x["annotation"]
                            settings.add((x["model"], x["prompt_sha256"], x["schema_sha256"],
                                          x.get("temperature"), x.get("inference_sha256")))
                    for doc_id, row in expected.items():
                        a = valid.get(doc_id)
                        if doc_id in conflicts:
                            decision, reason = "drop", "duplicate_qwen_disagreement"
                        elif a is None:
                            decision, reason = "drop", "missing_valid_qwen_result"
                        elif strict_keep(a, row["current_bucket"]):
                            decision, reason = "keep_candidate", "strict_qwen_keep"
                        else:
                            decision, reason = "drop", "qwen_not_strict_same_bucket_keep"
                        record = {"document_id": doc_id, "source_id": row["source_id"],
                                  "source_file": row["source_file"], "raw_row": row["raw_row"],
                                  "bucket": row["current_bucket"], "reason": reason}
                        (keeps if decision == "keep_candidate" else rejects).write(json.dumps(record) + "\n")
                        counts[reason] += 1
        if len(settings) != 1:
            raise ValueError(f"mixed Qwen protocol: {settings}")
        meta = {"stage": "nonwiki_qwen_strict_candidate_gate_not_training_approved",
                "training_approved": False, "input_manifest_sha256": sha256(manifest_path),
                "finished_sha256": sha256(data / "FINISHED8.json"),
                "reconciliation_conflicts": len(conflicts),
                "reconciliation_conflicts_sha256": sha256(conflicts_path) if conflicts_path.exists() else None,
                "by_reason": dict(counts), "scoreable_documents": manifest["total_documents"],
                "overlong_documents": manifest["overlong_documents"],
                "protocol": list(next(iter(settings))),
                "strict_keep_sha256": sha256(tmp / "strict_keep_candidates.jsonl"),
                "rejected_sha256": sha256(tmp / "rejected_or_unscored.jsonl"),
                "pending": ["second_judge_or_targeted_review", "near_dedup", "benchmark_contamination", "license_review", "tokenizer_exact_count"]}
        if sum(counts.values()) != manifest["total_documents"]:
            raise ValueError("gate coverage mismatch")
        (tmp / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        tmp.rename(output)
        print(json.dumps(meta, ensure_ascii=False))
    except BaseException:
        shutil.rmtree(tmp)
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    e = sub.add_parser("export")
    e.add_argument("--root", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--limit-per-source", type=int, default=0)
    p = sub.add_parser("prepare")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--prompt", type=Path, required=True)
    g = sub.add_parser("gate")
    g.add_argument("--data-dir", type=Path, required=True)
    g.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    {"export": export, "prepare": prepare, "gate": gate}[args.action](args)


if __name__ == "__main__":
    main()
