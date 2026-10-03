#!/usr/bin/env python3
"""Repartition four immutable Qwen shards across six GPUs and KV-safe bands."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile


BANDS = (("heavy", 3001, 4096, 12), ("medium", 1801, 3000, 20), ("light", 0, 1800, 32))
GPUS = (0, 2, 3, 4, 5, 6)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"output exists: {args.output_dir}")
    source = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    if len(source["shards"]) != 4:
        raise ValueError("expected four source shards")
    groups: dict[str, list[dict]] = {name: [] for name, *_ in BANDS}
    all_seen: set[str] = set()
    for i, meta in enumerate(source["shards"]):
        path = args.input_dir / meta["file"]
        if sha256_file(path) != meta["sha256"]:
            raise ValueError(f"source shard {i} checksum mismatch")
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                doc_id = row["document_id"]
                if doc_id in all_seen:
                    raise ValueError(f"duplicate document_id: {doc_id}")
                all_seen.add(doc_id)
                count += 1
                tokens = row["prompt_tokens_est"]
                for name, low, high, _ in BANDS:
                    if low <= tokens <= high:
                        groups[name].append(row)
                        break
                else:
                    raise ValueError(f"prompt token estimate outside bands: {tokens}")
        if count != meta["documents"]:
            raise ValueError(f"source shard {i} count mismatch")
    if len(all_seen) != source["scoreable_documents"]:
        raise ValueError("source document count mismatch")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qwen-bands-", dir=args.output_dir.parent) as temp:
        out = Path(temp)
        counts = {name: [0] * len(GPUS) for name, *_ in BANDS}
        token_sums = {name: [0] * len(GPUS) for name, *_ in BANDS}
        costs = {name: [0] * len(GPUS) for name, *_ in BANDS}
        handles = {}
        for name, *_ in BANDS:
            (out / name).mkdir()
            for i in range(len(GPUS)):
                handles[name, i] = (out / name / f"shard-{i}.input.jsonl").open("w", encoding="utf-8")
        try:
            for name, *_ in BANDS:
                for row in sorted(groups[name], key=lambda r: (-r["prompt_tokens_est"], r["document_id"])):
                    i = min(range(len(GPUS)), key=lambda j: (costs[name][j], counts[name][j], j))
                    handles[name, i].write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    counts[name][i] += 1
                    token_sums[name][i] += row["prompt_tokens_est"]
                    costs[name][i] += row["prompt_tokens_est"] + 200
        finally:
            for handle in handles.values():
                handle.close()
        report = {"stage": "kv_safe_qwen_input_bands_not_training_approved",
                  "source_manifest_sha256": sha256_file(args.input_dir / "manifest.json"),
                  "total_documents": source["scoreable_documents"], "gpu_indices": GPUS, "bands": []}
        for name, low, high, workers in BANDS:
            files = []
            for i, gpu in enumerate(GPUS):
                path = out / name / f"shard-{i}.input.jsonl"
                files.append({"file": f"{name}/{path.name}", "gpu": gpu, "documents": counts[name][i],
                              "prompt_tokens": token_sums[name][i], "sha256": sha256_file(path)})
            report["bands"].append({"name": name, "min_prompt_tokens": low, "max_prompt_tokens": high,
                                    "workers_per_gpu": workers, "documents": sum(counts[name]),
                                    "prompt_tokens": sum(token_sums[name]), "shards": files})
        if sum(band["documents"] for band in report["bands"]) != report["total_documents"]:
            raise ValueError("band count mismatch")
        (out / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        out.rename(args.output_dir)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
