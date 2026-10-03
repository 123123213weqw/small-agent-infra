#!/usr/bin/env python3
"""Separate math/code reference-NLL panel; never added to training packs.

All candidate 32-token windows are screened against every token position in
the bound training pack. uint64 hashes are only an index: hits are verified
by exact token comparison. This is not a semantic contamination certificate.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from training_eval_common_v1 import sha, write_json


def window_hashes(tokens, width=32):
    import numpy as np
    values = np.asarray(tokens, dtype=np.uint64)
    count = len(values) - width + 1
    if count <= 0: return np.empty(0, dtype=np.uint64)
    result = np.zeros(count, dtype=np.uint64)
    # Overflow is deliberate arithmetic modulo 2**64.
    for offset in range(width):
        result *= np.uint64(1000003)
        result += values[offset:offset + count] + np.uint64(1)
    return result


def main():
    import numpy as np
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    p = argparse.ArgumentParser()
    for key in ("data", "training_pack", "tokenizer", "output"): p.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    started = time.monotonic()
    tokenizer = Tokenizer.from_file(str(a.tokenizer))
    manifest = json.loads((a.training_pack / "manifest.json").read_text())
    if sha(a.tokenizer) != manifest["tokenizer_sha256"]: raise ValueError("tokenizer binding differs")
    rows = []
    for index, row in enumerate(pq.read_table(a.data / "gsm8k/test.parquet").to_pylist()):
        rows.append({"id": f"gsm8k/test/{index}", "domain": "reference_math",
                     "text": "Question: " + row["question"] + "\nAnswer: " + row["answer"],
                     "source": "openai/gsm8k:test", "revision": "740312add88f781978c0658806c59bc2815b9866"})
    for name, version in (("HumanEvalPlus", "v0.1.10"), ("MbppPlus", "v0.2.0")):
        for line in (a.data / "evalplus" / f"{name}-{version}.jsonl").read_text().splitlines():
            row = json.loads(line)
            rows.append({"id": row["task_id"], "domain": "reference_code",
                         "text": row["prompt"] + row["canonical_solution"], "source": name, "revision": version})
    seen, unique = set(), []
    for row in rows:
        digest = __import__("hashlib").sha256(row["text"].encode()).hexdigest()
        if digest in seen: continue
        seen.add(digest)
        ids = tokenizer.encode(row.pop("text"), add_special_tokens=False).ids + [2]
        if not 33 <= len(ids) <= 2048: continue
        unique.append(dict(row, text_sha256=digest, input_ids=ids))
    index = defaultdict(list)
    for i, row in enumerate(unique):
        for offset, h in enumerate(window_hashes(row["input_ids"])):
            index[int(h)].append((i, offset))
    bloom = np.zeros(1 << 24, dtype=np.bool_)
    mask = np.uint64(len(bloom) - 1)
    hashes = np.asarray(list(index), dtype=np.uint64)
    bloom[(hashes & mask).astype(np.int64)] = True
    rejected, bindings, scanned = {}, [], 0
    for entry in manifest["shards"]:
        if entry["split"] != "train": continue
        path = a.training_pack / entry["bin"]
        if sha(path) != entry["bin_sha256"]: raise ValueError("training shard changed")
        bindings.append({"path": str(path), "sha256": entry["bin_sha256"]})
        stream = np.memmap(path, dtype="<u2", mode="r")
        for base in range(0, max(0, len(stream) - 31), 262144):
            block = stream[base: min(len(stream), base + 262144 + 31)]
            hh = window_hashes(block)
            possible = np.flatnonzero(bloom[(hh & mask).astype(np.int64)])
            for offset in possible:
                hits = index.get(int(hh[offset]))
                if hits is None: continue
                segment = block[offset:offset + 32]
                for doc, pos in hits:
                    if doc not in rejected and np.array_equal(segment, unique[doc]["input_ids"][pos:pos + 32]):
                        rejected[doc] = {"shard": entry["bin"], "training_offset": base + int(offset), "candidate_offset": pos}
            scanned += len(hh)
        write_json(a.output / "progress.json", {"scanned_windows": scanned, "rejected_documents": len(rejected), "last_shard": entry["bin"], "seconds": time.monotonic() - started})
        print(json.dumps({"shard": entry["bin"], "scanned_windows": scanned, "rejected_documents": len(rejected)}), flush=True)
    accepted = [row for i, row in enumerate(unique) if i not in rejected]
    counts = Counter(row["domain"] for row in accepted)
    if any(counts[name] < 128 for name in ("reference_math", "reference_code")):
        raise ValueError("insufficient contamination-screened panel coverage")
    panel = a.output / "documents.jsonl"
    panel.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted))
    write_json(a.output / "rejected.json", [{"id": unique[i]["id"], **hit} for i, hit in rejected.items()])
    write_json(a.output / "manifest.json", {
        "version": "training_reference_panel_v2", "training_eligible": False,
        "metric_scope": "separate_reference_document_NLL_not_original_corpus_PPL_not_generation_accuracy",
        "documents": dict(counts), "predicted_tokens": dict(Counter({name: sum(len(r["input_ids"]) - 1 for r in accepted if r["domain"] == name) for name in counts})),
        "candidate_documents": len(unique), "rejected_overlap_documents": len(rejected),
        "screen": "all_32_token_windows_full_training_pack_uint64_index_exact_verified",
        "semantic_contamination_certified": False, "training_pack_manifest_sha256": sha(a.training_pack / "manifest.json"),
        "tokenizer_sha256": sha(a.tokenizer), "source_data_manifest_sha256": sha(a.data / "manifest.json"),
        "source_files": [{"path": str(x), "sha256": sha(x)} for x in [a.data / "gsm8k/test.parquet", a.data / "evalplus/HumanEvalPlus-v0.1.10.jsonl", a.data / "evalplus/MbppPlus-v0.2.0.jsonl"]],
        "scanned_training_shards": bindings, "scanned_windows": scanned,
        "artifacts": {"documents.jsonl": {"sha256": sha(panel), "bytes": panel.stat().st_size}}, "seconds": time.monotonic() - started,
    })


if __name__ == "__main__": main()
