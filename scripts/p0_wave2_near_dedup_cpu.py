#!/usr/bin/env python3
"""CPU-only, non-destructive near-duplicate audit for FineWeb-Edu Wave2.

Phase 1 writes bottom-k word-5-gram sketches of every post-domain Wave2
candidate and the frozen Wave1 pool. Phase 2 performs approximate retrieval
followed by exact shingle-Jaccard checks; neither phase deletes source text.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import heapq
import json
from pathlib import Path
import re


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[\u3400-\u4dbf\u4e00-\u9fff]|[^\s]")
SHINGLE_SIZE = 5
SKETCH_SIZE = 64
PAD = (1 << 64) - 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def shingle_hashes(text: str) -> set[int]:
    import xxhash

    tokens = TOKEN_PATTERN.findall(text.casefold())
    if len(tokens) < SHINGLE_SIZE:
        return {xxhash.xxh64_intdigest("\x1f".join(tokens))} if tokens else set()
    return {xxhash.xxh64_intdigest("\x1f".join(tokens[i:i + SHINGLE_SIZE]))
            for i in range(len(tokens) - SHINGLE_SIZE + 1)}


def make_sketch(text: str) -> tuple[list[int], str]:
    values = shingle_hashes(text)
    sketch = heapq.nsmallest(SKETCH_SIZE, values)
    sketch.extend([PAD] * (SKETCH_SIZE - len(sketch)))
    norm = re.sub(r"\s+", " ", text.casefold()).strip()
    return sketch, hashlib.sha256(norm.encode()).hexdigest()


def schema():
    import pyarrow as pa
    return pa.schema([
        ("origin", pa.string()), ("source_file", pa.string()),
        ("raw_row", pa.int64()), ("content_sha256", pa.string()),
        ("tokenizer_v1_tokens", pa.int64()), ("chars", pa.int32()),
        ("norm_sha256", pa.string()), ("split", pa.string()),
        ("sketch", pa.list_(pa.uint64(), SKETCH_SIZE)),
    ])


def write_rows(writer, batch: list[dict]) -> None:
    import pyarrow as pa
    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=schema()))
        batch.clear()


def sketch_wave2_one(root: str, output_dir: str, name: str) -> dict:
    import pyarrow.parquet as pq

    root, output_dir = Path(root), Path(output_dir)
    raw = root / "raw" / name
    index = root / "audit/prefilter_v1" / f"{name}.candidate_index.parquet"
    done = json.loads(index.with_suffix(".done.json").read_text())
    index_hash = sha256(index)
    if index_hash != done["index_sha256"]:
        raise ValueError(f"candidate index SHA mismatch: {index}")
    excluded_file = root / "audit/quality_gate_v1/excluded_known_domain.parquet"
    excluded = set()
    for row in pq.read_table(excluded_file, columns=["source_file", "raw_row"]).to_pylist():
        if Path(row["source_file"]).name == name:
            excluded.add(row["raw_row"])
    pointers = {}
    pf_index = pq.ParquetFile(index)
    for batch in pf_index.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            if row["raw_row"] not in excluded:
                pointers[row["raw_row"]] = row
    if len(pointers) + len(excluded) != done["counts"]["candidate_rows"]:
        raise ValueError(f"candidate count mismatch: {name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{name}.sketch.parquet"
    marker = output.with_suffix(".done.json")
    if marker.exists():
        old = json.loads(marker.read_text())
        if (not output.exists() or sha256(output) != old["output_sha256"]
                or old["index_sha256"] != index_hash
                or old["rows"] != len(pointers)):
            raise ValueError(f"completed sketch integrity mismatch: {output}")
        return old
    if output.exists():
        raise FileExistsError(output)
    temp = output.with_suffix(".parquet.tmp")
    temp.unlink(missing_ok=True)
    pf = pq.ParquetFile(raw)
    observed = kept = 0
    try:
        with pq.ParquetWriter(temp, schema(), compression="zstd") as writer:
            buffer = []
            for batch in pf.iter_batches(batch_size=2048, columns=["text"]):
                for value in batch.column(0).to_pylist():
                    pointer = pointers.get(observed)
                    if pointer is not None:
                        if not isinstance(value, str):
                            raise ValueError(f"missing raw text {name}:{observed}")
                        text = normalize(value)
                        if hashlib.sha256(text.encode()).hexdigest() != pointer["content_sha256"]:
                            raise ValueError(f"raw text SHA mismatch {name}:{observed}")
                        sketch, norm_sha = make_sketch(text)
                        buffer.append({"origin": "wave2", "source_file": pointer["source_file"],
                                       "raw_row": observed, "content_sha256": pointer["content_sha256"],
                                       "tokenizer_v1_tokens": pointer["tokenizer_v1_tokens"],
                                       "chars": pointer["chars"], "norm_sha256": norm_sha,
                                       "split": "candidate", "sketch": sketch})
                        kept += 1
                        if len(buffer) >= 2048:
                            write_rows(writer, buffer)
                    observed += 1
            write_rows(writer, buffer)
        if kept != len(pointers) or observed != done["counts"]["raw_rows"]:
            raise ValueError(f"raw/sketch row mismatch {name}: {kept}/{len(pointers)}")
        result = {"stage": "wave2_bottomk_sketch_not_training_approved", "file": name,
                  "rows": kept, "raw_rows_scanned": observed, "index_sha256": index_hash,
                  "output_sha256": sha256(temp), "sketch_size": SKETCH_SIZE,
                  "shingle_size": SHINGLE_SIZE}
        temp.replace(output)
        marker.write_text(json.dumps(result, indent=2) + "\n")
        return result
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def sketch_wave2(root: Path, output: Path, workers: int) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    names = [Path(item["path"]).name for item in manifest["items"]]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(sketch_wave2_one, str(root), str(output), name) for name in names]
        for future in as_completed(tasks):
            result = future.result()
            results.append(result)
            print(json.dumps({"completed": len(results), "total": len(names),
                              "file": result["file"], "rows": result["rows"]}), flush=True)
    results.sort(key=lambda x: x["file"])
    summary = {"stage": "wave2_bottomk_sketch_not_training_approved",
               "rows": sum(r["rows"] for r in results), "files": results,
               "algorithm": {"word_shingle_size": SHINGLE_SIZE,
                             "bottom_k": SKETCH_SIZE, "hash": "xxhash64",
                             "exact_verification_required": True}}
    expected = json.loads((root / "audit/quality_gate_v1/REPORT.json").read_text())["post_domain_rows"]
    if summary["rows"] != expected:
        raise ValueError(f"sketch count {summary['rows']} != quality gate {expected}")
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "complete", "rows": summary["rows"]}), flush=True)


def sketch_wave1(pool: Path, output: Path) -> None:
    import pyarrow.parquet as pq

    manifest_path = pool / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("stage") != "consolidate_candidates":
        raise ValueError("unexpected Wave1 pool manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    marker = output.with_suffix(".done.json")
    if marker.exists():
        old = json.loads(marker.read_text())
        if (old["manifest_sha256"] != sha256(manifest_path)
                or old["output_sha256"] != sha256(output)):
            raise ValueError("completed Wave1 sketch integrity mismatch")
        print(json.dumps({"status": "already_done", "rows": old["rows"]}), flush=True)
        return
    if output.exists():
        raise FileExistsError(output)
    temp = output.with_suffix(".parquet.tmp")
    temp.unlink(missing_ok=True)
    count = 0
    try:
        with pq.ParquetWriter(temp, schema(), compression="zstd") as writer:
            buffer = []
            for item in manifest["shards"]:
                source = pool / item["domain"] / item["file"]
                if source.stat().st_size != item["bytes"] or sha256(source) != item["sha256"]:
                    raise ValueError(f"Wave1 shard integrity mismatch: {source}")
                pf = pq.ParquetFile(source)
                local = 0
                for batch in pf.iter_batches(batch_size=1024,
                                             columns=["text", "document_id", "split", "token_count"]):
                    for row in batch.to_pylist():
                        text = normalize(row["text"])
                        sketch, norm_sha = make_sketch(text)
                        buffer.append({"origin": "wave1", "source_file": f"{item['domain']}/{item['file']}",
                                       "raw_row": local, "content_sha256": row["document_id"],
                                       "tokenizer_v1_tokens": row["token_count"],
                                       "chars": len(text), "norm_sha256": norm_sha,
                                       "split": row["split"], "sketch": sketch})
                        local += 1
                        count += 1
                        if len(buffer) >= 2048:
                            write_rows(writer, buffer)
                if local != item["rows"]:
                    raise ValueError(f"Wave1 shard row mismatch: {source}")
            write_rows(writer, buffer)
        if count != manifest["counts"]["retained_documents"]:
            raise ValueError("Wave1 sketch count mismatch")
        result = {"stage": "wave1_reference_bottomk_sketch", "rows": count,
                  "manifest_sha256": sha256(manifest_path), "output_sha256": sha256(temp)}
        temp.replace(output)
        marker.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def load_sketches(wave2_root: Path, wave1_pool: Path, sketch_dir: Path):
    import numpy as np
    import pyarrow.parquet as pq

    wave1 = sketch_dir / "wave1.sketch.parquet"
    wave1_done = json.loads(wave1.with_suffix(".done.json").read_text())
    if (sha256(wave1) != wave1_done["output_sha256"]
            or sha256(wave1_pool / "manifest.json") != wave1_done["manifest_sha256"]):
        raise ValueError("Wave1 sketch lineage mismatch")
    wave2_summary = json.loads((sketch_dir / "sketch_wave2/SUMMARY.json").read_text())
    gate = json.loads((wave2_root / "audit/quality_gate_v1/REPORT.json").read_text())
    if wave2_summary["rows"] != gate["post_domain_rows"]:
        raise ValueError("Wave2 sketch count differs from domain gate")
    paths = [wave1] + [sketch_dir / "sketch_wave2" / f"{item['file']}.sketch.parquet"
                       for item in wave2_summary["files"]]
    expected_hashes = [wave1_done["output_sha256"]] + [
        item["output_sha256"] for item in wave2_summary["files"]]
    columns = ("origin", "source_file", "raw_row", "content_sha256",
               "tokenizer_v1_tokens", "chars", "norm_sha256", "split")
    meta = {name: [] for name in columns}
    sketch_parts = []
    for path, digest in zip(paths, expected_hashes):
        if sha256(path) != digest:
            raise ValueError(f"sketch SHA mismatch: {path}")
        table = pq.read_table(path)
        n = table.num_rows
        values = table.column("sketch").combine_chunks().values.to_numpy(zero_copy_only=False)
        sketch_parts.append(np.asarray(values, dtype=np.uint64).reshape(n, SKETCH_SIZE))
        for name in columns:
            meta[name].extend(table.column(name).to_pylist())
        print(json.dumps({"loaded_sketch": path.name, "rows": n}), flush=True)
    sketches = np.concatenate(sketch_parts, axis=0)
    if len(sketches) != wave1_done["rows"] + wave2_summary["rows"]:
        raise ValueError("combined sketch row mismatch")
    return sketches, meta, {"wave1_sketch_sha256": expected_hashes[0],
                            "wave2_sketch_sha256": expected_hashes[1:],
                            "quality_gate_report_sha256": sha256(
                                wave2_root / "audit/quality_gate_v1/REPORT.json")}


def candidate_pairs(sketches, meta: dict, *, max_bucket: int = 128,
                    max_candidate_pairs: int = 20_000_000):
    import numpy as np

    n = len(sketches)
    if n >= (1 << 32):
        raise ValueError("32-bit document index exhausted")
    valid = sketches[:, 7] != PAD
    valid_ids = np.flatnonzero(valid).astype(np.uint32)
    m = len(valid_ids)
    num_pairs = 28
    dtype = np.dtype([("h1", "<u8"), ("h2", "<u8"), ("doc", "<u4")])
    postings = np.empty(m * num_pairs, dtype=dtype)
    position = 0
    for left in range(8):
        for right in range(left + 1, 8):
            part = postings[position:position + m]
            part["h1"] = sketches[valid_ids, left]
            part["h2"] = sketches[valid_ids, right]
            part["doc"] = valid_ids
            position += m
    postings.sort(order=("h1", "h2", "doc"))
    same = ((postings["h1"][1:] == postings["h1"][:-1])
            & (postings["h2"][1:] == postings["h2"][:-1]))
    boundaries = np.flatnonzero(~same).astype(np.int64) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(postings)]))
    repeated = np.flatnonzero(ends - starts > 1)
    token_lengths = np.asarray(meta["tokenizer_v1_tokens"], dtype=np.int64)
    wave1_count = sum(origin == "wave1" for origin in meta["origin"])
    votes = defaultdict(int)
    stats = Counter({"documents_with_8_sketch_values": m,
                     "postings": len(postings), "repeated_keys": len(repeated)})
    for group_idx in repeated:
        start, end = int(starts[group_idx]), int(ends[group_idx])
        size = end - start
        if size > max_bucket:
            stats["high_frequency_keys_skipped"] += 1
            stats["high_frequency_postings_skipped"] += size
            continue
        ids = postings["doc"][start:end]
        for i in range(size - 1):
            a = int(ids[i])
            for j in range(i + 1, size):
                b = int(ids[j])
                if a == b or (a < wave1_count and b < wave1_count):
                    continue
                x, y = (a, b) if a < b else (b, a)
                small, large = sorted((token_lengths[x], token_lengths[y]))
                if small < .70 * large:
                    continue
                key = x * n + y
                votes[key] += 1
                if len(votes) > max_candidate_pairs:
                    raise MemoryError("too many sketch collisions; abort instead of truncating")
    stats["unique_pair_collisions"] = len(votes)
    pairs = []
    for packed, vote in votes.items():
        if vote < 3:
            continue
        a, b = divmod(packed, n)
        # Exact Jaccard >= .82 should have high overlap in bottom-k sketches.
        # This prefilter is deliberately permissive; final decisions use full sets.
        sa, sb = sketches[a], sketches[b]
        overlap = len(set(map(int, sa)) & set(map(int, sb)) - {PAD})
        if overlap >= 20:
            pairs.append((a, b, vote, overlap))
    stats["pairs_with_3_votes"] = sum(v >= 3 for v in votes.values())
    stats["pairs_for_exact_verification"] = len(pairs)
    return pairs, stats


def full_shingles_for_ids(ids: set[int], meta: dict, wave2_root: Path, wave1_pool: Path):
    import pyarrow.parquet as pq

    by_file = defaultdict(list)
    for doc_id in ids:
        origin = meta["origin"][doc_id]
        source = meta["source_file"][doc_id]
        if origin == "wave2":
            path = wave2_root / "raw" / Path(source).name
        else:
            path = wave1_pool / source
        by_file[path].append(doc_id)
    values = {}
    for path, document_ids in sorted(by_file.items()):
        pf = pq.ParquetFile(path)
        offsets = [0]
        for group in range(pf.num_row_groups):
            offsets.append(offsets[-1] + pf.metadata.row_group(group).num_rows)
        groups = defaultdict(list)
        for doc_id in document_ids:
            row = meta["raw_row"][doc_id]
            group = bisect_right(offsets, row) - 1
            if group < 0 or group >= pf.num_row_groups:
                raise ValueError(f"raw pointer out of bounds {path}:{row}")
            groups[group].append(doc_id)
        for group, group_ids in sorted(groups.items()):
            texts = pf.read_row_group(group, columns=["text"]).column(0)
            for doc_id in group_ids:
                text = texts[meta["raw_row"][doc_id] - offsets[group]].as_py()
                if not isinstance(text, str):
                    raise ValueError("raw text missing during exact Jaccard")
                text = normalize(text)
                if (meta["origin"][doc_id] == "wave2" and
                        hashlib.sha256(text.encode()).hexdigest() != meta["content_sha256"][doc_id]):
                    raise ValueError("Wave2 raw text SHA mismatch during exact Jaccard")
                values[doc_id] = shingle_hashes(text)
        print(json.dumps({"verified_file": str(path), "documents": len(document_ids)}), flush=True)
    if len(values) != len(ids):
        raise ValueError("incomplete raw text extraction for exact Jaccard")
    return values


def match(wave2_root: Path, wave1_pool: Path, sketch_dir: Path, output: Path,
          threshold: float = .82) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    sketches, meta, lineage = load_sketches(wave2_root, wave1_pool, sketch_dir)
    candidate, collision_stats = candidate_pairs(sketches, meta)
    normalized = {}
    edges = []
    for doc_id, digest in enumerate(meta["norm_sha256"]):
        old = normalized.get(digest)
        if old is None:
            normalized[digest] = doc_id
        elif meta["origin"][doc_id] == "wave2":
            edges.append((old, doc_id, 1.0, "normalized_exact"))
    collision_stats["normalized_exact_edges"] = len(edges)
    needed = {i for a, b, _, _ in candidate for i in (a, b)}
    exact = full_shingles_for_ids(needed, meta, wave2_root, wave1_pool)
    for a, b, vote, overlap in candidate:
        union = len(exact[a] | exact[b])
        score = len(exact[a] & exact[b]) / union if union else 1.0
        if score >= threshold:
            edges.append((a, b, score, "jaccard_5gram"))
    collision_stats["exact_jaccard_edges"] = sum(reason == "jaccard_5gram"
                                                  for _, _, _, reason in edges)

    parent = list(range(len(sketches)))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, _, _ in edges:
        parent[find(b)] = find(a)
    clusters = defaultdict(list)
    for a, b, _, _ in edges:
        clusters[find(a)].extend((a, b))
    exclusion_rows = []
    for members in clusters.values():
        unique = set(members)
        anchors = [i for i in unique if meta["origin"][i] == "wave1"]
        if anchors:
            winner = min(anchors, key=lambda i: meta["content_sha256"][i])
        else:
            winner = min(unique, key=lambda i: (-meta["tokenizer_v1_tokens"][i],
                                                meta["content_sha256"][i]))
        for loser in sorted(unique):
            if loser == winner or meta["origin"][loser] == "wave1":
                continue
            exclusion_rows.append({
                "content_sha256": meta["content_sha256"][loser],
                "source_file": meta["source_file"][loser], "raw_row": meta["raw_row"][loser],
                "tokenizer_v1_tokens": meta["tokenizer_v1_tokens"][loser],
                "kept_origin": meta["origin"][winner],
                "kept_content_sha256": meta["content_sha256"][winner],
                "cluster_size": len(unique),
                "disposition": "near_duplicate_candidate_not_training_approved",
            })
    edge_rows = [{"left_origin": meta["origin"][a],
                  "left_sha256": meta["content_sha256"][a],
                  "right_origin": meta["origin"][b],
                  "right_sha256": meta["content_sha256"][b],
                  "jaccard": score, "reason": reason}
                 for a, b, score, reason in edges]
    edge_path = output / "verified_edges.parquet"
    exclusion_path = output / "wave2_exclusion_index.parquet"
    edge_schema = pa.schema([("left_origin", pa.string()), ("left_sha256", pa.string()),
                             ("right_origin", pa.string()), ("right_sha256", pa.string()),
                             ("jaccard", pa.float64()), ("reason", pa.string())])
    exclusion_schema = pa.schema([("content_sha256", pa.string()), ("source_file", pa.string()),
                                  ("raw_row", pa.int64()), ("tokenizer_v1_tokens", pa.int64()),
                                  ("kept_origin", pa.string()), ("kept_content_sha256", pa.string()),
                                  ("cluster_size", pa.int32()), ("disposition", pa.string())])
    pq.write_table(pa.Table.from_pylist(edge_rows, schema=edge_schema), edge_path,
                   compression="zstd")
    pq.write_table(pa.Table.from_pylist(exclusion_rows, schema=exclusion_schema),
                   exclusion_path, compression="zstd")
    summary = {"stage": "approximate_retrieval_exact_jaccard_audit_not_training_approved",
               "lineage": lineage, "algorithm": {
                   "token_5gram_bottom_k": SKETCH_SIZE,
                   "retrieval": "all pairs among 8 lowest hashes, 3 shared pair keys, 20/64 sketch overlap",
                   "max_key_bucket": 128, "length_ratio_floor": .70,
                   "exact_jaccard_threshold": threshold,
                   "approximate_retrieval_not_exhaustive": True},
               "documents": len(sketches), "wave1_reference_documents": sum(
                   origin == "wave1" for origin in meta["origin"]),
               "wave2_candidates": sum(origin == "wave2" for origin in meta["origin"]),
               "collision_stats": dict(collision_stats),
               "verified_edges": len(edge_rows),
               "wave2_excluded_documents": len(exclusion_rows),
               "wave2_excluded_tokens": sum(r["tokenizer_v1_tokens"] for r in exclusion_rows),
               "wave2_excluded_against_wave1": sum(r["kept_origin"] == "wave1" for r in exclusion_rows),
               "wave2_excluded_intra_wave2": sum(r["kept_origin"] == "wave2" for r in exclusion_rows),
               "outputs_sha256": {edge_path.name: sha256(edge_path),
                                  exclusion_path.name: sha256(exclusion_path)},
               "pending": ["external_evaluation_decontamination", "source_license_review",
                           "Qwen_quality_labels", "training_pack"]}
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "complete", "excluded": len(exclusion_rows),
                      "tokens": summary["wave2_excluded_tokens"]}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    wave2 = sub.add_parser("sketch-wave2")
    wave2.add_argument("--root", type=Path, required=True)
    wave2.add_argument("--output", type=Path, required=True)
    wave2.add_argument("--workers", type=int, default=4)
    wave1 = sub.add_parser("sketch-wave1")
    wave1.add_argument("--pool", type=Path, required=True)
    wave1.add_argument("--output", type=Path, required=True)
    matcher = sub.add_parser("match")
    matcher.add_argument("--root", type=Path, required=True)
    matcher.add_argument("--wave1-pool", type=Path, required=True)
    matcher.add_argument("--sketch-dir", type=Path, required=True)
    matcher.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "sketch-wave2":
        if not 1 <= args.workers <= 16:
            p.error("workers must be between 1 and 16")
        sketch_wave2(args.root.resolve(), args.output.resolve(), args.workers)
    elif args.command == "sketch-wave1":
        sketch_wave1(args.pool.resolve(), args.output.resolve())
    else:
        match(args.root.resolve(), args.wave1_pool.resolve(),
              args.sketch_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
