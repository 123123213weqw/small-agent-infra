#!/usr/bin/env python3
"""Conservative, pointer-only prefilter for pinned non-Wikipedia raw shards.

This is an audit queue, not a training approval or a token pack. Source text
stays in the original Parquet files; outputs contain row pointers and hashes.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import re
import shutil
import tempfile


LICENSES = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "cc0-1.0", "unlicense"}
LANGUAGES = {"Python", "Shell", "JavaScript"}
VENDOR_PARTS = {"node_modules", "vendor", "_vendor", "third_party", "third-party", "site-packages", "dist-packages", ".venv", "venv", "build", "dist"}
MATH_SIGNAL = re.compile(r"(?i)(?:\\(?:frac|sum|int|sqrt|begin)|[=≤≥∑∫√]|±|\b(?:equation|theorem|lemma|proof|integral|derivative|matrix|probability|formula|algebra|geometry|fraction|polynomial|inequality)\b)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def repeated_line_fraction(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 1.0
    counts = Counter(lines)
    return sum(n for n in counts.values() if n > 1) / len(lines)


def classify(source: str, row: dict, text: str) -> list[str]:
    reasons = []
    length = len(text)
    if source == "github_code_clean_new":
        if row.get("language") not in LANGUAGES:
            reasons.append("language_out_of_scope")
        if str(row.get("license") or "").lower() not in LICENSES:
            reasons.append("license_not_whitelisted")
        path = str(row.get("path") or "").replace("\\", "/")
        if set(x.casefold() for x in path.split("/")[:-1]) & VENDOR_PARTS:
            reasons.append("vendored_or_build_path")
        if length < 160 or length > 30_000:
            reasons.append("length_out_of_range")
        lines = [x for x in text.splitlines() if x.strip()]
        if len(lines) < 4:
            reasons.append("too_few_code_lines")
        if max(map(len, lines), default=0) > 5_000:
            reasons.append("minified_or_generated")
        if row.get("language") == "Python" and 160 <= length <= 30_000:
            try:
                ast.parse(text)
            except (SyntaxError, ValueError, UnicodeError, MemoryError):
                reasons.append("python3_parse_failure")
    else:
        if length < 500 or length > 20_000:
            reasons.append("length_out_of_range")
        if row.get("language") != "en" or float(row.get("language_score") or 0) < 0.8:
            reasons.append("language_not_confident_en")
        if int(row.get("int_score") or 0) < 4:
            reasons.append("upstream_score_below_4")
        if source == "finemath_4plus_new" and not MATH_SIGNAL.search(text):
            reasons.append("no_math_signal")
    if not text:
        reasons.append("empty_text")
    if text.count("\ufffd") / max(length, 1) > 0.001:
        reasons.append("replacement_characters")
    if 160 <= length <= 30_000 and repeated_line_fraction(text) > 0.6:
        reasons.append("repeated_lines")
    if "enable javascript and cookies" in text.lower():
        reasons.append("non_content_page")
    return reasons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sample-per-source", type=int, default=24)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.sample_per_source < 1:
        raise ValueError("sample-per-source must be positive")

    import pyarrow as pa
    import pyarrow.parquet as pq

    root = args.root.resolve()
    plan_path = root / "download_plan_v1.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("version") != "p0_overnight_upstream_v1" or len(plan.get("items", [])) != 6:
        raise ValueError("unexpected source plan")
    if {x["source_id"] for x in plan["items"]} != {
        "fineweb_edu_new", "finemath_4plus_new", "github_code_clean_new"
    }:
        raise ValueError("unexpected source set")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".nonwiki-prefilter-", dir=output.parent))
    schema = pa.schema([
        ("source_id", pa.string()), ("source_file", pa.string()), ("raw_row", pa.int64()),
        ("content_sha256", pa.string()), ("locator", pa.string()), ("family", pa.string()),
        ("language", pa.string()), ("record_license", pa.string()),
        ("chars", pa.int32()), ("upstream_token_count", pa.int64()),
        ("upstream_score", pa.float64()),
    ])
    writers = {}
    counters = defaultdict(Counter)
    score_hist = defaultdict(Counter)
    host_hist = defaultdict(Counter)
    samples = defaultdict(list)
    seen_hashes = set()
    files = []
    try:
        for item in plan["items"]:
            source = item["source_id"]
            path = root / "raw" / source / Path(item["path"]).name
            if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                raise ValueError(f"raw integrity mismatch: {path}")
            pf = pq.ParquetFile(path)
            expected = ({"code", "repo_name", "path", "language", "license", "size"}
                        if source == "github_code_clean_new" else
                        {"text", "url", "language", "language_score", "token_count", "score", "int_score"})
            if not expected <= set(pf.schema_arrow.names):
                raise ValueError(f"schema mismatch: {path}")
            writer = writers.get(source)
            if writer is None:
                writer = pq.ParquetWriter(tmp / f"{source}.candidate_index.parquet", schema, compression="zstd")
                writers[source] = writer
            row_offset = 0
            for group_id in range(pf.num_row_groups):
                cols = list(expected)
                rows = pf.read_row_group(group_id, columns=cols).to_pylist()
                kept = []
                for local_row, row in enumerate(rows):
                    counters[source]["raw_rows"] += 1
                    content = row.get("code" if source == "github_code_clean_new" else "text")
                    text = normalize(content) if isinstance(content, str) else ""
                    if source == "github_code_clean_new":
                        score_hist[source][str(row.get("language"))] += 1
                        host_hist[source][str(row.get("license"))] += 1
                    else:
                        score_hist[source][str(row.get("int_score"))] += 1
                        from urllib.parse import urlparse
                        host_hist[source][urlparse(str(row.get("url") or "")).hostname or ""] += 1
                    reasons = classify(source, row, text)
                    for reason in reasons:
                        counters[source]["reason:" + reason] += 1
                    if reasons:
                        counters[source]["prefilter_rejected"] += 1
                        continue
                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if digest in seen_hashes:
                        counters[source]["exact_duplicate_new_pool"] += 1
                        continue
                    seen_hashes.add(digest)
                    locator = (f"{row.get('repo_name')}/{row.get('path')}" if source == "github_code_clean_new"
                               else str(row.get("url") or ""))
                    family = (str(row.get("repo_name") or "") if source == "github_code_clean_new"
                              else locator)
                    upstream_count = row.get("token_count")
                    kept.append({
                        "source_id": source, "source_file": path.name,
                        "raw_row": row_offset + local_row, "content_sha256": digest,
                        "locator": locator, "family": family,
                        "language": str(row.get("language") or ""),
                        "record_license": str(row.get("license") or ""),
                        "chars": len(text), "upstream_token_count": int(upstream_count or 0),
                        "upstream_score": float(row.get("score") or 0),
                    })
                    counters[source]["candidate_rows"] += 1
                    counters[source]["candidate_chars"] += len(text)
                    counters[source]["candidate_upstream_tokens"] += int(upstream_count or 0)
                    # Keep the lowest hashed IDs, independent of Parquet row-group order.
                    sample_key = int(hashlib.sha256(f"{source}\0{digest}".encode()).hexdigest()[:16], 16)
                    sample = {"source_id": source, "source_file": path.name,
                              "raw_row": row_offset + local_row, "locator": locator,
                              "content_sha256": digest, "text": text}
                    heap = samples[source]
                    if len(heap) < args.sample_per_source:
                        heapq.heappush(heap, (-sample_key, sample))
                    elif sample_key < -heap[0][0]:
                        heapq.heapreplace(heap, (-sample_key, sample))
                if kept:
                    writer.write_table(pa.Table.from_pylist(kept, schema=schema))
                row_offset += len(rows)
            if row_offset != pf.metadata.num_rows:
                raise ValueError(f"Parquet row count changed: {path}")
            files.append({"source_id": source, "name": path.name, "rows": row_offset,
                          "raw_sha256": item["sha256"]})
            print(json.dumps({"file": str(path), "rows": row_offset,
                              "candidate_so_far": counters[source]["candidate_rows"]}), flush=True)
        for writer in writers.values():
            writer.close()
        for source, heap in samples.items():
            with (tmp / f"{source}.sample.jsonl").open("w", encoding="utf-8") as handle:
                for _, sample in sorted(heap, key=lambda x: -x[0]):
                    handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        report = {
            "stage": "deterministic_prefilter_candidates_not_training_approved",
            "version": "p0_nonwiki_prefilter_v1", "plan_sha256": sha256_file(plan_path),
            "script_sha256": sha256_file(Path(__file__)), "sample_per_source": args.sample_per_source,
            "rules": {"web_math_chars": [500, 20000], "code_chars": [160, 30000],
                      "language_score_min": 0.8, "int_score_min": 4,
                      "code_licenses": sorted(LICENSES)},
            "files": files, "by_source": {k: dict(v) for k, v in counters.items()},
            "score_or_language_hist": {k: dict(v) for k, v in score_hist.items()},
            "host_or_license_top20": {k: v.most_common(20) for k, v in host_hist.items()},
            "outputs": {},
            "pending": ["candidate_quality_model_review", "old_pool_exact_and_near_dedup",
                        "benchmark_contamination_scan", "source_and_record_license_review",
                        "tokenizer_v1_exact_count", "split_and_pack"],
        }
        for path in sorted(tmp.iterdir()):
            if path.is_file():
                report["outputs"][path.name] = {"bytes": path.stat().st_size,
                                                  "sha256": sha256_file(path)}
        (tmp / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.rename(output)
        print(json.dumps({"output": str(output), "by_source": report["by_source"]}, ensure_ascii=False), flush=True)
    except BaseException:
        for writer in writers.values():
            writer.close()
        shutil.rmtree(tmp)
        raise


if __name__ == "__main__":
    main()
