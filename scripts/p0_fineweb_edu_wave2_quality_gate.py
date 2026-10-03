#!/usr/bin/env python3
"""Non-destructive Wave2 domain gate and deterministic, stratified review export.

This produces audit artifacts only. It does not approve training data.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
from urllib.parse import urlsplit


DENY_HOSTS = ("coursehero.com", "lessonplanet.com", "teacherspayteachers.com")
RISK_HOSTS = ("reference.com", "enotes.com", "slideplayer.com",
              "studocu.com", "brainly.com", "quizlet.com")
SCORE_EDGES = (3.7, 4.1)
CHAR_EDGES = (1500, 5000)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def host_matches(hostname: str, root: str) -> bool:
    return hostname == root or hostname.endswith("." + root)


def stratum(row: dict, head: set[str], hostname: str) -> str:
    score = bisect_right(SCORE_EDGES, row["upstream_score"])
    length = bisect_right(CHAR_EDGES, row["chars"])
    return f"score{score}_length{length}_{'head' if hostname in head else 'tail'}"


def offer(heaps: dict, name: str, row: dict, limit: int, salt: str) -> None:
    priority = int.from_bytes(hashlib.sha256(
        (salt + row["content_sha256"]).encode()).digest()[:8], "big")
    entry = (-priority, row["content_sha256"], row)
    bucket = heaps[name]
    if len(bucket) < limit:
        heapq.heappush(bucket, entry)
    elif entry > bucket[0]:
        heapq.heapreplace(bucket, entry)


def validated_indexes(root: Path) -> tuple[list[Path], dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "audit/prefilter_v1/SUMMARY.json").read_text())
    if (summary["status"] != "candidate_only_not_training_approved"
            or summary["revision"] != manifest["revision"] or len(manifest["items"]) != 13):
        raise ValueError("unexpected source manifest or prefilter summary")
    index_dir = root / "audit/prefilter_v1"
    paths = []
    total_rows = total_tokens = 0
    index_hashes = {}
    for item in manifest["items"]:
        name = Path(item["path"]).name
        path = index_dir / f"{name}.candidate_index.parquet"
        done = json.loads(path.with_suffix(".done.json").read_text())
        digest = sha256_file(path)
        if digest != done["index_sha256"] or done["raw_sha256"] != item["sha256"]:
            raise ValueError(f"candidate index integrity failure: {path}")
        total_rows += done["counts"]["candidate_rows"]
        total_tokens += done["counts"]["candidate_tokenizer_v1_tokens"]
        index_hashes[path.name] = digest
        paths.append(path)
    if (total_rows != summary["candidate_rows"]
            or total_tokens != summary["candidate_tokenizer_v1_tokens"]):
        raise ValueError("candidate index counts differ from summary")
    return paths, {"manifest_revision": manifest["revision"],
                   "source_index_sha256": index_hashes,
                   "candidate_rows": total_rows,
                   "candidate_tokenizer_v1_tokens": total_tokens}


def iter_rows(paths: list[Path]):
    import pyarrow.parquet as pq
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=8192):
            yield from batch.to_pylist()


def gate(root: Path, output: Path, per_stratum: int, per_risk_host: int) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths, lineage = validated_indexes(root)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    # Two passes keep memory bounded while defining the head-domain stratum
    # from the candidate population, not from the chosen audit sample.
    host_counts = Counter()
    denied_counts, denied_tokens = Counter(), Counter()
    for row in iter_rows(paths):
        hostname = host(row["locator"])
        denied = next((x for x in DENY_HOSTS if host_matches(hostname, x)), None)
        if denied:
            denied_counts[denied] += 1
            denied_tokens[denied] += row["tokenizer_v1_tokens"]
        else:
            host_counts[hostname] += 1
    head = {name for name, _ in host_counts.most_common(100)}
    populations, population_tokens = Counter(), Counter()
    heaps: dict[str, list] = defaultdict(list)
    risk_heaps: dict[str, list] = defaultdict(list)
    denied_rows = []
    for row in iter_rows(paths):
        hostname = host(row["locator"])
        denied = next((x for x in DENY_HOSTS if host_matches(hostname, x)), None)
        if denied:
            denied_rows.append({"content_sha256": row["content_sha256"],
                                "source_file": row["source_file"],
                                "raw_row": row["raw_row"],
                                "deny_host": denied,
                                "tokenizer_v1_tokens": row["tokenizer_v1_tokens"]})
            continue
        name = stratum(row, head, hostname)
        populations[name] += 1
        population_tokens[name] += row["tokenizer_v1_tokens"]
        offer(heaps, name, row, per_stratum, "wave2-stratum-v1:")
        for risky in RISK_HOSTS:
            if host_matches(hostname, risky):
                offer(risk_heaps, risky, row, per_risk_host, "wave2-risk-v1:")
                break

    selected = {}
    for name, heap in heaps.items():
        for _, _, row in heap:
            selected[row["content_sha256"]] = {**row, "sampling_group": name,
                                                  "sampling_design": "stratified"}
    stratified_count = len(selected)
    risk_selected = Counter()
    for name, heap in risk_heaps.items():
        for _, _, row in heap:
            if row["content_sha256"] not in selected:
                selected[row["content_sha256"]] = {**row, "sampling_group": name,
                                                      "sampling_design": "targeted_risk"}
                risk_selected[name] += 1
    chosen = sorted(selected.values(), key=lambda x: (x["source_file"], x["raw_row"]))
    if sum(populations.values()) + len(denied_rows) != lineage["candidate_rows"]:
        raise ValueError("gate partition does not reconcile")
    if len({x["content_sha256"] for x in chosen}) != len(chosen):
        raise ValueError("duplicate selected document")

    denied_path = output / "excluded_known_domain.parquet"
    pq.write_table(pa.Table.from_pylist(denied_rows), denied_path, compression="zstd")
    selected_path = output / "selected_review_index.parquet"
    pq.write_table(pa.Table.from_pylist(chosen), selected_path, compression="zstd")
    input_path = output / "deepseek_review.input.jsonl"
    export_texts(root, chosen, input_path)
    report = {"status": "audit_sample_only_not_training_approved",
              "lineage": lineage,
              "known_domain_excluded_rows": len(denied_rows),
              "known_domain_excluded_tokens": sum(denied_tokens.values()),
              "known_domain_by_host_rows": dict(denied_counts),
              "known_domain_by_host_tokens": dict(denied_tokens),
              "post_domain_rows": sum(populations.values()),
              "post_domain_tokenizer_v1_tokens": (lineage["candidate_tokenizer_v1_tokens"]
                                                  - sum(denied_tokens.values())),
              "stratum_population_rows": dict(sorted(populations.items())),
              "stratum_population_tokens": dict(sorted(population_tokens.items())),
              "stratified_sample_rows": stratified_count,
              "targeted_additional_rows": dict(risk_selected),
              "total_review_rows": len(chosen),
              "top100_hosts": sorted(head),
              "outputs_sha256": {p.name: sha256_file(p) for p in
                                  (denied_path, selected_path, input_path)},
              "pending": ["near_dedup", "evaluation_decontamination",
                          "source_and_license_review", "second_judge",
                          "mixture_selection", "pack_and_split"]}
    (output / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def export_texts(root: Path, selected: list[dict], output: Path) -> None:
    import pyarrow.parquet as pq

    grouped = defaultdict(list)
    for row in selected:
        grouped[row["source_file"]].append(row)
    with output.open("w", encoding="utf-8") as dest:
        count = 0
        for source_file, rows in sorted(grouped.items()):
            raw = root / "raw" / Path(source_file).name
            pf = pq.ParquetFile(raw)
            offsets = [0]
            for group in range(pf.num_row_groups):
                offsets.append(offsets[-1] + pf.metadata.row_group(group).num_rows)
            by_group = defaultdict(list)
            for row in rows:
                group = bisect_right(offsets, row["raw_row"]) - 1
                if not 0 <= group < pf.num_row_groups:
                    raise ValueError("raw pointer outside Parquet file")
                by_group[group].append(row)
            for group in sorted(by_group):
                values = pf.read_row_group(group, columns=["text"]).column(0)
                for row in sorted(by_group[group], key=lambda x: x["raw_row"]):
                    text = values[row["raw_row"] - offsets[group]].as_py()
                    if not isinstance(text, str):
                        raise ValueError("raw text missing")
                    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
                    if hashlib.sha256(text.encode()).hexdigest() != row["content_sha256"]:
                        raise ValueError("raw text hash mismatch")
                    payload = {"document_id": row["content_sha256"],
                               "source_id": "fineweb_edu_wave2_v1",
                               "current_bucket": "general_en", "language": "en",
                               "document_text": text}
                    audit_row = {"document_id": row["content_sha256"],
                                 "source_id": "fineweb_edu_wave2_v1",
                                 "current_bucket": "general_en",
                                 "source_file": row["source_file"],
                                 "raw_row": row["raw_row"],
                                 "sampling_design": row["sampling_design"],
                                 "sampling_group": row["sampling_group"],
                                 "upstream_score": row["upstream_score"],
                                 "text_chars": row["chars"],
                                 "tokenizer_v1_tokens": row["tokenizer_v1_tokens"],
                                 "review_payload": json.dumps(payload, ensure_ascii=False,
                                                              separators=(",", ":"))}
                    dest.write(json.dumps(audit_row, ensure_ascii=False,
                                          separators=(",", ":")) + "\n")
                    count += 1
        if count != len(selected):
            raise ValueError("export count mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=100)
    parser.add_argument("--per-risk-host", type=int, default=25)
    args = parser.parse_args()
    if args.per_stratum < 1 or args.per_risk_host < 0:
        parser.error("sample sizes must be non-negative, with per-stratum > 0")
    print(json.dumps(gate(args.root.resolve(), args.output.resolve(), args.per_stratum,
                          args.per_risk_host), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
