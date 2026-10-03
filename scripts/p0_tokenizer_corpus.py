#!/usr/bin/env python3
"""Build a deterministic, domain-balanced corpus for the P0 tokenizer bake-off."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator


VERSION = "p0_tokenizer_corpus_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, document_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{document_id}".encode("utf-8")).hexdigest()


def load_sources(normalized_root: Path, domains: set[str]) -> list[dict[str, Any]]:
    sources = []
    for directory in sorted(normalized_root.iterdir()):
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("stage") != "normalize":
            continue
        paths = []
        observed_domains = set()
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise SystemExit("PyArrow is required") from error
        for shard in manifest["shards"]:
            path = directory / shard["file"]
            if path.stat().st_size != shard["bytes"]:
                raise ValueError(f"normalized shard byte mismatch: {path}")
            # Read only dictionary/statistical metadata here; content hashes are
            # verified once below before any row is admitted.
            table = pq.read_table(path, columns=["domain"])
            observed_domains.update(value.as_py() for chunk in table.column(0).chunks for value in chunk.unique())
            paths.append((path, shard))
        relevant = observed_domains & domains
        if not relevant:
            continue
        if len(relevant) != 1 or observed_domains != relevant:
            raise ValueError(f"source must map to exactly one configured domain: {directory} -> {observed_domains}")
        sources.append(
            {
                "source_id": manifest["source_id"],
                "domain": next(iter(relevant)),
                "directory": directory,
                "manifest_path": manifest_path,
                "manifest_sha256": sha256_file(manifest_path),
                "shards": paths,
            }
        )
    missing = domains - {source["domain"] for source in sources}
    if missing:
        raise ValueError(f"no normalized source for domains: {sorted(missing)}")
    return sources


def iter_rows(source: dict[str, Any], batch_size: int = 4096) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    columns = [
        "document_id", "source_id", "source_revision", "source_locator", "family_id",
        "domain", "language", "text", "normalized_sha256", "license_status", "split", "rule_keep",
    ]
    for path, _ in source["shards"]:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            yield from batch.to_pylist()


def verify_shards(sources: list[dict[str, Any]]) -> None:
    for source in sources:
        for path, metadata in source["shards"]:
            if sha256_file(path) != metadata["sha256"]:
                raise ValueError(f"normalized shard sha256 mismatch: {path}")


def collision(text: str, special_tokens: list[str]) -> bool:
    return any(token in text for token in special_tokens)


def availability_and_holdout(
    sources: list[dict[str, Any]], seed: int, special_tokens: list[str], holdout_per_domain: int
) -> tuple[dict[str, Counter[str]], dict[str, list[dict[str, Any]]]]:
    availability: dict[str, Counter[str]] = defaultdict(Counter)
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    for source in sources:
        domain = source["domain"]
        for row in iter_rows(source):
            if not row["rule_keep"]:
                availability[domain]["rule_rejected"] += 1
                continue
            if collision(row["text"], special_tokens):
                availability[domain]["special_collision"] += 1
                continue
            chars = len(row["text"])
            if row["split"] == "train":
                availability[domain]["train_documents"] += 1
                availability[domain]["train_characters"] += chars
                availability[domain]["train_utf8_bytes"] += len(row["text"].encode("utf-8"))
                continue
            availability[domain]["heldout_available"] += 1
            rank = int(stable_rank(seed + 1, row["document_id"]), 16)
            item = (-rank, row["document_id"], row)
            heap = heaps[domain]
            if len(heap) < holdout_per_domain:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)
    holdout = {
        domain: [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]
        for domain, heap in heaps.items()
    }
    return availability, holdout


def select_domain(
    sources: list[dict[str, Any]],
    domain: str,
    target_chars: int,
    available_chars: int,
    seed: int,
    special_tokens: list[str],
    global_hashes: set[str],
    family_counts: Counter[str],
    max_per_family: int,
    oversample: float,
) -> tuple[list[dict[str, Any]], Counter[str], float]:
    probability = min(1.0, target_chars / available_chars * oversample)
    threshold = int(probability * ((1 << 256) - 1))
    sampled = []
    counters: Counter[str] = Counter()
    for source in sources:
        if source["domain"] != domain:
            continue
        for row in iter_rows(source):
            if not row["rule_keep"] or row["split"] != "train" or collision(row["text"], special_tokens):
                continue
            rank = stable_rank(seed, row["document_id"])
            if int(rank, 16) > threshold:
                continue
            sampled.append((rank, row))
            counters["sampled_documents"] += 1
            counters["sampled_characters"] += len(row["text"])
    sampled.sort(key=lambda item: (item[0], item[1]["document_id"]))
    selected = []
    selected_chars = 0
    for rank, row in sampled:
        if selected_chars >= target_chars:
            break
        content_hash = row["normalized_sha256"]
        if content_hash in global_hashes:
            counters["exact_duplicates_removed"] += 1
            continue
        family = row["family_id"]
        if family_counts[family] >= max_per_family:
            counters["family_cap_removed"] += 1
            continue
        chars = len(row["text"])
        selected.append(
            {
                **{key: row[key] for key in (
                    "document_id", "source_id", "source_revision", "source_locator", "family_id",
                    "domain", "language", "text", "normalized_sha256", "license_status", "split",
                )},
                "tokenizer_sample_rank": rank,
                "character_count": chars,
                "utf8_byte_count": len(row["text"].encode("utf-8")),
            }
        )
        selected_chars += chars
        global_hashes.add(content_hash)
        family_counts[family] += 1
    if selected_chars < target_chars:
        raise ValueError(
            f"{domain}: sampled {counters['sampled_characters']} chars but only selected "
            f"{selected_chars} for target {target_chars}; increase oversample"
        )
    counters["selected_documents"] = len(selected)
    counters["selected_characters"] = selected_chars
    counters["selected_utf8_bytes"] = sum(row["utf8_byte_count"] for row in selected)
    return selected, counters, probability


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd", use_dictionary=True)
    return {
        "file": path.name,
        "rows": len(rows),
        "characters": sum(len(row["text"]) for row in rows),
        "utf8_bytes": sum(len(row["text"].encode("utf-8")) for row in rows),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    targets = config["corpus"]["domain_character_targets"]
    if sum(targets.values()) != config["corpus"]["target_characters"]:
        raise ValueError("domain character targets do not sum to target_characters")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    sources = load_sources(args.normalized_root.resolve(), set(targets))
    verify_shards(sources)
    special_tokens = config["shared"]["special_tokens"]
    availability, holdout = availability_and_holdout(
        sources, config["seed"], special_tokens, config["evaluation"]["heldout_documents_per_domain"]
    )
    for domain, target in targets.items():
        if availability[domain]["train_characters"] < target:
            raise ValueError(
                f"{domain}: only {availability[domain]['train_characters']} train chars for target {target}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    shards = []
    domain_reports = {}
    holdout_reports = []
    global_hashes: set[str] = set()
    family_counts: Counter[str] = Counter()
    try:
        for domain, target in targets.items():
            selected, counters, probability = select_domain(
                sources, domain, target, availability[domain]["train_characters"], config["seed"],
                special_tokens, global_hashes, family_counts,
                config["corpus"]["max_documents_per_family"], args.oversample,
            )
            shard = write_parquet(temporary / "train" / domain / "part-00000.parquet", selected)
            shard["domain"] = domain
            shards.append(shard)
            domain_reports[domain] = {
                "target_characters": target,
                "available": dict(availability[domain]),
                "sampling_probability": probability,
                "selection": dict(counters),
            }
            eval_rows = []
            for row in holdout.get(domain, []):
                eval_rows.append(
                    {
                        **{key: row[key] for key in (
                            "document_id", "source_id", "source_revision", "source_locator", "family_id",
                            "domain", "language", "text", "normalized_sha256", "license_status", "split",
                        )},
                        "character_count": len(row["text"]),
                        "utf8_byte_count": len(row["text"].encode("utf-8")),
                    }
                )
            holdout_report = write_parquet(temporary / "heldout" / domain / "part-00000.parquet", eval_rows)
            holdout_report["domain"] = domain
            holdout_reports.append(holdout_report)

        licenses = Counter()
        for source in sources:
            for row in iter_rows(source):
                if row["rule_keep"] and row["split"] == "train":
                    licenses[row["license_status"]] += 1
        manifest = {
            "version": VERSION,
            "stage": "tokenizer_training_corpus",
            "config": {"file": str(args.config.resolve()), "sha256": sha256_file(args.config)},
            "normalized_root": str(args.normalized_root.resolve()),
            "sources": [
                {
                    "source_id": source["source_id"], "domain": source["domain"],
                    "manifest_sha256": source["manifest_sha256"],
                    "shards": len(source["shards"]),
                }
                for source in sources
            ],
            "selection": {
                "seed": config["seed"],
                "rank": "sha256(seed || document_id)",
                "oversample": args.oversample,
                "exact_deduplication": True,
                "near_deduplication": False,
                "max_documents_per_family": config["corpus"]["max_documents_per_family"],
                "special_token_collision_rejected": True,
            },
            "counts": {
                "train_documents": sum(shard["rows"] for shard in shards),
                "train_characters": sum(shard["characters"] for shard in shards),
                "train_utf8_bytes": sum(shard["utf8_bytes"] for shard in shards),
                "heldout_documents": sum(shard["rows"] for shard in holdout_reports),
                "heldout_characters": sum(shard["characters"] for shard in holdout_reports),
            },
            "domain_reports": domain_reports,
            "train_shards": shards,
            "heldout_shards": holdout_reports,
            "license_gate": {
                "statuses_in_normalized_train_pool": dict(sorted(licenses.items())),
                "training_eligible": bool(licenses) and set(licenses) <= {"approved"},
            },
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oversample", type=float, default=1.35)
    args = parser.parse_args()
    if args.oversample <= 1:
        raise ValueError("oversample must be > 1")
    build(args)


if __name__ == "__main__":
    main()
