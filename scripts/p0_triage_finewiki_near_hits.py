#!/usr/bin/env python3
"""Conservative title-aware overlay for FineWiki's verified near-similar pairs.

Different-title pairs are review-only: near-identical prose may describe
different entities. For same-title pairs, a greedy independent-set policy
keeps one direct Jaccard >=0.82 match for every proposed removal.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import unicodedata

import p0_data_pipeline as core


def normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKC", title).casefold()
    return "".join(char for char in text if char.isalnum())


def selected_titles(root: Path, metadata: dict[str, dict]) -> tuple[dict[str, str], dict[str, str], int]:
    import pyarrow.parquet as pq

    clean_root = root / "clean_rule_v1"
    clean_manifest = json.loads((clean_root / "manifest.json").read_text())
    files = {item["output_file"]: item for item in clean_manifest["files"]}
    if clean_manifest["stage"] != "finewiki_zh_rule_clean_audit_only" or clean_manifest["training_eligible"]:
        raise ValueError("unexpected FineWiki clean stage")
    wanted = defaultdict(dict)
    for document_id, row in metadata.items():
        if row["source"] != "finewiki":
            continue
        pos = row["row"]
        if pos in wanted[row["file"]]:
            raise ValueError("duplicate source pointer")
        wanted[row["file"]][pos] = document_id
    original, normalized = {}, {}
    missing_headings = 0
    for name, positions in wanted.items():
        path = clean_root / name
        if name not in files or core.sha256_file(path) != files[name]["output_sha256"]:
            raise ValueError(f"FineWiki clean checksum mismatch: {path}")
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["document_id", "text"]):
            for row in batch.to_pylist():
                document_id = positions.get(offset)
                if document_id is not None:
                    if document_id != row["document_id"]:
                        raise ValueError("title source pointer mismatch")
                    first = row["text"].splitlines()[0] if row["text"] else ""
                    if not first.startswith("# "):
                        missing_headings += 1
                        first = ""
                    title = first[2:].strip()
                    original[document_id] = title
                    normalized[document_id] = normalize_title(title)
                offset += 1
    if len(normalized) != sum(len(x) for x in wanted.values()):
        raise ValueError("missing pair-member title")
    return original, normalized, missing_headings


def greedy_direct_cover(graph: dict[str, dict[str, float]], metadata: dict[str, dict]) -> dict[str, tuple[str, float]]:
    """Return {removed_id: (directly-matched retained_id, Jaccard)}."""
    split_priority = {"test": 2, "validation": 1, "train": 0}

    def rank(document_id: str):
        row = metadata[document_id]
        return (-split_priority[row["split"]], -row["token_count"], document_id)

    removed = {}
    retained = set()
    for keeper in sorted(graph, key=rank):
        if keeper in removed:
            continue
        retained.add(keeper)
        for other in sorted(graph[keeper]):
            if other not in retained and other not in removed:
                removed[other] = (keeper, graph[keeper][other])
    if retained & set(removed) or len(retained) + len(removed) != len(graph):
        raise ValueError("direct cover partition invalid")
    for document_id, (keeper, similarity) in removed.items():
        if keeper not in retained or graph[document_id][keeper] != similarity:
            raise ValueError("removal has no direct retained match")
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    import pyarrow.parquet as pq

    near_root = args.candidate_root / "audit/full_near_dedup_v1"
    near_path = near_root / "manifest.json"
    near = json.loads(near_path.read_text())
    hits_path = near_root / "near_hits.jsonl"
    signature_path = near_root / "signature_index.parquet"
    if (near["stage"] != "finewiki_priority_full_near_dedup_audit_only" or
            near["training_eligible"] or
            core.sha256_file(hits_path) != near["near_hits_sha256"] or
            core.sha256_file(signature_path) != near["signature_index_sha256"]):
        raise ValueError("near-dedup audit integrity mismatch")
    pairs = []
    ids = set()
    counts = Counter()
    for line in hits_path.read_text().splitlines():
        row = json.loads(line)
        counts[f"group:{row['group']}"] += 1
        if row["group"] == "finewiki_internal":
            pairs.append(row)
            ids.update((row["left_document_id"], row["right_document_id"]))
    if counts["group:finewiki_internal"] != near["exact_check"].get("near_hits:finewiki_internal", 0):
        raise ValueError("near-hit pair count mismatch")
    metadata = {}
    for batch in pq.ParquetFile(signature_path).iter_batches(batch_size=8192, columns=[
        "document_id", "source", "split", "file", "row", "token_count",
    ]):
        for row in batch.to_pylist():
            if row["document_id"] in ids:
                if row["document_id"] in metadata:
                    raise ValueError("duplicate signature index ID")
                metadata[row["document_id"]] = row
    if len(metadata) != len(ids):
        raise ValueError("missing near-hit metadata")
    titles, normalized, missing_headings = selected_titles(args.candidate_root, metadata)
    graph = defaultdict(dict)
    review_pairs = []
    for row in pairs:
        left, right = row["left_document_id"], row["right_document_id"]
        if normalized[left] and normalized[left] == normalized[right]:
            counts["same_title_pairs"] += 1
            graph[left][right] = row["jaccard"]
            graph[right][left] = row["jaccard"]
        else:
            counts["different_or_unknown_title_pairs"] += 1
            review_pairs.append({
                "left_document_id": left, "left_title": titles[left],
                "right_document_id": right, "right_title": titles[right],
                "jaccard": row["jaccard"], "left_split": row["left_split"],
                "right_split": row["right_split"],
            })
    removed = greedy_direct_cover(graph, metadata)
    counts["near_hit_documents"] = len(ids)
    counts["same_title_graph_documents"] = len(graph)
    counts["missing_title_headings"] = missing_headings
    counts["proposed_drop_documents"] = len(removed)
    counts["proposed_drop_tokens"] = sum(metadata[k]["token_count"] for k in removed)
    counts["proposed_remaining_documents"] = near["counts"]["finewiki_documents"] - len(removed)
    counts["proposed_remaining_tokens"] = near["selection_counts"]["selected_tokens"] - counts["proposed_drop_tokens"]
    for document_id, (keeper, _) in removed.items():
        counts[f"drop_split:{metadata[document_id]['split']}"] += 1
        if metadata[document_id]["split"] != metadata[keeper]["split"]:
            counts["cross_split_proposed_drops"] += 1
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finewiki-near-triage-", dir=args.output_dir.parent) as tmp:
        stage = Path(tmp) / args.output_dir.name
        stage.mkdir()
        overlay = stage / "same_title_direct_cover_drops.jsonl"
        with overlay.open("w") as out:
            for document_id in sorted(removed):
                keeper, similarity = removed[document_id]
                out.write(core.canonical_json({
                    "document_id": document_id, "split": metadata[document_id]["split"],
                    "token_count": metadata[document_id]["token_count"],
                    "keeper_document_id": keeper, "keeper_split": metadata[keeper]["split"],
                    "direct_jaccard": similarity,
                    "normalized_title_sha256": hashlib.sha256(normalized[document_id].encode()).hexdigest(),
                }) + "\n")
        review = stage / "different_title_review_pairs.jsonl"
        with review.open("w") as out:
            for row in review_pairs:
                out.write(core.canonical_json(row) + "\n")
        report = {
            "stage": "finewiki_near_hit_title_aware_triage_audit_only",
            "near_manifest_sha256": core.sha256_file(near_path),
            "counts": dict(counts),
            "drop_overlay_sha256": core.sha256_file(overlay),
            "different_title_review_sha256": core.sha256_file(review),
            "policy": "same normalized first-line title only; preserve test then validation then train, then longer token count; every proposed drop directly Jaccard>=0.82 to a retained keeper",
            "limitations": ["title equality does not prove identical facts", "LSH retrieval has incomplete recall",
                            "different-title pairs require separate review", "does not approve corpus for training"],
            "training_eligible": False,
        }
        (stage / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        stage.rename(args.output_dir)
    print(json.dumps({"complete": True, "counts": dict(counts)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
