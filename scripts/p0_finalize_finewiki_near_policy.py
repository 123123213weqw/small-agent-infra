#!/usr/bin/env python3
"""Strict audit overlay: quarantine ambiguous titles, directly cover same-title hits.

Every observed near-hit edge has at least one excluded endpoint. This is an
audit candidate selection, not training approval or exhaustive deduplication.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile

import p0_data_pipeline as core
from p0_triage_finewiki_near_hits import greedy_direct_cover


def edge_key(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("self near-hit edge")
    return tuple(sorted((left, right)))


def build_policy(near_hits: list[dict], different_pairs: list[dict],
                 metadata: dict[str, dict]) -> tuple[dict[str, dict], dict]:
    different = {edge_key(x["left_document_id"], x["right_document_id"])
                 for x in different_pairs}
    quarantine = {document_id for pair in different for document_id in pair}
    same_graph = defaultdict(dict)
    counts = Counter()
    hit_keys = set()
    for hit in near_hits:
        if hit["group"] != "finewiki_internal":
            continue
        left, right = hit["left_document_id"], hit["right_document_id"]
        key = edge_key(left, right)
        if key in hit_keys:
            raise ValueError("duplicate near-hit pair")
        hit_keys.add(key)
        if key in different:
            counts["different_title_pairs"] += 1
        else:
            counts["same_title_pairs"] += 1
            if left not in quarantine and right not in quarantine:
                same_graph[left][right] = hit["jaccard"]
                same_graph[right][left] = hit["jaccard"]
    if different - hit_keys or counts["different_title_pairs"] != len(different):
        raise ValueError("title-review pair mismatch against verified near hits")
    if not quarantine <= metadata.keys():
        raise ValueError("missing quarantine metadata")
    direct_drops = greedy_direct_cover(same_graph, metadata)
    if quarantine & direct_drops.keys():
        raise ValueError("quarantined document also marked direct-drop")
    excluded = {}
    for document_id in quarantine:
        excluded[document_id] = {"reason": "different_title_near_pair_quarantine",
                                 "keeper_document_id": None, "direct_jaccard": None}
    for document_id, (keeper, score) in direct_drops.items():
        if keeper in quarantine or keeper in excluded:
            raise ValueError("direct keeper is excluded")
        excluded[document_id] = {"reason": "same_title_direct_near_duplicate",
                                 "keeper_document_id": keeper, "direct_jaccard": score}
    for hit in near_hits:
        if hit["group"] == "finewiki_internal" and (hit["left_document_id"] not in excluded and
                                                      hit["right_document_id"] not in excluded):
            raise ValueError("observed near-hit edge remains among candidates")
    counts["quarantine_documents"] = len(quarantine)
    counts["direct_duplicate_documents"] = len(direct_drops)
    counts["excluded_documents"] = len(excluded)
    counts["excluded_tokens"] = sum(metadata[k]["token_count"] for k in excluded)
    for document_id, detail in excluded.items():
        counts[f"excluded_split:{metadata[document_id]['split']}"] += 1
        counts[f"excluded_tokens_by_reason:{detail['reason']}"] += metadata[document_id]["token_count"]
    return excluded, dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    import pyarrow.parquet as pq

    near_root = args.candidate_root / "audit/full_near_dedup_v1"
    triage_root = args.candidate_root / "audit/near_hit_title_triage_v1"
    near_path, triage_path = near_root / "manifest.json", triage_root / "manifest.json"
    near, triage = json.loads(near_path.read_text()), json.loads(triage_path.read_text())
    hits_path = near_root / "near_hits.jsonl"
    different_path = triage_root / "different_title_review_pairs.jsonl"
    signature_path = near_root / "signature_index.parquet"
    if (near["stage"] != "finewiki_priority_full_near_dedup_audit_only" or
            triage["stage"] != "finewiki_near_hit_title_aware_triage_audit_only" or
            near["training_eligible"] or triage["training_eligible"] or
            triage["near_manifest_sha256"] != core.sha256_file(near_path) or
            core.sha256_file(hits_path) != near["near_hits_sha256"] or
            core.sha256_file(signature_path) != near["signature_index_sha256"] or
            core.sha256_file(different_path) != triage["different_title_review_sha256"]):
        raise ValueError("near-hit or title-triage provenance mismatch")
    near_hits = [json.loads(x) for x in hits_path.read_text().splitlines()]
    different_pairs = [json.loads(x) for x in different_path.read_text().splitlines()]
    wanted = {hit[k] for hit in near_hits if hit["group"] == "finewiki_internal"
              for k in ("left_document_id", "right_document_id")}
    metadata = {}
    for batch in pq.ParquetFile(signature_path).iter_batches(batch_size=8192, columns=[
        "document_id", "source", "split", "token_count",
    ]):
        for row in batch.to_pylist():
            if row["document_id"] in wanted:
                if row["document_id"] in metadata or row["source"] != "finewiki":
                    raise ValueError("duplicate or non-FineWiki near-hit metadata")
                metadata[row["document_id"]] = row
    if len(metadata) != len(wanted):
        raise ValueError("missing near-hit metadata")
    excluded, counts = build_policy(near_hits, different_pairs, metadata)
    if (counts["same_title_pairs"] != triage["counts"]["same_title_pairs"] or
            counts["different_title_pairs"] != triage["counts"]["different_or_unknown_title_pairs"]):
        raise ValueError("title pair counts drifted")
    counts["remaining_documents"] = near["counts"]["finewiki_documents"] - counts["excluded_documents"]
    counts["remaining_tokens"] = near["selection_counts"]["selected_tokens"] - counts["excluded_tokens"]
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finewiki-near-policy-", dir=args.output_dir.parent) as tmp:
        stage = Path(tmp) / args.output_dir.name
        stage.mkdir()
        output = stage / "exclusion_overlay.jsonl"
        with output.open("w") as out:
            for document_id in sorted(excluded):
                out.write(core.canonical_json({
                    "document_id": document_id, "split": metadata[document_id]["split"],
                    "token_count": metadata[document_id]["token_count"],
                    **excluded[document_id],
                }) + "\n")
        report = {
            "stage": "finewiki_strict_near_policy_overlay_audit_only",
            "near_manifest_sha256": core.sha256_file(near_path),
            "triage_manifest_sha256": core.sha256_file(triage_path),
            "counts": counts, "exclusion_overlay_sha256": core.sha256_file(output),
            "policy": "quarantine both endpoints of all different-title near-hit pairs; on remaining same-title graph use direct-match greedy cover, preferring test then validation then train and longer records",
            "guarantee": "no observed LSH-retrieved Jaccard>=0.82 FineWiki internal edge has both endpoints in the candidate remainder",
            "limitations": ["LSH retrieval is approximate", "different-title quarantine may discard useful unique facts",
                            "does not address other sources or extraction defects beyond prior overlays"],
            "training_eligible": False,
        }
        (stage / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        stage.rename(args.output_dir)
    print(json.dumps({"complete": True, "counts": counts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
