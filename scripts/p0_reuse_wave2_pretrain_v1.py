#!/usr/bin/env python3
"""Reuse frozen Wave2 candidates for a versioned, CPU-only pretraining experiment.

No API calls, GPU allocation, text rewriting, training launch or old-gate mutation.
Existing candidate/near-dedup/evaluation receipts are checked, not invented.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p0_data_pipeline import assign_split, build_family_id, normalize_text, normalize_url_for_family

SPLITS = ("train", "validation", "test")
RESERVED = ("<pad>", "<bos>", "<eos>", "<unk>", "<|system|>", "<|user|>",
            "<|assistant|>", "<|tool|>", "<|tool_result|>", "<|end|>",
            "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
            "<|memory|>", "<|memory_end|>")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def body_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("x", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def valid_url(value: str) -> bool:
    try:
        p = urlsplit(value)
        return p.scheme.lower() in {"http", "https"} and bool(p.hostname) and not p.username
    except (TypeError, ValueError):
        return False


def page_alias(url: str) -> str:
    """Conservative URL page grouping, joining HTTP/HTTPS in addition to old families."""
    p = urlsplit(normalize_url_for_family(url))
    host = (p.hostname or "").lower().removeprefix("www.")
    port = p.port
    if port and port not in {80, 443}:
        host += f":{port}"
    path = p.path.rstrip("/") or "/"
    return body_sha("page_alias\0" + host + path)


def hard_reason(text: str, url: str, policy: dict) -> str | None:
    h = policy["hard_filter"]
    if not valid_url(url):
        return "missing_or_invalid_source_url"
    try:
        page_alias(url)
    except ValueError:
        return "missing_or_invalid_source_url"
    if len(text.strip()) < h["min_chars"]:
        return "missing_or_tiny_body"
    if h["reject_nul"] and "\0" in text:
        return "nul_encoding_artifact"
    if text.count("\ufffd") / max(1, len(text)) > h["max_replacement_fraction"]:
        return "severe_encoding_damage"
    if h["reject_reserved_token_literals"] and any(x in text for x in RESERVED):
        return "reserved_training_token_literal"
    return None


def safe_filename(name: str) -> str:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"unsafe filename {name!r}")
    return name


def verify_file(path: Path, expected_sha: str, expected_bytes: int | None = None) -> None:
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"size mismatch: {path}")
    if sha(path) != expected_sha:
        raise ValueError(f"SHA mismatch: {path}")


def prior_sets(priors: list[Path]) -> tuple[set, set, set, list]:
    import pyarrow.parquet as pq
    bodies, families, pages, receipts = set(), set(), set(), []
    for source in priors:
        m = read_json(source / "manifest.json")
        if m.get("stage") != "accepted_real" or not m.get("license_gate", {}).get("training_eligible"):
            raise ValueError("prior source not approved")
        count = 0
        for s in m["shards"]:
            path = source / safe_filename(s["domain"]) / safe_filename(s["file"])
            verify_file(path, s["sha256"], s["bytes"])
            q = pq.ParquetFile(path)
            cols = ["text", "family_id", "split"]
            cols += ["url"] if "url" in q.schema_arrow.names else ["source_locator"]
            for b in q.iter_batches(batch_size=256, columns=cols):
                for row in b.to_pylist():
                    if row["split"] not in SPLITS:
                        raise ValueError("unassigned prior split")
                    bodies.add(body_sha(row["text"]))
                    bodies.add(body_sha(normalize_text(row["text"])))
                    families.add(row["family_id"])
                    url = row.get("url")
                    if not url:
                        loc = row.get("source_locator", "")
                        for part in loc.split("|"):
                            if part.startswith("url="):
                                url = part[4:]
                                break
                    if url and valid_url(url):
                        try:
                            pages.add(page_alias(url))
                        except ValueError:
                            pass
                    count += 1
        if count != m["counts"]["accepted_documents"]:
            raise ValueError("prior source count mismatch")
        receipts.append({"path": str(source), "manifest_sha256": sha(source / "manifest.json"), "documents": count})
    return bodies, families, pages, receipts


def lineage(wave: Path, policy: dict) -> tuple[dict, dict]:
    summary_path = wave / "audit/acceptance_v1/SUMMARY.json"
    summary = read_json(summary_path)
    if (summary["documents"] != policy["expected_candidate_documents"]
            or summary["exact_text_tokens_excluding_eos"] != policy["expected_candidate_text_tokens"]
            or summary["tokenizer_sha256"] != policy["tokenizer_sha256"]):
        raise ValueError("candidate population/tokenizer changed")
    upstream = read_json(wave / "manifest.json")
    if (upstream["dataset"] != policy["source_dataset"]
            or upstream["revision"] != policy["source_revision"]
            or upstream["upstream_license_label"] != policy["source_license_label"]):
        raise ValueError("source identity mismatch")
    raw_hashes = {Path(item["path"]).name: item["sha256"] for item in upstream["items"]}
    if {x["file"]: x["raw_sha256"] for x in summary["shards"]} != raw_hashes:
        raise ValueError("candidate/raw provenance mismatch")
    recon = wave / "audit/acceptance_v1/reports/RECONCILIATION.json"
    verify_file(recon, summary["reconciliation_sha256"])
    r = read_json(recon)
    if r["selected_documents"] != summary["documents"]:
        raise ValueError("reconciliation population changed")
    near_root = wave / "audit/near_dedup_v1/verified_v1"
    near = read_json(near_root / "SUMMARY.json")
    ev_root = wave / "audit/eval_screen_v1"
    ev = read_json(ev_root / "SUMMARY.json")
    if r["inherited_near_dedup"] != near:
        raise ValueError("inherited near-dedup receipt changed")
    if (r["inherited_eval_screen"]["hit_documents_quarantined"] != ev["hit_documents_quarantined"]
            or r["inherited_eval_screen"]["documents_scanned"] != ev["documents_scanned"]):
        raise ValueError("inherited evaluation receipt changed")
    for name, digest in near["outputs_sha256"].items():
        verify_file(near_root / safe_filename(name), digest)
    verify_file(ev_root / "quarantine_index.parquet", ev["quarantine_index_sha256"])
    references = []
    for ref in ev["reference_files"].values():
        path = wave / "audit/eval_reference_v1" / ref["relative_path"]
        verify_file(path, ref["sha256"], ref["bytes"])
        references.append({"path": str(path), "sha256": ref["sha256"]})
    return summary, {
        "candidate_summary_sha256": sha(summary_path),
        "upstream_manifest_sha256": sha(wave / "manifest.json"),
        "reconciliation_sha256": sha(recon),
        "near_summary_sha256": sha(near_root / "SUMMARY.json"),
        "near_exclusion_sha256": near["outputs_sha256"]["wave2_exclusion_index.parquet"],
        "eval_summary_sha256": sha(ev_root / "SUMMARY.json"),
        "eval_quarantine_sha256": ev["quarantine_index_sha256"],
        "evaluation_reference_files": references,
    }


def freeze(args) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    policy = read_json(args.policy)
    if args.output.exists():
        raise FileExistsError(args.output)
    summary, evidence = lineage(args.wave_root, policy)
    tok_path = args.tokenizer / "tokenizer.json"
    verify_file(tok_path, policy["tokenizer_sha256"])
    tok = read_json(args.tokenizer / "manifest.json")
    if tok["stage"] != "frozen_tokenizer":
        raise ValueError("unfrozen tokenizer")
    seen, prior_families, prior_pages, prior_receipts = prior_sets(args.prior)
    inherited_exclusions = set()
    for path in (args.wave_root / "audit/near_dedup_v1/verified_v1/wave2_exclusion_index.parquet",
                 args.wave_root / "audit/eval_screen_v1/quarantine_index.parquet"):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["source_file", "raw_row"]):
            for row in batch.to_pylist():
                inherited_exclusions.add((Path(row["source_file"]).name, row["raw_row"]))
    families = {}
    counts, reasons, discarded_tokens = Counter(), Counter(), Counter()
    split_docs, split_tokens, routes = Counter(), Counter(), Counter()
    selected_ids = set()
    candidate_ids = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.tmp-", dir=args.output.parent))
    schema = pa.schema([
        ("document_id", pa.string()), ("text", pa.large_string()),
        ("normalized_sha256", pa.string()), ("family_id", pa.string()),
        ("page_alias", pa.string()), ("domain", pa.string()), ("split", pa.string()),
        ("source_id", pa.string()), ("source_revision", pa.string()),
        ("source_locator", pa.string()), ("source_file", pa.string()), ("raw_row", pa.int64()),
        ("upstream_dump", pa.string()), ("upstream_file_path", pa.string()),
        ("upstream_score", pa.float64()),
        ("license_status", pa.string()), ("review_route", pa.string()),
        ("own_text_tokens", pa.int64()),
    ])
    try:
        domain = tmp / policy["domain"]
        domain.mkdir()
        shards = []
        with (tmp / "accepted_index.jsonl").open("x", encoding="utf-8") as index, (tmp / "excluded_index.jsonl").open("x", encoding="utf-8") as excluded:
            for item in summary["shards"]:
                name = safe_filename(item["file"])
                path = args.wave_root / "audit/acceptance_v1/candidates" / name
                verify_file(path, item["output_sha256"], item["output_bytes"])
                dest = domain / name
                writer = pq.ParquetWriter(dest, schema, compression="zstd")
                scanned = candidate_tokens = accepted = 0
                try:
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                        output_rows = []
                        for row in batch.to_pylist():
                            text, did, url = row["text"], row["document_id"], row["url"]
                            nt = row["exact_text_tokens"]
                            if did != body_sha(text) or did in candidate_ids or not isinstance(nt, int) or nt <= 0:
                                raise ValueError("candidate body/identity/count mismatch")
                            if (row["dataset_revision"] != policy["source_revision"]
                                    or row["upstream_license_label"] != policy["source_license_label"]
                                    or row["source_file"] != name
                                    or row["split"] != "unassigned" or row["training_approved"] is not False
                                    or row["review_route"] not in {"qwen_candidate", "deepseek_long_candidate"}):
                                raise ValueError("candidate provenance/state mismatch")
                            if (name, row["raw_row"]) in inherited_exclusions:
                                raise ValueError("candidate resurrects an inherited duplicate/eval quarantine")
                            candidate_ids.add(did)
                            scanned += 1
                            candidate_tokens += nt
                            canonical_body = body_sha(normalize_text(text))
                            reason = hard_reason(text, url, policy)
                            family = alias = None
                            if reason is None:
                                family = build_family_id({"url": url}, ["url"], "fineweb_edu_new", url)
                                alias = page_alias(url)
                                if did in seen or canonical_body in seen:
                                    reason = "exact_body_duplicate_prior_or_new"
                                elif family in prior_families or alias in prior_pages:
                                    reason = "page_family_overlaps_prior_pool"
                            if reason:
                                reasons[reason] += 1
                                discarded_tokens[reason] += nt
                                excluded.write(json.dumps({"document_id": did, "source_file": name,
                                                           "raw_row": row["raw_row"], "text_tokens": nt,
                                                           "reason": reason}, sort_keys=True) + "\n")
                                continue
                            split = families.setdefault(alias, assign_split(alias, policy["split_seed"],
                                policy["train_fraction"], policy["validation_fraction"]))
                            seen.update((did, canonical_body))
                            selected_ids.add(did)
                            result = {
                                "document_id": did, "text": text, "normalized_sha256": did,
                                "family_id": family, "page_alias": alias, "domain": policy["domain"],
                                "split": split, "source_id": row["source_id"],
                                "source_revision": row["dataset_revision"], "source_locator": url,
                                "source_file": name, "raw_row": row["raw_row"],
                                "upstream_dump": row["upstream_dump"], "upstream_file_path": row["upstream_file_path"],
                                "upstream_score": row["upstream_score"],
                                "license_status": "approved_internal_experiment_source_level",
                                "review_route": row["review_route"], "own_text_tokens": nt,
                            }
                            output_rows.append(result)
                            index.write(json.dumps({k: result[k] for k in (
                                "document_id", "family_id", "page_alias", "domain", "split", "normalized_sha256",
                                "own_text_tokens", "source_file", "raw_row", "review_route")}, sort_keys=True) + "\n")
                            counts["accepted_documents"] += 1
                            counts["accepted_text_tokens_excluding_eos"] += nt
                            split_docs[split] += 1
                            split_tokens[split] += nt
                            routes[row["review_route"]] += 1
                            accepted += 1
                        if output_rows:
                            writer.write_table(pa.Table.from_pylist(output_rows, schema=schema))
                finally:
                    writer.close()
                if scanned != item["documents"] or candidate_tokens != item["text_tokens_excluding_eos"]:
                    raise ValueError("candidate shard accounting mismatch")
                shards.append({"domain": policy["domain"], "file": name, "rows": accepted,
                               "bytes": dest.stat().st_size, "sha256": sha(dest)})
                print(json.dumps({"stage": "reuse_freeze", "file": name,
                                  "scanned": scanned, "accepted": accepted,
                                  "total_accepted": counts["accepted_documents"]}), flush=True)
        if len(candidate_ids) != summary["documents"]:
            raise ValueError("candidate coverage mismatch")
        if counts["accepted_text_tokens_excluding_eos"] + sum(discarded_tokens.values()) != summary["exact_text_tokens_excluding_eos"]:
            raise ValueError("token accounting mismatch")
        if counts["accepted_documents"] + sum(reasons.values()) != summary["documents"]:
            raise ValueError("document accounting mismatch")
        if not all(split_docs[s] for s in SPLITS):
            raise ValueError("empty split")
        counts["candidate_documents"] = summary["documents"]
        counts["accepted_tokens_with_one_eos_each"] = counts["accepted_text_tokens_excluding_eos"] + counts["accepted_documents"]
        manifest = {
            "stage": "accepted_real", "version": args.output.name,
            "purpose": policy["purpose"], "generated_at": datetime.now(timezone.utc).isoformat(),
            "license_gate": {"training_eligible": True, "distribution_eligible": False,
                             "scope": "internal_pretraining_experiment_only", "source_license_label": policy["source_license_label"],
                             "basis": "reuse prior internal FineWeb source policy; underlying page rights not individually certified"},
            "quality_gate": {"type": "reuse_existing_selected_candidates_plus_hard_integrity_filter",
                             "unanimous_dual_judgment_required": False,
                             "independent_factual_accuracy_certified": False,
                             "old_strict_gate_unchanged": True},
            "source_dataset": policy["source_dataset"], "source_revision": policy["source_revision"],
            "tokenizer": {"vocab_size": tok["vocab_size"], "tokenizer_json_sha256": policy["tokenizer_sha256"]},
            "inputs": {**evidence, "policy_sha256": sha(args.policy), "script_sha256": sha(Path(__file__)),
                       "tokenizer_manifest_sha256": sha(args.tokenizer / "manifest.json"), "prior_sources": prior_receipts,
                       "candidate_shards": [{"file": x["file"], "sha256": x["output_sha256"]} for x in summary["shards"]]},
            "counts": dict(counts), "excluded_documents_by_reason": dict(reasons),
            "excluded_text_tokens_by_reason": dict(discarded_tokens), "review_routes": dict(routes),
            "splits": {s: {"documents": split_docs[s], "text_tokens": split_tokens[s]} for s in SPLITS},
            "domains": {policy["domain"]: {"documents": counts["accepted_documents"],
                                           "text_tokens": counts["accepted_text_tokens_excluding_eos"]}},
            "family_split_conflicts": 0, "shards": shards,
            "accepted_index": {"file": "accepted_index.jsonl", "sha256": sha(tmp / "accepted_index.jsonl")},
            "excluded_index": {"file": "excluded_index.jsonl", "sha256": sha(tmp / "excluded_index.jsonl")},
            "new_api_calls": 0, "training_started": False, "limitations": policy["notes"],
        }
        write_json(tmp / "manifest.json", manifest)
        tmp.rename(args.output)
        return {"output": str(args.output), "counts": dict(counts), "splits": manifest["splits"], "excluded": dict(reasons)}
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def pack(args) -> dict:
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from small_agent.data.sharded_pack import TokenizedDocument, pack_sharded
    m = read_json(args.source / "manifest.json")
    if m["stage"] != "accepted_real" or m["purpose"] != "internal_pretraining_experiment_not_per_document_fact_certification":
        raise ValueError("not a reuse experiment source")
    t = read_json(args.tokenizer / "manifest.json")
    tok_path = args.tokenizer / "tokenizer.json"
    verify_file(tok_path, m["tokenizer"]["tokenizer_json_sha256"])
    if t["stage"] != "frozen_tokenizer" or t["artifacts"]["tokenizer.json"]["sha256"] != sha(tok_path):
        raise ValueError("tokenizer manifest changed")
    tokenizer = Tokenizer.from_file(str(tok_path))
    special_ids = set(t["special_token_ids"].values())
    stats = Counter()

    def documents():
        for shard in m["shards"]:
            path = args.source / shard["domain"] / shard["file"]
            verify_file(path, shard["sha256"], shard["bytes"])
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                rows = batch.to_pylist()
                enc = tokenizer.encode_batch([r["text"] for r in rows], add_special_tokens=False)
                dec = tokenizer.decode_batch([e.ids for e in enc], skip_special_tokens=False)
                for r, e, text in zip(rows, enc, dec):
                    if (text != r["text"] or len(e.ids) != r["own_text_tokens"]
                            or body_sha(r["text"]) != r["document_id"] or special_ids.intersection(e.ids)):
                        raise ValueError("token count/roundtrip/body/special-token failure")
                    stats["documents"] += 1
                    stats["text_tokens"] += len(e.ids)
                    yield TokenizedDocument(r["document_id"], r["split"], r["domain"], e.ids)
            print(json.dumps({"stage": "reuse_pack", "source_file": shard["file"], **stats}), flush=True)
        if stats["documents"] != m["counts"]["accepted_documents"] or stats["text_tokens"] != m["counts"]["accepted_text_tokens_excluding_eos"]:
            raise ValueError("packed source accounting mismatch")

    mixture = {domain: 1.0 / len(m["domains"]) for domain in m["domains"]}
    result = pack_sharded(documents(), args.output, sequence_length=2048, target_shard_tokens=32_000_000,
        vocab_size=t["vocab_size"], eos_token_id=t["special_token_ids"]["<eos>"],
        tokenizer_sha256=sha(tok_path), mixture=mixture, license_gate=m["license_gate"],
        source_manifest_sha256=sha(args.source / "manifest.json"))
    return {"output": str(args.output), "shards": len(result["shards"]), **stats}


def check_source(args) -> dict:
    m = read_json(args.source / "manifest.json")
    if m.get("stage") != "accepted_real" or m.get("purpose") != "internal_pretraining_experiment_not_per_document_fact_certification":
        raise ValueError("unexpected reuse source")
    rows = 0
    for item in m["shards"]:
        verify_file(args.source / safe_filename(item["domain"]) / safe_filename(item["file"]), item["sha256"], item["bytes"])
        rows += item["rows"]
    if rows != m["counts"]["accepted_documents"]:
        raise ValueError("source shard row counts mismatch")
    for kind in ("accepted_index", "excluded_index"):
        verify_file(args.source / m[kind]["file"], m[kind]["sha256"])
    return {"status": "source_files_hash_verified", "documents": rows}


def check_pack(args) -> dict:
    from small_agent.data.sharded_format import load_sharded_manifest
    m, _ = load_sharded_manifest(args.pack)
    return {"status": "pack_files_hash_and_index_verified", "shards": len(m["shards"])}


def verify(args) -> dict:
    import numpy as np
    from small_agent.data.sharded_format import load_sharded_manifest
    from small_agent.data import ShardedTokenDataset
    source = read_json(args.source / "manifest.json")
    m, shards = load_sharded_manifest(args.pack)
    if m["source_manifest_sha256"] != sha(args.source / "manifest.json"):
        raise ValueError("pack/source mismatch")
    verify_file(args.source / "accepted_index.jsonl", source["accepted_index"]["sha256"])
    verify_file(args.source / "excluded_index.jsonl", source["excluded_index"]["sha256"])
    families, aliases, seen = {}, {}, set()
    with (args.source / "accepted_index.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["document_id"] in seen:
                raise ValueError("duplicate frozen ID")
            seen.add(row["document_id"])
            for key, store in (("family_id", families), ("page_alias", aliases)):
                old = store.setdefault(row[key], row["split"])
                if old != row["split"]:
                    raise ValueError("family crosses splits")
    if len(seen) != source["counts"]["accepted_documents"]:
        raise ValueError("accepted index count mismatch")
    eos = docs = tokens = checked_samples = 0
    by_split = defaultdict(Counter)
    for entry in m["shards"]:
        values = np.memmap(args.pack / entry["bin"], dtype="<u2", mode="r")
        if int(values.max()) >= m["vocab_size"]:
            raise ValueError("token out of vocabulary")
        count = int(np.count_nonzero(values == m["eos_token_id"]))
        if count != entry["documents"]:
            raise ValueError("EOS/document count mismatch")
        eos += count
        docs += entry["documents"]
        tokens += entry["tokens"]
        by_split[entry["split"]]["documents"] += entry["documents"]
        by_split[entry["split"]]["stream_tokens"] += entry["tokens"]
        by_split[entry["split"]]["samples"] += entry["samples"]
    if docs != len(seen) or tokens != source["counts"]["accepted_tokens_with_one_eos_each"]:
        raise ValueError("pack total mismatch")
    for split in SPLITS:
        expected = source["splits"][split]
        if (by_split[split]["documents"] != expected["documents"]
                or by_split[split]["stream_tokens"] != expected["documents"] + expected["text_tokens"]):
            raise ValueError("split token accounting mismatch")
        dataset = ShardedTokenDataset(args.pack, split, 2048, verify_hashes=False)
        offset = 0
        for shard in dataset.shards:
            for ref in sorted({offset, offset + shard.samples // 2, offset + shard.samples - 1}):
                item = dataset[ref]
                n = int(item["valid_tokens"])
                if not np.array_equal(item["input_ids"][:n].numpy(), item["labels"][:n].numpy()):
                    raise ValueError("labels shifted in loader")
                if n < 2048 and not bool((item["labels"][n:] == -100).all()):
                    raise ValueError("padding labels not masked")
                checked_samples += 1
            offset += shard.samples
    result = {
        "status": "data_ready_for_internal_pretraining_experiment",
        "source_manifest_sha256": sha(args.source / "manifest.json"), "pack_manifest_sha256": sha(args.pack / "manifest.json"),
        "documents": docs, "text_tokens": tokens - eos, "stream_tokens_with_eos": tokens,
        "splits": {s: dict(by_split[s]) for s in SPLITS}, "all_shard_hashes_and_indexes_verified": True,
        "all_token_ranges_and_eos_counts_verified": True, "cpu_loader_samples_checked": checked_samples,
        "family_split_conflicts": 0, "new_api_calls": 0, "training_started": False,
        "quality_accuracy_certified": False, "distribution_eligible": False, "limitations": source["limitations"],
    }
    write_json(args.output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--wave-root", type=Path, required=True)
    f.add_argument("--policy", type=Path, required=True)
    f.add_argument("--prior", type=Path, action="append", required=True)
    f.add_argument("--tokenizer", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    k = sub.add_parser("pack")
    k.add_argument("--source", type=Path, required=True)
    k.add_argument("--tokenizer", type=Path, required=True)
    k.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--source", type=Path, required=True)
    v.add_argument("--pack", type=Path, required=True)
    v.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("check-source")
    c.add_argument("--source", type=Path, required=True)
    c = sub.add_parser("check-pack")
    c.add_argument("--pack", type=Path, required=True)
    args = p.parse_args()
    result = {"freeze": freeze, "pack": pack, "verify": verify,
              "check-source": check_source, "check-pack": check_pack}[args.command](args)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
