#!/usr/bin/env python3
"""Verify a copied Wave2 pack on L40 and inventory it with the prior pack.

This publishes only the copied pack, not a combined pack or training run.
No model inference, GPU allocation, or corpus rewrite is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

SPLITS = ("train", "validation", "test")
FIELDS = ("documents", "text_tokens", "stream_tokens", "samples")
COMPATIBLE = ("version", "dtype", "index_dtype", "sequence_length", "stride",
              "vocab_size", "eos_token_id", "tokenizer_sha256")


def sha(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(manifest: dict) -> dict:
    splits = {s: Counter({f: 0 for f in FIELDS}) for s in SPLITS}
    domains = defaultdict(lambda: {s: Counter({f: 0 for f in FIELDS}) for s in SPLITS})
    for item in manifest["shards"]:
        split = item["split"]
        if split not in SPLITS:
            raise ValueError("unknown split")
        values = {"documents": int(item["documents"]), "stream_tokens": int(item["tokens"]),
                  "samples": int(item["samples"])}
        if values["documents"] < 1 or values["stream_tokens"] <= values["documents"] or values["samples"] < 1:
            raise ValueError("invalid shard accounting")
        values["text_tokens"] = values["stream_tokens"] - values["documents"]
        splits[split].update(values)
        domains[item["domain"]][split].update(values)
    total = {f: sum(splits[s][f] for s in SPLITS) for f in FIELDS}
    return {"total": total, "splits": {s: dict(v) for s, v in splits.items()},
            "domains": {d: {s: dict(v) for s, v in sv.items()} for d, sv in sorted(domains.items())},
            "shards": len(manifest["shards"]),
            "one_pass_train_loss_tokens": splits["train"]["samples"] * manifest["stride"]}


def match_ready(stats: dict, ready: dict) -> None:
    for a, b in (("documents", "documents"), ("text_tokens", "text_tokens"),
                 ("stream_tokens", "stream_tokens_with_eos")):
        if stats["total"][a] != ready[b]:
            raise ValueError("V100 ready total mismatch: " + a)
    for split in SPLITS:
        for field in ("documents", "stream_tokens", "samples"):
            if stats["splits"][split][field] != ready["splits"][split][field]:
                raise ValueError("V100 ready split mismatch: " + split + "/" + field)


def combine_stats(packs: dict[str, dict]) -> dict:
    result = {"total": {f: 0 for f in FIELDS},
              "splits": {s: {f: 0 for f in FIELDS} for s in SPLITS}, "domains": {},
              "shards": 0, "one_pass_train_loss_tokens": 0}
    for stats in packs.values():
        if set(result["domains"]) & set(stats["domains"]):
            raise ValueError("domain overlap: inventory must not double count packs")
        result["domains"].update(stats["domains"])
        result["shards"] += stats["shards"]
        result["one_pass_train_loss_tokens"] += stats["one_pass_train_loss_tokens"]
        for f in FIELDS:
            result["total"][f] += stats["total"][f]
            for split in SPLITS:
                result["splits"][split][f] += stats["splits"][split][f]
    n = result["splits"]["train"]["text_tokens"]
    result["natural_train_body_token_fraction"] = {
        d: v["train"]["text_tokens"] / n for d, v in result["domains"].items()}
    return result


def verify_pack(root: Path) -> tuple[dict, dict, int]:
    import numpy as np
    from small_agent.data.sharded_format import load_sharded_manifest
    from small_agent.data.sharded_dataset import ShardedTokenDataset
    manifest, shards = load_sharded_manifest(root, verify_hashes=True)
    if not manifest["license_gate"].get("training_eligible") or manifest["license_gate"].get("distribution_eligible"):
        raise ValueError("pack scope differs from internal training experiment")
    for item, info in zip(manifest["shards"], shards):
        values = np.memmap(info.bin_path, dtype="<u2", mode="r", shape=(info.tokens,))
        eos = 0
        for start in range(0, info.tokens, 4_000_000):
            block = values[start:start + 4_000_000]
            if int(block.max()) >= manifest["vocab_size"]:
                raise ValueError("out-of-range token: " + str(info.bin_path))
            eos += int(np.count_nonzero(block == manifest["eos_token_id"]))
        if eos != item["documents"]:
            raise ValueError("document/EOS mismatch: " + str(info.bin_path))
        del values
    checked = 0
    for split in SPLITS:
        dataset = ShardedTokenDataset(root, split, manifest["sequence_length"], verify_hashes=False)
        offset = 0
        for info in dataset.shards:
            for local in sorted({0, info.samples // 2, info.samples - 1}):
                batch = dataset[offset + local]
                ids = batch["input_ids"].numpy()
                labels = batch["labels"].numpy()
                mask = batch["attention_mask"].numpy()
                valid = int(batch["valid_tokens"])
                if (not np.array_equal(labels[:valid], ids[:valid])
                        or not np.all(labels[valid:] == -100)
                        or not np.all(mask[:valid] == 1) or not np.all(mask[valid:] == 0)):
                    raise ValueError("CPU loader labels/padding mismatch")
                checked += 1
            offset += info.samples
        del dataset
    stats = summarize(manifest)
    print(json.dumps({"verified_pack": str(root), **stats["total"], "loader_samples": checked}), flush=True)
    return manifest, stats, checked


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--prior-pack", required=True, type=Path)
    p.add_argument("--prior-ready", required=True, type=Path)
    p.add_argument("--runtime-code", required=True, type=Path)
    p.add_argument("--tokenizer", required=True, type=Path)
    args = p.parse_args()
    sys.path.insert(0, str(args.runtime_code / "src"))
    root = args.root.resolve()
    incoming, final = root / ".token_pack.incoming", root / "token_pack"
    if incoming.exists() and final.exists():
        raise ValueError("both incoming and published pack exist; inspect before proceeding")
    pack = final if final.exists() else incoming
    if not pack.is_dir():
        raise ValueError("missing transferred pack")
    inputs_path = root / "reports/TRANSFER_INPUTS.json"
    inputs = read(inputs_path)
    if inputs["version"] != "wave2_l40_transfer_inputs_v1" or inputs["destination_root"] != str(root):
        raise ValueError("wrong transfer receipt/destination")
    for item in inputs["files"]:
        relative = Path(item["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe receipt path")
        path = (pack / relative.name) if relative.parts[0] == "token_pack" else root / relative
        if path.is_symlink() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("transfer file failed size/hash check: " + str(path))
    ready = read(root / "reports/V100_READY.json")
    source_path = root / "reports/SOURCE_MANIFEST.json"
    source = read(source_path)
    if ready["status"] != "data_ready_for_internal_pretraining_experiment":
        raise ValueError("invalid V100 ready status")
    if sha(pack / "manifest.json") != ready["pack_manifest_sha256"] or sha(source_path) != ready["source_manifest_sha256"]:
        raise ValueError("ready manifest bindings differ")
    if sha(root / "reports/REUSE_POLICY.json") != source["inputs"]["policy_sha256"]:
        raise ValueError("policy binding differs")
    prior_ready = read(args.prior_ready)
    frozen = {x["path"]: x["sha256"] for x in prior_ready["frozen_files"]}
    if frozen.get(str(args.prior_pack / "manifest.json")) != sha(args.prior_pack / "manifest.json"):
        raise ValueError("prior pack no longer matches its frozen readiness receipt")
    new, new_stats, new_checked = verify_pack(pack)
    prior, prior_stats, prior_checked = verify_pack(args.prior_pack)
    for field in COMPATIBLE:
        if new[field] != prior[field]:
            raise ValueError("incompatible source packs: " + field)
    if new["source_manifest_sha256"] != sha(source_path) or new["tokenizer_sha256"] != sha(args.tokenizer):
        raise ValueError("source/tokenizer binding differs")
    match_ready(new_stats, ready)
    if (new_stats["total"]["documents"] != source["counts"]["accepted_documents"]
            or new_stats["total"]["text_tokens"] != source["counts"]["accepted_text_tokens_excluding_eos"]):
        raise ValueError("source total count differs")
    for split in SPLITS:
        for field in ("documents", "text_tokens"):
            if source["splits"][split][field] != new_stats["splits"][split][field]:
                raise ValueError("source split count differs")
    if (prior_stats["total"]["documents"] != prior_ready["ready_pool_documents"]
            or prior_stats["total"]["text_tokens"] != prior_ready["ready_pool_text_tokens"]
            or prior_stats["splits"]["train"]["stream_tokens"] != prior_ready["ready_pool_train_stream_tokens"]):
        raise ValueError("prior readiness accounting mismatch")
    result = {
        "status": "transferred_and_data_integrity_verified",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "transfer_inputs_sha256": sha(inputs_path), "verifier_sha256": sha(Path(__file__)),
        "new_pack": {"path": str(final), "manifest_sha256": sha(pack / "manifest.json"), **new_stats},
        "prior_pack": {"path": str(args.prior_pack), "manifest_sha256": sha(args.prior_pack / "manifest.json"), **prior_stats},
        "combined_inventory": combine_stats({"prior": prior_stats, "wave2": new_stats}),
        "checks": {"all_transfer_files_hash_verified": True,
                   "both_packs_all_bin_idx_hashes_verified": True,
                   "all_index_records_verified": True, "all_token_ranges_and_eos_counts_verified": True,
                   "cpu_loader_samples_checked": new_checked + prior_checked,
                   "same_frozen_tokenizer": True, "v100_ready_accounting_matches": True},
        "corpus_downloaded_to_mac": False, "new_pack_bytes": sum(s["bin_bytes"] + s["idx_bytes"] for s in new["shards"]),
        "training_started_by_this_job": False, "combined_pack_created": False,
        "training_configuration_modified": False, "distribution_eligible": False,
        "quality_accuracy_certified": False,
        "limitations": ready["limitations"] + [
            "Combined inventory is a sum of two separate compatible packs, not a training mixture or a merged pack.",
            "Cross-pool exact-body/page-family isolation is inherited from V100 freeze; bodies remain on V100.",
            "Natural domain proportions are inventory proportions, not a recommended training sampling policy."],
    }
    # Publish only after all checks, with no overwrite of an existing final pack.
    if pack == incoming:
        if final.exists():
            raise FileExistsError(final)
        incoming.rename(final)
    out = root / "reports/L40_TRANSFER_STATS.json"
    fd, temporary = tempfile.mkstemp(prefix=".transfer-stats-", dir=out.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temporary, out)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({"report": str(out), "status": result["status"],
                      "combined_total": result["combined_inventory"]["total"],
                      "combined_splits": result["combined_inventory"]["splits"]}), flush=True)


if __name__ == "__main__":
    main()
