#!/usr/bin/env python3
"""Combine compatible immutable token packs without duplicating token storage."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from small_agent.data.sharded_format import load_sharded_manifest, sha256_file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--add", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--add-weight", type=float, default=0.20)
    p.add_argument(
        "--balance-by-train-samples", action="store_true",
        help="set integer mixture weights to train sample counts for a one-pass schedule",
    )
    a = p.parse_args()
    if not 0 < a.add_weight < 1:
        raise ValueError("add weight must be strictly between zero and one")
    if a.output.exists():
        raise FileExistsError(a.output)
    base, _ = load_sharded_manifest(a.base)
    add, _ = load_sharded_manifest(a.add)
    compatible = (
        "version", "dtype", "index_dtype", "sequence_length", "stride",
        "vocab_size", "eos_token_id", "tokenizer_sha256",
    )
    for field in compatible:
        if base[field] != add[field]:
            raise ValueError(f"incompatible packs: {field}")
    if not base["license_gate"].get("training_eligible") or not add["license_gate"].get("training_eligible"):
        raise ValueError("both source packs must be training eligible")
    if base["stage"] != "approved_sharded_token_stream" or add["stage"] != "approved_sharded_token_stream":
        raise ValueError("both sources must be approved token packs")
    if set(base["mixture"]) & set(add["mixture"]):
        raise ValueError("source domains must be disjoint for explicit mixture control")
    mixture = {k: float(v) * (1-a.add_weight) for k,v in base["mixture"].items()}
    mixture.update({k: float(v) * a.add_weight for k,v in add["mixture"].items()})
    total = sum(mixture.values())
    mixture = {k: v/total for k,v in mixture.items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="." + a.output.name + ".tmp-", dir=a.output.parent))
    shards = []
    try:
        for source_name, root, source in (("base", a.base, base), ("add", a.add, add)):
            for old in source["shards"]:
                entry = dict(old)
                new_id = len(shards)
                for suffix in ("bin", "idx"):
                    new_name = f"{entry['split']}-{new_id:05d}.{suffix}"
                    os.link(root / old[suffix], temp / new_name)
                    entry[suffix] = new_name
                entry["shard_id"] = new_id
                entry["source_pack"] = source_name
                shards.append(entry)
        if a.balance_by_train_samples:
            sample_counts = Counter()
            for entry in shards:
                if entry["split"] == "train":
                    sample_counts[entry["domain"]] += int(entry["samples"])
            mixture = dict(sorted(sample_counts.items()))
        manifest = {k: base[k] for k in compatible}
        manifest.update({
            "stage": "approved_sharded_token_stream",
            "license_gate": {
                "training_eligible": True,
                "distribution_eligible": False,
                "scope": "internal_only",
                "restriction": "FineWiki source and derived weights require release review",
            },
            "source_manifest_sha256": hashlib.sha256((
                sha256_file(a.base / "manifest.json") + "\n" +
                sha256_file(a.add / "manifest.json")
            ).encode()).hexdigest(),
            "source_packs": {
                "base": {"path": str(a.base.resolve()), "manifest_sha256": sha256_file(a.base / "manifest.json")},
                "add": {"path": str(a.add.resolve()), "manifest_sha256": sha256_file(a.add / "manifest.json")},
            },
            "mixture": mixture, "shards": shards,
            "mixture_basis": "train_sample_counts" if a.balance_by_train_samples else "requested_add_weight",
            "storage": "hardlinks_to_immutable_source_packs",
        })
        (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        load_sharded_manifest(temp)
        temp.rename(a.output)
        print(json.dumps({
            "output": str(a.output), "shards": len(shards), "mixture": mixture,
            "samples": sum(x["samples"] for x in shards),
        }, ensure_ascii=False))
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
