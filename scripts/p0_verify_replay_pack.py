#!/usr/bin/env python3
"""Verify an offline replay's gate, frozen source and mmap token pack."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--code-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    gate = json.loads((args.gate / "gate_summary.json").read_text())
    source = json.loads((args.source / "manifest.json").read_text())
    pack = json.loads((args.pack / "manifest.json").read_text())
    if gate["input_documents"] != 100 or gate["accepted_documents"] != 80:
        raise ValueError("replay gate count changed")
    if source["counts"]["accepted_documents"] != 80 or source["family_split_conflicts"]:
        raise ValueError("replay freeze count or family split failed")
    if pack["source_manifest_sha256"] != sha(args.source / "manifest.json"):
        raise ValueError("token pack source manifest mismatch")
    sys.path.insert(0, str(args.code_root / "src"))
    from small_agent.data import ShardedTokenDataset
    samples = {}
    for split in ("train", "validation", "test"):
        dataset = ShardedTokenDataset(args.pack, split, 2048)
        if len(dataset):
            dataset[0]
            dataset[len(dataset) - 1]
        samples[split] = len(dataset)
    report = {"stage": "offline_replay_pack_verified_not_new_training_data",
              "gate_documents": gate["input_documents"], "accepted_documents": 80,
              "source_manifest_sha256": sha(args.source / "manifest.json"),
              "pack_manifest_sha256": sha(args.pack / "manifest.json"), "samples": samples}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
