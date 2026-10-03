#!/usr/bin/env python3
"""Export only normalized text hashes from the small approved/gen corpora."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


CORPORA = {
    "approved_real_v1": Path("p0_1m_approved_rebuild_v1/accepted_real_v1"),
    "generated_verified_v2": Path("p0_gen_verified_v2"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with args.output.open("x", encoding="utf-8") as handle:
        for name, rel in CORPORA.items():
            files = sorted((args.root / rel).glob("*/part-*.parquet"))
            if not files:
                raise FileNotFoundError(rel)
            for path in files:
                pf = pq.ParquetFile(path)
                for batch in pf.iter_batches(batch_size=4096, columns=["normalized_sha256", "split"]):
                    data = batch.to_pydict()
                    for digest, split in zip(data["normalized_sha256"], data["split"]):
                        if not isinstance(digest, str) or len(digest) != 64:
                            raise ValueError(f"bad digest in {path}")
                        handle.write(json.dumps({"normalized_sha256": digest, "corpus": name,
                                                 "split": split}, sort_keys=True) + "\n")
                        counts[name] += 1
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    report = {"stage": "existing_corpus_hash_export", "counts": dict(counts),
              "sha256": digest, "output": str(args.output)}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
