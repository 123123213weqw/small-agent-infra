#!/usr/bin/env python3
"""Count visibly empty foreign-language annotations in rule-cleaned FineWiki."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import p0_data_pipeline as core
from p0_sample_finewiki_quality import stratum


LANGUAGES = ("英語|英语|法語|法语|德語|德语|俄語|俄语|韓語|韩语|日語|日语|"
             "葡萄牙語|葡萄牙语|拉丁語|拉丁语|西班牙語|西班牙语|義大利語|意大利语")
EMPTY_LANGUAGE = re.compile(rf"(?:{LANGUAGES})[：:]\s*[；;，,、）)]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import pyarrow.parquet as pq
    clean_root = args.root / "clean_rule_v1"
    manifest_path = clean_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["stage"] != "finewiki_zh_rule_clean_audit_only":
        raise ValueError("unexpected clean manifest")
    counts = Counter()
    for item in manifest["files"]:
        path = clean_root / item["output_file"]
        if core.sha256_file(path) != item["output_sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["text", "chars", "has_math"]):
            for row in batch.to_pylist():
                key = stratum(row["chars"], row["has_math"])
                counts["documents"] += 1
                counts[f"documents:{key}"] += 1
                matches = EMPTY_LANGUAGE.findall(row["text"])
                if matches:
                    counts["flagged_documents"] += 1
                    counts[f"flagged_documents:{key}"] += 1
                    counts["empty_annotation_occurrences"] += len(matches)
    if counts["documents"] != manifest["counts"]["kept"]:
        raise ValueError("row count mismatch")
    report = {
        "stage": "finewiki_zh_empty_language_annotation_audit_only",
        "clean_manifest_sha256": core.sha256_file(manifest_path),
        "script_sha256": core.sha256_file(Path(__file__)),
        "counts": dict(counts),
        "regex": EMPTY_LANGUAGE.pattern,
        "interpretation": "Visible extraction artifact flag, not an automatic document rejection.",
        "training_eligible": False,
    }
    core.atomic_write_text(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
