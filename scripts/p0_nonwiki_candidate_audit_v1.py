#!/usr/bin/env python3
"""Audit new non-Wikipedia candidates against Wave1 and a fixed Qwen probe.

No raw text is copied. This creates an exclusion *proposal*, not a training set.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.parse import unquote, urlsplit


SOURCES = ("fineweb_edu_new", "finemath_4plus_new", "github_code_clean_new")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locator_key(locator: str) -> str:
    if locator.startswith("repo_name="):
        return "code:" + locator.casefold()
    if "|url=" in locator:
        locator = locator.split("|url=", 1)[1]
    elif locator.startswith("url="):
        locator = locator[4:]
    if not locator.startswith(("http://", "https://")):
        locator = "http://" + locator
    try:
        parsed = urlsplit(locator)
        host = (parsed.hostname or "").casefold()
        path = unquote(parsed.path).rstrip("/").casefold()
        query = "?" + parsed.query if parsed.query else ""
        return "url:" + host + path + query
    except ValueError:
        return "bad:" + locator.casefold()


def new_locator_key(source: str, locator: str) -> str:
    if source == "github_code_clean_new":
        parts = locator.split("/", 2)
        if len(parts) == 3:
            return locator_key(f"repo_name={parts[0]}/{parts[1]}|path={parts[2]}")
        return "bad:" + locator.casefold()
    return locator_key(locator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq

    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    old_path = root / "wave1_consolidated_100m_v1/unified_index.parquet"
    candidate_dir = root / "upstream_nonwiki_v1/audit/prefilter_v1"
    probe_dir = root / "upstream_nonwiki_v1/audit/qwen_probe_v1"
    probe_manifest = json.loads((probe_dir / "manifest.json").read_text())
    probe_input = probe_dir / "input.jsonl"
    probe_result = probe_dir / "qwen_v1.results.jsonl"
    if file_sha256(probe_input) != probe_manifest["input_sha256"]:
        raise ValueError("Qwen probe input hash mismatch")
    input_rows = [json.loads(line) for line in probe_input.open()]
    results = [json.loads(line) for line in probe_result.open()]
    if len(input_rows) != 54 or len(results) != 54:
        raise ValueError("incomplete Qwen probe")
    input_ids = {row["document_id"] for row in input_rows}
    result_ids = [row["annotation"]["document_id"] for row in results]
    if len(set(result_ids)) != 54 or set(result_ids) != input_ids:
        raise ValueError("Qwen probe ID mismatch")
    decisions = defaultdict(Counter)
    reasons = defaultdict(Counter)
    for row in results:
        if row["status"] != "ok":
            raise ValueError("Qwen probe error")
        source = row["source"]["source_id"]
        annotation = row["annotation"]
        decisions[source][annotation["decision"]] += 1
        reasons[source].update(annotation.get("reason_codes", []))

    old_table = pq.read_table(old_path, columns=["normalized_sha256", "source_locator"])
    old_hashes = set(old_table.column("normalized_sha256").to_pylist())
    old_locators = {locator_key(x) for x in old_table.column("source_locator").to_pylist()}
    del old_table

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".nonwiki-candidate-audit-", dir=output.parent))
    schema = pa.schema([
        ("source_id", pa.string()), ("source_file", pa.string()), ("raw_row", pa.int64()),
        ("content_sha256", pa.string()), ("exclusion_reason", pa.string()),
        ("locator_key_sha256", pa.string()),
    ])
    counts = defaultdict(Counter)
    input_hashes = {}
    try:
        for source in SOURCES:
            candidate_path = candidate_dir / f"{source}.candidate_index.parquet"
            input_hashes[source] = file_sha256(candidate_path)
            pf = pq.ParquetFile(candidate_path)
            writer = pq.ParquetWriter(tmp / f"{source}.exclusions.parquet", schema, compression="zstd")
            seen_locator = set()
            for group in range(pf.num_row_groups):
                records = pf.read_row_group(group).to_pylist()
                exclusions = []
                for row in records:
                    counts[source]["candidate_rows"] += 1
                    key = new_locator_key(source, row["locator"])
                    flags = []
                    if row["content_sha256"] in old_hashes:
                        flags.append("old_exact_hash")
                    if key in old_locators:
                        flags.append("old_locator")
                    if key in seen_locator:
                        flags.append("new_repeated_locator")
                    seen_locator.add(key)
                    # Generated/vendor paths are high-precision code exclusions.
                    if source == "github_code_clean_new":
                        parts = row["locator"].replace("\\", "/").casefold().split("/")
                        if any(part in {"pods", "target support files", "generated", "gen"}
                               for part in parts[2:-1]):
                            flags.append("code_generated_or_vendor_path_v2")
                    if flags:
                        for flag in flags:
                            counts[source][flag] += 1
                        exclusions.append({
                            "source_id": source, "source_file": row["source_file"],
                            "raw_row": row["raw_row"], "content_sha256": row["content_sha256"],
                            "exclusion_reason": ",".join(flags),
                            "locator_key_sha256": hashlib.sha256(key.encode()).hexdigest(),
                        })
                    else:
                        counts[source]["unflagged_audit_candidates"] += 1
                if exclusions:
                    writer.write_table(pa.Table.from_pylist(exclusions, schema=schema))
            writer.close()
        report = {
            "stage": "overlap_and_qwen_probe_audit_not_training_approved",
            "version": "p0_nonwiki_candidate_audit_v1",
            "wave1_index": str(old_path), "wave1_index_sha256": file_sha256(old_path),
            "wave1_rows": len(old_hashes), "candidate_index_sha256": input_hashes,
            "probe_input_sha256": file_sha256(probe_input),
            "probe_result_sha256": file_sha256(probe_result),
            "probe_selection": probe_manifest["selection"],
            "probe_by_source": {s: {"decisions": dict(decisions[s]), "reason_codes": dict(reasons[s])}
                                for s in SOURCES},
            "overlap_by_source": {s: dict(counts[s]) for s in SOURCES},
            "interpretation": "All candidate counts remain pretraining-unapproved. The 54 short-document Qwen probe is diagnostic, not a representative acceptance estimate. Locator duplicates require review before exclusion because URLs may have different content.",
            "pending": ["stratified_long_and_domain_quality_review", "near_dedup", "benchmark_contamination", "license_review", "tokenizer_v1_exact_count", "split_and_pack"],
            "output_sha256": {p.name: file_sha256(p) for p in tmp.glob("*.parquet")},
        }
        (tmp / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        tmp.rename(output)
        print(json.dumps(report, ensure_ascii=False))
    except BaseException:
        shutil.rmtree(tmp)
        raise


if __name__ == "__main__":
    main()
