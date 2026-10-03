#!/usr/bin/env python3
"""Apply narrow high-precision exclusions to pinned non-Wikipedia candidate indexes.

The output remains an *unapproved candidate queue*, not a training corpus.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlsplit


SOURCES = ("fineweb_edu_new", "finemath_4plus_new", "github_code_clean_new")
DENY_HOSTS = frozenset({"coursehero.com", "teacherspayteachers.com", "lessonplanet.com"})
DENY_CODE_PARTS = frozenset({"pods", "target support files", "generated", "gen"})


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exclusion_reasons(source: str, locator: str, content_hash: str, old_hashes: set[str]) -> list[str]:
    reasons = []
    if content_hash in old_hashes:
        reasons.append("wave1_exact_content_duplicate")
    if source == "github_code_clean_new":
        parts = locator.replace("\\", "/").casefold().split("/")
        if any(part in DENY_CODE_PARTS for part in parts[2:-1]):
            reasons.append("vendored_or_generated_code_path")
    else:
        host = (urlsplit(locator).hostname or "").casefold().removeprefix("www.")
        if host in DENY_HOSTS:
            reasons.append("pilot_confirmed_noisy_source_domain")
    return reasons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq

    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    old_path = root / "wave1_consolidated_100m_v1/unified_index.parquet"
    index_dir = root / "upstream_nonwiki_v1/audit/prefilter_v1"
    domain_probe = root / "upstream_nonwiki_v1/audit/domain_probe_v1"
    results_path = domain_probe / "qwen_v1.results.jsonl"
    rows = [json.loads(line) for line in results_path.open()]
    if len(rows) != 90 or any(row["status"] != "ok" for row in rows):
        raise ValueError("incomplete domain probe")
    supported = defaultdict(Counter)
    for row in rows:
        source = row["source"]["source_id"]
        category = row["source"]["category"]
        host = category.casefold().removeprefix("www.")
        if host in DENY_HOSTS:
            supported[host][row["annotation"]["decision"]] += 1
    for host in DENY_HOSTS:
        if supported[host]["drop"] < 6 or supported[host]["keep"]:
            raise ValueError(f"insufficient Qwen pilot support for {host}: {supported[host]}")
    old_hashes = set(pq.read_table(old_path, columns=["normalized_sha256"]).column(0).to_pylist())
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".nonwiki-v2-", dir=output.parent))
    counters = defaultdict(Counter)
    index_hashes = {}
    try:
        for source in SOURCES:
            path = index_dir / f"{source}.candidate_index.parquet"
            index_hashes[source] = digest(path)
            pf = pq.ParquetFile(path)
            writer = pq.ParquetWriter(tmp / f"{source}.candidate_index.parquet", pf.schema_arrow, compression="zstd")
            for group in range(pf.num_row_groups):
                entries = pf.read_row_group(group).to_pylist()
                kept = []
                for row in entries:
                    counters[source]["input_candidates"] += 1
                    reasons = exclusion_reasons(source, row["locator"], row["content_sha256"], old_hashes)
                    if reasons:
                        for reason in reasons:
                            counters[source][reason] += 1
                    else:
                        kept.append(row)
                        counters[source]["output_candidates"] += 1
                        counters[source]["output_upstream_tokens"] += row["upstream_token_count"]
                if kept:
                    writer.write_table(pa.Table.from_pylist(kept, schema=pf.schema_arrow))
            writer.close()
        report = {
            "stage": "high_precision_filtered_candidates_not_training_approved",
            "version": "p0_nonwiki_high_precision_v2",
            "wave1_index_sha256": digest(old_path), "input_index_sha256": index_hashes,
            "domain_probe_result_sha256": digest(results_path),
            "pilot_evidence": {host: dict(supported[host]) for host in sorted(DENY_HOSTS)},
            "domain_exclusions": sorted(DENY_HOSTS),
            "code_path_exclusions": sorted(DENY_CODE_PARTS),
            "by_source": {source: dict(counters[source]) for source in SOURCES},
            "outputs": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                        for p in tmp.glob("*.parquet")},
            "note": "Only old exact hashes and narrow source/path rules are excluded. Near duplicates, same-URL different-content documents, math validity, benchmark contamination, and license remain unreviewed. Candidate counts are not train-ready tokens.",
        }
        (tmp / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        tmp.rename(output)
        print(json.dumps(report, ensure_ascii=False))
    except BaseException:
        shutil.rmtree(tmp)
        raise


if __name__ == "__main__":
    main()
