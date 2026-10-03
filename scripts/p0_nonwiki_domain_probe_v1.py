#!/usr/bin/env python3
"""Build a fixed, source-stratified Qwen diagnostic probe from candidate pointers."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlsplit


WEB_DOMAINS = ("en.wikipedia.org", "www.reference.com", "www.lessonplanet.com", "www.teacherspayteachers.com")
MATH_DOMAINS = ("www.physicsforums.com", "math.stackexchange.com", "www.jiskha.com", "mathhelpforum.com", "www.coursehero.com", "nrich.maths.org")
CODE_LANGUAGES = ("Python", "Shell", "JavaScript")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq

    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    audit = root / "upstream_nonwiki_v1/audit"
    old_probe = {json.loads(line)["document_id"] for line in
                 (audit / "qwen_probe_v1/input.jsonl").open()}
    bins = defaultdict(list)
    candidate_hashes = {}
    for source, groups, size in (
        ("fineweb_edu_new", WEB_DOMAINS, 6),
        ("finemath_4plus_new", MATH_DOMAINS, 8),
        ("github_code_clean_new", CODE_LANGUAGES, 6),
    ):
        path = audit / "prefilter_v1" / f"{source}.candidate_index.parquet"
        candidate_hashes[source] = sha256(path)
        pf = pq.ParquetFile(path)
        for group in range(pf.num_row_groups):
            rows = pf.read_row_group(group).to_pylist()
            for row in rows:
                if row["content_sha256"] in old_probe or not 900 <= row["chars"] <= 4800:
                    continue
                category = (row["language"] if source == "github_code_clean_new" else
                            (urlsplit(row["locator"]).hostname or "").lower())
                if category not in groups:
                    continue
                rank = hashlib.sha256((source + "\0" + category + "\0" + row["content_sha256"]).encode()).hexdigest()
                bins[(source, category)].append((rank, row))
        for category in groups:
            bins[(source, category)] = sorted(bins[(source, category)])[:size]
            if len(bins[(source, category)]) != size:
                raise ValueError(f"insufficient {source}/{category}: {len(bins[(source, category)])}")

    selected = [row for group in bins.values() for _, row in group]
    if len(selected) != 90:
        raise ValueError(f"expected 90 rows, got {len(selected)}")
    by_file = defaultdict(list)
    for row in selected:
        by_file[(row["source_id"], row["source_file"])].append(row)

    raw = root / "upstream_nonwiki_v1/raw"
    text_by_id = {}
    for (source, name), rows in by_file.items():
        pf = pq.ParquetFile(raw / source / name)
        offsets = [0]
        for group in range(pf.num_row_groups):
            offsets.append(offsets[-1] + pf.metadata.row_group(group).num_rows)
        grouped = defaultdict(list)
        for row in rows:
            group = bisect_right(offsets, row["raw_row"]) - 1
            grouped[group].append(row)
        field = "code" if source == "github_code_clean_new" else "text"
        for group, entries in grouped.items():
            values = pf.read_row_group(group, columns=[field]).column(0)
            for row in entries:
                text = values[row["raw_row"] - offsets[group]].as_py()
                text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
                if hashlib.sha256(text.encode()).hexdigest() != row["content_sha256"]:
                    raise ValueError(f"raw pointer/hash mismatch: {source}/{name}/{row['raw_row']}")
                text_by_id[row["content_sha256"]] = text

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".nonwiki-domain-probe-", dir=output.parent))
    try:
        counts = Counter()
        with (tmp / "input.jsonl").open("w", encoding="utf-8") as f:
            for row in sorted(selected, key=lambda x: (x["source_id"], x["content_sha256"])):
                source = row["source_id"]
                counts[source] += 1
                bucket = {"fineweb_edu_new": "general_en", "finemath_4plus_new": "math_en",
                          "github_code_clean_new": "code"}[source]
                payload = {"document_id": row["content_sha256"], "source_id": source,
                           "source_locator": row["locator"], "current_bucket": bucket,
                           "language": row["language"], "document_text": text_by_id[row["content_sha256"]]}
                record = {"document_id": row["content_sha256"], "source_id": source,
                          "source_file": row["source_file"], "raw_row": row["raw_row"],
                          "chars": row["chars"], "category":
                          row["language"] if source == "github_code_clean_new" else urlsplit(row["locator"]).hostname,
                          "current_bucket": bucket,
                          "review_payload": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest = {"stage": "domain_stratified_quality_probe_not_training_approved",
                    "candidate_index_sha256": candidate_hashes,
                    "input_sha256": sha256(tmp / "input.jsonl"), "rows": len(selected),
                    "by_source": dict(counts), "web_domains": WEB_DOMAINS,
                    "math_domains": MATH_DOMAINS, "code_languages": CODE_LANGUAGES,
                    "char_range": [900, 4800],
                    "selection": "lowest stable hash per named domain/language; excludes 54-document prior probe"}
        (tmp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        tmp.rename(output)
        print(json.dumps(manifest, ensure_ascii=False))
    except BaseException:
        shutil.rmtree(tmp)
        raise


if __name__ == "__main__":
    main()
