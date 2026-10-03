#!/usr/bin/env python3
"""Map Qwen-kept Ultra-FineWeb rows back to upstream source labels.

Produces metadata only; no document text and no training approval.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import pyarrow.parquet as pq


RAW_NAMES = {
    "ultrafineweb_en": "ultrafineweb-en-part-0037-of-2048.parquet",
    "ultrafineweb_zh": "ultrafineweb-zh-part-029-of-256.parquet",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected: dict[str, dict[int, dict]] = defaultdict(dict)
    with args.index.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source_id = row["source_id"]
            if source_id in RAW_NAMES:
                raw_row = row["raw_row"]
                if raw_row in selected[source_id]:
                    raise ValueError(f"Repeated raw row: {source_id}:{raw_row}")
                selected[source_id][raw_row] = row

    counts = Counter()
    statuses: dict[str, Counter] = defaultdict(Counter)
    mapping = args.output_dir / "web_upstream_source_by_id.jsonl"
    with mapping.open("w", encoding="utf-8") as out:
        for source_id, filename in RAW_NAMES.items():
            parquet = args.raw_dir / source_id / filename
            pf = pq.ParquetFile(parquet)
            row_offset = 0
            seen = 0
            for batch in pf.iter_batches(batch_size=8192, columns=["source"]):
                sources = batch.column(0).to_pylist()
                for local_row, upstream_source in enumerate(sources):
                    raw_row = row_offset + local_row
                    entry = selected[source_id].get(raw_row)
                    if entry is None:
                        continue
                    if not isinstance(upstream_source, str) or not upstream_source:
                        raise ValueError(f"Missing upstream source: {source_id}:{raw_row}")
                    record = {
                        "document_id": entry["document_id"],
                        "source_id": source_id,
                        "raw_row": raw_row,
                        "upstream_source": upstream_source,
                        "postfilter_status": entry["postfilter_status"],
                    }
                    out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    counts[(source_id, upstream_source)] += 1
                    statuses[(source_id, upstream_source)][entry["postfilter_status"]] += 1
                    seen += 1
                row_offset += len(sources)
            if row_offset != pf.metadata.num_rows or seen != len(selected[source_id]):
                raise ValueError(
                    f"Row mismatch for {source_id}: scanned={row_offset}, "
                    f"expected={pf.metadata.num_rows}, matched={seen}, "
                    f"selected={len(selected[source_id])}"
                )

    summary = {
        "scope": "Qwen-kept Ultra-FineWeb records only; metadata audit, not license approval",
        "total_mapped": sum(counts.values()),
        "by_source": [
            {
                "source_id": source_id,
                "upstream_source": upstream_source,
                "count": count,
                "postfilter_status": dict(sorted(statuses[(source_id, upstream_source)].items())),
            }
            for (source_id, upstream_source), count in sorted(counts.items())
        ],
    }
    (args.output_dir / "web_source_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
