#!/usr/bin/env python3
"""Reproducible, small Hugging Face dataset-viewer audit probe (not a bulk ingest)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode, quote
from urllib.request import urlopen


def fetch_json(url: str) -> dict:
    for attempt in range(5):
        try:
            with urlopen(url, timeout=25) as response:
                return json.load(response)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1 + attempt * 2)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1 or args.output.exists() or args.manifest.exists():
        raise ValueError("count must be positive and output paths must not exist")
    dataset_url = f"https://huggingface.co/api/datasets/{quote(args.dataset, safe='/')}"
    before = fetch_json(dataset_url)
    revision = before["sha"]
    probe = fetch_json("https://datasets-server.huggingface.co/rows?" + urlencode({
        "dataset": args.dataset, "config": args.config, "split": args.split,
        "offset": 0, "length": 1,
    }))
    total = probe["num_rows_total"]
    chosen = []
    seen_offsets = set()
    for attempt in range(args.count * 8):
        # Stratify across the full row-index range rather than taking the first page.
        bucket = attempt % args.count
        span = max(1, total // args.count)
        entropy = int(hashlib.sha256(f"{args.seed}\0{attempt}".encode()).hexdigest(), 16)
        offset = min(total - 1, bucket * span + entropy % span)
        if offset in seen_offsets:
            continue
        seen_offsets.add(offset)
        payload = fetch_json("https://datasets-server.huggingface.co/rows?" + urlencode({
            "dataset": args.dataset, "config": args.config, "split": args.split,
            "offset": offset, "length": 1,
        }))
        if not payload.get("rows"):
            continue
        row = payload["rows"][0]["row"]
        body = row.get("text", "")
        if not isinstance(body, str) or not 200 <= len(body) <= 20_000:
            continue
        source_id = row.get("id") or str(offset)
        document_id = hashlib.sha256(f"{args.dataset}\0{revision}\0{source_id}".encode()).hexdigest()
        envelope = {
            "document_id": document_id,
            "source_id": f"{args.dataset}/{args.config}",
            "source_locator": source_id,
            "current_bucket": args.domain,
            "language": row.get("language"),
            "quality_score": row.get("quality_score"),
            "document_text": body,
        }
        chosen.append({
            "document_id": document_id,
            "current_bucket": args.domain,
            "row_index": offset,
            "text_chars": len(body),
            "review_payload": json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        })
        if len(chosen) >= args.count:
            break
    after = fetch_json(dataset_url)
    if after["sha"] != revision or len(chosen) < args.count:
        raise RuntimeError("dataset revision changed or too few qualifying rows")
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in chosen)
    manifest = {
        "stage": "hf_viewer_source_probe_not_training_approved",
        "dataset": args.dataset, "revision": revision, "config": args.config,
        "split": args.split, "domain": args.domain, "seed": args.seed,
        "selected_documents": len(chosen), "attempted_offsets": len(seen_offsets),
        "license": before.get("cardData", {}).get("license"),
        "input_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "note": "Viewer sample only; not a verified raw shard or trainable corpus.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
