#!/usr/bin/env python3
"""Replay conservative, deterministic prefilters on the four-source audit probe.

No source data is modified. Hard rejects are limited to vendored/generated Python
paths and exact duplicate text. Ambiguous quality indicators are review flags only.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


VENDOR_COMPONENTS = frozenset({
    ".venv", "venv", "virtualenv", "site-packages", "dist-packages",
    ".eggs", "_vendor", "vendor", "third_party", "third-party",
    "node_modules", "__pycache__", ".tox", ".nox",
})
MATH_SOURCE = "ultradata_math_l2_preview"
CODE_SOURCE = "ultradata_code_l2_py"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def code_relative_path(locator: str) -> str:
    # The locator is "owner/repo/relative/path"; never classify owner/repo.
    parts = locator.replace("\\", "/").split("/", 2)
    return parts[2] if len(parts) == 3 else ""


def classify_hard(row: dict, text: str) -> list[str]:
    hard = []
    source = row["source_id"]
    if not text.strip():
        hard.append("empty_text")
    if source == CODE_SOURCE:
        relpath = code_relative_path(row["source_locator"])
        components = {part.casefold() for part in relpath.replace("\\", "/").split("/")[:-1]}
        if components & VENDOR_COMPONENTS:
            hard.append("vendored_environment_path")
    return hard


def classify(row: dict, text: str) -> tuple[list[str], list[str]]:
    hard = classify_hard(row, text)
    flags = []
    source = row["source_id"]
    if source == CODE_SOURCE:
        # Python 2 and snippets may fail Python 3 parsing. This is only a flag.
        try:
            ast.parse(text)
        except (SyntaxError, ValueError, UnicodeError):
            flags.append("python3_parse_failure")
        if len(text.strip()) < 160:
            flags.append("short_code")
    elif source == MATH_SOURCE:
        if len(text.strip()) < 300:
            flags.append("short_math")
        if re.search(r"\b(up vote|down vote|favorite)\b", text[:500], re.I):
            flags.append("forum_chrome")
    else:
        if len(text.strip()) < (220 if source == "ultrafineweb_en" else 120):
            flags.append("short_web")
        if re.search(r"(?i)all rights reserved|cookie policy|privacy policy", text[:500]):
            flags.append("web_chrome")
    return hard, flags


def load_probe(root: Path) -> dict[str, dict]:
    rows = {}
    for path in sorted((root / "inputs").glob("*/*.input.jsonl")):
        for row in read_jsonl(path):
            envelope = json.loads(row["review_payload"])
            doc_id = row["document_id"]
            if envelope["document_id"] != doc_id or doc_id in rows:
                raise ValueError(f"probe identity collision: {doc_id}")
            rows[doc_id] = {**row, "source_locator": envelope["source_locator"],
                            "document_text": envelope["document_text"]}
    return rows


def load_qwen(root: Path, ids: set[str]) -> dict[str, dict]:
    results = {}
    for path in sorted((root / "results").glob("*/*.results.jsonl")):
        for row in read_jsonl(path):
            doc_id, = json.loads(row["source_id"])
            if doc_id not in ids:
                raise ValueError(f"Qwen result outside probe: {doc_id}")
            if row["status"] == "ok":
                if row["annotation"]["document_id"] != doc_id or doc_id in results:
                    raise ValueError(f"Qwen result identity collision: {doc_id}")
                results[doc_id] = row["annotation"]
    return results


def load_deepseek(root: Path | None, ids: set[str]) -> dict[str, dict]:
    if root is None:
        return {}
    selected = {row["document_id"] for row in
                json.loads((root / "deepseek_review180.input.manifest.json").read_text())["items"]}
    if not selected <= ids:
        raise ValueError("DeepSeek review contains records outside the probe")
    results = {}
    for row in read_jsonl(root / "deepseek_review180.results.jsonl"):
        doc_id, = json.loads(row["source_id"])
        if row["status"] == "ok":
            if row["annotation"]["document_id"] != doc_id:
                raise ValueError(f"DeepSeek result identity mismatch: {doc_id}")
            results[doc_id] = row["annotation"]
    if set(results) != selected:
        raise ValueError("DeepSeek review is not complete")
    return results


def replay(probe: Path, deepseek: Path | None) -> dict:
    rows = load_probe(probe)
    if len(rows) != 4000:
        raise ValueError(f"expected 4000 probe documents; found {len(rows)}")
    qwen = load_qwen(probe, set(rows))
    ds = load_deepseek(deepseek, set(rows))
    # Prefer a non-rejected original as the representative of an exact-copy
    # group; otherwise a vendored copy could hide the only usable original.
    preliminary = {}
    for doc_id, row in rows.items():
        preliminary[doc_id] = classify(row, row["document_text"])
    representatives = {}
    for doc_id, row in sorted(rows.items(), key=lambda item: (bool(preliminary[item[0]][0]), item[0])):
        digest = hashlib.sha256(row["document_text"].encode("utf-8")).hexdigest()
        representatives.setdefault(digest, doc_id)
    per_source = defaultdict(Counter)
    hard_by_reason = Counter()
    flags_by_reason = Counter()
    dropped = []
    for doc_id, row in sorted(rows.items()):
        source = row["source_id"]
        counter = per_source[source]
        counter["input"] += 1
        text = row["document_text"]
        hard, flags = preliminary[doc_id]
        hard, flags = list(hard), list(flags)
        # Exact decoded text only. Whitespace normalization would corrupt code.
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if representatives[digest] != doc_id:
            hard.append("exact_duplicate_text")
        for flag in flags:
            flags_by_reason[flag] += 1
        if flags:
            counter["review_flagged"] += 1
        qa, da = qwen.get(doc_id), ds.get(doc_id)
        counter["qwen_" + (qa["decision"] if qa else "error")] += 1
        if da:
            counter["deepseek_sample"] += 1
        if hard:
            counter["hard_reject"] += 1
            if qa and qa["decision"] == "keep":
                counter["hard_reject_qwen_keep"] += 1
            if da and qa and da["decision"] == qa["decision"] == "keep":
                counter["hard_reject_both_keep"] += 1
            if da and qa and da["decision"] == "keep" and qa["decision"] == "drop":
                counter["hard_reject_ds_keep_qwen_drop"] += 1
            hard_by_reason.update(hard)
            dropped.append({"document_id": doc_id, "source_id": source,
                            "source_locator": row["source_locator"],
                            "hard_reasons": hard, "qwen": qa["decision"] if qa else None,
                            "deepseek": da["decision"] if da else None})
        else:
            counter["pass_hard_filter"] += 1
    return {
        "stage": "prefilter_replay_not_training_approval",
        "probe_documents": len(rows), "qwen_ok": len(qwen), "deepseek_reviewed": len(ds),
        "rules": {"hard": ["empty_text", "vendored_environment_path", "exact_duplicate_text"],
                  "review_only": sorted(flags_by_reason)},
        "per_source": {k: dict(sorted(v.items())) for k, v in sorted(per_source.items())},
        "hard_by_reason": dict(hard_by_reason), "flags_by_reason": dict(flags_by_reason),
        "hard_rejected": dropped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--deepseek-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.probe_dir, args.deepseek_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "hard_rejected"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
