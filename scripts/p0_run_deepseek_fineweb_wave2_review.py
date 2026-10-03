#!/usr/bin/env python3
"""Run Wave2 audit against the verified user key; never print the key."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--input-name", default="deepseek_full.input.jsonl")
    p.add_argument("--output-name", default="deepseek_full.results.jsonl")
    p.add_argument("--response-format", choices=("json_object", "json_schema"),
                   default=os.environ.get("P0_REVIEW_RESPONSE_FORMAT", "json_object"))
    p.add_argument("--review-dir", type=Path, required=True)
    p.add_argument("--key-file", type=Path, required=True)
    p.add_argument("--expected-key-sha256", required=True)
    args = p.parse_args()
    matches = []
    for raw in args.key_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip().removeprefix("export ")
        if line.startswith("DEEPSEEK_API_KEY="):
            matches.append(line.split("=", 1)[1].strip().strip("\"'"))
    if len(matches) != 1 or hashlib.sha256(matches[0].encode()).hexdigest() != args.expected_key_sha256:
        raise ValueError("explicit user key fingerprint mismatch")
    os.environ["DEEPSEEK_API_KEY"] = matches[0]
    root = Path("/srv/small-agent/datasets/small-agent-p0/p0_500m_v1")
    python = "/srv/small-agent/venvs/deepspec_gemma4/bin/python"
    command = [
        python, str(root / "code/scripts/annotate_jsonl_v2.py"),
        "--input", str(args.review_dir / args.input_name),
        "--output", str(args.review_dir / args.output_name),
        "--prompt-file", str(root / "code/configs/data/p0_teacher_quality_prompt_v2.txt"),
        "--schema-file", str(root / "code/schemas/p0_teacher_quality_v2.json"),
        "--text-field", "review_payload", "--id-fields", "document_id",
        "--carry-fields", "source_id,current_bucket,raw_row,source_file,sampling_design,sampling_group,upstream_score,text_chars,tokenizer_v1_tokens",
        "--api-base", "https://api.deepseek.com", "--api-key-env", "DEEPSEEK_API_KEY",
        "--model", "deepseek-flash", "--thinking", "disabled", "--temperature", "0",
        "--response-format", args.response_format, "--require-document-id-match",
        "--workers", str(args.workers), "--max-tokens", str(args.max_tokens),
        "--timeout", "120", "--max-retries", "1",
    ]
    if args.limit:
        command.extend(("--limit", str(args.limit)))
    os.execv(python, command)


if __name__ == "__main__":
    main()
