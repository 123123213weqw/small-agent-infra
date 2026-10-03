#!/usr/bin/env python3
"""Prepare, validate, audit, and freeze the DeepSeek P0 synthetic fill."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CANDIDATE = load_module("p0_candidate_pipeline_for_synthetic", "scripts/p0_candidate_pipeline.py")
TEACHER = load_module("p0_teacher_pilot_for_synthetic", "scripts/p0_teacher_pilot.py")
VERSION = "p0_synthetic_fill_v1"
FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def command_prepare(args: argparse.Namespace) -> None:
    config = json.loads(args.topics.read_text(encoding="utf-8"))
    rows = []
    for domain in ("technical", "math_zh"):
        for index, topic in enumerate(config[domain]):
            seed_id = f"p0syn-{domain.replace('_', '-')}-{index:02d}"
            payload = {
                "seed_id": seed_id,
                "domain": domain,
                "topic": topic,
                "target_han_characters": [1800, 2800],
                "purpose": "P0中文技术与数学预训练语料补充",
            }
            rows.append(
                {
                    "seed_id": seed_id,
                    "domain": domain,
                    "topic": topic,
                    "generation_request": canonical_json(payload),
                }
            )
    atomic_write(args.output, "".join(canonical_json(row) + "\n" for row in rows))
    report = {
        "version": VERSION,
        "topics_sha256": sha256_file(args.topics),
        "requests": len(rows),
        "by_domain": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "output_sha256": sha256_file(args.output),
    }
    atomic_write(args.output.with_suffix(".manifest.json"), json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def matching_successes(requests: list[dict[str, Any]], results_path: Path) -> tuple[dict[str, Any], list[str]]:
    request_by_id = {row["seed_id"]: row for row in requests}
    successful: dict[str, Any] = {}
    for result in load_jsonl(results_path):
        source = result.get("source") or {}
        seed_id = source.get("seed_id")
        request = request_by_id.get(seed_id)
        if request is None or result.get("status") != "ok":
            continue
        if result.get("text_sha256") != sha256_text(request["generation_request"]):
            continue
        annotation = result["annotation"]
        if annotation.get("seed_id") != seed_id or annotation.get("domain") != request["domain"]:
            continue
        if seed_id in successful:
            raise ValueError(f"duplicate successful generation: {seed_id}")
        successful[seed_id] = result
    return successful, sorted(set(request_by_id) - set(successful))


def evaluate_fraction(expression: str) -> Fraction:
    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            if abs(node.value) > 10**9:
                raise ValueError("integer too large")
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("division by zero")
                return left / right
            if isinstance(node.op, ast.Pow):
                if right.denominator != 1 or abs(right.numerator) > 12:
                    raise ValueError("unsafe exponent")
                return left ** right.numerator
        raise ValueError(f"unsupported arithmetic node: {type(node).__name__}")

    value = visit(tree)
    if abs(value.numerator) > 10**18 or value.denominator > 10**18:
        raise ValueError("result too large")
    return value


def validate_code_blocks(text: str) -> list[str]:
    reasons: list[str] = []
    if text.count("```") != len(FENCE_RE.findall(text)) * 2:
        return ["unbalanced_code_fence"]
    for language, code in FENCE_RE.findall(text):
        language = language.casefold()
        try:
            if language in ("python", "py"):
                ast.parse(code)
            elif language == "json":
                json.loads(code)
            elif language in ("bash", "sh"):
                completed = subprocess.run(
                    ["bash", "-n"], input=code, text=True, capture_output=True, timeout=5
                )
                if completed.returncode:
                    reasons.append("invalid_bash_block")
            else:
                reasons.append("unsupported_code_fence")
        except (SyntaxError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired):
            reasons.append(f"invalid_{language or 'plain'}_block")
    return reasons


def validate_generation(annotation: dict[str, Any]) -> list[str]:
    text = annotation["text"]
    reasons: list[str] = []
    han_count = len(HAN_RE.findall(text))
    # Formula-heavy mathematics legitimately has a lower Han ratio than prose.
    # These are corruption guards, not subjective quality thresholds; the
    # independent teacher gate below handles educational quality.
    minimum_han = 400 if annotation["domain"] == "math_zh" else 700
    minimum_fraction = 0.15 if annotation["domain"] == "math_zh" else 0.22
    if han_count < minimum_han:
        reasons.append("too_short_chinese")
    if han_count > 4200 or len(text) > 9000:
        reasons.append("too_long")
    if han_count / max(len(text), 1) < minimum_fraction:
        reasons.append("low_chinese_fraction")
    lowered = text.casefold()
    banned = ("作为ai", "作为 ai", "语言模型", "根据你的要求", "生成任务", "占位符", "todo", "http://", "https://")
    if any(marker in lowered for marker in banned):
        reasons.append("meta_or_placeholder")
    reasons.extend(validate_code_blocks(text))
    cases = annotation["verification_cases"]
    if annotation["domain"] == "technical" and cases:
        reasons.append("unexpected_math_checks")
    if annotation["domain"] == "math_zh":
        if len(cases) < 3:
            reasons.append("insufficient_math_checks")
        for case in cases:
            expression, expected = case["expression"], str(case["expected"])
            try:
                actual = evaluate_fraction(expression)
                target = Fraction(expected)
                if actual != target:
                    reasons.append("wrong_math_check")
            except (ValueError, SyntaxError, ZeroDivisionError):
                reasons.append("invalid_math_check")
            if expression not in text or expected not in text:
                reasons.append("math_check_missing_from_text")
    return sorted(set(reasons))


def load_accepted_documents(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("stage") != "accepted_real":
        raise ValueError("real pool is not frozen accepted data")
    rows = []
    for shard in manifest["shards"]:
        path = directory / shard["domain"] / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"real shard hash mismatch: {path}")
        rows.extend(pq.read_table(path, columns=["document_id", "text", "normalized_sha256"]).to_pylist())
    return manifest, rows


def quality_request(record: dict[str, Any]) -> dict[str, Any]:
    envelope = {
        "document_id": record["document_id"],
        "source_id": "deepseek_synthetic_v1",
        "source_locator": record["seed_id"],
        "current_bucket": record["domain"],
        "language": "zh",
        "token_count": record["token_count"],
        "text_selection": {"mode": "full", "original_chars": len(record["text"]), "review_chars": len(record["text"])},
        "document_text": record["text"],
    }
    return {
        "document_id": record["document_id"],
        "source_id": "deepseek_synthetic_v1",
        "current_bucket": record["domain"],
        "language": "zh",
        "token_count": record["token_count"],
        "text_selection": envelope["text_selection"],
        "review_payload": canonical_json(envelope),
    }


def command_validate(args: argparse.Namespace) -> None:
    from tokenizers import Tokenizer

    requests = load_jsonl(args.requests)
    successes, missing = matching_successes(requests, args.results)
    real_manifest, real_rows = load_accepted_documents(args.accepted_real)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    deduper = CANDIDATE.MinHashDeduper(seed=args.seed, shingle_size=5, num_perm=32, bands=8, threshold=0.82)
    exact = set()
    for row in sorted(real_rows, key=lambda item: item["document_id"]):
        exact.add(row["normalized_sha256"])
        deduper.find_duplicate(row["document_id"], row["text"])

    request_by_id = {row["seed_id"]: row for row in requests}
    accepted = []
    rejected = [{"seed_id": seed_id, "reason_codes": ["generation_schema_failure"]} for seed_id in missing]
    for seed_id in sorted(successes):
        annotation = successes[seed_id]["annotation"]
        reasons = validate_generation(annotation)
        normalized_text = annotation["text"].strip().replace("\r\n", "\n").replace("\r", "\n")
        normalized_sha = sha256_text(normalized_text)
        document_id = sha256_text(f"deepseek_synthetic_v1\0{normalized_text}")
        if normalized_sha in exact:
            reasons.append("exact_duplicate")
        duplicate_of = deduper.find_duplicate(document_id, normalized_text)
        if duplicate_of:
            reasons.append("near_duplicate")
        if reasons:
            rejected.append({"seed_id": seed_id, "reason_codes": sorted(set(reasons)), "duplicate_of": duplicate_of})
            continue
        exact.add(normalized_sha)
        token_count = len(tokenizer.encode(normalized_text, add_special_tokens=False).ids) + 1
        accepted.append(
            {
                "seed_id": seed_id,
                "domain": annotation["domain"],
                "topic": request_by_id[seed_id]["topic"],
                "title": annotation["title"],
                "text": normalized_text,
                "verification_cases": annotation["verification_cases"],
                "document_id": document_id,
                "normalized_sha256": normalized_sha,
                "token_count": token_count,
                "generation_result_sha256": sha256_text(canonical_json(successes[seed_id])),
            }
        )
    atomic_write(args.candidates, "".join(canonical_json(row) + "\n" for row in accepted))
    atomic_write(args.quality_requests, "".join(canonical_json(quality_request(row)) + "\n" for row in accepted))
    atomic_write(args.rejected, "".join(canonical_json(row) + "\n" for row in rejected))
    report = {
        "version": VERSION,
        "generation_requested": len(requests),
        "generation_successful": len(successes),
        "automatic_validation_kept": len(accepted),
        "automatic_validation_rejected": len(rejected),
        "kept_tokens": sum(row["token_count"] for row in accepted),
        "by_domain": {
            domain: {
                "documents": sum(row["domain"] == domain for row in accepted),
                "tokens": sum(row["token_count"] for row in accepted if row["domain"] == domain),
            }
            for domain in ("technical", "math_zh")
        },
        "rejection_reasons": dict(sorted(Counter(code for row in rejected for code in row["reason_codes"]).items())),
        "tokenizer": real_manifest["tokenizer"],
        "outputs": {
            "candidates_sha256": sha256_file(args.candidates),
            "quality_requests_sha256": sha256_file(args.quality_requests),
            "rejected_sha256": sha256_file(args.rejected),
        },
    }
    atomic_write(args.candidates.with_suffix(".manifest.json"), json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def stable_rank(seed: int, document_id: str) -> str:
    return sha256_text(f"{seed}\0{document_id}")


def quality_successes(requests: list[dict[str, Any]], results_path: Path) -> tuple[dict[str, Any], list[str]]:
    request_by_id = {row["document_id"]: row for row in requests}
    successful = {}
    for result in load_jsonl(results_path):
        source = result.get("source") or {}
        document_id = source.get("document_id")
        request = request_by_id.get(document_id)
        if request is None or result.get("status") != "ok":
            continue
        if result.get("text_sha256") != sha256_text(request["review_payload"]):
            continue
        annotation = result["annotation"]
        if annotation.get("document_id") != document_id:
            continue
        if document_id in successful:
            raise ValueError(f"duplicate successful quality result: {document_id}")
        successful[document_id] = result
    return successful, sorted(set(request_by_id) - set(successful))


def command_freeze(args: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"output directory is not empty: {args.output}")
    candidates = load_jsonl(args.candidates)
    candidate_by_id = {row["document_id"]: row for row in candidates}
    requests = load_jsonl(args.quality_requests)
    successes, missing = quality_successes(requests, args.quality_results)
    eligible: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    audit = []
    for document_id, candidate in candidate_by_id.items():
        result = successes.get(document_id)
        annotation = result["annotation"] if result else None
        keep = bool(annotation and TEACHER.passes_gate(annotation))
        if keep:
            eligible[candidate["domain"]].append((candidate, annotation))
        else:
            audit.append(
                {
                    "document_id": document_id,
                    "seed_id": candidate["seed_id"],
                    "domain": candidate["domain"],
                    "route": "drop",
                    "reason_codes": annotation["reason_codes"] if annotation else ["teacher_schema_failure"],
                }
            )

    quotas = {"technical": args.technical_tokens, "math_zh": args.math_tokens}
    selected: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for domain, target in quotas.items():
        running = 0
        ranked = sorted(eligible[domain], key=lambda item: (stable_rank(args.seed, item[0]["document_id"]), item[0]["document_id"]))
        for candidate, annotation in ranked:
            if running >= target:
                audit.append(
                    {
                        "document_id": candidate["document_id"], "seed_id": candidate["seed_id"],
                        "domain": domain, "route": "reserve", "reason_codes": ["quota_filled"],
                    }
                )
                continue
            selected[domain].append((candidate, annotation))
            running += candidate["token_count"]
        if running < target:
            raise ValueError(f"{domain}: only {running} eligible tokens for target {target}")

    args.output.mkdir(parents=True, exist_ok=True)
    shards = []
    total_documents = total_tokens = 0
    for domain, items in sorted(selected.items()):
        rows = []
        for candidate, annotation in sorted(items, key=lambda item: item[0]["document_id"]):
            document_id = candidate["document_id"]
            bucket = int(document_id[:8], 16) / 0xFFFFFFFF
            split = "train" if bucket < 0.98 else "validation" if bucket < 0.99 else "test"
            rows.append(
                {
                    "document_id": document_id,
                    "source_id": "deepseek_synthetic_v1",
                    "source_revision": "deepseek-flash",
                    "source_locator": candidate["seed_id"],
                    "family_id": sha256_text(f"synthetic-family\0{candidate['seed_id']}"),
                    "domain": domain,
                    "language": "zh",
                    "title": candidate["title"],
                    "text": candidate["text"],
                    "raw_sha256": candidate["normalized_sha256"],
                    "normalized_sha256": candidate["normalized_sha256"],
                    "license_status": "generated",
                    "pipeline_version": VERSION,
                    "split": split,
                    "rule_keep": True,
                    "rule_reason_codes": [],
                    "source_metadata_json": canonical_json(
                        {"topic": candidate["topic"], "verification_cases": candidate["verification_cases"]}
                    ),
                    "token_count": candidate["token_count"],
                    "sample_rank": stable_rank(args.seed, document_id),
                    "teacher_decision": annotation["decision"],
                    "teacher_confidence": annotation["confidence"],
                    "teacher_quality": annotation["quality"],
                    "teacher_completeness": annotation["completeness"],
                    "teacher_educational_value": annotation["educational_value"],
                    "teacher_format_integrity": annotation["format_integrity"],
                    "teacher_reason_codes": annotation["reason_codes"],
                    "teacher_evidence": annotation["evidence"],
                }
            )
        directory = args.output / domain
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "part-00000.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        tokens = sum(row["token_count"] for row in rows)
        total_documents += len(rows)
        total_tokens += tokens
        shards.append(
            {"domain": domain, "file": path.name, "rows": len(rows), "tokens": tokens,
             "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    audit.sort(key=lambda row: row["document_id"])
    audit_path = args.output / "selection_audit.parquet"
    pq.write_table(pa.Table.from_pylist(audit), audit_path, compression="zstd")
    manifest = {
        "version": VERSION,
        "stage": "accepted_synthetic",
        "generator": {"model": "deepseek-flash", "temperature": 0.6, "thinking": "disabled"},
        "quality_gate": {"model": "deepseek-flash", "temperature": 0, "thinking": "disabled"},
        "inputs": {
            "candidates": {"sha256": sha256_file(args.candidates)},
            "quality_requests": {"sha256": sha256_file(args.quality_requests)},
            "quality_results": {"sha256": sha256_file(args.quality_results)},
        },
        "targets": quotas,
        "counts": {
            "validated_candidates": len(candidates),
            "quality_successful": len(successes),
            "quality_missing": len(missing),
            "quality_gate_eligible": sum(len(items) for items in eligible.values()),
            "selected_documents": total_documents,
            "selected_tokens": total_tokens,
        },
        "shards": shards,
        "selection_audit": {"file": audit_path.name, "rows": len(audit), "sha256": sha256_file(audit_path)},
    }
    atomic_write(args.output / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--topics", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(func=command_prepare)
    validate = sub.add_parser("validate")
    validate.add_argument("--requests", type=Path, required=True)
    validate.add_argument("--results", type=Path, required=True)
    validate.add_argument("--accepted-real", type=Path, required=True)
    validate.add_argument("--tokenizer-json", type=Path, required=True)
    validate.add_argument("--candidates", type=Path, required=True)
    validate.add_argument("--quality-requests", type=Path, required=True)
    validate.add_argument("--rejected", type=Path, required=True)
    validate.add_argument("--seed", type=int, default=20260923)
    validate.set_defaults(func=command_validate)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--candidates", type=Path, required=True)
    freeze.add_argument("--quality-requests", type=Path, required=True)
    freeze.add_argument("--quality-results", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--technical-tokens", type=int, default=28097)
    freeze.add_argument("--math-tokens", type=int, default=28097)
    freeze.add_argument("--seed", type=int, default=20260923)
    freeze.set_defaults(func=command_freeze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
