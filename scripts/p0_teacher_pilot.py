#!/usr/bin/env python3
"""Prepare and summarize the deterministic P0 DeepSeek quality pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable


PILOT_VERSION = "p0_teacher_pilot_v2"
DOMAINS = ("general_zh", "general_en", "math_en", "code_python", "code_shell", "code_other")


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


def pilot_rank(seed: str, document_id: str) -> str:
    return sha256_text(f"{seed}\0{document_id}")


def cheap_reject(record: dict[str, Any]) -> list[str]:
    """Only high-precision rules belong here; semantic filtering stays with teachers."""
    text = record["text"]
    lowered = text.casefold()
    locator = record.get("source_locator", "").casefold().replace("\\", "/")
    reasons: list[str] = []
    if record["domain"].startswith("code_"):
        if re.search(r"(?:^|[=|/])(?:node_modules|vendor|vendors)/", locator):
            reasons.append("vendored_dependency_path")
        if re.search(r"(?:^|/)(?:package-lock\.json|yarn\.lock|pnpm-lock\.yaml)(?:$|[|/])", locator):
            reasons.append("dependency_lock_file")
        if re.search(r"(?:^|/)[^/]+\.min\.js(?:$|[|/])", locator):
            reasons.append("minified_javascript_path")
    paywall_markers = (
        "want to see the full answer",
        "want to see this answer and more",
        "unlock the full answer",
        "see solution\n*response times",
        "submit your documents and get free plagiarism report",
    )
    if any(marker in lowered for marker in paywall_markers):
        reasons.append("paywall_or_answer_preview")
    feedback_signature = ("不感兴趣", "广告软文", "标题夸张", "感谢您的反馈")
    if sum(marker in text for marker in feedback_signature) >= 3:
        reasons.append("embedded_recommendation_feedback_widget")
    if re.search(r"关键词\s*[:：].{0,40}(彩票|博彩|时时彩|排列三)", text, re.IGNORECASE):
        reasons.append("lottery_keyword_spam")
    return sorted(set(reasons))


def review_text(text: str, document_id: str, max_chars: int = 100_000) -> tuple[str, dict[str, Any]]:
    if len(text) <= max_chars:
        return text, {"mode": "full", "original_chars": len(text), "review_chars": len(text)}
    head, middle, tail = 8_000, 6_000, 8_000
    available_start = head
    available_end = len(text) - tail - middle
    span = max(available_end - available_start, 0)
    offset = int(document_id[:16], 16) % (span + 1)
    middle_start = available_start + offset
    selected = (
        text[:head]
        + f"\n\n<<<省略；确定性中段从字符 {middle_start:,} 开始>>>\n\n"
        + text[middle_start : middle_start + middle]
        + f"\n\n<<<省略；结尾从字符 {len(text)-tail:,} 开始>>>\n\n"
        + text[-tail:]
    )
    return selected, {
        "mode": "head_middle_tail",
        "original_chars": len(text),
        "review_chars": len(selected),
        "middle_start": middle_start,
    }


def select_stratified(
    records: Iterable[dict[str, Any]], per_domain: int, seed: str
) -> list[dict[str, Any]]:
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not cheap_reject(record):
            eligible[record["domain"]].append(record)
    selected: list[dict[str, Any]] = []
    for domain in DOMAINS:
        ranked = sorted(
            eligible[domain],
            key=lambda record: (pilot_rank(seed, record["document_id"]), record["document_id"]),
        )
        if len(ranked) < per_domain:
            raise ValueError(f"{domain}: only {len(ranked)} eligible records for target {per_domain}")
        selected.extend(ranked[:per_domain])
    return selected


def select_all_eligible(records: Iterable[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    """Return every cheap-prefilter survivor in deterministic order."""
    eligible = [record for record in records if not cheap_reject(record)]
    return sorted(
        eligible,
        key=lambda record: (pilot_rank(seed, record["document_id"]), record["document_id"]),
    )


def load_pool(pool: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit("PyArrow is required") from error
    manifest_path = pool / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "consolidate_candidates":
        raise ValueError("pool is not consolidated")
    records: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        path = pool / shard["domain"] / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"shard hash mismatch: {path}")
        records.extend(pq.read_table(path).to_pylist())
    return manifest, records


def make_request(record: dict[str, Any]) -> dict[str, Any]:
    selected_text, selection = review_text(record["text"], record["document_id"])
    envelope = {
        "document_id": record["document_id"],
        "source_id": record["source_id"],
        "source_locator": record["source_locator"],
        "current_bucket": record["domain"],
        "language": record["language"],
        "token_count": record["token_count"],
        "text_selection": selection,
        "document_text": selected_text,
    }
    return {
        "document_id": record["document_id"],
        "source_id": record["source_id"],
        "current_bucket": record["domain"],
        "language": record["language"],
        "token_count": record["token_count"],
        "text_selection": selection,
        "review_payload": canonical_json(envelope),
    }


def command_prepare(args: argparse.Namespace) -> None:
    pool = args.pool.resolve()
    manifest, records = load_pool(pool)
    prefilter = []
    for record in records:
        reasons = cheap_reject(record)
        if reasons:
            prefilter.append(
                {
                    "document_id": record["document_id"],
                    "source_id": record["source_id"],
                    "domain": record["domain"],
                    "token_count": record["token_count"],
                    "reason_codes": reasons,
                }
            )
    if args.all_eligible:
        selected = select_all_eligible(records, args.seed)
        selection_mode = "all_eligible"
    else:
        selected = select_stratified(records, args.per_domain, args.seed)
        selection_mode = "stratified"
    requests = [make_request(record) for record in selected]
    output = args.output.resolve()
    atomic_write(output, "".join(canonical_json(row) + "\n" for row in requests))
    exclusions_path = output.with_name("prefilter_exclusions.jsonl")
    atomic_write(exclusions_path, "".join(canonical_json(row) + "\n" for row in prefilter))
    summary = {
        "pilot_version": PILOT_VERSION,
        "selection_mode": selection_mode,
        "seed": args.seed,
        "source_pool": str(pool),
        "source_manifest_sha256": sha256_file(pool / "manifest.json"),
        "input_documents": len(records),
        "input_tokens": sum(row["token_count"] for row in records),
        "prefilter_rejected_documents": len(prefilter),
        "prefilter_rejected_tokens": sum(row["token_count"] for row in prefilter),
        "prefilter_reasons": dict(sorted(Counter(code for row in prefilter for code in row["reason_codes"]).items())),
        "pilot_documents": len(requests),
        "pilot_domains": dict(sorted(Counter(row["current_bucket"] for row in requests).items())),
        "text_selection_modes": dict(sorted(Counter(row["text_selection"]["mode"] for row in requests).items())),
        "requests": {"file": output.name, "bytes": output.stat().st_size, "sha256": sha256_file(output)},
        "prefilter_exclusions": {
            "file": exclusions_path.name,
            "bytes": exclusions_path.stat().st_size,
            "sha256": sha256_file(exclusions_path),
        },
    }
    atomic_write(output.with_name("prepare_manifest.json"), json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def passes_gate(annotation: dict[str, Any]) -> bool:
    return (
        annotation["decision"] == "keep"
        and annotation["confidence"] >= 0.85
        and annotation["quality"] >= 3
        and annotation["completeness"] >= 4
        and annotation["educational_value"] >= 3
        and annotation["format_integrity"] >= 4
        and annotation["recommended_bucket"] != "drop"
    )


def command_summarize(args: argparse.Namespace) -> None:
    requests = {row["document_id"]: row for row in load_jsonl(args.requests)}
    results = load_jsonl(args.results)
    ok: dict[str, dict[str, Any]] = {}
    errors = 0
    for result in results:
        source = result.get("source") or {}
        document_id = source.get("document_id")
        request = requests.get(document_id)
        if request is None:
            continue
        if result.get("text_sha256") != sha256_text(request["review_payload"]):
            # A regenerated request may coexist with an old append-only result.
            # Never apply a score to text different from the current request.
            continue
        if result.get("status") != "ok":
            errors += 1
            continue
        annotation = result["annotation"]
        if document_id not in requests or annotation["document_id"] != document_id:
            raise ValueError(f"document id mismatch: {document_id}")
        if document_id in ok:
            raise ValueError(f"duplicate successful result: {document_id}")
        ok[document_id] = result
    missing = sorted(set(requests) - set(ok))
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    reasons: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    admitted = []
    for document_id, result in ok.items():
        request = requests[document_id]
        annotation = result["annotation"]
        route = "keep" if passes_gate(annotation) else "drop"
        by_domain[request["current_bucket"]][route] += 1
        reasons.update(annotation["reason_codes"])
        buckets[annotation["recommended_bucket"]] += 1
        admitted.append(
            {
                "document_id": document_id,
                "current_bucket": request["current_bucket"],
                "token_count": request["token_count"],
                "route": route,
                "teacher_status": "ok",
                **annotation,
            }
        )
    for document_id in missing:
        request = requests[document_id]
        by_domain[request["current_bucket"]]["drop"] += 1
        admitted.append(
            {
                "document_id": document_id,
                "current_bucket": request["current_bucket"],
                "token_count": request["token_count"],
                "route": "drop",
                "teacher_status": "schema_failure",
                "decision": None,
                "confidence": None,
                "quality": None,
                "completeness": None,
                "educational_value": None,
                "format_integrity": None,
                "recommended_bucket": "drop",
                "reason_codes": ["teacher_schema_failure"],
                "evidence": "模型多次返回不满足 JSON Schema 的结果；严格模式下自动剔除。",
            }
        )
    output = args.output.resolve()
    atomic_write(output, "".join(canonical_json(row) + "\n" for row in sorted(admitted, key=lambda x: x["document_id"])))
    report = {
        "pilot_version": PILOT_VERSION,
        "requested": len(requests),
        "successful": len(ok),
        "error_attempt_records": errors,
        "unresolved_documents": len(missing),
        "unresolved_document_ids": missing,
        "gate_keep": sum(row["route"] == "keep" for row in admitted),
        "gate_drop": sum(row["route"] == "drop" for row in admitted),
        "gate_keep_tokens": sum(row["token_count"] for row in admitted if row["route"] == "keep"),
        "gate_drop_tokens": sum(row["token_count"] for row in admitted if row["route"] == "drop"),
        "by_domain": {domain: dict(counts) for domain, counts in sorted(by_domain.items())},
        "reason_codes": dict(reasons.most_common()),
        "recommended_buckets": dict(buckets.most_common()),
        "usage": dict(
            Counter(
                {key: sum((row.get("usage") or {}).get(key, 0) for row in ok.values())
                 for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
            )
        ),
        "served_models": dict(Counter(row.get("served_model", "unknown") for row in ok.values())),
        "scoring_settings": [
            {
                "requested_model": model,
                "prompt_sha256": prompt_hash,
                "schema_sha256": schema_hash,
                "temperature": temperature,
                "thinking": "disabled"
            }
            for model, prompt_hash, schema_hash, temperature in sorted(
                {
                    (
                        row.get("model"), row.get("prompt_sha256"),
                        row.get("schema_sha256"), row.get("temperature")
                    )
                    for row in ok.values()
                },
                key=lambda item: tuple(str(value) for value in item),
            )
        ],
        "inputs": {
            "requests": {"file": args.requests.name, "bytes": args.requests.stat().st_size, "sha256": sha256_file(args.requests)},
            "results": {"file": args.results.name, "bytes": args.results.stat().st_size, "sha256": sha256_file(args.results)},
        },
        "output": {"file": output.name, "bytes": output.stat().st_size, "sha256": sha256_file(output)},
    }
    atomic_write(output.with_suffix(".summary.json"), json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    atomic_write(
        output.with_suffix(".html"),
        render_pilot_html(admitted, requests, report),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def render_pilot_html(
    adjudicated: list[dict[str, Any]],
    requests: dict[str, dict[str, Any]],
    report: dict[str, Any],
) -> str:
    rows = sorted(adjudicated, key=lambda row: (row["current_bucket"], row["route"], row["document_id"]))
    domain_buttons = "".join(
        f'<button data-domain="{html.escape(domain)}">{html.escape(domain)}</button>'
        for domain in sorted({row["current_bucket"] for row in rows})
    )
    cards = []
    for row in rows:
        request = requests[row["document_id"]]
        payload = json.loads(request["review_payload"])
        text = payload["document_text"]
        preview = text if len(text) <= 1800 else text[:1200] + "\n\n……\n\n" + text[-600:]
        scores = " / ".join(
            f"{name}={row.get(name)}"
            for name in ("quality", "completeness", "educational_value", "format_integrity")
        )
        reasons = ", ".join(row["reason_codes"]) or "无"
        cards.append(
            f'''<article class="card {row['route']}" data-route="{row['route']}" data-domain="{html.escape(row['current_bucket'])}">
<header><b>{html.escape(row['current_bucket'])}</b><span class="route">{row['route'].upper()}</span></header>
<p><b>推荐桶：</b>{html.escape(str(row['recommended_bucket']))}　<b>置信度：</b>{html.escape(str(row.get('confidence')))}　{html.escape(scores)}</p>
<p><b>原因：</b>{html.escape(reasons)}</p><p><b>证据：</b>{html.escape(row['evidence'])}</p>
<pre>{html.escape(preview)}</pre><small>{html.escape(row['document_id'])}</small></article>'''
        )
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepSeek P0 初筛报告</title><style>
body{{margin:0;background:#f4f6fa;color:#182033;font:14px/1.55 system-ui}}main{{max-width:1100px;margin:auto;padding:24px 16px 70px}}
.summary,.card{{background:white;border:1px solid #dbe0e8;border-radius:10px;padding:16px;margin:12px 0}}h1{{margin:0 0 8px}}.stats{{display:flex;gap:10px;flex-wrap:wrap}}.stat{{background:#eef2f8;padding:7px 10px;border-radius:7px}}
nav{{position:sticky;top:0;background:#f4f6fa;padding:10px 0;display:flex;gap:6px;flex-wrap:wrap;z-index:2}}button{{border:1px solid #ccd3df;background:white;border-radius:999px;padding:6px 10px;cursor:pointer}}button.active{{background:#2457d6;color:white}}
header{{display:flex;justify-content:space-between}}.route{{font-weight:800}}.keep .route{{color:#18864b}}.drop .route{{color:#c0392b}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:330px;overflow:auto;background:#111827;color:#e5e7eb;padding:12px;border-radius:7px}}small{{color:#778197;overflow-wrap:anywhere}}
</style></head><body><main><section class="summary"><h1>DeepSeek P0 初筛报告</h1><p>只需大致浏览保留/剔除是否符合直觉，不需要逐条标注。</p>
<div class="stats"><span class="stat">请求 {report['requested']}</span><span class="stat">有效 {report['successful']}</span><span class="stat">保留 {report['gate_keep']} / {report['gate_keep_tokens']:,} 数据 token</span><span class="stat">剔除 {report['gate_drop']} / {report['gate_drop_tokens']:,} 数据 token</span><span class="stat">Schema 失败 {report['unresolved_documents']}</span><span class="stat">API token {report['usage']['total_tokens']:,}</span></div></section>
<nav><button class="active" data-route="all" data-domain="all">全部</button><button data-route="keep" data-domain="all">只看保留</button><button data-route="drop" data-domain="all">只看剔除</button>{domain_buttons}</nav>{''.join(cards)}
</main><script>document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{{document.querySelectorAll('nav button').forEach(x=>x.classList.remove('active'));b.classList.add('active');let r=b.dataset.route,d=b.dataset.domain;document.querySelectorAll('.card').forEach(x=>x.hidden=(r!=='all'&&x.dataset.route!==r)||(d!=='all'&&x.dataset.domain!==d));}});</script></body></html>'''


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--pool", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--per-domain", type=int, default=50)
    prepare.add_argument(
        "--all-eligible",
        action="store_true",
        help="score every record that survives the deterministic cheap prefilter",
    )
    prepare.add_argument("--seed", default=PILOT_VERSION)
    prepare.set_defaults(func=command_prepare)
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--requests", type=Path, required=True)
    summarize.add_argument("--results", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.set_defaults(func=command_summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
