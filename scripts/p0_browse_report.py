#!/usr/bin/env python3
"""Create a deterministic, read-only HTML spot-check report from a P0 pool."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable


REPORT_VERSION = "p0_quick_browse_v1"


def audit_rank(seed: str, document_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{document_id}".encode("utf-8")).hexdigest()


def choose_samples(
    rows: Iterable[dict[str, Any]], per_domain: int, seed: str
) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    selected: list[dict[str, Any]] = []
    for domain in sorted(by_domain):
        ranked = sorted(
            by_domain[domain],
            key=lambda row: (audit_rank(seed, row["document_id"]), row["document_id"]),
        )
        selected.extend(ranked[:per_domain])
    return selected


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


def load_selected_records(pool: Path, per_domain: int, seed: str):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit("PyArrow is required to build the browse report") from error
    manifest = json.loads((pool / "manifest.json").read_text(encoding="utf-8"))
    index_meta = manifest.get("unified_index")
    if manifest.get("stage") != "consolidate_candidates" or not index_meta:
        raise ValueError("input must be a consolidated candidate pool")
    index_path = pool / index_meta["file"]
    if sha256_file(index_path) != index_meta["sha256"]:
        raise ValueError("unified index sha256 mismatch")
    index_rows = pq.read_table(index_path).to_pylist()
    selected_index = choose_samples(index_rows, per_domain, seed)
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_index:
        by_path[row["candidate_relpath"]].append(row)
    records: list[dict[str, Any]] = []
    for relpath in sorted(by_path):
        table = pq.read_table(pool / relpath).to_pylist()
        for index_row in by_path[relpath]:
            record = table[index_row["row_in_shard"]]
            if record["document_id"] != index_row["document_id"]:
                raise ValueError(f"index location mismatch: {index_row['document_id']}")
            records.append(record)
    return manifest, sorted(records, key=lambda row: (row["domain"], audit_rank(seed, row["document_id"])))


def preview_text(text: str, head_chars: int = 1800, tail_chars: int = 1000) -> str:
    if len(text) <= head_chars + tail_chars + 200:
        return text
    omitted = len(text) - head_chars - tail_chars
    return f"{text[:head_chars]}\n\n…… 中间省略 {omitted:,} 个字符 ……\n\n{text[-tail_chars:]}"


def render_html(records: list[dict[str, Any]], source_manifest: dict[str, Any], seed: str) -> str:
    domains = Counter(row["domain"] for row in records)
    nav = "".join(
        f'<button type="button" data-domain="{html.escape(domain)}">{html.escape(domain)} ({count})</button>'
        for domain, count in sorted(domains.items())
    )
    cards = []
    for number, row in enumerate(records, 1):
        reasons = row.get("rule_reason_codes") or []
        reason_text = ", ".join(reasons) if reasons else "规则过滤已通过；没有已知规则告警"
        metadata = {
            "来源": row["source_id"],
            "语言": row["language"],
            "token": f"{row['token_count']:,}",
            "字符": f"{len(row['text']):,}",
            "split": row["split"],
            "license": row["license_status"],
            "family": row["family_id"],
            "定位": row["source_locator"],
        }
        chips = "".join(
            f'<span class="chip"><b>{html.escape(key)}</b> {html.escape(str(value))}</span>'
            for key, value in metadata.items()
        )
        cards.append(
            f"""
<article class="sample" data-domain="{html.escape(row['domain'])}">
  <header><span class="number">#{number}</span><h2>{html.escape(row['domain'])}</h2></header>
  <div class="chips">{chips}</div>
  <p class="rule"><b>自动规则：</b>{html.escape(reason_text)}</p>
  <pre class="preview">{html.escape(preview_text(row['text']))}</pre>
  <details><summary>展开完整正文</summary><pre>{html.escape(row['text'])}</pre></details>
  <p class="docid">document_id: {html.escape(row['document_id'])}</p>
</article>"""
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P0 候选数据快速浏览</title>
<style>
:root{{--bg:#f5f7fa;--card:#fff;--line:#d9dee8;--ink:#172033;--muted:#687386;--accent:#2457d6}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,-apple-system,sans-serif}}
main{{max-width:1100px;margin:auto;padding:28px 18px 80px}} .intro,.sample{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:16px}}
h1{{margin:0 0 8px;font-size:26px}} h2{{font-size:18px;margin:0}} .muted,.docid{{color:var(--muted)}}
.toolbar{{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px;position:sticky;top:0;background:rgba(245,247,250,.95);padding:10px 0;z-index:2}}
button{{border:1px solid var(--line);background:white;border-radius:999px;padding:7px 12px;cursor:pointer}} button.active{{background:var(--accent);color:white;border-color:var(--accent)}}
header{{display:flex;align-items:center;gap:10px}} .number{{font-weight:700;color:var(--accent)}} .chips{{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}}
.chip{{background:#eef2f8;border-radius:6px;padding:3px 7px;font-size:12px;max-width:100%;overflow-wrap:anywhere}} .rule{{background:#f0f8f1;padding:8px 10px;border-radius:6px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#111827;color:#e5e7eb;border-radius:8px;padding:14px;max-height:560px;overflow:auto;font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}}
details summary{{cursor:pointer;color:var(--accent);font-weight:600;margin:8px 0}} .docid{{font:11px ui-monospace,monospace;overflow-wrap:anywhere}}
</style></head><body><main>
<section class="intro"><h1>P0 候选数据快速浏览</h1>
<p>这里只用于发现<strong>成片的系统性问题</strong>。不需要逐条打标签；每个桶随便翻几条即可。</p>
<p class="muted">报告版本：{REPORT_VERSION} ｜ 抽样种子：{html.escape(seed)} ｜ 样本：{len(records)} ｜ 来源候选：{source_manifest['counts']['retained_documents']} 篇 / {source_manifest['counts']['retained_tokens']:,} token</p>
</section><nav class="toolbar"><button class="active" type="button" data-domain="all">全部 ({len(records)})</button>{nav}</nav>
{''.join(cards)}
</main><script>
document.querySelectorAll('button[data-domain]').forEach(function(btn){{btn.addEventListener('click',function(){{
 document.querySelectorAll('button[data-domain]').forEach(function(x){{x.classList.remove('active')}}); btn.classList.add('active');
 var d=btn.dataset.domain; document.querySelectorAll('.sample').forEach(function(x){{x.hidden=d!=='all'&&x.dataset.domain!==d}});
}})}});
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-domain", type=int, default=10)
    parser.add_argument("--seed", default=REPORT_VERSION)
    args = parser.parse_args()
    if args.per_domain <= 0:
        raise ValueError("per-domain must be positive")
    manifest, records = load_selected_records(args.pool.resolve(), args.per_domain, args.seed)
    report = render_html(records, manifest, args.seed)
    atomic_write(args.output.resolve(), report)
    report_hash = sha256_file(args.output.resolve())
    sidecar = {
        "report_version": REPORT_VERSION,
        "seed": args.seed,
        "source_pool": str(args.pool.resolve()),
        "source_manifest_sha256": sha256_file(args.pool.resolve() / "manifest.json"),
        "per_domain": args.per_domain,
        "counts": dict(sorted(Counter(row["domain"] for row in records).items())),
        "documents": [row["document_id"] for row in records],
        "html": {"file": args.output.name, "bytes": args.output.stat().st_size, "sha256": report_hash},
    }
    sidecar_path = args.output.with_suffix(".manifest.json")
    atomic_write(sidecar_path, json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(sidecar, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
