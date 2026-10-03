#!/usr/bin/env python3
"""TensorBoard logging and a standalone memory-eviction viewer."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


def create_summary_writer(log_dir: Path, enabled: bool = True) -> Any | None:
    """Create TensorBoard lazily so importing experiments does not require it."""
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:  # pragma: no cover - depends on the runtime image
        raise RuntimeError(
            "TensorBoard is not installed. Run `python -m pip install -r "
            "requirements-phase0-b2.txt` or pass --no-tensorboard."
        ) from exc
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def trace_as_markdown(trace: dict[str, Any]) -> str:
    """Render one eviction decision in TensorBoard's Text panel."""
    lines = [
        f"**目标：** {_markdown_escape(trace['goal'])}",
        "",
        f"split=`{trace['split']}` · episode=`{trace['episode']}` · "
        f"decision=`{trace['decision']}` · regret=`{trace['regret']:.4f}`",
        "",
        "| 操作 | 记录 | 年龄 | 真实效用 | 预测效用 |",
        "|---|---|---:|---:|---:|",
    ]
    for record in trace["records"]:
        flags = []
        if record["evicted"]:
            flags.append("**模型淘汰**")
        if record["oracle"]:
            flags.append("Oracle可淘汰")
        if record["is_candidate"]:
            flags.append("新候选")
        lines.append(
            f"| {' / '.join(flags) or '保留'} | {_markdown_escape(record['text'])} | "
            f"{record['age']} | {record['utility']:.4f} | {record['score']:.4f} |"
        )
    return "\n".join(lines)


def log_evaluation_to_tensorboard(
    writer: Any | None,
    rows: Sequence[dict[str, object]],
    summary: dict[str, object],
    metadata: dict[str, object],
    traces: Sequence[dict[str, Any]],
) -> None:
    if writer is None:
        return

    writer.add_text("run/config", f"```json\n{json.dumps(metadata, indent=2, ensure_ascii=False)}\n```")
    for key, value in summary["scores"].items():
        split, policy = str(key).split("/", 1)
        writer.add_scalar(f"evaluation/{split}/success/{policy}", float(value), 0)
    for split, item in summary["decision"].items():
        writer.add_scalar(
            f"evaluation/{split}/oracle_fifo_gap", float(item["oracle_fifo_gap"]), 0
        )
        writer.add_scalar(
            f"evaluation/{split}/text_gap_recovered", float(item["text_gap_recovered"]), 0
        )

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["policy"]))].append(row)
    metrics = (
        "clause_accuracy",
        "required_recall",
        "eviction_accuracy",
        "eviction_regret",
        "stale_value_rate",
    )
    for (split, policy), items in grouped.items():
        for metric in metrics:
            writer.add_scalar(
                f"evaluation/{split}/{metric}/{policy}",
                _mean([float(row[metric]) for row in items]),
                0,
            )

    # Keep TensorBoard readable: detailed browsing remains in eviction_viewer.html.
    for trace in traces[:18]:
        tag = f"memory_eviction/{trace['split']}/episode_{trace['episode']}/step_{trace['decision']:03d}"
        writer.add_text(tag, trace_as_markdown(trace), 0)
    writer.flush()


def write_eviction_artifacts(output: Path, traces: Sequence[dict[str, Any]]) -> None:
    """Write auditable JSONL plus a dependency-free interactive HTML viewer."""
    output.mkdir(parents=True, exist_ok=True)
    jsonl = output / "eviction_trace.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")

    # Escape '<' so generated record text cannot terminate the JSON script tag.
    payload = json.dumps(list(traces), ensure_ascii=False).replace("<", "\\u003c")
    template = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>记忆淘汰查看器</title>
<style>
:root{color-scheme:light dark;--bg:#f6f7f9;--panel:#fff;--text:#19202a;--muted:#687386;--line:#d8dee8;--accent:#315efb;--bad:#c73535;--good:#237a4b;--new:#7a4bc2}
@media(prefers-color-scheme:dark){:root{--bg:#101318;--panel:#171b22;--text:#e9edf4;--muted:#9ba7b8;--line:#343c49;--accent:#7d9cff;--bad:#ff8585;--good:#73d49d;--new:#c5a0ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,"PingFang SC",sans-serif}.shell{max-width:1280px;margin:auto;padding:22px}h1{font-size:22px;margin:0 0 16px}.controls{display:grid;grid-template-columns:repeat(3,minmax(160px,1fr)) auto auto;gap:10px;align-items:end;margin-bottom:14px}label{display:grid;gap:5px;color:var(--muted)}select,button{min-height:38px;border:1px solid var(--line);border-radius:7px;background:var(--panel);color:var(--text);padding:7px 10px}button{cursor:pointer}.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:15px;margin-bottom:14px}.summary{display:grid;grid-template-columns:2fr repeat(4,1fr);gap:14px}.label{color:var(--muted);font-size:12px}.value{font-weight:600;margin-top:3px}.goal{font-weight:500}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;min-width:860px}th,td{text-align:left;padding:10px 8px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-size:12px;font-weight:500}.record{min-width:320px}.flag{display:inline-block;margin-right:6px;font-size:12px;font-weight:600}.evicted{color:var(--bad)}.oracle{color:var(--good)}.candidate{color:var(--new)}.bar{width:140px;height:8px;background:color-mix(in srgb,var(--line) 60%,transparent);border-radius:6px;overflow:hidden;margin-top:5px}.fill{height:100%;background:var(--accent)}tr.model-evicted{background:color-mix(in srgb,var(--bad) 8%,transparent)}.empty{color:var(--muted)}@media(max-width:760px){.shell{padding:14px}.controls{grid-template-columns:1fr 1fr}.controls label:first-child{grid-column:1/-1}.summary{grid-template-columns:1fr 1fr}.summary>div:first-child{grid-column:1/-1}}
</style>
</head>
<body>
<main class="shell">
<h1>记忆淘汰查看器</h1>
<div class="controls" aria-label="选择淘汰决策">
  <label>测试集<select id="split"></select></label>
  <label>Episode<select id="episode"></select></label>
  <label>决策<select id="decision"></select></label>
  <button id="prev" type="button">上一步</button>
  <button id="next" type="button">下一步</button>
</div>
<section class="panel summary" aria-live="polite">
  <div><div class="label">目标</div><div class="value goal" id="goal"></div></div>
  <div><div class="label">任务</div><div class="value" id="task"></div></div>
  <div><div class="label">容量</div><div class="value" id="capacity"></div></div>
  <div><div class="label">本步 regret</div><div class="value" id="regret"></div></div>
  <div><div class="label">最终成功</div><div class="value" id="success"></div></div>
</section>
<section class="panel table-wrap">
<table><thead><tr><th>状态</th><th class="record">记录</th><th>年龄</th><th>真实效用</th><th>预测效用</th></tr></thead><tbody id="records"></tbody></table>
</section>
</main>
<script id="trace-data" type="application/json">__TRACE_DATA__</script>
<script>
const data=JSON.parse(document.getElementById('trace-data').textContent);const $=id=>document.getElementById(id);const split=$('split'),episode=$('episode'),decision=$('decision');
const unique=a=>[...new Set(a)];const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function options(el,values,selected){el.innerHTML=values.map(v=>`<option value="${esc(v)}" ${String(v)===String(selected)?'selected':''}>${esc(v)}</option>`).join('')}
function traces(){return data.filter(x=>x.split===split.value&&String(x.episode)===episode.value)}
function syncEpisodes(){const values=unique(data.filter(x=>x.split===split.value).map(x=>x.episode));options(episode,values,values[0]);syncDecisions()}
function syncDecisions(){const rows=traces();options(decision,rows.map(x=>x.decision),rows[0]?.decision);render()}
function bar(v){const p=Math.max(0,Math.min(100,Number(v)*100));return `<div>${Number(v).toFixed(4)}</div><div class="bar"><div class="fill" style="width:${p}%"></div></div>`}
function render(){const t=traces().find(x=>String(x.decision)===decision.value);if(!t){$('records').innerHTML='<tr><td colspan="5" class="empty">没有记录</td></tr>';return}$('goal').textContent=t.goal;$('task').textContent=t.task;$('capacity').textContent=t.capacity;$('regret').textContent=Number(t.regret).toFixed(4);$('success').textContent=t.episode_success?'是':'否';$('records').innerHTML=t.records.map(r=>{const flags=[r.evicted?'<span class="flag evicted">模型淘汰</span>':'',r.oracle?'<span class="flag oracle">Oracle可淘汰</span>':'',r.is_candidate?'<span class="flag candidate">新候选</span>':''].join('');return `<tr class="${r.evicted?'model-evicted':''}"><td>${flags||'保留'}</td><td class="record">${esc(r.text)}<div class="label">${esc(r.uid)}</div></td><td>${r.age}</td><td>${bar(r.utility)}</td><td>${bar(r.score)}</td></tr>`}).join('')}
split.addEventListener('change',syncEpisodes);episode.addEventListener('change',syncDecisions);decision.addEventListener('change',render);$('prev').onclick=()=>{decision.selectedIndex=Math.max(0,decision.selectedIndex-1);render()};$('next').onclick=()=>{decision.selectedIndex=Math.min(decision.options.length-1,decision.selectedIndex+1);render()};
const splits=unique(data.map(x=>x.split));options(split,splits,splits[0]);syncEpisodes();
</script>
</body></html>'''
    (output / "eviction_viewer.html").write_text(
        template.replace("__TRACE_DATA__", payload), encoding="utf-8"
    )
