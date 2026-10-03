#!/usr/bin/env python3
"""Generate a pinned, resumable V100→L40→V100 non-Wikipedia review run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex


VROOT = "/srv/small-agent/datasets/small-agent-p0/p0_500m_v1"
LROOT = "/srv/small-agent/datasets/small-agent-p0/p0_500m_v1"
L_QWEN = "/srv/small-agent/experiments/qwen38-fp8-serving-v1"
L_PY = "/srv/small-agent/venvs/vllm-qwen38-fp8-v1/bin/python"
V_PY = "/home/user/anaconda3/bin/python"


def item(host: str, path: str) -> dict:
    return {"host": host, "path": path}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--limit-per-source", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,48}", args.batch_id):
        raise ValueError("batch-id must be lowercase ASCII letters/digits/underscores")
    if args.limit_per_source < 0:
        raise ValueError("limit-per-source must be nonnegative")
    if args.output.exists():
        raise FileExistsError(args.output)
    repo = Path(__file__).resolve().parent.parent
    vdir = f"{VROOT}/upstream_nonwiki_v1/audit/auto_review_{args.batch_id}"
    ldir = f"{LROOT}/nonwiki_new_quality_probe_v1/auto_review_{args.batch_id}"
    vscript = f"{vdir}/code/scripts/p0_nonwiki_review_batch_v1.py"
    lcode = f"{ldir}/code"
    lscript = f"{lcode}/scripts/p0_nonwiki_review_batch_v1.py"
    p0 = f"{vdir}/source.input.jsonl"
    p0m = f"{vdir}/source.input.manifest.json"
    p1 = f"{ldir}/source.input.jsonl"
    p1m = f"{ldir}/source.input.manifest.json"
    band = f"{ldir}/input_bands6_v1"
    gate = f"{ldir}/gate_v1"
    local_scripts = ["p0_nonwiki_review_batch_v1.py", "p0_run_nonwiki_qwen6_auto.sh",
                     "p0_run_qwen_wave1_6.sh", "annotate_jsonl.py", "p0_verify_qwen_six.py"]
    deploy = (
        f"ssh CPU_NODE 'mkdir -p {shlex.quote(vdir + '/code/scripts')}' && "
        f"ssh GPU_NODE 'mkdir -p {shlex.quote(lcode + '/scripts')}' && "
        f"scp -q {shlex.quote(str(repo / 'scripts/p0_nonwiki_review_batch_v1.py'))} "
        f"CPU_NODE:{shlex.quote(vscript)} && "
        + " && ".join(
            f"scp -q {shlex.quote(str(repo / 'scripts' / name))} "
            f"GPU_NODE:{shlex.quote(lcode + '/scripts/' + ('p0_run_nonwiki_qwen6.sh' if name == 'p0_run_qwen_wave1_6.sh' else name))}"
            for name in local_scripts
        )
    )
    vcode = item("CPU_NODE", vscript)
    lcode_item = item("GPU_NODE", lscript)
    stages = [
        {"name": "deploy_review_code", "kind": "command", "host": "local", "command": deploy,
         "inputs": [item("local", str(repo / "scripts" / n)) for n in local_scripts],
         "outputs": [vcode, lcode_item] + [
             item("GPU_NODE", f"{lcode}/scripts/{'p0_run_nonwiki_qwen6.sh' if n == 'p0_run_qwen_wave1_6.sh' else n}")
             for n in local_scripts[1:]]},
        {"name": "export_raw_pointer_text", "kind": "command", "host": "CPU_NODE",
         "command": f"{V_PY} {shlex.quote(vscript)} export --root {VROOT} --output {shlex.quote(p0)} --limit-per-source {args.limit_per_source}",
         "inputs": [vcode, item("CPU_NODE", f"{VROOT}/upstream_nonwiki_v1/audit/high_precision_v2/report.json")]
                   + [item("CPU_NODE", f"{VROOT}/upstream_nonwiki_v1/audit/high_precision_v2/{s}.candidate_index.parquet")
                      for s in ("fineweb_edu_new", "finemath_4plus_new", "github_code_clean_new")],
         "outputs": [item("CPU_NODE", p0), item("CPU_NODE", p0m)]},
        {"name": "copy_review_text", "kind": "copy", "inputs": [item("CPU_NODE", p0)],
         "outputs": [item("GPU_NODE", p1)]},
        {"name": "copy_review_manifest", "kind": "copy", "inputs": [item("CPU_NODE", p0m)],
         "outputs": [item("GPU_NODE", p1m)]},
        {"name": "qwen_exact_preflight", "kind": "command", "host": "GPU_NODE",
         "command": f"{L_PY} {shlex.quote(lscript)} prepare --input {shlex.quote(p1)} --output-dir {shlex.quote(band)} "
                    f"--model-path /srv/small-agent/models/Qwen3.8-27B-FP8 "
                    f"--prompt {LROOT}/code/configs/data/p0_teacher_quality_prompt_v2.txt",
         "inputs": [item("GPU_NODE", p1), item("GPU_NODE", p1m), lcode_item,
                    item("GPU_NODE", "/srv/small-agent/models/Qwen3.8-27B-FP8/tokenizer.json"),
                    item("GPU_NODE", f"{LROOT}/code/configs/data/p0_teacher_quality_prompt_v2.txt")],
         "outputs": [item("GPU_NODE", f"{band}/manifest.json"),
                     item("GPU_NODE", f"{band}/overlong.jsonl")]},
        {"name": "qwen_six_card_rating", "kind": "command", "host": "GPU_NODE",
         "command": f"DATA={shlex.quote(ldir)} CODE={shlex.quote(lcode)} "
                    f"bash {shlex.quote(lcode + '/scripts/p0_run_nonwiki_qwen6_auto.sh')} ",
         "inputs": [item("GPU_NODE", f"{band}/manifest.json"),
                    item("GPU_NODE", f"{lcode}/scripts/p0_run_nonwiki_qwen6_auto.sh"),
                    item("GPU_NODE", f"{lcode}/scripts/p0_run_nonwiki_qwen6.sh"),
                    item("GPU_NODE", f"{LROOT}/code/schemas/p0_teacher_quality_v2.json")],
         "outputs": [item("GPU_NODE", f"{ldir}/FINISHED6.json")],
         "checks": [{"host": "GPU_NODE", "command":
                     f"{L_PY} -c 'import json; x=json.load(open(\"{ldir}/FINISHED6.json\")); "
                     "assert x[\"expected\"]>0 and x[\"unattempted\"]==0'"}]},
        {"name": "strict_qwen_gate", "kind": "command", "host": "GPU_NODE",
         "command": f"{L_PY} {shlex.quote(lscript)} gate --data-dir {shlex.quote(ldir)} --output-dir {shlex.quote(gate)}",
         "inputs": [item("GPU_NODE", f"{ldir}/FINISHED6.json"),
                    item("GPU_NODE", f"{band}/manifest.json"), lcode_item],
         "outputs": [item("GPU_NODE", f"{gate}/manifest.json"),
                     item("GPU_NODE", f"{gate}/strict_keep_candidates.jsonl"),
                     item("GPU_NODE", f"{gate}/rejected_or_unscored.jsonl")]},
    ]
    for name in ("manifest.json", "strict_keep_candidates.jsonl", "rejected_or_unscored.jsonl"):
        stages.append({"name": "return_gate_" + name.replace(".", "_"), "kind": "copy",
                       "inputs": [item("GPU_NODE", f"{gate}/{name}")],
                       "outputs": [item("CPU_NODE", f"{vdir}/gate_v1/{name}")]})
    stages.append({"name": "return_overlong_quarantine", "kind": "copy",
                   "inputs": [item("GPU_NODE", f"{band}/overlong.jsonl")],
                   "outputs": [item("CPU_NODE", f"{vdir}/overlong_quarantine.jsonl")]})
    config = {"version": 1, "description":
              "Pinned new non-Wikipedia v2 candidates -> six-GPU Qwen strict candidate gate; no paid calls or training approval",
              "stages": stages}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
