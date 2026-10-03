#!/usr/bin/env python3
"""pack_p0_shards_v2.py — 保留代码缩进，打包知识点与推理链蒸馏语料。

对齐 p0_1m_frozen_v1 的 schema 与约定:
  * 15 个字段 (document_id / family_id / normalized_sha256 / split / token_count / ...)
  * 域桶: math_zh, code_python
  * license_status = generated (合成数据)
  * family 级切分: 同一 family 绝不跨 train/val/test (P0 报告 family_split_conflicts=0)
  * 与已有 P0 成品做精确去重 (normalized_sha256)

用法:
  python3 p0_pack_generated_shards_v2.py --outdir <dir> [--existing-p0 <p0_frozen_dir>]
"""
import argparse, ast, hashlib, json, os, re, sys
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

TOK_PATH = "/srv/small-agent/models/Qwen3-0.6B-Base-tokenizer/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/tokenizer.json"
TOK_SHA = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
SEQ_LEN = 2048

SCHEMA = pa.schema([
    ("document_id", pa.string()), ("source_id", pa.string()), ("source_revision", pa.string()),
    ("source_locator", pa.string()), ("family_id", pa.string()), ("domain", pa.string()),
    ("language", pa.string()), ("text", pa.string()), ("normalized_sha256", pa.string()),
    ("license_status", pa.string()), ("split", pa.string()), ("token_count", pa.int64()),
    ("sample_rank", pa.string()), ("source_metadata_json", pa.string()),
    ("teacher_metadata_json", pa.string()),
])

def norm_text(t):
    t = (t or "").replace("\r\n", "\n").replace("\r", "\n")
    # Python 的缩进具有语义；不能全文压缩连续空格或逐行 lstrip。
    return t.strip("\n")

def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()

def split_of(family_id, test_mod=50, val_mod=50):
    """family 级确定性切分: ~2% test, ~2% validation, 其余 train。"""
    h = int(family_id[:12], 16)
    if h % test_mod == 0:
        return "test"
    if h % test_mod == 1:
        return "validation"
    return "train"

# ---------------------------------------------------------------- 载入两个来源
def load_knowledge(path):
    """知识点语料: 中文讲解 + 可执行验证代码 (945 条)。"""
    prefix = re.compile(r"^\s*\[(?:math|code)\]\s*")
    docs = []
    for i, line in enumerate(open(path)):
        r = json.loads(line)
        topic = prefix.sub("", r["topic"]).strip()     # v1 生成的 topic 可能带 [math]/[code] 前缀
        text = (f"# {topic}\n\n{r['passage']}\n\n"
                f"```python\n{r['verify']}\n```\n")
        docs.append({
            "text": text,
            "family_key": "kv:" + r.get("topic_key", topic),
            "domain": "math_zh" if r["kind"] == "math" else "code_python",
            "source_id": "gen_verified_v2",
            "source_revision": "deepseek-v4-flash",
            "source_locator": f"kv-{r['kind']}-{i:05d}",
            "meta": {"topic": topic, "kind": r["kind"],
                     "asserts": r.get("asserts"), "gen_attempts": r.get("gen_attempts"),
                     "verify_ms": r.get("verify_ms"), "variant": r.get("variant")},
            "teacher": {"verification": "exact_execution",
                        "verify_code_lines": len(r["verify"].splitlines()),
                        "asserts": r.get("asserts")},
        })
    return docs

def load_distill(path, thinking_dir=None):
    """推理链蒸馏: 题目→分步解题 / 任务→思路+代码 (588 条)。"""
    docs = []
    for i, line in enumerate(open(path)):
        r = json.loads(line)
        if r["track"] == "math":
            text = f"# 题目\n{r['problem']}\n\n# 解题过程\n{r['solution'].strip()}\n"
            fam = "dm:" + r["family"]
            dom = "math_zh"
        else:
            text = r["doc"]
            fam = "dc:" + r["family"]
            dom = "code_python"
        docs.append({
            "text": text,
            "family_key": fam,
            "domain": dom,
            "source_id": "distill_reasoning_v1",
            "source_revision": "deepseek-v4-flash",
            "source_locator": f"dist-{r['track']}-{i:05d}",
            "meta": {"track": r["track"], "family": r["family"],
                     "variation": r.get("variation", ""), "attempts": r.get("attempts"),
                     "thinking_chars": len(r.get("thinking", ""))},
            "thinking": r.get("thinking", ""),
            "teacher": {"verification": r.get("verdict"),
                        "ground_truth": r.get("gt") if r["track"] == "math" else None,
                        "answer_extracted": r.get("answer") if r["track"] == "math" else None},
        })
    return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knowledge", default="/home/user/train-data/out/corpus.jsonl")
    ap.add_argument("--distill", default="/srv/small-agent/datasets/small-agent-p0/p0_distill_v1/accepted.jsonl")
    ap.add_argument("--existing-p0", default="/srv/small-agent/datasets/small-agent-p0/p0_1m_smoke_v1/p0_1m_frozen_v1")
    ap.add_argument("--outdir", default="/srv/small-agent/datasets/small-agent-p0/p0_gen_verified_v2")
    ap.add_argument("--save-thinking", action="store_true", help="额外导出思考链 (单独目录)")
    args = ap.parse_args()

    if os.path.isdir(args.outdir) and os.listdir(args.outdir):
        raise FileExistsError(f"refusing to overwrite existing dataset: {args.outdir}")
    os.makedirs(args.outdir, exist_ok=True)
    if hashlib.sha256(open(TOK_PATH, "rb").read()).hexdigest() != TOK_SHA:
        raise ValueError("tokenizer.json SHA-256 mismatch")
    tok = Tokenizer.from_file(TOK_PATH)
    print(f"tokenizer vocab={tok.get_vocab_size()}")

    # 已有 P0 成品里的 normalized_sha256, 用于跨数据集去重
    existing = set()
    if os.path.isdir(args.existing_p0):
        for dom in os.listdir(args.existing_p0):
            d = os.path.join(args.existing_p0, dom)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".parquet"):
                    t = pq.ParquetFile(os.path.join(d, f)).read(columns=["normalized_sha256"])
                    existing.update(t.column("normalized_sha256").to_pylist())
    print(f"已有 P0 文档指纹 {len(existing)} 个 (用于跨集去重)")

    kn = load_knowledge(args.knowledge)
    di = load_distill(args.distill)
    docs = kn + di
    print(f"载入 {len(docs)} 条 (知识点 {len(kn)} + 蒸馏 {len(di)})")

    # 规范化 + 去重
    rows, seen, drop_self, drop_p0 = [], set(), 0, 0
    python_blocks_checked = 0
    for d in docs:
        text = norm_text(d["text"])
        if not text:
            continue
        nh = sha(text)
        if nh in seen:
            drop_self += 1
            continue
        if nh in existing:
            drop_p0 += 1
            continue
        for block in re.findall(r"```(?:python|py)\s*\n(.*?)```", text, flags=re.I | re.S):
            try:
                ast.parse(block)
            except SyntaxError as error:
                raise ValueError(
                    f"invalid final Python block in {d['source_locator']}: {error}"
                ) from error
            python_blocks_checked += 1
        seen.add(nh)
        rows.append({
            "document_id": nh,
            "source_id": d["source_id"],
            "source_revision": d["source_revision"],
            "source_locator": d["source_locator"],
            "family_id": sha(d["family_key"]),
            "domain": d["domain"],
            "language": "zh",
            "text": text,
            "normalized_sha256": nh,
            "license_status": "generated",
            "split": split_of(sha(d["family_key"])),
            "token_count": len(tok.encode(text).ids),
            "sample_rank": sha("rank:" + nh),
            "source_metadata_json": json.dumps(d["meta"], ensure_ascii=False, sort_keys=True),
            "teacher_metadata_json": json.dumps(d["teacher"], ensure_ascii=False, sort_keys=True),
            "_thinking": d.get("thinking", ""),
        })
    print(f"去重: 内部重复 {drop_self} | 与已有 P0 重复 {drop_p0} | 保留 {len(rows)}")

    # family 级切分完整性检查
    fam_split = {}
    for r in rows:
        fam_split.setdefault(r["family_id"], set()).add(r["split"])
    conflicts = {f: s for f, s in fam_split.items() if len(s) > 1}
    assert not conflicts, f"family 跨集冲突: {list(conflicts)[:3]}"
    print(f"family 数 {len(fam_split)} | 跨集冲突 0")

    # 写分片
    shards = []
    for dom in sorted({r["domain"] for r in rows}):
        sub = [r for r in rows if r["domain"] == dom]
        if not sub:
            continue
        out_dir = os.path.join(args.outdir, dom)
        os.makedirs(out_dir, exist_ok=True)
        sub.sort(key=lambda r: r["sample_rank"])           # 确定性顺序
        tbl = pa.Table.from_pylist(
            [{k: r[k] for k in SCHEMA.names} for r in sub], schema=SCHEMA)
        fp = os.path.join(out_dir, "part-00000.parquet")
        pq.write_table(tbl, fp, compression="zstd")
        h = hashlib.sha256(open(fp, "rb").read()).hexdigest()
        shards.append({"domain": dom, "file": "part-00000.parquet", "rows": len(sub),
                       "tokens": sum(r["token_count"] for r in sub),
                       "bytes": os.path.getsize(fp), "sha256": h})
        print(f"  {dom:14s} {len(sub):5d} 行  {sum(r['token_count'] for r in sub):7,d} tokens  "
              f"{os.path.getsize(fp):9,d} bytes")

    # 思考链单独导出 (可选)
    thinking_files = {}
    if args.save_thinking:
        td = os.path.join(args.outdir, "thinking")
        os.makedirs(td, exist_ok=True)
        n = 0
        with open(os.path.join(td, "thinking.jsonl"), "w") as fh:
            for r in rows:
                if r["_thinking"]:
                    fh.write(json.dumps({"document_id": r["document_id"], "domain": r["domain"],
                                         "thinking": r["_thinking"]}, ensure_ascii=False) + "\n")
                    n += 1
        fp = os.path.join(td, "thinking.jsonl")
        thinking_files["thinking.jsonl"] = {"rows": n, "bytes": os.path.getsize(fp)}
        print(f"思考链导出 {n} 条 -> thinking/thinking.jsonl")

    split_counts = {}
    for r in rows:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1

    manifest = {
        "version": "p0_gen_verified_v2",
        "stage": "p0_gen_verified_packed",
        "normalization": "newline_only_preserve_python_indentation_v2",
        "final_python_syntax": {"blocks_checked": python_blocks_checked, "parse_failed": 0},
        "inputs": {
            "knowledge": {"file": args.knowledge,
                          "sha256": hashlib.sha256(open(args.knowledge, "rb").read()).hexdigest()},
            "distill": {"file": args.distill,
                        "sha256": hashlib.sha256(open(args.distill, "rb").read()).hexdigest()},
            "existing_p0": {"dir": args.existing_p0, "fingerprints": len(existing)},
        },
        "tokenizer": {"model_id": "Qwen/Qwen3-0.6B-Base",
                      "tokenizer_json": TOK_PATH, "tokenizer_json_sha256": TOK_SHA,
                      "sequence_length": SEQ_LEN, "vocab_size": tok.get_vocab_size()},
        "counts": {
            "documents": len(rows),
            "tokens": sum(r["token_count"] for r in rows),
            "families": len(fam_split), "family_split_conflicts": 0,
            "dropped_internal_dup": drop_self, "dropped_vs_existing_p0": drop_p0,
            "by_source": {s: sum(1 for r in rows if r["source_id"] == s)
                          for s in sorted({r["source_id"] for r in rows})},
            "by_domain": {d: sum(1 for r in rows if r["domain"] == d)
                          for d in sorted({r["domain"] for r in rows})},
        },
        "splits": split_counts,
        "license_gate": {"statuses": {"generated": len(rows)}, "training_eligible": None,
                         "note": "全部为教师模型生成的合成数据, license_status=generated"},
        "provenance": {
            "knowledge.corpus": "945 条知识点语料, 每条自带可执行 assert 并实际跑通",
            "distill.accepted": "588 条推理链蒸馏, 数学答案由本地 Python 独立算出, 代码与参考实现随机对拍",
        },
        "shards": shards,
        "thinking": thinking_files or None,
    }
    mp = os.path.join(args.outdir, "manifest.json")
    json.dump(manifest, open(mp, "w"), ensure_ascii=False, indent=2)
    print()
    print(json.dumps({k: manifest[k] for k in ("counts", "splits", "shards")},
                     ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
