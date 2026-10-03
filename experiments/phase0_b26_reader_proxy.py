#!/usr/bin/env python3
"""B2.6: frozen-reader, equal-evidence-budget memory selection proxy.

This does not implement an incremental KV cache or train a utility model.
It asks whether selecting useful expired records makes a frozen reader more
accurate than retaining the most recent expired records.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def episode(seed: int, hops: int, length: int = 128, recent: int = 16):
    rng = random.Random(seed)
    labels = [f"n{n:04d}" for n in rng.sample(range(10000), 2 * length + hops + 8)]
    chain = labels[: hops + 1]
    decoy_nodes = labels[hops + 1 :]
    records = []
    for i in range(length):
        records.append((decoy_nodes[2 * i], decoy_nodes[2 * i + 1], False))
    # Each required edge is in the expired region, distributed across the
    # stream; no unrelated edge touches a chain node.
    positions = sorted(rng.sample(range(4, length - recent - 4), hops))
    for j, pos in enumerate(positions):
        records[pos] = (chain[j], chain[j + 1], True)
    answer = chain[-1]
    negatives = rng.sample([r[1] for r in records if not r[2]], 3)
    choices = [answer, *negatives]
    rng.shuffle(choices)
    return {"records": records, "start": chain[0], "answer": answer,
            "choices": choices, "letter": "ABCD"[choices.index(answer)],
            "positions": positions, "hops": hops}


def choose_old(ep, policy: str, capacity: int = 12, recent: int = 16):
    old = ep["records"][:-recent]
    if policy == "full_context":
        return list(range(len(old)))
    if policy == "oracle":
        required = [i for i, (_, _, needed) in enumerate(old) if needed]
        filler = [i for i in range(len(old) - 1, -1, -1) if i not in required]
        return sorted((required + filler[:capacity - len(required)]))
    memory = []
    for i, (a, b, _) in enumerate(old):
        memory.append(i)
        if len(memory) <= capacity:
            continue
        if policy == "recency":
            memory.pop(0)
            continue
        if policy == "lexical":
            score = lambda j: int(ep["start"] in old[j][:2])
        elif policy == "graph":
            reachable = {ep["start"]}
            changed = True
            while changed:
                before = len(reachable)
                for j in memory:
                    src, dst, _ = old[j]
                    if src in reachable:
                        reachable.add(dst)
                changed = len(reachable) != before
            score = lambda j: int(old[j][0] in reachable)
        else:
            raise ValueError(policy)
        # Oldest lowest-score record is evicted. No hindsight labels used.
        victim = min(memory, key=lambda j: (score(j), j))
        memory.remove(victim)
    return sorted(memory)


def prompt(ep, indices, recent: int = 16):
    old = ep["records"][:-recent]
    selected = [old[i] for i in indices] + ep["records"][-recent:]
    lines = [f"{i + 1}. {a} -> {b}" for i, (a, b, _) in enumerate(selected)]
    options = "\n".join(f"{letter}. {value}" for letter, value in zip("ABCD", ep["choices"]))
    return ("Read the directed mappings below. Starting from " + ep["start"]
            + f", follow exactly {ep['hops']} arrows. Which node is reached? "
            "Only the listed mappings may be used. Reply with one letter: A, B, C, or D.\n\n"
            + "Mappings:\n" + "\n".join(lines) + "\n\nOptions:\n" + options + "\nAnswer:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2600)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(args.device).eval()
    rows = []
    policies = ["recency", "lexical", "graph", "oracle", "full_context"]
    for hops in (2, 4):
        for trial in range(args.n):
            ep = episode(args.seed + 1000 * hops + trial, hops)
            for policy in policies:
                indices = choose_old(ep, policy)
                user = prompt(ep, indices)
                messages = [{"role": "system", "content": "Answer the user's question accurately and concisely."},
                            {"role": "user", "content": user}]
                ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                    return_tensors="pt").to(args.device)
                with torch.inference_mode():
                    output = model.generate(ids, attention_mask=torch.ones_like(ids),
                                            max_new_tokens=8, do_sample=False,
                                            pad_token_id=tokenizer.eos_token_id)
                response = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()
                match = re.search(r"\b([ABCD])\b", response.upper())
                prediction = match.group(1) if match else ""
                selected_required = sum(ep["records"][i][2] for i in indices)
                rows.append({"hops": hops, "trial": trial, "policy": policy,
                             "correct": int(prediction == ep["letter"]),
                             "complete_evidence": int(selected_required == hops),
                             "required_kept": selected_required,
                             "input_tokens": int(ids.shape[1]), "response": response,
                             "prediction": prediction, "target": ep["letter"],
                             "positions": ep["positions"]})
            if (trial + 1) % 4 == 0:
                print(f"hops={hops} completed={trial + 1}/{args.n}", flush=True)
    summary = {}
    for hops in (2, 4):
        summary[str(hops)] = {}
        for policy in policies:
            subset = [r for r in rows if r["hops"] == hops and r["policy"] == policy]
            summary[str(hops)][policy] = {
                "accuracy": sum(r["correct"] for r in subset) / len(subset),
                "complete_evidence_rate": sum(r["complete_evidence"] for r in subset) / len(subset),
                "mean_input_tokens": sum(r["input_tokens"] for r in subset) / len(subset),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"config": {"model": args.model, "n": args.n,
                        "seed": args.seed, "length": 128, "recent": 16, "capacity": 12},
                        "summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
