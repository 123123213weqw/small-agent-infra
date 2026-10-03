#!/usr/bin/env python3
"""B2.7: connect a frozen B2.4 utility predictor to the B2.6 reader proxy.

The predictor sees Chinese D-family renderings matching its training input.
The reader sees the same selected directed edges under all policies. Labels
are never passed into predictor inputs; oracle is an evaluation-only bound.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.phase0_ab import Episode, Record
from experiments.phase0_b21_data import B21Renderer
from experiments.phase0_b21_model import UtilitySetModel
from experiments.phase0_b21_train import load_config, predict_groups, resolve_seed_config
from experiments.phase0_b26_reader_proxy import choose_old, episode, prompt


CAPACITY = 12
RECENT = 16


def native_episode(raw: dict, seed: int) -> Episode:
    eid = f"e{seed}"
    records = tuple(Record(uid=f"{eid}:r{i}", position=i, kind="edge",
                           key=a, value=b, refs=(a, b))
                    for i, (a, b, _) in enumerate(raw["records"]))
    required = tuple(r.uid for r, source in zip(records, raw["records"]) if source[2])
    return Episode(eid=eid, task="relation_chain", records=records,
                   clauses=(required,), goal_tokens=frozenset({raw["start"], raw["answer"]}),
                   target_key=raw["start"])


def predictor_group(ep: Episode, renderer: B21Renderer, memory: list[Record], candidate: Record):
    records = [*memory, candidate]
    goal, _ = renderer.goal(ep)
    return {
        "goal": goal,
        "recent_context": [
            {"text": renderer.record(r)} for r in ep.records[max(0, candidate.position - 2):candidate.position]
        ],
        "records": [{"uid": r.uid, "text": renderer.record(r),
                     "event_index": r.position,
                     "relative_age": candidate.position - r.position,
                     "is_candidate": r.uid == candidate.uid,
                     "source": "tool"} for r in records],
        # Required by the existing collator but not read by model.forward().
        "utilities": [0.0] * len(records), "oracle_eviction_indices": [0],
    }


def predictor_select(model, config, native: list[Episode], device: torch.device):
    renderers = [B21Renderer("D", f"entity_{ep.eid}") for ep in native]
    memory = [[] for _ in native]
    for position in range(len(native[0].records) - RECENT):
        pending = []
        for k, ep in enumerate(native):
            candidate = ep.records[position]
            if len(memory[k]) < CAPACITY:
                memory[k].append(candidate)
                continue
            pending.append((k, predictor_group(ep, renderers[k], memory[k], candidate)))
        if not pending:
            continue
        scores = predict_groups(model, [item[1] for item in pending], config, device)
        for (k, group), values in zip(pending, scores):
            victim = min(range(len(values)), key=lambda j: (values[j], group["records"][j]["event_index"]))
            candidate = native[k].records[position]
            combined = [*memory[k], candidate]
            memory[k] = [r for j, r in enumerate(combined) if j != victim]
        if (position + 1) % 16 == 0:
            print(f"predictor decisions processed through position {position}", flush=True)
    return [[r.position for r in retained] for retained in memory]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2600)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reader", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    config = resolve_seed_config(load_config(args.config))
    model = UtilitySetModel(config).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"])
    model.eval()
    raw, native, keys = [], [], []
    for hops in (2, 4):
        for trial in range(args.n):
            key = args.seed + 1000 * hops + trial
            e = episode(key, hops)
            raw.append(e)
            native.append(native_episode(e, key))
            keys.append((hops, trial))
    with torch.inference_mode():
        predictor_indices = predictor_select(model, config, native, device)
    del model
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.reader)
    reader = AutoModelForCausalLM.from_pretrained(args.reader, dtype=torch.float16).to(device).eval()
    policies = ("recency", "lexical", "graph", "oracle", "predictor", "full_context")
    rows = []
    for k, (e, (hops, trial)) in enumerate(zip(raw, keys)):
        target = e["letter"]
        for policy in policies:
            indices = predictor_indices[k] if policy == "predictor" else choose_old(e, policy)
            text = prompt(e, indices)
            messages = [{"role": "system", "content": "Answer the user's question accurately and concisely."},
                        {"role": "user", "content": text}]
            ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                return_tensors="pt").to(device)
            with torch.inference_mode():
                output = reader.generate(ids, attention_mask=torch.ones_like(ids),
                                         max_new_tokens=8, do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()
            match = re.search(r"\b([ABCD])\b", answer.upper())
            prediction = match.group(1) if match else ""
            kept = sum(e["records"][i][2] for i in indices if i < len(e["records"]) - RECENT)
            needed_old = sum(x[2] for x in e["records"][:-RECENT])
            rows.append({"hops": hops, "trial": trial, "policy": policy,
                         "correct": int(prediction == target), "prediction": prediction,
                         "target": target, "response": answer,
                         "required_old_kept": kept, "required_old_total": needed_old,
                         "complete_evidence": int(kept == needed_old),
                         "input_tokens": int(ids.shape[1])})
        if (trial + 1) % 8 == 0:
            print(f"reader hops={hops} completed={trial + 1}/{args.n}", flush=True)
    summary = {}
    for hops in (2, 4):
        summary[str(hops)] = {}
        for policy in policies:
            subset = [r for r in rows if r["hops"] == hops and r["policy"] == policy]
            summary[str(hops)][policy] = {
                "accuracy": sum(r["correct"] for r in subset) / len(subset),
                "complete_evidence_rate": sum(r["complete_evidence"] for r in subset) / len(subset),
                "mean_required_old_kept": sum(r["required_old_kept"] for r in subset) / len(subset),
                "mean_input_tokens": sum(r["input_tokens"] for r in subset) / len(subset),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"config": {"config": str(args.config),
                           "checkpoint": str(args.checkpoint), "n": args.n,
                           "seed": args.seed, "capacity": CAPACITY, "recent": RECENT},
                           "summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
