"""Train a contextual-record Cross-Attention Reader before memory selection.

The pretrained LM is frozen in this experiment. Every old tool record is
encoded independently by that LM. Only the new Reader is optimized. This
tests *reading*, not FIFO or utility-based retention.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
MODEL_SHA256 = "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--init-adapter", type=Path)
    p.add_argument("--slots", type=int, choices=(1, 8), required=True)
    p.add_argument("--max-record-tokens", type=int, default=48)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-samples", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--read-scale", type=float, default=0.1)
    p.add_argument("--value-source", choices=("contextual", "token_embedding"), default="contextual")
    p.add_argument("--slot-loss-weight", type=float, default=0.5)
    variants = ("standard", "heldout_ids", "paraphrase", "delay", "mixed",
                "unseen_natural", "delay64", "id_extreme",
                "record_paraphrase_only", "query_paraphrase_only")
    p.add_argument("--train-variant", choices=variants, default="standard")
    p.add_argument("--eval-variant", choices=variants, default="standard")
    p.add_argument("--control-only", action="store_true")
    p.add_argument("--seed", type=int, default=260923)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def make_episode(rng: random.Random, slots: int, variant: str = "standard"):
    if variant == "mixed":
        variant = rng.choice(("standard", "heldout_ids", "paraphrase", "delay"))
    key_pool = range(10000, 11000) if variant == "id_extreme" else (
        range(1000, 2000) if variant == "heldout_ids" else range(1000)
    )
    keys = rng.sample(key_pool, slots)
    values = [rng.randrange(10) for _ in range(slots)]
    target_slot = rng.randrange(slots)
    if variant == "unseen_natural":
        records = [f"Calculator report: token K{k:03d} was assigned number {v}."
                   for k, v in zip(keys, values)]
        query = (f"According to the earlier calculator report, what number belongs to "
                 f"token K{keys[target_slot]:03d}?\nAnswer: ")
    elif variant in ("paraphrase", "record_paraphrase_only", "query_paraphrase_only"):
        if variant == "query_paraphrase_only":
            records = [f"[tool result] ID K{k:03d}: value {v}." for k, v in zip(keys, values)]
        else:
            records = [f"Tool message: item K{k:03d} has digit {v}." for k, v in zip(keys, values)]
        if variant == "record_paraphrase_only":
            query = (
                "An earlier tool returned records. Retrieve the value for "
                f"ID K{keys[target_slot]:03d}. Reply with one digit.\nAnswer: "
            )
        else:
            query = f"What digit did the tool report for item K{keys[target_slot]:03d}?\nAnswer: "
    else:
        records = [f"[tool result] ID K{k:03d}: value {v}." for k, v in zip(keys, values)]
        query = (
            "An earlier tool returned records. Retrieve the value for "
            f"ID K{keys[target_slot]:03d}. Reply with one digit.\nAnswer: "
        )
    if variant in ("delay", "delay64"):
        recent_keys = rng.sample(range(3000, 5000), 64 if variant == "delay64" else 32)
        recent = "\n".join(
            f"[recent tool] ID N{k:04d}: value {rng.randrange(10)}."
            for k in recent_keys
        )
        query = recent + "\n" + query
    return records, query, str(values[target_slot]), target_slot


def make_batch(tokenizer, seed: int, indices: list[int], slots: int, max_record_tokens: int,
               device, variant: str = "standard"):
    pad = tokenizer.pad_token_id
    episodes = [make_episode(random.Random(seed + i * 1000003), slots, variant) for i in indices]
    prefixes = [tokenizer.encode(ep[1], add_special_tokens=False) for ep in episodes]
    answers = [tokenizer.encode(ep[2], add_special_tokens=False) for ep in episodes]
    records = [[tokenizer.encode(r, add_special_tokens=False) for r in ep[0]] for ep in episodes]
    if not all(len(a) == 1 for a in answers):
        raise AssertionError("Every digit must be one token")
    record_len = max(len(r) for group in records for r in group)
    if record_len > max_record_tokens:
        raise ValueError(f"Record length {record_len} exceeds cap {max_record_tokens}")
    query_len = max(map(len, prefixes))
    batch_size = len(indices)
    input_ids = torch.full((batch_size, query_len), pad, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.empty(batch_size, dtype=torch.long)
    memory_ids = torch.full((batch_size, slots, record_len), pad, dtype=torch.long)
    memory_mask = torch.zeros_like(memory_ids, dtype=torch.bool)
    for b, (prefix, answer, group) in enumerate(zip(prefixes, answers, records)):
        # The target answer is deliberately absent from the Reader input.
        input_ids[b, : len(prefix)] = torch.tensor(prefix)
        attention_mask[b, : len(prefix)] = 1
        labels[b] = answer[0]
        for s, record in enumerate(group):
            memory_ids[b, s, : len(record)] = torch.tensor(record)
            memory_mask[b, s, : len(record)] = True
    target_slot = torch.tensor([ep[3] for ep in episodes], dtype=torch.long)
    prefix_len = torch.tensor([len(p) for p in prefixes], dtype=torch.long)
    tensors = (input_ids, attention_mask, labels, memory_ids, memory_mask, target_slot, prefix_len)
    return tuple(x.to(device) for x in tensors), episodes


class ContextualMemoryReader(nn.Module):
    def __init__(self, base, heads: int = 8, read_scale: float = 0.1,
                 value_source: str = "contextual"):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()
        self.read_scale = read_scale
        self.value_source = value_source
        width = base.config.hidden_size
        self.memory_attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        # Identity at step zero. There is no explicit global gate, but the
        # projection can still learn to ignore memory; evaluate wrong-memory
        # controls to detect that failure mode.
        nn.init.zeros_(self.memory_attention.out_proj.weight)
        nn.init.zeros_(self.memory_attention.out_proj.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, input_ids, attention_mask, labels, memory_ids, memory_mask,
                target_slot, prefix_len, slot_loss_weight=0.0):
        with torch.no_grad():
            query_states = self.base.model(
                input_ids=input_ids, attention_mask=attention_mask,
                use_cache=False, return_dict=True,
            ).last_hidden_state
            batch, slots, record_len = memory_ids.shape
            flat_ids = memory_ids.reshape(batch * slots, record_len)
            flat_mask = memory_mask.reshape(batch * slots, record_len)
            record_states = self.base.model(
                input_ids=flat_ids, attention_mask=flat_mask.long(),
                use_cache=False, return_dict=True,
            ).last_hidden_state
            if self.value_source == "token_embedding":
                value_states = self.base.get_input_embeddings()(flat_ids)
            else:
                value_states = record_states
        width = record_states.shape[-1]
        record_states = record_states.reshape(batch, slots * record_len, width)
        value_states = value_states.reshape(batch, slots * record_len, width)
        visible = memory_mask.reshape(batch, slots * record_len)
        query_state = query_states[torch.arange(batch, device=query_states.device),
                                   prefix_len - 1].unsqueeze(1)
        read, weights = self.memory_attention(
            query_state.float(), record_states.float(), value_states.float(),
            key_padding_mask=~visible, need_weights=True, average_attn_weights=True,
        )
        logits = self.base.lm_head(
            (query_state.float() + self.read_scale * read.float()).squeeze(1)
            .to(self.base.lm_head.weight.dtype)
        ).float()
        answer_loss = F.cross_entropy(logits, labels)
        query_weights = weights[:, 0]
        slot_prob = query_weights.reshape(batch, slots, record_len).sum(dim=-1)
        slot_loss = -torch.log(slot_prob.gather(1, target_slot[:, None]).clamp_min(1e-9)).mean()
        loss = answer_loss + slot_loss_weight * slot_loss
        return loss, answer_loss, slot_loss, logits, slot_prob


def evaluate(model, tokenizer, args, device):
    model.eval()
    digit_ids = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    totals = {"nll": 0.0, "wrong_memory_nll": 0.0, "accuracy": 0,
              "digit_accuracy": 0, "slot_accuracy": 0, "n": 0}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for start in range(0, args.eval_samples, args.batch_size):
            stop = min(args.eval_samples, start + args.batch_size)
            batch, episodes = make_batch(
                tokenizer, args.seed + 100000000, list(range(start, stop)),
                args.slots, args.max_record_tokens, device, args.eval_variant,
            )
            _, answer_loss, _, logits, slot_prob = model(*batch)
            wrong = list(batch)
            wrong[3] = torch.roll(batch[3], 1, 0)
            wrong[4] = torch.roll(batch[4], 1, 0)
            _, wrong_loss, _, _, _ = model(*wrong)
            count = stop - start
            totals["nll"] += answer_loss.item() * count
            totals["wrong_memory_nll"] += wrong_loss.item() * count
            answer_logits = logits
            predictions = answer_logits.argmax(dim=-1)
            digit_predictions = torch.tensor(digit_ids, device=device)[answer_logits[:, digit_ids].argmax(dim=-1)]
            targets = batch[2]
            totals["accuracy"] += (predictions == targets).sum().item()
            totals["digit_accuracy"] += (digit_predictions == targets).sum().item()
            totals["slot_accuracy"] += (slot_prob.argmax(dim=-1) == batch[5]).sum().item()
            totals["n"] += count
    n = totals.pop("n")
    nll = totals["nll"] / n
    wrong_nll = totals["wrong_memory_nll"] / n
    return {"nll": nll, "wrong_memory_nll": wrong_nll,
            "memory_nll_advantage": wrong_nll - nll,
            "accuracy": totals["accuracy"] / n,
            "digit_accuracy": totals["digit_accuracy"] / n,
            "slot_accuracy": totals["slot_accuracy"] / n,
            "n": n}


def evaluate_full_context(base, tokenizer, args, device):
    """Check that the unmodified Base LM can solve the held-out task format."""
    digit_ids = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    totals = {"nll": 0.0, "accuracy": 0, "digit_accuracy": 0, "n": 0}
    pad = tokenizer.pad_token_id
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for start in range(0, args.eval_samples, args.batch_size):
            stop = min(args.eval_samples, start + args.batch_size)
            episodes = [
                make_episode(random.Random(args.seed + 100000000 + i * 1000003),
                             args.slots, args.eval_variant)
                for i in range(start, stop)
            ]
            prompts = [tokenizer.encode("\n".join(ep[0]) + "\n" + ep[1],
                                        add_special_tokens=False) for ep in episodes]
            ids = torch.full((len(prompts), max(map(len, prompts))), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for b, prompt in enumerate(prompts):
                ids[b, : len(prompt)] = torch.tensor(prompt, device=device)
                mask[b, : len(prompt)] = 1
            hidden = base.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                return_dict=True).last_hidden_state
            positions = torch.tensor([len(p) - 1 for p in prompts], device=device)
            states = hidden[torch.arange(len(prompts), device=device), positions]
            logits = base.lm_head(states).float()
            targets = torch.tensor([tokenizer.encode(ep[2], add_special_tokens=False)[0]
                                    for ep in episodes], device=device)
            totals["nll"] += F.cross_entropy(logits, targets, reduction="sum").item()
            totals["accuracy"] += (logits.argmax(dim=-1) == targets).sum().item()
            choices = torch.tensor(digit_ids, device=device)[logits[:, digit_ids].argmax(dim=-1)]
            totals["digit_accuracy"] += (choices == targets).sum().item()
            totals["n"] += len(prompts)
    n = totals.pop("n")
    return {k: v / n for k, v in totals.items()} | {"n": n}


def main():
    from torch.utils.tensorboard import SummaryWriter

    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    if args.control_only:
        control = evaluate_full_context(base.eval(), tokenizer, args, device)
        result = {"kind": "full_context_control", "variant": args.eval_variant,
                  "slots": args.slots, **control}
        (args.run_dir / "control.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result, flush=True)
        return
    model = ContextualMemoryReader(
        base, read_scale=args.read_scale, value_source=args.value_source,
    ).to(device)
    if args.init_adapter:
        saved = torch.load(args.init_adapter, map_location="cpu", weights_only=True)
        model.memory_attention.load_state_dict(saved["adapter"])
    optimizer = torch.optim.AdamW(
        model.memory_attention.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=0.01,
    )
    manifest = {
        "kind": "reader_v2_curriculum", "base": "Qwen/Qwen3-0.6B-Base",
        "base_revision": MODEL_REVISION, "base_weight_sha256": MODEL_SHA256,
        "slots": args.slots, "max_record_tokens": args.max_record_tokens,
        "seed": args.seed, "steps": args.steps, "batch_size": args.batch_size,
        "lr": args.lr, "read_scale": args.read_scale,
        "value_source": args.value_source,
        "slot_loss_weight": args.slot_loss_weight,
        "train_variant": args.train_variant, "eval_variant": args.eval_variant,
        "init_adapter": str(args.init_adapter) if args.init_adapter else None,
    }
    (args.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    start_time = time.monotonic()
    with (args.run_dir / "metrics.jsonl").open("w") as log, SummaryWriter(str(args.run_dir / "tb")) as writer:
        initial = evaluate(model, tokenizer, args, device)
        row = {"step": 0, **initial}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(row, flush=True)
        for name, value in initial.items():
            writer.add_scalar(f"eval/{name}", value, 0)
        for step in range(1, args.steps + 1):
            model.train()
            indices = [(step - 1) * args.batch_size + j for j in range(args.batch_size)]
            batch, _ = make_batch(tokenizer, args.seed, indices, args.slots,
                                  args.max_record_tokens, device, args.train_variant)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, answer_loss, slot_loss, _, _ = model(
                    *batch, slot_loss_weight=args.slot_loss_weight if args.slots > 1 else 0.0,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at {step}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.memory_attention.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                result = evaluate(model, tokenizer, args, device)
                row = {"step": step, "train_loss": loss.item(),
                       "train_answer_loss": answer_loss.item(),
                       "train_slot_loss": slot_loss.item(),
                       **result, "elapsed_s": round(time.monotonic() - start_time, 1)}
                log.write(json.dumps(row) + "\n")
                log.flush()
                writer.add_scalar("train/answer_loss", answer_loss.item(), step)
                writer.add_scalar("train/slot_loss", slot_loss.item(), step)
                for name, value in result.items():
                    writer.add_scalar(f"eval/{name}", value, step)
                writer.flush()
                print(row, flush=True)
    torch.save({"adapter": {k: v.cpu() for k, v in model.memory_attention.state_dict().items()},
                "manifest": manifest}, args.run_dir / "adapter.pt")


if __name__ == "__main__":
    main()
