"""Minimal trainable memory Reader on an open Base LM.

This is an engineering/readability smoke test, NOT a FIFO-vs-utility result.
The Base LM is frozen; the memory encoder, cross-attention, and gate are
trained on generated key/value tool records. No future answer enters memory.
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
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-samples", type=int, default=64)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--record-tokens", type=int, default=48)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gate-init", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=260923)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def make_episode(rng: random.Random, slots: int) -> tuple[list[str], str, str]:
    # Random per-episode binding prevents solving this by memorizing global IDs.
    keys = rng.sample(range(1000), slots)
    values = [rng.randrange(10) for _ in range(slots)]
    query_index = rng.randrange(slots)
    records = [f"[tool result] ID K{k:03d}: value {v}." for k, v in zip(keys, values)]
    prefix = (
        "An earlier tool returned records. Retrieve the value for "
        f"ID K{keys[query_index]:03d}. Reply with one digit.\nAnswer: "
    )
    return records, prefix, str(values[query_index])


def encode_batch(tokenizer, seed: int, indices: list[int], slots: int, record_tokens: int, device):
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError("Tokenizer needs a pad token ID")
    prefixes: list[list[int]] = []
    answers: list[list[int]] = []
    memories: list[list[list[int]]] = []
    for index in indices:
        rng = random.Random(seed + index * 1000003)
        records, prefix, answer = make_episode(rng, slots)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        if not answer_ids:
            raise AssertionError("Empty answer tokenization")
        prefixes.append(prefix_ids)
        answers.append(answer_ids)
        record_ids = [tokenizer.encode(r, add_special_tokens=False) for r in records]
        if max(map(len, record_ids)) > record_tokens:
            raise ValueError("Record was truncated; increase --record-tokens")
        memories.append(record_ids)

    max_length = max(len(p) + len(a) for p, a in zip(prefixes, answers))
    input_ids = torch.full((len(indices), max_length), pad, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    memory_ids = torch.full((len(indices), slots, record_tokens), pad, dtype=torch.long)
    memory_mask = torch.zeros_like(memory_ids, dtype=torch.bool)
    for b, (prefix, answer, records) in enumerate(zip(prefixes, answers, memories)):
        ids = prefix + answer
        input_ids[b, : len(ids)] = torch.tensor(ids)
        attention_mask[b, : len(ids)] = 1
        labels[b, len(prefix) : len(ids)] = torch.tensor(answer)
        for s, record in enumerate(records):
            memory_ids[b, s, : len(record)] = torch.tensor(record)
            memory_mask[b, s, : len(record)] = True
    return tuple(x.to(device) for x in (input_ids, attention_mask, labels, memory_ids, memory_mask)), prefixes, answers


class MemoryReader(nn.Module):
    def __init__(self, base: nn.Module, heads: int = 8, gate_init: float = 0.0):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()
        hidden = base.config.hidden_size
        self.record_encoder = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=2 * hidden, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.record_norm = nn.LayerNorm(hidden)
        self.memory_attention = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()  # always frozen in this smoke test
        return self

    def forward(self, input_ids, attention_mask, labels, memory_ids, memory_mask):
        with torch.no_grad():
            hidden = self.base.model(
                input_ids=input_ids, attention_mask=attention_mask,
                use_cache=False, return_dict=True,
            ).last_hidden_state
            memory_embeds = self.base.get_input_embeddings()(memory_ids)
        batch, slots, record_length, width = memory_embeds.shape
        flat_embeds = memory_embeds.reshape(batch * slots, record_length, width).float()
        flat_mask = memory_mask.reshape(batch * slots, record_length)
        encoded = self.record_encoder(flat_embeds, src_key_padding_mask=~flat_mask)
        encoded = self.record_norm(encoded)
        encoded = encoded.reshape(batch, slots * record_length, width)
        memory_visible = memory_mask.reshape(batch, slots * record_length)
        read, _ = self.memory_attention(
            hidden.float(), encoded, encoded, key_padding_mask=~memory_visible,
            need_weights=False,
        )
        augmented = hidden.float() + torch.tanh(self.gate) * read
        logits = self.base.lm_head(augmented.to(self.base.lm_head.weight.dtype)).float()
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1),
            ignore_index=-100,
        )
        return loss, logits


def evaluate(model, tokenizer, args, device):
    model.eval()
    nll = 0.0
    wrong_memory_nll = 0.0
    correct = 0
    digit_correct = 0
    total = 0
    digit_ids = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for start in range(0, args.eval_samples, args.batch_size):
            stop = min(args.eval_samples, start + args.batch_size)
            batch, prefixes, answers = encode_batch(
                tokenizer, args.seed + 100000000, list(range(start, stop)),
                args.slots, args.record_tokens, device,
            )
            loss, logits = model(*batch)
            wrong_memory = list(batch)
            wrong_memory[3] = torch.roll(batch[3], shifts=1, dims=0)
            wrong_memory[4] = torch.roll(batch[4], shifts=1, dims=0)
            wrong_loss, _ = model(*wrong_memory)
            nll += loss.item() * (stop - start)
            wrong_memory_nll += wrong_loss.item() * (stop - start)
            for b, (prefix, answer) in enumerate(zip(prefixes, answers)):
                # All decimal digits are one token for this tokenizer; assert it.
                if len(answer) != 1:
                    raise AssertionError("Expected one answer token")
                answer_logits = logits[b, len(prefix) - 1]
                pred = int(answer_logits.argmax().item())
                correct += pred == answer[0]
                digit_pred = int(answer_logits[digit_ids].argmax().item())
                digit_correct += digit_ids[digit_pred] == answer[0]
                total += 1
    return {
        "nll": nll / total,
        "wrong_memory_nll": wrong_memory_nll / total,
        "memory_nll_advantage": (wrong_memory_nll - nll) / total,
        "accuracy": correct / total,
        "digit_accuracy": digit_correct / total,
        "n": total,
    }


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device(args.device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for digit in range(10):
        if len(tokenizer.encode(str(digit), add_special_tokens=False)) != 1:
            raise AssertionError("Every digit must tokenize as one token")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model = MemoryReader(base, gate_init=args.gate_init).to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=0.01,
    )
    manifest = {
        "kind": "reader_smoke_only", "base": "Qwen/Qwen3-0.6B-Base",
        "base_revision": BASE_REVISION,
        "slots": args.slots, "record_tokens": args.record_tokens,
        "seed": args.seed, "steps": args.steps, "batch_size": args.batch_size,
        "lr": args.lr, "gate_init": args.gate_init,
    }
    (args.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log_path = args.run_dir / "metrics.jsonl"
    start_time = time.monotonic()
    with log_path.open("w") as log, SummaryWriter(str(args.run_dir / "tb")) as writer:
        initial = evaluate(model, tokenizer, args, device)
        log.write(json.dumps({"step": 0, **initial}) + "\n")
        log.flush()
        for name, value in initial.items():
            writer.add_scalar(f"eval/{name}", value, 0)
        print(f"step=0 {initial}", flush=True)
        for step in range(1, args.steps + 1):
            model.train()
            indices = [(step - 1) * args.batch_size + j for j in range(args.batch_size)]
            batch, _, _ = encode_batch(tokenizer, args.seed, indices, args.slots, args.record_tokens, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, _ = model(*batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                result = evaluate(model, tokenizer, args, device)
                row = {"step": step, "train_loss": loss.item(), **result,
                       "gate": model.gate.item(), "elapsed_s": round(time.monotonic() - start_time, 1)}
                log.write(json.dumps(row) + "\n")
                log.flush()
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/gate", model.gate.item(), step)
                for name, value in result.items():
                    writer.add_scalar(f"eval/{name}", value, step)
                writer.flush()
                print(row, flush=True)
    torch.save({
        "adapter": {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("base.")},
        "manifest": manifest,
    }, args.run_dir / "adapter.pt")


if __name__ == "__main__":
    main()
