"""Token-weighted causal-language-model validation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ValidationResult:
    loss: float
    perplexity: float
    predicted_tokens: int
    batches: int


def evaluate_causal_lm(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
    max_batches: int | None = None,
) -> ValidationResult:
    was_training = model.training
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.int64)
    batch_count = 0
    try:
        with torch.no_grad():
            for batch in batches:
                if max_batches is not None and batch_count >= max_batches:
                    break
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                # Hugging Face causal LMs compare logits[:, :-1] with
                # labels[:, 1:], so count exactly those non-ignored targets.
                targets = torch.count_nonzero(labels[:, 1:] != -100)
                if not targets:
                    continue
                context = (
                    torch.autocast(device_type=device.type, dtype=autocast_dtype)
                    if autocast_dtype is not None and device.type != "cpu"
                    else nullcontext()
                )
                with context:
                    output = model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                if not torch.isfinite(output.loss):
                    raise FloatingPointError("non-finite validation loss")
                loss_sum += output.loss.detach().double() * targets
                token_count += targets
                batch_count += 1
    finally:
        model.train(was_training)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    if not token_count:
        raise ValueError("validation produced zero predicted tokens")
    loss = float(loss_sum / token_count)
    return ValidationResult(
        loss=loss,
        perplexity=math.exp(min(loss, 80.0)),
        predicted_tokens=int(token_count),
        batches=batch_count,
    )
