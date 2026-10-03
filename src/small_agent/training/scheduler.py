"""Token-count based learning-rate schedules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from torch.optim import Optimizer


@dataclass(frozen=True)
class WarmupCosineConfig:
    maximum_learning_rate: float
    minimum_learning_rate: float
    warmup_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        if self.maximum_learning_rate <= 0:
            raise ValueError("maximum_learning_rate must be positive")
        if not 0 <= self.minimum_learning_rate <= self.maximum_learning_rate:
            raise ValueError("minimum_learning_rate must be in [0, maximum_learning_rate]")
        if self.warmup_tokens < 0 or self.total_tokens <= self.warmup_tokens:
            raise ValueError("total_tokens must be greater than non-negative warmup_tokens")


def warmup_cosine_lr(tokens_seen: int, config: WarmupCosineConfig) -> float:
    if tokens_seen < 0:
        raise ValueError("tokens_seen must be non-negative")
    if config.warmup_tokens and tokens_seen < config.warmup_tokens:
        return config.maximum_learning_rate * tokens_seen / config.warmup_tokens
    progress = (tokens_seen - config.warmup_tokens) / (
        config.total_tokens - config.warmup_tokens
    )
    progress = min(1.0, max(0.0, progress))
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + (
        config.maximum_learning_rate - config.minimum_learning_rate
    ) * coefficient


class TokenLRScheduler:
    """Set optimizer LR from the number of causal targets already consumed."""

    VERSION = 1

    def __init__(self, optimizer: Optimizer, config: WarmupCosineConfig) -> None:
        self.optimizer = optimizer
        self.config = config
        self.tokens_seen = 0
        self._set_lr(warmup_cosine_lr(0, config))

    def _set_lr(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def step(self, added_tokens: int) -> float:
        if added_tokens <= 0:
            raise ValueError("added_tokens must be positive")
        self.tokens_seen += added_tokens
        learning_rate = warmup_cosine_lr(self.tokens_seen, self.config)
        self._set_lr(learning_rate)
        return learning_rate

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "config": asdict(self.config),
            "tokens_seen": self.tokens_seen,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {"version": self.VERSION, "config": asdict(self.config)}
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"scheduler state mismatch for {key}")
        tokens_seen = int(state["tokens_seen"])
        if tokens_seen < 0:
            raise ValueError("invalid scheduler token count")
        self.tokens_seen = tokens_seen
        self._set_lr(warmup_cosine_lr(tokens_seen, self.config))
