#!/usr/bin/env python3
"""Models, collation, and losses for the B2.1 joint-ranking experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import Dataset


PAD, CLS = 256, 257
VOCAB_SIZE = 258
SOURCE_IDS = {"user": 0, "assistant": 1, "tool": 2}


def encode_bytes(text: str, maximum: int) -> tuple[torch.Tensor, torch.Tensor]:
    raw = list(text.encode("utf-8"))[: maximum - 1]
    ids = [CLS, *raw]
    mask = [True] * len(ids)
    ids.extend([PAD] * (maximum - len(ids)))
    mask.extend([False] * (maximum - len(mask)))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def context_text(group: dict[str, Any]) -> str:
    recent = " ".join(item["text"] for item in group["recent_context"])
    return f"[目标] {group['goal']}\n[近期上下文] {recent}"


def pointwise_text(group: dict[str, Any], index: int) -> str:
    records = group["records"]
    others = [item for j, item in enumerate(records) if j != index]
    selected: list[dict[str, Any]] = []
    for item in [*others[:2], *others[-2:]]:
        if item["uid"] not in {existing["uid"] for existing in selected}:
            selected.append(item)
    other_text = " ".join(item["text"] for item in selected)
    return f"[候选记录] {records[index]['text']}\n[其他竞争记录] {other_text}"


class DecisionGroupDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: Path):
        self.path = path
        with path.open(encoding="utf-8") as handle:
            self.groups = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.groups[index]


class GroupCollator:
    def __init__(
        self,
        architecture: str,
        record_bytes: int = 256,
        context_bytes: int = 512,
        pointwise_bytes: int = 384,
    ):
        if architecture not in {"pointwise", "joint", "joint_partial"}:
            raise ValueError(architecture)
        self.architecture = architecture
        self.record_bytes = record_bytes
        self.context_bytes = context_bytes
        self.pointwise_bytes = pointwise_bytes

    def __call__(self, groups: Sequence[dict[str, Any]]) -> dict[str, Any]:
        batch = len(groups)
        records = max(len(group["records"]) for group in groups)
        record_length = self.pointwise_bytes if self.architecture == "pointwise" else self.record_bytes
        record_ids = torch.full((batch, records, record_length), PAD, dtype=torch.long)
        record_token_mask = torch.zeros((batch, records, record_length), dtype=torch.bool)
        # Avoid all-padding sequences inside TransformerEncoder; set_mask still hides them.
        record_ids[:, :, 0] = CLS
        record_token_mask[:, :, 0] = True
        set_mask = torch.zeros((batch, records), dtype=torch.bool)
        context_ids = torch.full((batch, self.context_bytes), PAD, dtype=torch.long)
        context_token_mask = torch.zeros((batch, self.context_bytes), dtype=torch.bool)
        ages = torch.zeros((batch, records), dtype=torch.long)
        event_indices = torch.zeros((batch, records), dtype=torch.long)
        candidates = torch.zeros((batch, records), dtype=torch.long)
        sources = torch.zeros((batch, records), dtype=torch.long)
        utilities = torch.zeros((batch, records), dtype=torch.float32)
        oracle_mask = torch.zeros((batch, records), dtype=torch.bool)

        for b, group in enumerate(groups):
            ctx_ids, ctx_mask = encode_bytes(context_text(group), self.context_bytes)
            context_ids[b], context_token_mask[b] = ctx_ids, ctx_mask
            count = len(group["records"])
            set_mask[b, :count] = True
            utilities[b, :count] = torch.tensor(group["utilities"], dtype=torch.float32)
            oracle_mask[b, group["oracle_eviction_indices"]] = True
            for i, record in enumerate(group["records"]):
                text = pointwise_text(group, i) if self.architecture == "pointwise" else record["text"]
                ids, token_mask = encode_bytes(text, record_length)
                record_ids[b, i], record_token_mask[b, i] = ids, token_mask
                ages[b, i] = min(255, int(record["relative_age"]))
                event_indices[b, i] = min(255, int(record["event_index"]))
                candidates[b, i] = int(bool(record["is_candidate"]))
                sources[b, i] = SOURCE_IDS[record["source"]]

        return {
            "record_ids": record_ids,
            "record_token_mask": record_token_mask,
            "set_mask": set_mask,
            "context_ids": context_ids,
            "context_token_mask": context_token_mask,
            "ages": ages,
            "event_indices": event_indices,
            "candidates": candidates,
            "sources": sources,
            "utilities": utilities,
            "oracle_mask": oracle_mask,
            "groups": list(groups),
        }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


class ByteEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        layers: int,
        heads: int,
        ff_dim: int,
        dropout: float,
        maximum_bytes: int,
    ):
        super().__init__()
        self.token = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.position = nn.Embedding(maximum_bytes, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d_model))

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        hidden = self.token(ids) + self.position(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        return hidden[:, 0]


class UtilitySetModel(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.architecture = config["architecture"]
        self.d_model = int(config["d_model"])
        self.heads = int(config["heads"])
        byte_layers = int(config["pointwise_layers"] if self.architecture == "pointwise" else config["record_layers"])
        maximum = max(int(config["context_bytes"]), int(config["pointwise_bytes"]), int(config["record_bytes"]))
        self.byte_encoder = ByteEncoder(
            self.d_model,
            byte_layers,
            self.heads,
            int(config["ff_dim"]),
            float(config["dropout"]),
            maximum,
        )
        self.context_projection = nn.Linear(self.d_model, self.d_model)
        self.age_embedding = nn.Embedding(256, self.d_model)
        self.event_embedding = nn.Embedding(256, self.d_model)
        self.candidate_embedding = nn.Embedding(2, self.d_model)
        self.source_embedding = nn.Embedding(len(SOURCE_IDS), self.d_model)

        self.set_encoder: nn.TransformerEncoder | None = None
        if self.architecture != "pointwise":
            layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.heads,
                dim_feedforward=int(config["ff_dim"]),
                dropout=float(config["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.set_encoder = nn.TransformerEncoder(
                layer,
                int(config["set_layers"]),
                norm=nn.LayerNorm(self.d_model),
            )
        self.partial_window = int(config.get("partial_window", 4))
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )

    def _partial_attention_mask(self, set_mask: torch.Tensor) -> torch.Tensor:
        batch, width = set_mask.shape
        masks = torch.ones((batch, width, width), dtype=torch.bool, device=set_mask.device)
        for b in range(batch):
            valid = int(set_mask[b].sum().item())
            for query in range(valid):
                candidates = sorted(range(valid), key=lambda key: (abs(key - query), key))
                allowed = []
                for index in (query, 0, valid - 1, *candidates):
                    if index not in allowed:
                        allowed.append(index)
                    if len(allowed) >= self.partial_window:
                        break
                masks[b, query, allowed] = False
            for query in range(valid, width):
                # Padded queries are discarded, but must retain one unmasked,
                # non-padding key to avoid an all-masked softmax/NaN.
                masks[b, query, 0] = False
        return masks.repeat_interleave(self.heads, dim=0)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        ids = batch["record_ids"]
        token_mask = batch["record_token_mask"]
        batch_size, records, length = ids.shape
        encoded = self.byte_encoder(ids.reshape(-1, length), token_mask.reshape(-1, length))
        encoded = encoded.reshape(batch_size, records, self.d_model)
        context = self.byte_encoder(batch["context_ids"], batch["context_token_mask"])
        hidden = (
            encoded
            + self.context_projection(context).unsqueeze(1)
            + self.age_embedding(batch["ages"])
            + self.event_embedding(batch["event_indices"])
            + self.candidate_embedding(batch["candidates"])
            + self.source_embedding(batch["sources"])
        )
        if self.set_encoder is not None:
            attention_mask = None
            if self.architecture == "joint_partial":
                attention_mask = self._partial_attention_mask(batch["set_mask"])
            hidden = self.set_encoder(
                hidden,
                mask=attention_mask,
                src_key_padding_mask=~batch["set_mask"],
            )
        scores = torch.sigmoid(self.head(hidden).squeeze(-1))
        return scores.masked_fill(~batch["set_mask"], 1.0)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def utility_losses(
    predictions: torch.Tensor,
    batch: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    targets = batch["utilities"]
    mask = batch["set_mask"]
    positive_weight = float(config.get("positive_regression_weight", 1.0))
    weights = torch.where(targets > 0, positive_weight, 1.0) * mask
    regression = (weights * torch.square(predictions - targets)).sum() / weights.sum().clamp_min(1.0)

    true_difference = targets.unsqueeze(2) - targets.unsqueeze(1)
    predicted_difference = predictions.unsqueeze(2) - predictions.unsqueeze(1)
    pair_mask = (
        (true_difference > 1e-8)
        & mask.unsqueeze(2)
        & mask.unsqueeze(1)
    )
    pair_values = (
        true_difference.abs()
        * torch.relu(float(config.get("rank_margin", 0.1)) - predicted_difference)
        * pair_mask
    )
    ranking = pair_values.sum() / pair_mask.sum().clamp_min(1)

    logits = -predictions / float(config.get("eviction_temperature", 0.1))
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    oracle = batch["oracle_mask"] & mask
    target_probabilities = oracle / oracle.sum(dim=1, keepdim=True).clamp_min(1)
    eviction = -(target_probabilities * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()

    total = (
        float(config.get("regression_weight", 1.0)) * regression
        + float(config.get("ranking_weight", 0.0)) * ranking
        + float(config.get("eviction_weight", 0.0)) * eviction
    )
    return {"total": total, "regression": regression, "ranking": ranking, "eviction": eviction}
