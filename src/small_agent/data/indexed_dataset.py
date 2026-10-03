"""Deterministic indexed access to the frozen P0 token streams.

The packed corpus is stored as one uint16 NumPy stream per split.  A training
sample is a fixed-length window.  Adjacent logical samples overlap by one
token, so Hugging Face's internal causal-label shift does not lose the target
at a block boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset


PACKED_STAGES = {"internal_smoke_token_stream", "approved_real_token_stream"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") not in PACKED_STAGES:
        raise ValueError(f"unexpected packed corpus stage: {manifest.get('stage')!r}")
    if manifest.get("dtype") != "uint16":
        raise ValueError(f"unsupported packed dtype: {manifest.get('dtype')!r}")
    if not 0 <= manifest["eos_token_id"] < manifest["vocab_size"] <= 65536:
        raise ValueError("packed vocabulary metadata is inconsistent with uint16")
    return manifest


@dataclass(frozen=True)
class SplitReport:
    split: str
    documents: int
    tokens: int
    eos_tokens: int
    sha256: str


def validate_split(
    data_dir: Path,
    manifest: dict[str, Any],
    split: str,
    *,
    verify_hash: bool = True,
    verify_boundaries: bool = True,
) -> tuple[np.memmap, SplitReport]:
    """Open one split and fail closed on checksum or document-boundary drift."""
    if split not in manifest["splits"]:
        raise KeyError(f"unknown split: {split}")
    info = manifest["splits"][split]
    path = data_dir / info["file"]
    if path.stat().st_size != info["bytes"]:
        raise ValueError(f"packed file size mismatch: {path}")
    if verify_hash and sha256_file(path) != info["sha256"]:
        raise ValueError(f"packed file hash mismatch: {path}")
    stream = np.load(path, mmap_mode="r", allow_pickle=False)
    if stream.ndim != 1 or stream.dtype != np.uint16:
        raise ValueError(f"expected one-dimensional uint16 stream: {path}")
    if len(stream) != info["tokens_including_eos"]:
        raise ValueError(f"packed token count mismatch: {path}")
    if len(stream) and int(stream.max()) >= manifest["vocab_size"]:
        raise ValueError(f"token outside vocabulary: {path}")

    spans = manifest["document_spans"][split]
    if len(spans) != info["documents"]:
        raise ValueError(f"document span count mismatch for {split}")
    eos = manifest["eos_token_id"]
    if verify_boundaries:
        expected_start = 0
        document_ids: set[str] = set()
        for span in spans:
            start, end = int(span["start"]), int(span["end"])
            document_id = span["document_id"]
            if document_id in document_ids:
                raise ValueError(f"duplicate document id in {split}: {document_id}")
            document_ids.add(document_id)
            if start != expected_start or end <= start or end > len(stream):
                raise ValueError(f"non-contiguous document span in {split}: {span}")
            if int(stream[end - 1]) != eos:
                raise ValueError(f"document does not end in EOS: {document_id}")
            if np.any(stream[start : end - 1] == eos):
                raise ValueError(f"EOS collision inside document: {document_id}")
            expected_start = end
        if expected_start != len(stream):
            raise ValueError(f"document spans do not cover the {split} stream")
        actual_eos = int(np.count_nonzero(stream == eos))
        if actual_eos != info["eos_tokens"] or actual_eos != len(spans):
            raise ValueError(f"EOS count mismatch for {split}")

    return stream, SplitReport(
        split=split,
        documents=len(spans),
        tokens=len(stream),
        eos_tokens=info["eos_tokens"],
        sha256=info["sha256"],
    )


class IndexedTokenDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed windows over a frozen split-safe token stream.

    Each returned tensor has ``sequence_length`` tokens.  The default stride
    is ``sequence_length - 1`` because a causal LM fed ``labels=input_ids``
    predicts all tokens except the first one.  The one-token overlap therefore
    gives every interior stream token exactly one prediction target.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        sequence_length: int,
        *,
        tail_policy: Literal["drop", "pad"] = "drop",
        pad_token_id: int = 0,
        verify_hash: bool = True,
        verify_boundaries: bool = True,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if tail_policy not in ("drop", "pad"):
            raise ValueError("tail_policy must be 'drop' or 'pad'")
        self.data_dir = Path(data_dir).resolve()
        self.split = split
        self.sequence_length = sequence_length
        self.stride = sequence_length - 1
        self.tail_policy = tail_policy
        self.pad_token_id = pad_token_id
        self.manifest = load_manifest(self.data_dir)
        if not 0 <= pad_token_id < self.manifest["vocab_size"]:
            raise ValueError("pad_token_id is outside the vocabulary")
        self.stream, self.split_report = validate_split(
            self.data_dir,
            self.manifest,
            split,
            verify_hash=verify_hash,
            verify_boundaries=verify_boundaries,
        )
        if tail_policy == "pad":
            # A stream of N tokens contains N-1 causal targets.  Each sample
            # predicts at most ``stride`` targets.
            self.num_samples = (max(len(self.stream) - 1, 0) + self.stride - 1) // self.stride
        elif len(self.stream) < sequence_length:
            self.num_samples = 0
        else:
            self.num_samples = 1 + (len(self.stream) - sequence_length) // self.stride

    def __len__(self) -> int:
        return self.num_samples

    def token_offset(self, sample_id: int) -> int:
        if sample_id < 0:
            sample_id += self.num_samples
        if not 0 <= sample_id < self.num_samples:
            raise IndexError(sample_id)
        return sample_id * self.stride

    def __getitem__(self, sample_id: int) -> dict[str, torch.Tensor]:
        if sample_id < 0:
            sample_id += self.num_samples
        start = self.token_offset(sample_id)
        end = min(start + self.sequence_length, len(self.stream))
        valid_length = end - start
        # Copy because np.load(..., mmap_mode="r") is read-only and embedding
        # indices must be torch.long.  It also keeps DataLoader batches detached
        # from the underlying mmap lifetime.
        tokens = np.full(self.sequence_length, self.pad_token_id, dtype=np.int64)
        tokens[:valid_length] = self.stream[start:end]
        attention_mask = np.zeros(self.sequence_length, dtype=np.int64)
        attention_mask[:valid_length] = 1
        labels = tokens.copy()
        labels[valid_length:] = -100
        return {
            "input_ids": torch.from_numpy(tokens),
            "labels": torch.from_numpy(labels),
            "attention_mask": torch.from_numpy(attention_mask),
            "sample_id": torch.tensor(sample_id, dtype=torch.int64),
            "token_offset": torch.tensor(start, dtype=torch.int64),
            "valid_tokens": torch.tensor(valid_length, dtype=torch.int64),
        }

    def coverage_report(self) -> dict[str, int | float | str]:
        if not self.num_samples:
            covered_end = 0
            predicted_tokens = 0
        else:
            final_start = (self.num_samples - 1) * self.stride
            covered_end = min(final_start + self.sequence_length, len(self.stream))
            if self.tail_policy == "pad":
                predicted_tokens = max(len(self.stream) - 1, 0)
            else:
                predicted_tokens = self.num_samples * self.stride
        return {
            "stream_tokens": len(self.stream),
            "samples": self.num_samples,
            "sequence_length": self.sequence_length,
            "stride": self.stride,
            "tail_policy": self.tail_policy,
            "predicted_tokens": predicted_tokens,
            "unpredicted_initial_tokens": min(1, len(self.stream)),
            "dropped_tail_tokens": len(self.stream) - covered_end,
            "prediction_coverage": predicted_tokens / max(len(self.stream) - 1, 1),
        }
