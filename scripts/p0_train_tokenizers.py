#!/usr/bin/env python3
"""Train and evaluate the two frozen P0 tokenizer candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterable, Iterator


VERSION = "p0_train_tokenizers_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_and_verify_corpus(corpus: Path) -> dict[str, Any]:
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "tokenizer_training_corpus":
        raise ValueError("input is not a tokenizer training corpus")
    for section, prefix in (("train_shards", "train"), ("heldout_shards", "heldout")):
        for shard in manifest[section]:
            path = corpus / prefix / shard["domain"] / shard["file"]
            if path.stat().st_size != shard["bytes"] or sha256_file(path) != shard["sha256"]:
                raise ValueError(f"corpus shard integrity failure: {path}")
    return manifest


def iter_shard_rows(corpus: Path, shards: list[dict[str, Any]], prefix: str) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for shard in shards:
        path = corpus / prefix / shard["domain"] / shard["file"]
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256, columns=["document_id", "domain", "text"]):
            yield from batch.to_pylist()


def iter_training_text(corpus: Path, manifest: dict[str, Any]) -> Iterator[str]:
    for row in iter_shard_rows(corpus, manifest["train_shards"], "train"):
        yield row["text"]


def base_manifest(
    *, variant: str, config_path: Path, corpus: Path, corpus_manifest: dict[str, Any],
    output: Path, started: float,
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "stage": "trained_tokenizer_candidate",
        "variant": variant,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started, 3),
        "config": {"file": str(config_path.resolve()), "sha256": sha256_file(config_path)},
        "corpus": {
            "directory": str(corpus.resolve()),
            "manifest_sha256": sha256_file(corpus / "manifest.json"),
            "documents": corpus_manifest["counts"]["train_documents"],
            "characters": corpus_manifest["counts"]["train_characters"],
        },
        "output_directory": str(output.resolve()),
    }


def train_bytelevel(config: dict[str, Any], config_path: Path, corpus: Path, output: Path) -> dict[str, Any]:
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers

    if output.exists():
        raise FileExistsError(output)
    corpus_manifest = load_and_verify_corpus(corpus)
    shared = config["shared"]
    variant = config["variants"]["bytelevel_bpe_32k"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    started = time.time()
    try:
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.normalizer = None
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=variant["add_prefix_space"], use_regex=True
        )
        tokenizer.decoder = decoders.ByteLevel()
        specials = [
            AddedToken(token, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)
            for token in shared["special_tokens"]
        ]
        trainer = trainers.BpeTrainer(
            vocab_size=shared["vocab_size"],
            min_frequency=2,
            show_progress=False,
            special_tokens=specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            max_token_length=variant["max_token_length"],
        )
        tokenizer.train_from_iterator(
            iter_training_text(corpus, corpus_manifest),
            trainer=trainer,
            length=corpus_manifest["counts"]["train_documents"],
        )
        tokenizer_path = temporary / "tokenizer.json"
        tokenizer.save(str(tokenizer_path), pretty=False)
        vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
        if vocab_size != shared["vocab_size"]:
            raise ValueError(f"bytelevel vocab size {vocab_size} != {shared['vocab_size']}")
        special_ids = {token: tokenizer.token_to_id(token) for token in shared["special_tokens"]}
        expected_ids = {token: index for index, token in enumerate(shared["special_tokens"])}
        if special_ids != expected_ids:
            raise ValueError(f"unexpected ByteLevel special ids: {special_ids}")
        manifest = base_manifest(
            variant="bytelevel_bpe_32k", config_path=config_path, corpus=corpus,
            corpus_manifest=corpus_manifest, output=output, started=started,
        )
        manifest.update(
            {
                "library": {"name": "tokenizers", "version": __import__("tokenizers").__version__},
                "vocab_size": vocab_size,
                "special_token_ids": special_ids,
                "artifacts": {
                    "tokenizer.json": {
                        "bytes": tokenizer_path.stat().st_size,
                        "sha256": sha256_file(tokenizer_path),
                    }
                },
            }
        )
        atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def train_sentencepiece(config: dict[str, Any], config_path: Path, corpus: Path, output: Path) -> dict[str, Any]:
    import sentencepiece as spm

    if output.exists():
        raise FileExistsError(output)
    corpus_manifest = load_and_verify_corpus(corpus)
    shared = config["shared"]
    variant = config["variants"]["sentencepiece_bpe_32k"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    started = time.time()
    try:
        prefix = temporary / "tokenizer"
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter_training_text(corpus, corpus_manifest),
            model_prefix=str(prefix),
            model_type="bpe",
            vocab_size=shared["vocab_size"],
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            pad_piece=shared["special_tokens"][0],
            bos_piece=shared["special_tokens"][1],
            eos_piece=shared["special_tokens"][2],
            unk_piece=shared["special_tokens"][3],
            user_defined_symbols=shared["special_tokens"][4:],
            normalization_rule_name=variant["normalization_rule_name"],
            remove_extra_whitespaces=variant["remove_extra_whitespaces"],
            add_dummy_prefix=variant["add_dummy_prefix"],
            byte_fallback=variant["byte_fallback"],
            allow_whitespace_only_pieces=variant["allow_whitespace_only_pieces"],
            max_sentencepiece_length=variant["max_sentencepiece_length"],
            character_coverage=1.0,
            input_sentence_size=0,
            shuffle_input_sentence=False,
            max_sentence_length=2_000_000,
            num_threads=min(32, os.cpu_count() or 1),
            train_extremely_large_corpus=True,
            hard_vocab_limit=True,
            minloglevel=1,
        )
        model_path = prefix.with_suffix(".model")
        vocab_path = prefix.with_suffix(".vocab")
        processor = spm.SentencePieceProcessor(model_file=str(model_path))
        vocab_size = processor.vocab_size()
        if vocab_size != shared["vocab_size"]:
            raise ValueError(f"sentencepiece vocab size {vocab_size} != {shared['vocab_size']}")
        special_ids = {token: processor.piece_to_id(token) for token in shared["special_tokens"]}
        expected_ids = {token: index for index, token in enumerate(shared["special_tokens"])}
        if special_ids != expected_ids:
            raise ValueError(f"unexpected SentencePiece special ids: {special_ids}")
        manifest = base_manifest(
            variant="sentencepiece_bpe_32k", config_path=config_path, corpus=corpus,
            corpus_manifest=corpus_manifest, output=output, started=started,
        )
        manifest.update(
            {
                "library": {"name": "sentencepiece", "version": spm.__version__},
                "vocab_size": vocab_size,
                "special_token_ids": special_ids,
                "artifacts": {
                    path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                    for path in (model_path, vocab_path)
                },
            }
        )
        atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


class ByteLevelAdapter:
    def __init__(self, path: Path, special_tokens: list[str]):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(path))
        self.special_ids = {self.tokenizer.token_to_id(token) for token in special_tokens}
        self.unk_id = self.tokenizer.token_to_id("<unk>")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def is_byte_fallback(self, token_id: int) -> bool:
        return False


class SentencePieceAdapter:
    def __init__(self, path: Path, special_tokens: list[str]):
        import sentencepiece as spm

        self.tokenizer = spm.SentencePieceProcessor(model_file=str(path))
        self.special_ids = {self.tokenizer.piece_to_id(token) for token in special_tokens}
        self.unk_id = self.tokenizer.unk_id()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def is_byte_fallback(self, token_id: int) -> bool:
        piece = self.tokenizer.id_to_piece(token_id)
        return piece.startswith("<0x") and piece.endswith(">")


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def first_mismatch(left: str, right: str) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def evaluate_rows(adapter: Any, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    domains: dict[str, Counter[str]] = defaultdict(Counter)
    lengths: dict[str, list[int]] = defaultdict(list)
    failures = []
    for row in rows:
        domain, text = row["domain"], row["text"]
        ids = adapter.encode(text)
        decoded = adapter.decode(ids)
        stats = domains[domain]
        stats["documents"] += 1
        stats["characters"] += len(text)
        stats["utf8_bytes"] += len(text.encode("utf-8"))
        stats["tokens"] += len(ids)
        stats["unknown_tokens"] += sum(token_id == adapter.unk_id for token_id in ids)
        stats["special_tokens_emitted"] += sum(token_id in adapter.special_ids for token_id in ids)
        stats["byte_fallback_tokens"] += sum(adapter.is_byte_fallback(token_id) for token_id in ids)
        lengths[domain].append(len(ids))
        if decoded != text:
            stats["roundtrip_failures"] += 1
            if len(failures) < 20:
                failures.append(
                    {
                        "document_id": row["document_id"],
                        "domain": domain,
                        "mismatch_character": first_mismatch(text, decoded),
                        "original_characters": len(text),
                        "decoded_characters": len(decoded),
                    }
                )
    by_domain = {}
    total = Counter()
    all_lengths = []
    for domain, stats in sorted(domains.items()):
        total.update(stats)
        all_lengths.extend(lengths[domain])
        by_domain[domain] = {
            **dict(stats),
            "tokens_per_character": stats["tokens"] / max(stats["characters"], 1),
            "tokens_per_utf8_byte": stats["tokens"] / max(stats["utf8_bytes"], 1),
            "document_tokens_p50": percentile(lengths[domain], 0.50),
            "document_tokens_p95": percentile(lengths[domain], 0.95),
        }
    return {
        "totals": {
            **dict(total),
            "tokens_per_character": total["tokens"] / max(total["characters"], 1),
            "tokens_per_utf8_byte": total["tokens"] / max(total["utf8_bytes"], 1),
            "document_tokens_p50": percentile(all_lengths, 0.50),
            "document_tokens_p95": percentile(all_lengths, 0.95),
        },
        "by_domain": by_domain,
        "roundtrip_failure_samples": failures,
    }


def iter_frozen_rows(directory: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("stage") != "p0_1m_smoke_frozen":
        raise ValueError("frozen eval pool has unexpected stage")
    for shard in manifest["shards"]:
        path = directory / shard["domain"] / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"frozen eval shard hash mismatch: {path}")
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256, columns=["document_id", "domain", "text"]):
            yield from batch.to_pylist()


def command_evaluate(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    corpus_manifest = load_and_verify_corpus(args.corpus)
    variants = {
        "bytelevel_bpe_32k": ByteLevelAdapter(
            args.artifacts / "bytelevel_bpe_32k" / "tokenizer.json", config["shared"]["special_tokens"]
        ),
        "sentencepiece_bpe_32k": SentencePieceAdapter(
            args.artifacts / "sentencepiece_bpe_32k" / "tokenizer.model", config["shared"]["special_tokens"]
        ),
    }
    report = {
        "version": VERSION,
        "stage": "tokenizer_bakeoff_evaluation",
        "config_sha256": sha256_file(args.config),
        "corpus_manifest_sha256": sha256_file(args.corpus / "manifest.json"),
        "frozen_eval_manifest_sha256": sha256_file(args.frozen_eval / "manifest.json"),
        "variants": {},
    }
    heldout_rows = list(iter_shard_rows(args.corpus, corpus_manifest["heldout_shards"], "heldout"))
    frozen_rows = list(iter_frozen_rows(args.frozen_eval))
    qualified = []
    for name, adapter in variants.items():
        heldout = evaluate_rows(adapter, heldout_rows)
        frozen = evaluate_rows(adapter, frozen_rows)
        hard_gate = {
            "heldout_roundtrip_failures": heldout["totals"].get("roundtrip_failures", 0),
            "frozen_roundtrip_failures": frozen["totals"].get("roundtrip_failures", 0),
            "unknown_tokens": heldout["totals"].get("unknown_tokens", 0) + frozen["totals"].get("unknown_tokens", 0),
            "special_tokens_emitted": heldout["totals"].get("special_tokens_emitted", 0) + frozen["totals"].get("special_tokens_emitted", 0),
        }
        passed = all(value == 0 for value in hard_gate.values())
        report["variants"][name] = {"hard_gate": {**hard_gate, "passed": passed}, "heldout": heldout, "frozen_p0": frozen}
        if passed:
            qualified.append((frozen["totals"]["tokens"], heldout["totals"]["tokens"], name))
    qualified.sort()
    report["winner"] = qualified[0][2] if qualified else None
    report["selection_rule"] = (
        "Pass exact roundtrip, zero unknown and zero special collision; then minimize frozen-P0 tokens, "
        "using heldout tokens as tie-breaker."
    )
    atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_freeze(args: argparse.Namespace) -> None:
    """Freeze the selected bake-off winner into an immutable HF-compatible bundle."""
    if args.output.exists():
        raise FileExistsError(args.output)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    winner = evaluation.get("winner")
    if winner != "bytelevel_bpe_32k":
        raise ValueError(f"expected bytelevel_bpe_32k winner, got {winner!r}")
    if not evaluation["variants"][winner]["hard_gate"]["passed"]:
        raise ValueError("selected tokenizer did not pass the hard gate")
    if evaluation["config_sha256"] != sha256_file(args.config):
        raise ValueError("evaluation/config hash mismatch")

    candidate = args.artifacts / winner
    candidate_manifest_path = candidate / "manifest.json"
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if candidate_manifest.get("variant") != winner:
        raise ValueError("candidate manifest variant mismatch")
    if candidate_manifest["config"]["sha256"] != sha256_file(args.config):
        raise ValueError("candidate/config hash mismatch")
    source_tokenizer = candidate / "tokenizer.json"
    expected_hash = candidate_manifest["artifacts"]["tokenizer.json"]["sha256"]
    if sha256_file(source_tokenizer) != expected_hash:
        raise ValueError("candidate tokenizer hash mismatch")

    special_tokens = config["shared"]["special_tokens"]
    special_ids = candidate_manifest["special_token_ids"]
    if special_ids != {token: index for index, token in enumerate(special_tokens)}:
        raise ValueError("special token IDs are not the frozen 0..14 mapping")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.tmp-", dir=args.output.parent))
    try:
        shutil.copy2(source_tokenizer, temporary / "tokenizer.json")
        shutil.copy2(args.config, temporary / "tokenizer_training_config.json")
        shutil.copy2(args.evaluation, temporary / "bakeoff_evaluation.json")
        atomic_json(
            temporary / "special_tokens_map.json",
            {
                "pad_token": special_tokens[0],
                "bos_token": special_tokens[1],
                "eos_token": special_tokens[2],
                "unk_token": special_tokens[3],
                "additional_special_tokens": special_tokens[4:],
            },
        )
        atomic_json(
            temporary / "tokenizer_config.json",
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": args.model_max_length,
                "padding_side": "right",
                "truncation_side": "right",
                "clean_up_tokenization_spaces": False,
                "add_bos_token": False,
                "add_eos_token": False,
                "pad_token": special_tokens[0],
                "bos_token": special_tokens[1],
                "eos_token": special_tokens[2],
                "unk_token": special_tokens[3],
                "additional_special_tokens": special_tokens[4:],
            },
        )
        artifact_names = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer_training_config.json",
            "bakeoff_evaluation.json",
        ]
        manifest = {
            "version": VERSION,
            "stage": "frozen_tokenizer",
            "name": args.output.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selected_variant": winner,
            "selection_rule": evaluation["selection_rule"],
            "vocab_size": candidate_manifest["vocab_size"],
            "model_max_length": args.model_max_length,
            "special_token_ids": special_ids,
            "source": {
                "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
                "evaluation_sha256": sha256_file(args.evaluation),
                "config_sha256": sha256_file(args.config),
                "corpus_manifest_sha256": evaluation["corpus_manifest_sha256"],
                "frozen_eval_manifest_sha256": evaluation["frozen_eval_manifest_sha256"],
            },
            "hard_gate": evaluation["variants"][winner]["hard_gate"],
            "artifacts": {
                name: {
                    "bytes": (temporary / name).stat().st_size,
                    "sha256": sha256_file(temporary / name),
                }
                for name in artifact_names
            },
        }
        atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, args.output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train-bytelevel", "train-sentencepiece"):
        train = sub.add_parser(name)
        train.add_argument("--config", type=Path, required=True)
        train.add_argument("--corpus", type=Path, required=True)
        train.add_argument("--output", type=Path, required=True)
        train.set_defaults(train_variant=name)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--corpus", type=Path, required=True)
    evaluate.add_argument("--artifacts", type=Path, required=True)
    evaluate.add_argument("--frozen-eval", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(func=command_evaluate)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--evaluation", type=Path, required=True)
    freeze.add_argument("--artifacts", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--model-max-length", type=int, default=2048)
    freeze.set_defaults(func=command_freeze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "func"):
        args.func(args)
        return
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.train_variant == "train-bytelevel":
        train_bytelevel(config, args.config, args.corpus.resolve(), args.output.resolve())
    else:
        train_sentencepiece(config, args.config, args.corpus.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
