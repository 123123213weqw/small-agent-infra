"""Pure, deterministic measurements for checkpoint generation smoke tests."""

from __future__ import annotations

from typing import Any, Sequence


def repeated_ngram_rate(token_ids: Sequence[int], n: int = 4) -> float:
    """Fraction of n-gram positions whose n-gram occurred earlier."""
    if n < 1:
        raise ValueError("n must be positive")
    grams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    if not grams:
        return 0.0
    seen: set[tuple[int, ...]] = set()
    repeated = 0
    for gram in grams:
        if gram in seen:
            repeated += 1
        seen.add(gram)
    return repeated / len(grams)


def longest_identical_token_run(token_ids: Sequence[int]) -> int:
    best = current = 0
    previous: int | None = None
    for token in token_ids:
        current = current + 1 if token == previous else 1
        best = max(best, current)
        previous = token
    return best


def generation_metrics(token_ids: Sequence[int], text: str, eos_id: int) -> dict[str, Any]:
    ids = [int(token) for token in token_ids]
    eos_position = next((i + 1 for i, token in enumerate(ids) if token == eos_id), None)
    try:
        text.encode("utf-8", errors="strict")
        utf8_encodable = True
    except UnicodeEncodeError:
        utf8_encodable = False
    controls = sum(ord(char) < 32 and char not in "\n\r\t" for char in text)
    return {
        "generated_tokens": len(ids),
        "eos_reached": eos_position is not None,
        "eos_position": eos_position,
        "distinct_1_rate": len(set(ids)) / len(ids) if ids else 0.0,
        "repeated_4gram_rate": repeated_ngram_rate(ids, 4),
        "longest_identical_token_run": longest_identical_token_run(ids),
        "replacement_character_count": text.count("\ufffd"),
        "control_character_count": controls,
        "utf8_encodable": utf8_encodable,
        "whitespace_only": not text.strip(),
    }


def compare_cache_tokens(cached: Sequence[int], uncached: Sequence[int]) -> dict[str, Any]:
    a, b = list(cached), list(uncached)
    matching_prefix = 0
    for left, right in zip(a, b):
        if left != right:
            break
        matching_prefix += 1
    compared = max(len(a), len(b))
    matching_positions = sum(x == y for x, y in zip(a, b))
    exact = a == b
    return {
        "exact_match": exact,
        "matching_prefix_tokens": matching_prefix,
        "first_divergence_index": None if exact else matching_prefix,
        "position_agreement_rate": matching_positions / compared if compared else 1.0,
        "cached_tokens": len(a),
        "uncached_tokens": len(b),
    }


def summarize_step(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one prompt is required")
    modes = ("cached", "uncached")
    result: dict[str, Any] = {"prompts": len(samples)}
    for mode in modes:
        metrics = [sample[mode]["metrics"] for sample in samples]
        result[mode] = {
            "mean_repeated_4gram_rate": sum(m["repeated_4gram_rate"] for m in metrics) / len(metrics),
            "eos_reached_prompts": sum(m["eos_reached"] for m in metrics),
            "replacement_characters": sum(m["replacement_character_count"] for m in metrics),
            "control_characters": sum(m["control_character_count"] for m in metrics),
            "whitespace_only_prompts": sum(m["whitespace_only"] for m in metrics),
        }
    result["cache_comparison"] = {
        "exact_match_prompts": sum(sample["cache_comparison"]["exact_match"] for sample in samples),
        "mean_position_agreement_rate": sum(
            sample["cache_comparison"]["position_agreement_rate"] for sample in samples
        ) / len(samples),
    }
    return result
