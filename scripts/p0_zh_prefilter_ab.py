#!/usr/bin/env python3
"""Offline A/B: all-Qwen baseline versus rules versus a cheap CPU text filter.

The result is diagnostic only. It never promotes or rejects training documents.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time


SPAM_PATTERN = re.compile(r"六合彩|真人娱乐城|博彩平台|现金网|体育投注|时时彩|葡京赌场|博彩网站|赌博网站|bet365|m88bet", re.I)


def split_for_host(host: str) -> str:
    n = int.from_bytes(hashlib.sha256(host.encode()).digest()[:8], "big") % 100
    return "train" if n < 70 else "validation" if n < 85 else "test"


def threshold_at_clean_budget(scores, clean_labels, max_false_reject_rate: float) -> float:
    # Reject iff score > threshold. A tie at the boundary is retained.
    clean_scores = sorted((float(score) for score, clean in zip(scores, clean_labels) if clean), reverse=True)
    if not clean_scores:
        raise ValueError("no clean validation examples")
    allowed = math.floor(len(clean_scores) * max_false_reject_rate)
    return clean_scores[allowed] if allowed < len(clean_scores) else -1.0


def rule_reject(text: str) -> bool:
    return len(SPAM_PATTERN.findall(text)) >= 2


def metrics(rows, predictions) -> dict:
    counts = Counter()
    for row, blocked in zip(rows, predictions):
        counts["total"] += 1
        counts["teacher_reject"] += not row["qwen_strict_keep"]
        counts["teacher_keep"] += row["qwen_strict_keep"]
        counts["final_approved"] += row["final_approved"]
        counts["blocked"] += bool(blocked)
        counts["teacher_reject_blocked"] += bool(blocked) and not row["qwen_strict_keep"]
        counts["teacher_keep_blocked"] += bool(blocked) and row["qwen_strict_keep"]
        counts["final_approved_blocked"] += bool(blocked) and row["final_approved"]
        counts["reference_tokens_blocked"] += row["reference_tokens"] if blocked else 0
    return dict(counts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-train", type=int, default=24000)
    p.add_argument("--max-clean-false-reject", type=float, default=.005)
    args = p.parse_args()
    if args.output.exists() or args.max_train < 100 or not 0 < args.max_clean_false_reject < .1:
        raise ValueError("invalid output/training size/guard")
    import numpy as np
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import roc_auc_score

    data = [json.loads(line) for line in args.input.open() if line.strip()]
    if len({x["document_id"] for x in data}) != len(data):
        raise ValueError("duplicate document ID")
    groups = {split: [] for split in ("train", "validation", "test")}
    hosts = {split: set() for split in groups}
    for row in data:
        split = split_for_host(row["host"])
        groups[split].append(row)
        hosts[split].add(row["host"])
    if any(len(groups[split]) < 100 for split in groups):
        raise ValueError("host-disjoint split is too small")
    train = sorted(groups["train"], key=lambda x: hashlib.sha256(x["document_id"].encode()).hexdigest())[:args.max_train]
    validation, test = groups["validation"], groups["test"]
    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(2, 3), n_features=2**18,
                                   alternate_sign=False, binary=True, norm="l2", lowercase=False)
    started = time.monotonic()
    x_train = vectorizer.transform(x["excerpt"] for x in train)
    y_train = np.array([not x["qwen_strict_keep"] for x in train], dtype=int)
    clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=12, tol=1e-3,
                        class_weight="balanced", random_state=20260926)
    clf.fit(x_train, y_train)
    fit_seconds = time.monotonic() - started
    val_scores = clf.predict_proba(vectorizer.transform(x["excerpt"] for x in validation))[:, 1]
    threshold = threshold_at_clean_budget(val_scores, [x["qwen_strict_keep"] for x in validation],
                                           args.max_clean_false_reject)
    # Protect final-approved examples in validation with an additional zero-error guard.
    approved_val_scores = [float(score) for score, row in zip(val_scores, validation) if row["final_approved"]]
    if approved_val_scores:
        threshold = max(threshold, max(approved_val_scores))
    started = time.monotonic()
    test_scores = clf.predict_proba(vectorizer.transform(x["excerpt"] for x in test))[:, 1]
    infer_seconds = time.monotonic() - started
    model_result = metrics(test, test_scores > threshold)
    rule_result = metrics(test, [rule_reject(x["excerpt"]) for x in test])
    # Exploratory sensitivity check only: the primary .5% setting above was
    # specified before seeing held-out labels. Do not select a deployment
    # threshold by looking at these test outcomes.
    sensitivity = {}
    for budget in (0.0, 0.001, 0.005):
        candidate_threshold = threshold_at_clean_budget(
            val_scores, [x["qwen_strict_keep"] for x in validation], budget
        )
        if approved_val_scores:
            candidate_threshold = max(candidate_threshold, max(approved_val_scores))
        sensitivity[str(budget)] = {
            "threshold": float(candidate_threshold),
            "validation": metrics(validation, val_scores > candidate_threshold),
            "test": metrics(test, test_scores > candidate_threshold),
        }
    false_block_examples = [
        {"document_id": row["document_id"], "host": row["host"],
         "score": float(score), "final_approved": row["final_approved"],
         "excerpt_prefix": row["excerpt"][:320]}
        for row, score in zip(test, test_scores)
        if score > threshold and row["qwen_strict_keep"]
    ]
    report = {
        "stage": "zh_cpu_prefilter_ab_diagnostic_not_deployed",
        "policy": "site-host-disjoint train/validation/test; threshold chosen on validation only",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "split_documents": {k: len(v) for k, v in groups.items()},
        "split_hosts": {k: len(v) for k, v in hosts.items()},
        "trained_documents": len(train),
        "validation_guard": {"max_clean_false_reject_rate": args.max_clean_false_reject,
                             "zero_final_approved_blocked": True,
                             "threshold": float(threshold)},
        "test": {"baseline_all_qwen_documents": len(test),
                 "rule": rule_result, "cpu_model": model_result,
                 "model_auc_vs_qwen_labels": float(roc_auc_score(
                     [not x["qwen_strict_keep"] for x in test], test_scores)),
                 "false_block_examples": false_block_examples},
        "exploratory_threshold_sensitivity": sensitivity,
        "timing": {"fit_seconds": round(fit_seconds, 2),
                   "test_infer_seconds": round(infer_seconds, 2),
                   "test_docs_per_cpu_second": round(len(test) / max(infer_seconds, 1e-9), 1)},
        "limits": "Teacher labels are Qwen decisions, not independent human truth; new-source drift not measured.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
