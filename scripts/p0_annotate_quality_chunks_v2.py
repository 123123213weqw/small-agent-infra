#!/usr/bin/env python3
"""Use annotate_jsonl with an explicit, auditable reason-code deduplication.

The model sometimes repeats an otherwise valid reason code. This wrapper
removes only exact duplicate list entries, records that fact on the result,
and leaves every other schema requirement unchanged.
"""

from __future__ import annotations

import threading

import annotate_jsonl as base


_original_call_model = base.call_model
_original_score_job = base.score_job
_state = threading.local()


def call_model_with_reason_dedup(**kwargs):
    annotation, usage, served_model = _original_call_model(**kwargs)
    _state.normalization = None
    codes = annotation.get("reason_codes")
    if isinstance(codes, list) and all(isinstance(x, str) for x in codes):
        unique = list(dict.fromkeys(codes))
        if len(unique) < len(codes):
            annotation["reason_codes"] = unique
            _state.normalization = {
                "type": "exact_duplicate_reason_codes_removed",
                "removed_count": len(codes) - len(unique),
                "original_codes": codes,
            }
    return annotation, usage, served_model


def score_job_with_normalization(job, **kwargs):
    _state.normalization = None
    result = _original_score_job(job, **kwargs)
    if result["status"] == "ok" and _state.normalization is not None:
        result["model_output_normalization"] = _state.normalization
    return result


base.call_model = call_model_with_reason_dedup
base.score_job = score_job_with_normalization

if __name__ == "__main__":
    raise SystemExit(base.main())
