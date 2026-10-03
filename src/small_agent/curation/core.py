"""Pure quality/routing gates; a matching quotation is not a truth certificate."""
from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import math
import re
from collections import Counter
from typing import Any

from jsonschema import Draft202012Validator


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(value: Any) -> str:
    return sha(canonical(value))


def evidence_catalog(text: str) -> list[str]:
    """Bounded exact literals, not summaries, repairs, or a claim of entailment.

    Single-line spans make simple evidence selectable; overlapping full-body
    spans preserve cross-line context. Repeated/ambiguous literals are omitted.
    """
    if len(text) > 131072:
        raise ValueError("body too large for bounded literal evidence catalog")
    pieces = {}
    for block in [*text.splitlines(), text]:
        if len(block) < 8:
            continue
        # Line separators can be outside a selected span; the full-body pass
        # also supplies cross-line spans. No strip or body normalization.
        starts = [0] if len(block) <= 240 else list(range(0, len(block), 180))
        for start in starts:
            q = block[start:start + 240]
            if 8 <= len(q) <= 320 and text.find(q) == text.rfind(q):
                pieces.setdefault(q, None)
                if len(pieces) > 1024:
                    raise ValueError("literal evidence catalog exceeded bounded capacity")
    if not pieces:
        raise ValueError("no unique exact evidence literals available; do not invent quotes")
    return list(pieces)


def decoder_schema(doc: dict, schema: dict, judge: dict) -> dict:
    """Versioned wire constraints. Full semantic schema always remains in force."""
    value = copy.deepcopy({k: v for k, v in schema.items() if k not in ("$schema", "$id")})
    projection = judge["generation"].get("decoder_schema_projection", "identity")
    if projection == "identity":
        return value
    if projection not in ("omit_uniqueItems_v1", "literal_catalog_v1") or judge["endpoint"]["kind"] != "chat":
        raise ValueError("unsupported decoder schema projection")
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "array":
                node.pop("uniqueItems", None)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)
    walk(value)
    if projection == "literal_catalog_v1":
        literals = evidence_catalog(doc["document_text"])
        value["properties"]["anchors"]["items"]["enum"] = literals
        value["properties"]["findings"]["items"]["properties"]["quotes"]["items"]["enum"] = literals
    return value


def finite_json(value: Any) -> None:
    # JSON Schema's number type alone does not reject Python's NaN/Infinity.
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite number")
    if isinstance(value, dict):
        for v in value.values():
            finite_json(v)
    if isinstance(value, list):
        for v in value:
            finite_json(v)


def span(text: str, quote: str) -> dict:
    """Resolve an exact, unambiguous Unicode-codepoint span, not byte offsets."""
    start = text.find(quote)
    if not quote.strip() or start < 0:
        raise ValueError("evidence quote absent from reviewed body")
    if text.find(quote, start + 1) >= 0:
        raise ValueError("ambiguous evidence quote; provide a longer exact quote")
    return {"quote": quote, "start": start, "end": start + len(quote)}


def lint(text: str, *, content_kind: str = "text", language: str = "unknown") -> list[dict]:
    """High-precision *risk flags*. Never silently edit or certify a document.

    These narrow rules intentionally miss many real defects. Flagged records
    cannot pass automatically, including if both teachers overlook a flag.
    A new policy/review version is needed to resolve a false positive.
    """
    flags = []

    def add(code: str, start: int, end: int, explanation: str, severity: str = "hold") -> None:
        flags.append({"code": code, "severity": severity, "start": start, "end": end,
                      "quote": text[start:end], "explanation": explanation})

    # Pair only an explicit address with a nearby, explicitly named reverse node.
    pattern = (r"\b(?P<ip>(?:\d{1,3}\.){3}\d{1,3})\b"
               r"(?P<link>[^\n]{0,180}?(?:reverse|reversing|corresponding)[^\n]{0,80}?)"
               r"(?P<reverse>(?:\d{1,3}\.){3}\d{1,3})\.in-addr\.arpa")
    for match in re.finditer(pattern, text, re.I):
        try:
            ipaddress.IPv4Address(match["ip"])
            ipaddress.IPv4Address(match["reverse"])
        except ValueError:
            continue
        expected = ".".join(reversed(match["ip"].split(".")))
        if match["reverse"] != expected:
            add("dns_reverse_mismatch", match.start(), match.end(),
                f"explicit local pairing inconsistent; expected {expected}.in-addr.arpa")

    # An unfinished enumerated lead-in immediately followed by an obvious heading
    # or another unfinished lead-in. A normal list with actual items is not flagged.
    lines = list(re.finditer(r"[^\n]+", text))
    intro = re.compile(r"(?:may include|include the following|including|options|one could|as follows)\s*:\s*$", re.I)
    heading = re.compile(r"^(?:#{1,6}\s+|What\s+|How\s+|Less Common\s+|Conclusion\b|References\b|Copyright\b|Next Section\b)", re.I)
    for current, following in zip(lines, lines[1:]):
        a, b = current.group().strip(), following.group().strip()
        if intro.search(a) and (heading.search(b) or intro.search(b)):
            add("empty_list_transition", current.start(), following.end(),
                "list introducer has no visible item before the next heading/introducer")

    for match in re.finditer(r"(?:See the rest of (?:the|this) story|Read the rest of (?:the|this) (?:story|article))[^\n]{0,100}", text, re.I):
        add("explicit_continuation_preview", match.start(), match.end(),
            "explicit continuation elsewhere; not evidence of paid access")
    for match in re.finditer(r"(?:Which of the following|选择下列|下列哪)[^\n]{0,250}(?:\.\.\.|…)", text, re.I):
        add("truncated_question_marker", match.start(), match.end(),
            "question/options contain an explicit truncation marker; inspect context")

    # Only complete Python *files*, never prose, fragments, shell commands, or Rust.
    if content_kind == "complete_code_file" and language == "python":
        import ast
        try:
            ast.parse(text)  # parse only; NEVER execute/import the supplied code
        except (SyntaxError, ValueError, RecursionError) as exc:
            flags.append({"code": "python_syntax_failure", "severity": "hold",
                          "explanation": str(exc)[:240]})
    for match in re.finditer(r"(?:Join to answer|Join a community|professional writing compan(?:y|ies))[^\n]{0,100}", text, re.I):
        add("registration_or_promotion_present", match.start(), match.end(),
            "contextual notice only: does not imply a paywall or a worthless article", "info")
    return flags


def normalize_document(record: dict) -> dict:
    if isinstance(record.get("review_payload"), str):
        payload = json.loads(record["review_payload"])
        combined = {**record, **payload}
    else:
        combined = record
    text = combined.get("document_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("missing nonempty full document_text")
    body_sha = sha(text)
    if combined.get("document_id", body_sha) != body_sha:
        raise ValueError("document_id must equal SHA256 of exact UTF-8 body; no normalization here")
    for field in ("source_id", "source_locator"):
        if not isinstance(combined.get(field), str) or not combined[field].strip():
            raise ValueError(f"missing provenance field {field}")
    tokens = combined.get("token_count")
    if tokens is not None and (type(tokens) is not int or tokens < 1):
        raise ValueError("token_count must be a positive integer or unknown")
    doc = {"document_id": body_sha, "body_sha256": body_sha, "document_text": text,
           "source_id": combined["source_id"], "source_locator": combined["source_locator"],
           "source_revision": combined.get("source_revision", "unknown"),
           "license_status": combined.get("license_status", "unresolved"),
           "family_id": combined.get("family_id", "unknown"),
           "language": combined.get("language", "unknown"),
           "content_kind": combined.get("content_kind", "text"),
           "token_count": tokens, "tokenizer_sha256": combined.get("tokenizer_sha256"),
           "sampling_stratum": record.get("sampling_stratum", "unstratified"),
           "previous_bucket": combined.get("current_bucket"),
           "training_approved": False}
    # First judge/bucket/score and deterministic lint results are deliberately blind.
    payload = {k: doc[k] for k in ("document_id", "body_sha256", "source_id", "source_locator",
                                   "language", "content_kind", "document_text")}
    doc["review_payload"] = canonical(payload)
    doc["lint_flags"] = lint(text, content_kind=doc["content_kind"], language=doc["language"])
    return doc


def validate_annotation(annotation: dict, doc: dict, schema: dict) -> dict:
    finite_json(annotation)
    Draft202012Validator(schema).validate(annotation)
    if annotation["document_id"] != doc["document_id"] or annotation["body_sha256"] != doc["body_sha256"]:
        raise ValueError("annotation is for another body/version")
    # This is a full-document protocol. Chunk-only labels are not transferable.
    if annotation["coverage"] != "full_document":
        raise ValueError("incomplete coverage")
    grounded = {"anchors": [span(doc["document_text"], q) for q in annotation["anchors"]], "findings": []}
    for finding in annotation["findings"]:
        if finding["code"] in ("registration_prompt", "ad_present", "external_figure") and finding["severity"] != "info":
            raise ValueError("context notice alone is not a content rejection reason")
        grounded["findings"].append({**finding, "resolved_spans": [span(doc["document_text"], q) for q in finding["quotes"]]})
    adverse = [f for f in annotation["findings"] if f["severity"] in ("reject", "review")]
    if annotation["decision"] == "keep" and adverse:
        raise ValueError("keep contradicts adverse findings")
    if annotation["decision"] != "keep" and not adverse:
        raise ValueError("drop/uncertain requires grounded adverse finding")
    # Grounding proves presence, NOT entailment, factual correctness, or causal origin.
    return grounded


def quality_pass(annotation: dict, policy: dict) -> bool:
    return (annotation["decision"] == "keep"
            and annotation["confidence"] >= policy["minimum_confidence"]
            and all(annotation[k] >= v for k, v in policy["minimum_scores"].items())
            and not any(f["severity"] != "info" for f in annotation["findings"]))


def route(buckets: list[str], language: str) -> tuple[str | None, str]:
    unique = set(buckets)
    if len(unique) == 1:
        return buckets[0], "agree"
    # Do not punish a correct general/technical recategorization as low quality.
    broad = "general_zh" if language in ("zh", "Chinese", "chinese") else "general_en" if language in ("en", "English", "english") else None
    if broad and unique <= {broad, "technical"}:
        return broad, "compatible_general_technical_fallback"
    return None, "routing_uncertain_not_quality_failure"


def source_audit_decision(doc: dict, reviews: dict[str, dict], judges: dict, policy: dict) -> dict:
    """Single-teacher risk signals. Never eligible text or source approval."""
    if set(judges) != {"deepseek"} or judges["deepseek"]["family"] != "deepseek":
        raise ValueError("source audit requires exactly one DeepSeek reviewer")
    out = {"document_id": doc["document_id"], "body_sha256": doc["body_sha256"],
           "source_id": doc["source_id"], "token_count": doc["token_count"],
           "training_approved": False, "source_approved": False,
           "quality_status": "model_signal_not_independent_truth", "routing_status": "diagnostic_only",
           "bucket": None, "reasons": [], "lint_flags": doc["lint_flags"]}
    review = reviews.get("deepseek", {})
    state = review.get("status", "missing")
    if state != "valid":
        out.update(status="audit_invalid" if state in ("invalid", "conflict") else "audit_await_review",
                   reasons=["deepseek:" + state])
        return out
    label = review["annotation"]
    out["bucket"] = label["recommended_bucket"]
    if label["decision"] == "drop":
        out.update(status="audit_drop_signal", reasons=["model_recommends_drop_not_verified_defect"])
    elif (label["decision"] == "uncertain" or not quality_pass(label, policy)
          or any(f["severity"] == "hold" for f in doc["lint_flags"])):
        out.update(status="audit_review_signal", reasons=["uncertain_or_risk_signal"])
    else:
        out.update(status="audit_keep_signal", reasons=["not_source_or_corpus_approval"])
    return out


def decide(doc: dict, reviews: dict[str, dict], judges: dict, policy: dict) -> dict:
    if policy.get("review_purpose") == "source_quality_audit":
        return source_audit_decision(doc, reviews, judges, policy)
    out = {"document_id": doc["document_id"], "body_sha256": doc["body_sha256"],
           "source_id": doc["source_id"], "token_count": doc["token_count"],
           "training_approved": False, "quality_status": "unresolved", "routing_status": "unresolved",
           "bucket": None, "reasons": [], "lint_flags": doc["lint_flags"]}
    if len({j["family"] for j in judges.values()}) < 2:
        raise ValueError("two independent model families required (two seeds are not two judges)")
    pending = [name for name in judges if reviews.get(name, {}).get("status") != "valid"]
    if pending:
        out.update(status="quarantine" if any(reviews.get(n, {}).get("status") in ("invalid", "conflict") for n in pending) else "await_review",
                   reasons=[f"{n}:{reviews.get(n, {}).get('status', 'missing')}" for n in pending])
        return out
    labels = [reviews[name]["annotation"] for name in judges]
    if any(a["decision"] != "keep" for a in labels):
        decisions = {a["decision"] for a in labels}
        out.update(status="excluded", quality_status="excluded_by_policy",
                   reasons=["quality_disagreement" if len(decisions) > 1 else "judges_do_not_keep"])
        return out
    if not all(quality_pass(a, policy) for a in labels):
        out.update(status="excluded", quality_status="score_or_confidence_below_policy",
                   reasons=["quality_threshold"])
        return out
    out["quality_status"] = "both_judges_pass_not_ground_truth"
    out["bucket"], out["routing_status"] = route([a["recommended_bucket"] for a in labels], doc["language"])
    if any(f["severity"] == "hold" for f in doc["lint_flags"]):
        out.update(status="quarantine", reasons=["deterministic_risk_unresolved"])
    elif out["bucket"] is None:
        out.update(status="quarantine", reasons=["routing_uncertain"])
    else:
        out.update(status="candidate_eligible", reasons=[])
    return out


def summary(rows: list[dict]) -> dict:
    counts, tokens = Counter(), Counter()
    unknown = 0
    for r in rows:
        counts[r["status"]] += 1
        if r.get("token_count") is None:
            unknown += 1
        else:
            tokens[r["status"]] += r["token_count"]
    return {"documents": len(rows), "status_documents": dict(counts),
            "reported_body_tokens_by_status": dict(tokens), "unknown_token_count_documents": unknown,
            "new_training_approved_documents": 0, "training_started": False,
            "interpretation": "candidate/review agreement is not independently measured quality accuracy"}
