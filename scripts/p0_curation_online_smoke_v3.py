#!/usr/bin/env python3
"""Freeze a 40-document diagnostic plan, not a corpus acceptance sample."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from small_agent.curation.core import canonical, normalize_document, sha
from small_agent.curation.store import file_sha, read_jsonl, write_json


DIAGNOSTIC_IDS = {
    "ec095798d7c2e8842c04ce8f003eb7b0aa889f2290d661c99b52c740932945e3": "dns_numeric_contradiction",
    "9f012f621562b8324f6c124c79aa2a1cbe8d8e51735ffbfe7a00309405cf9538": "technical_with_incomplete_quiz",
    "0636011ab86a36be64faa92c11b422ec10644af90c7a884b64fb8b0464b8628a": "geology_routing_not_truth_certificate",
    "c2c1bb50da2c5fe12b220b148cec51cfc03175009911a5abff3411c4ebc7436d": "cpp_answer_context_dependency",
    "fe781109ba9af5c94d795057d2b59092a92ba4a781a15d1f54b6534002761069": "math_teaching_with_image_reference",
    "15483a7798ecefdc3facffcaf17f385dc4f4e0ca43d098f747a994be12b159b5": "empty_medical_lists_not_fact_check",
    "42e531cebceecdbf5a0da0ee832c0145ffda5e2295c0c5c6c70374a497d25c06": "empty_forest_options",
    "7eebbbb69930c04bf001ab123ab8e4f1f03abaec68528ca14a7a16f539ca0c54": "continuation_not_proven_paywall",
    "118d3dd6186a4bdebae296992c17af566fa2d769201f4905e67a296218d3cf6b": "incomplete_options_registration_not_paywall",
    "b75980b0d4bbe8c67fac21a8a694d5e1f7285f5bdc2dd86e87dbe9aa2b7af1ed": "unnatural_wording_not_generated_origin_proof",
    "089371705b9b09f2e9b7a1eec4c3a3c6639434dff619b461067f6e3b322de9aa": "ad_present_not_whole_document_verdict",
}


def controls() -> list[dict]:
    dns = """Understanding a reverse DNS name

A reverse DNS name represents an IPv4 address by putting its four octets in
reverse order and appending the suffix in-addr.arpa. To construct the name,
first split the address at the dots, reverse the sequence of four components,
and then join them with dots. This is a symbolic transformation of a name; it
does not by itself show whether any DNS record has been published.

IP address 192.0.2.17 has corresponding reverse node 17.2.0.192.in-addr.arpa.
The final octet becomes the first label, and the first octet becomes the last
label before the suffix. The address in this example has four components:
192, 0, 2, and 17. The reverse sequence is 17, 2, 0, and 192.

The transformation is reversible. Remove the fixed suffix, split the remaining
four labels, and reverse them again to reconstruct the original address.
This check is useful for detecting a mismatch between a written address and
its stated reverse name. It checks internal consistency, not the existence
of a live record or the identity of a machine on a network.
"""
    lists = """Checking a tokenized dataset before training

A data loader should use the same tokenizer version that created the token
files. A readable file is not enough: its contents must correspond to the
frozen source and configuration. Checksums detect file changes, while sample
inspection tests whether the loader interprets the format as intended.

Essential checks may include:
- Compare each shard checksum with the frozen manifest.
- Confirm token IDs fall inside the tokenizer vocabulary.
- Inspect examples at shard boundaries and at the end of a split.
- Restore a saved sampler state and compare the next sample sequence.

What Should Happen Next?
Only after those checks pass should a short engineering run begin. Keep a
separate validation split and record the exact run configuration. A successful
loader test does not prove that the model will learn useful language skills.
If a check fails, stop and fix the underlying file or configuration, rather
than changing a recorded checksum to make an altered file appear unchanged.
"""
    code = '''"""A small iterative factorial example.

Factorial multiplies the integers from one through n. The empty product is one,
so factorial(0) is one. Negative inputs are outside this example's domain.
The loop keeps a running product and uses constant auxiliary state.
This is a complete Python file, including deterministic assertions.
"""

def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be nonnegative")
    product = 1
    for factor in range(2, n + 1):
        product *= factor
    return product

def verify_examples() -> None:
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(5) == 120
    try:
        factorial(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative input was not rejected")

if __name__ == "__main__":
    verify_examples()
'''
    math = """Solving and checking a linear equation

Consider the equation 3x + 5 = 26. To find x, apply the same operation to both
sides, which preserves the equality. Subtract five from both sides:

3x = 26 - 5 = 21.

Then divide both sides by three:

x = 21 / 3 = 7.

The candidate solution is x = 7. Check it by substitution into the original
equation: 3 times 7 plus 5 equals 21 plus 5, which equals 26. Thus the left
and right sides agree. Substitution is a separate verification step and can
catch arithmetic mistakes made while isolating the variable.

More generally, for ax + b = c with a nonzero, subtract b and divide by a to
obtain x = (c - b) / a. The nonzero condition matters: division by zero is not
defined, and the case a = 0 must instead be handled by checking whether b = c.
"""
    pairs = [
        ("dns", dns, dns.replace("corresponding reverse node 17.2.0.192", "corresponding reverse node 192.0.2.17"), "text", "en", "dns_reverse_mismatch"),
        ("list", lists, lists.replace("- Compare each shard checksum with the frozen manifest.\n- Confirm token IDs fall inside the tokenizer vocabulary.\n- Inspect examples at shard boundaries and at the end of a split.\n- Restore a saved sampler state and compare the next sample sequence.\n", ""), "text", "en", "empty_list_transition"),
        ("python", code, code.replace("for factor in range(2, n + 1):", "for factor in range(2, n + 1"), "complete_code_file", "python", "python_syntax_failure"),
        ("math", math, math.replace("3x = 26 - 5 = 21.", "3x = 26 - 5 = 51.").replace("x = 21 / 3 = 7.", "x = 51 / 3 = 17.").replace("The candidate solution is x = 7.", "The candidate solution is x = 17."), "text", "en", "internal_arithmetic_contradiction"),
    ]
    out = []
    for name, good, bad, kind, lang, defect in pairs:
        for variant, text in (("good", good), ("bad", bad)):
            doc = {"document_id": sha(text), "document_text": text, "source_id": "curation_v3_authored_control",
                   # Expected labels must not leak through a /good or /bad URL.
                   "source_locator": f"internal://curation-v3-controls/{sha(text)}",
                   "source_revision": "authored_control_v1", "license_status": "internal_authored_control_not_training",
                   "family_id": f"authored_control_{name}", "content_kind": kind, "language": lang,
                   "token_count": None, "sampling_stratum": f"control_{variant}",
                   "control_name": f"{name}_{variant}", "expected_control_decision": "keep" if variant == "good" else "exclude",
                   "expected_observable_defect": None if variant == "good" else defect}
            out.append(doc)
    return out


def select(source: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    raw = list(read_jsonl(source))
    records = {normalize_document(r)["document_id"]: r for r in raw}
    if not DIAGNOSTIC_IDS.keys() <= records.keys():
        raise ValueError("fixed diagnostic cases missing")
    chosen = []
    for d, reason in DIAGNOSTIC_IDS.items():
        chosen.append({**records[d], "diagnostic_case": reason, "smoke_group": "previously_read_diagnostic"})
    rng = random.Random(20261001)
    for s in ("qwen_short", "qwen_middle", "qwen_long"):
        candidates = sorted(d for d, r in records.items() if r["sampling_stratum"] == s and d not in DIAGNOSTIC_IDS)
        for d in rng.sample(candidates, 7):
            chosen.append({**records[d], "smoke_group": "fresh_from_old_frozen_sample"})
    ctrl = controls()
    pilot = next(d for d in ctrl if d["control_name"] == "dns_good")
    chosen += [{**c, "smoke_group": "authored_control"} for c in ctrl if c["document_id"] != pilot["document_id"]]
    assert len(chosen) == 39 and len({normalize_document(r)["document_id"] for r in chosen}) == 39
    output.mkdir(parents=True)
    for name, rows in (("pilot.input.jsonl", [pilot]), ("calibration39.input.jsonl", chosen)):
        with (output / name).open("x", encoding="utf-8") as f:
            for row in rows:
                f.write(canonical(row) + "\n")
    plan = {"source_sha256": file_sha(source), "selection_program_sha256": file_sha(Path(__file__)),
            "seed": 20261001, "documents_total": 40, "pilot_documents": 1, "expansion_documents": 39,
            "known_diagnostics": 11, "fresh_real_documents": 21, "authored_controls": 8,
            "files": {n: file_sha(output / n) for n in ("pilot.input.jsonl", "calibration39.input.jsonl")},
            "cases": [{"document_id": normalize_document(r)["document_id"], "smoke_group": r.get("smoke_group", "authored_control"),
                       "control_name": r.get("control_name"), "expected_control_decision": r.get("expected_control_decision"),
                       "diagnostic_case": r.get("diagnostic_case")} for r in [pilot, *chosen]],
            "limits": ["diagnostic/development set, not independently representative corpus acceptance",
                       "authored controls test narrow consistency; not natural-language accuracy ground truth",
                       "fresh real documents are from the same previously frozen population, not source-disjoint references"],
            "training_approved": False}
    write_json(output / "PLAN.json", plan)
    return plan


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = select(args.input, args.output)
    import json
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, ensure_ascii=False, indent=2))
