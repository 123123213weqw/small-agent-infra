#!/usr/bin/env python3
"""Describe a frozen small online experiment, never certify a training corpus."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from small_agent.curation.core import quality_pass
from small_agent.curation.store import Batch, file_sha, write_json


def report(plan_path: Path, batch_paths: list[Path]) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    planned = {r["document_id"]: r for r in plan["cases"]}
    records, manifests = {}, {}
    for path in batch_paths:
        batch = Batch(path)
        try:
            manifests[str(path)] = batch.manifest_hash
            decisions = {r["document_id"]: r for r in batch.decisions()}
            for doc_id, doc in batch.docs.items():
                if doc_id in records or doc_id not in planned:
                    raise ValueError("duplicate or unplanned experiment document")
                judges = {}
                for row in batch.db.execute("SELECT * FROM jobs WHERE document_id=?", (doc_id,)):
                    a = json.loads(row["annotation"]) if row["annotation"] else None
                    judges[row["judge"]] = {"status": row["status"]}
                    if a:
                        judges[row["judge"]].update(decision=a["decision"], quality_pass=quality_pass(a, batch.policy),
                            bucket=a["recommended_bucket"], confidence=a["confidence"],
                            scores={k: a[k] for k in ("quality", "completeness", "educational_value", "format_integrity")},
                            finding_codes=[f["code"] for f in a["findings"]])
                p = planned[doc_id]
                records[doc_id] = {"document_id": doc_id, "group": p["smoke_group"],
                    "control_name": p.get("control_name"), "expected_control_decision": p.get("expected_control_decision"),
                    "body_token_count_carried_not_recounted": doc["token_count"],
                    "status": decisions[doc_id]["status"], "quality_status": decisions[doc_id]["quality_status"],
                    "routing_status": decisions[doc_id]["routing_status"], "reasons": decisions[doc_id]["reasons"],
                    "lint_codes": [f["code"] for f in doc["lint_flags"]], "judges": judges}
        finally:
            batch.close()
    if set(records) != set(planned):
        raise ValueError("all planned documents, including failed reviews, must be counted")
    rows = list(records.values())
    groups = {g: dict(Counter(r["status"] for r in rows if r["group"] == g)) for g in sorted({r["group"] for r in rows})}
    controls = [r for r in rows if r["control_name"]]
    return {"stage": "small_online_smoke_not_corpus_acceptance", "plan_sha256": file_sha(plan_path),
        "batch_manifest_sha256": manifests, "documents": len(rows), "groups": groups,
        "statuses": dict(Counter(r["status"] for r in rows)),
        "review_statuses": dict(Counter(j["status"] for r in rows for j in r["judges"].values())),
        "control_results": controls, "records": rows, "training_approved": False,
        "training_started": False, "independent_population_quality_accuracy_measured": False,
        "limits": ["authored controls measure only their narrow observed defects",
                   "known diagnostic examples were used in prompt development, not independent tests",
                   "fresh real examples are from the old audit population, not source-disjoint reference labels",
                   "grounded evidence and two-model agreement do not certify external factual truth",
                   "no release-check receipts, no new token approval and no training"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--batch", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("version experiment reports, do not overwrite")
    out = report(a.plan, a.batch)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.output, out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("records", "control_results")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
