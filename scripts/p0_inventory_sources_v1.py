#!/usr/bin/env python3
"""Read-only source inventory. Writes reports only; never approves or trains data.

Parquet row counts come from footers, not tokenizer counts. Candidate reports
may overlap and are NEVER added to the three explicitly enumerated frozen pools.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


RAW_ROOTS = ("upstream_v2", "upstream_nonwiki_v1", "upstream_ultradata_probe_v1",
             "upstream_l3_comparison_probe_v1", "fineweb_edu_wave2_v1", "finewiki_zh_2025_candidate_v1",
             "upstream_supply_v4")
FROZEN = ("pretrain_ready_v1/inputs/prior_approved_real_v2/manifest.json",
          "finewiki_zh_short_internal_v2/manifest.json",
          "pretrain_reuse_wave2_v1/frozen_source/manifest.json")
CANDIDATES = ("mdn_full_wave4_v1/audit/mdn_en_full_candidates_v1.manifest.json",
              "wave3_nonhf_pilot_v1/audit/quality_keep_index_v2.summary.json",
              "upstream_nonwiki_v1/audit/high_precision_v2/report.json")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(4 << 20), b""):
            h.update(part)
    return h.hexdigest()


def load_registry(paths: list[Path]) -> dict:
    entries = {}
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        for row in value.get("items", value.get("sources", [])):
            key = (row["source_id"], Path(row.get("path", row.get("remote_file", ""))).name)
            item = {"dataset": row.get("dataset"), "revision": row.get("revision"),
                    "expected_bytes": row.get("bytes", row.get("actual_bytes")),
                    "expected_sha256": row.get("sha256"),
                    "declared_policy": row.get("license_review", row.get("license", row.get("license_status"))),
                    "registry_sha256": sha_file(path)}
            if key in entries and entries[key] != item:
                raise ValueError(f"conflicting registry entry {key}")
            entries[key] = item
    return entries


def inspect_file(path: Path, root: Path, source: str, registry: dict, verify: bool) -> dict:
    before = path.stat()
    out = {"path": str(path.relative_to(root)), "bytes": before.st_size,
           "mtime_ns": before.st_mtime_ns, "rows": None, "own_tokenizer_tokens": None,
           "hash_status": "not_rechecked", "training_approved_by_inventory": False}
    declared = registry.get((source, path.name))
    if declared:
        out.update(declared)
        out["size_matches_registry"] = before.st_size == declared["expected_bytes"]
    else:
        out["registry_status"] = "no_matching_pinned_entry"
    try:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            p = pq.ParquetFile(path)
            out.update(rows=p.metadata.num_rows, columns=p.schema_arrow.names,
                       row_groups=p.metadata.num_row_groups, count_scope="parquet_footer")
        else:
            out["count_scope"] = "file_stat_only_no_full_jsonl_scan"
        if verify:
            out["observed_sha256"] = sha_file(path)
            out["hash_status"] = ("match" if out["observed_sha256"] == declared["expected_sha256"]
                                  else "mismatch") if declared and declared["expected_sha256"] else "computed_unpinned"
    except Exception as exc:
        out["inspection_error"] = type(exc).__name__
    after = path.stat()
    out["stable_during_inspection"] = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    if not out["stable_during_inspection"]:
        out["hash_status"] = "file_changed_hold"
    return out


def frozen_inventory(root: Path) -> tuple[list, dict]:
    pools = []
    reuse = root / FROZEN[-1]
    refs = {}
    if reuse.exists():
        refs = {Path(x["path"]).resolve(): x["manifest_sha256"]
                for x in json.loads(reuse.read_text())["inputs"].get("prior_sources", [])}
    totals = defaultdict(lambda: {"documents": 0, "body_tokens": 0, "stream_tokens_with_eos": 0})
    counted_paths = set()
    for rel in FROZEN:
        p = root / rel
        if not p.exists():
            pools.append({"path": rel, "status": "missing"})
            continue
        x = json.loads(p.read_text())
        digest = sha_file(p)
        duplicate = p.resolve() in counted_paths
        counted_paths.add(p.resolve())
        expected = refs.get(p.parent.resolve())
        binding = "match" if expected == digest else "mismatch" if expected else "not_bound_by_reuse_manifest"
        allowed = (x.get("stage") == "accepted_real" and x.get("license_gate", {}).get("training_eligible") is True
                   and x.get("family_split_conflicts") == 0 and binding != "mismatch" and not duplicate)
        splits = x.get("splits", {})
        if any(type(v.get(k)) is not int or v[k] < 0 for v in splits.values() for k in ("documents", "text_tokens")):
            raise ValueError("invalid frozen counts")
        item = {"path": rel, "manifest_sha256": digest, "stage": x.get("stage"),
                "license_gate": x.get("license_gate"), "quality_gate": x.get("quality_gate"),
                "prior_manifest_binding": binding, "splits": splits,
                "included_in_declared_frozen_totals": allowed,
                "count_evidence": "existing_manifest_not_full_shard_reverification"}
        pools.append(item)
        if allowed:
            for split, v in splits.items():
                totals[split]["documents"] += v["documents"]
                totals[split]["body_tokens"] += v["text_tokens"]
                totals[split]["stream_tokens_with_eos"] += v["text_tokens"] + v["documents"]
    return pools, dict(totals)


def collect(root: Path, registries: list[Path], verify_sources: set[str]) -> dict:
    started = time.monotonic()
    registry = load_registry(registries)
    groups = []
    for name in RAW_ROOTS:
        raw = root / name / "raw"
        paths = sorted(p for p in raw.rglob("*") if p.is_file() and p.suffix in (".parquet", ".jsonl")
                       and not p.is_symlink() and root.resolve() in p.resolve().parents) if raw.exists() else []
        by_source = defaultdict(list)
        for p in paths:
            by_source[p.parent.name if p.parent != raw else name].append(p)
        for source, files in sorted(by_source.items()):
            items = [inspect_file(p, root, source, registry, source in verify_sources and i == 0)
                     for i, p in enumerate(files)]
            rows = [i["rows"] for i in items if i["rows"] is not None]
            groups.append({"root": name, "source": source, "files": items,
                           "file_count": len(items), "bytes": sum(i["bytes"] for i in items),
                           "raw_rows_from_readable_parquet_footers": sum(rows) if rows else None,
                           "rows_counted_files": len(rows), "own_tokenizer_train_tokens": None,
                           "status": "raw_available_not_training_approved"})
    pools, totals = frozen_inventory(root)
    candidates = []
    for rel in CANDIDATES:
        p = root / rel
        if p.exists():
            x = json.loads(p.read_text())
            keys = ("stage", "training_approved", "counts", "deepseek_quality_keep_documents",
                    "deepseek_quality_keep_tokens", "nonquarantined_documents", "nonquarantined_tokens")
            candidates.append({"path": rel, "sha256": sha_file(p),
                               "declared": {k: x[k] for k in keys if k in x},
                               "added_to_ready_totals": False})
    return {"version": 1, "stage": "read_only_inventory_not_new_approval",
            "generated_at": datetime.now(timezone.utc).isoformat(), "root": str(root),
            "inventory_script_sha256": sha_file(Path(__file__)), "raw_groups": groups,
            "frozen_pools": pools, "declared_existing_frozen_totals": totals,
            "additional_candidate_reports": candidates, "new_approved_train_tokens": 0,
            "new_api_calls": 0, "training_started": False, "elapsed_s": time.monotonic() - started,
            "limitations": ["Raw rows are not usable documents or training token counts.",
                "Only the first file of each explicitly selected source is SHA rechecked; other files are not.",
                "Candidate and raw reports may overlap; no sum into ready inventory.",
                "Frozen totals inherit existing policies and checks, not fresh scientific quality certification."]}


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".inventory-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.link(name, path)  # Atomic publication, no overwrite on repeat invocation.
    finally:
        os.unlink(name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--registry", type=Path, action="append", default=[])
    p.add_argument("--verify-source", action="append", default=[], help="SHA-check first file only")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("immutable report already exists")
    out = collect(a.root.resolve(), a.registry, set(a.verify_source))
    write_new(a.output, out)
    print(json.dumps({"report": str(a.output), "raw_source_groups": len(out["raw_groups"]),
        "declared_frozen_totals": out["declared_existing_frozen_totals"],
        "new_approved_train_tokens": 0, "new_api_calls": 0, "elapsed_s": out["elapsed_s"]}))


if __name__ == "__main__":
    main()
