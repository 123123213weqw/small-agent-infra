"""Persistent rolling-batch state and a unique released-document ledger.

This control layer does not implement data processing, judge scientific quality,
or launch tasks. Only a verified release receipt plus a bound document index can
advance the counter. External measured checkers/handlers must still be connected.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import time


STAGES = ("ingest", "normalize", "dedup", "quality", "freeze", "pack", "verify")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(4 << 20), b""):
            h.update(part)
    return h.hexdigest()


def validate_recipe(recipe: dict) -> None:
    targets = recipe.get("targets_by_domain", {})
    if (recipe.get("version") != 4 or not targets or
            any(type(v) is not int or v <= 0 for v in targets.values()) or
            sum(targets.values()) != recipe.get("target_train_body_tokens")):
        raise ValueError("positive domain quotas must sum to total train body tokens")
    if recipe.get("stages") != list(STAGES) or recipe.get("auto_training") is not False:
        raise ValueError("stage chain and independent training boundary required")
    if recipe.get("maximum_unreleased_batches", 0) < 1:
        raise ValueError("bounded backlog required")
    checks = recipe.get("required_release_checks", [])
    if not checks or len(set(checks)) != len(checks) or len(recipe.get("tokenizer_sha256", "")) != 64:
        raise ValueError("release checks and frozen tokenizer required")


def recipe_identity(recipe: dict) -> str:
    # Operational CPU quotas / deployment state / paid budget are not corpus
    # semantics. Changing them must not erase or re-count released documents.
    keys = ("version", "purpose", "target_train_body_tokens", "counting_scope",
            "targets_by_domain", "tokenizer_sha256", "review_policy", "required_release_checks")
    return digest({key: recipe[key] for key in keys})


class SupplyControl:
    def __init__(self, database: Path, recipe: dict):
        validate_recipe(recipe)
        database.parent.mkdir(parents=True, exist_ok=True)
        self.recipe, self.recipe_sha = recipe, recipe_identity(recipe)
        self.db = sqlite3.connect(database, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS batches(
            batch_id TEXT PRIMARY KEY, domain TEXT NOT NULL, input_binding TEXT NOT NULL,
            stage INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'queued',
            reason TEXT, created REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS stage_receipts(
            batch_id TEXT, stage INTEGER, receipt_sha TEXT NOT NULL, files_json TEXT NOT NULL,
            PRIMARY KEY(batch_id,stage), FOREIGN KEY(batch_id) REFERENCES batches(batch_id));
          CREATE TABLE IF NOT EXISTS releases(
            batch_id TEXT PRIMARY KEY, receipt_sha TEXT UNIQUE NOT NULL, train_tokens INTEGER NOT NULL,
            FOREIGN KEY(batch_id) REFERENCES batches(batch_id));
          CREATE TABLE IF NOT EXISTS documents(
            body_sha TEXT PRIMARY KEY, batch_id TEXT NOT NULL, split TEXT NOT NULL,
            domain TEXT NOT NULL, tokens INTEGER NOT NULL,
            FOREIGN KEY(batch_id) REFERENCES batches(batch_id));
          CREATE INDEX IF NOT EXISTS documents_domain_split ON documents(domain,split);
          CREATE TABLE IF NOT EXISTS legacy_imports(
            batch_id TEXT PRIMARY KEY, receipt_sha TEXT NOT NULL,
            FOREIGN KEY(batch_id) REFERENCES batches(batch_id));
          CREATE TABLE IF NOT EXISTS legacy_families(
            family_id TEXT PRIMARY KEY, split TEXT NOT NULL);
        """)
        try:
            with self.db:
                old = self.db.execute("SELECT value FROM meta WHERE key='recipe_sha'").fetchone()
                if old and old[0] != self.recipe_sha:
                    raise ValueError("recipe changed; explicit versioned migration required")
                self.db.execute("INSERT OR IGNORE INTO meta VALUES ('recipe_sha',?)", (self.recipe_sha,))
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def status(self) -> dict:
        counts = {d: 0 for d in self.recipe["targets_by_domain"]}
        legacy = dict(counts)
        for row in self.db.execute("SELECT domain,SUM(tokens) FROM documents WHERE split='train' GROUP BY domain"):
            counts[row[0]] = row[1]
        for row in self.db.execute("SELECT d.domain,SUM(d.tokens) FROM documents d JOIN legacy_imports l "
                                   "ON d.batch_id=l.batch_id WHERE d.split='train' GROUP BY d.domain"):
            legacy[row[0]] = row[1]
        new = {d: counts[d] - legacy[d] for d in counts}
        remaining = {d: max(0, t - counts[d]) for d, t in self.recipe["targets_by_domain"].items()}
        return {"recipe_sha256": self.recipe_sha, "unique_ready_train_body_tokens": sum(new.values()),
                "ready_by_domain": new, "legacy_internal_train_body_tokens": sum(legacy.values()),
                "legacy_internal_by_domain": legacy, "accounted_unique_train_body_tokens": sum(counts.values()),
                "accounted_by_domain": counts, "remaining_by_domain": remaining,
                "legacy_imports": self.db.execute("SELECT COUNT(*) FROM legacy_imports").fetchone()[0],
                "legacy_quality_scope": "inherited_internal_only_not_new_v4_certification",
                "target_met": not any(remaining.values()),
                "batch_statuses": dict(self.db.execute("SELECT status,COUNT(*) FROM batches GROUP BY status").fetchall()),
                "candidates_counted_as_ready": False, "training_started": False}

    def register(self, domain: str, input_binding: dict) -> str:
        if domain not in self.recipe["targets_by_domain"] or not input_binding:
            raise ValueError("known domain and frozen input binding required")
        batch_id = digest({"domain": domain, "input_binding": input_binding, "recipe": self.recipe_sha})
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.db.execute("SELECT 1 FROM batches WHERE batch_id=?", (batch_id,)).fetchone():
                return batch_id
            if not self.status()["remaining_by_domain"][domain]:
                raise ValueError("domain quota already met")
            pending = self.db.execute("SELECT COUNT(*) FROM batches WHERE status NOT IN ('ready','quarantined','legacy_internal')").fetchone()[0]
            if pending >= self.recipe["maximum_unreleased_batches"]:
                raise ValueError("backpressure: release pending batches before adding data")
            self.db.execute("INSERT INTO batches(batch_id,domain,input_binding,created) VALUES (?,?,?,?)",
                            (batch_id, domain, json.dumps(input_binding, sort_keys=True), time.time()))
        return batch_id

    def next_domain(self, available_domains: set[str]) -> str | None:
        state = self.status()
        eligible = [d for d in self.recipe["targets_by_domain"]
                    if d in available_domains and state["remaining_by_domain"][d]]
        return min(eligible, key=lambda d: (state["accounted_by_domain"][d] /
                   self.recipe["targets_by_domain"][d], d)) if eligible else None

    def import_legacy(self, receipt_path: Path) -> dict:
        """Import audited old inventory, without manufacturing a new READY receipt.

        Audit binds real frozen Parquets, a document index and a physical pack
        verification report. Existing quality restrictions remain unchanged.
        """
        receipt = json.loads(receipt_path.read_text())
        receipt_sha = file_sha(receipt_path)
        if (receipt.get("stage") != "legacy_inventory_audit_internal_only" or
                receipt.get("tokenizer_sha256") != self.recipe["tokenizer_sha256"] or
                receipt.get("recipe_sha256") != self.recipe_sha):
            raise ValueError("legacy receipt identity mismatch")
        for item in receipt["evidence_files"]:
            if file_sha(Path(item["path"])) != item["sha256"]:
                raise ValueError("legacy evidence changed")
        pack = json.loads(Path(receipt["pack_verification_path"]).read_text())
        if (pack.get("stage") != "physical_pack_hashes_verified" or
                pack.get("tokenizer_sha256") != self.recipe["tokenizer_sha256"] or
                pack.get("manifest_sha256") != receipt.get("pack_manifest_sha256") or
                pack.get("body_tokens_by_raw_domain_split") != receipt.get("body_tokens_by_raw_domain_split") or
                pack.get("documents_by_raw_domain_split") != receipt.get("documents_by_raw_domain_split")):
            raise ValueError("legacy physical pack counters mismatch")
        bound = {x["path"] for x in receipt["evidence_files"]}
        index = Path(receipt["document_index_path"])
        if str(index) not in bound or receipt["pack_verification_path"] not in bound:
            raise ValueError("legacy index and physical report must be hash bound")
        batch = digest({"legacy_input_identity":receipt["input_identity"], "recipe":self.recipe_sha})
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            old = self.db.execute("SELECT receipt_sha FROM legacy_imports WHERE batch_id=?", (batch,)).fetchone()
            if old:
                if old[0] != receipt_sha:
                    raise ValueError("legacy import changed")
                return self.status()
            self.db.execute("INSERT INTO batches(batch_id,domain,input_binding,status,created) VALUES (?,?,?,?,?)",
                            (batch,"legacy",json.dumps(receipt["input_identity"],sort_keys=True),"legacy_internal",time.time()))
            totals = Counter(); indexed_documents = Counter()
            with index.open() as f:
                for line in f:
                    d = json.loads(line)
                    if (d["domain"] not in self.recipe["targets_by_domain"] or
                            d["split"] not in ("train","validation","test") or
                            type(d["body_tokens"]) is not int or d["body_tokens"] <= 0 or
                            len(d["body_sha256"]) != 64 or any(c not in '0123456789abcdef' for c in d["body_sha256"])):
                        raise ValueError("invalid legacy document")
                    previous = self.db.execute("SELECT split,domain,tokens FROM documents WHERE body_sha=?",
                                               (d["body_sha256"],)).fetchone()
                    if previous and tuple(previous) != (d["split"],d["domain"],d["body_tokens"]):
                        raise ValueError("legacy duplicate split/domain/token conflict")
                    if not previous:
                        self.db.execute("INSERT INTO documents VALUES (?,?,?,?,?)",
                                        (d["body_sha256"],batch,d["split"],d["domain"],d["body_tokens"]))
                    if d.get("family_id"):
                        family = self.db.execute("SELECT split FROM legacy_families WHERE family_id=?",
                                                 (d["family_id"],)).fetchone()
                        if family and family[0] != d["split"]:
                            raise ValueError("legacy family split conflict")
                        self.db.execute("INSERT OR IGNORE INTO legacy_families VALUES (?,?)",
                                        (d["family_id"],d["split"]))
                    totals[(d["raw_domain"],d["split"])] += d["body_tokens"]
                    indexed_documents[(d["raw_domain"],d["split"])] += 1
            if {f"{d}/{s}":n for (d,s),n in totals.items()} != receipt["body_tokens_by_raw_domain_split"]:
                raise ValueError("legacy index totals mismatch")
            if {f"{d}/{s}":n for (d,s),n in indexed_documents.items()} != receipt["documents_by_raw_domain_split"]:
                raise ValueError("legacy index document counts mismatch")
            self.db.execute("INSERT INTO legacy_imports VALUES (?,?)",(batch,receipt_sha))
        return self.status()

    def commit_stage(self, batch_id: str, stage: str, files: list[Path]) -> None:
        if stage not in STAGES or not files:
            raise ValueError("known stage and durable measured output required")
        bindings = {str(p.resolve()): file_sha(p) for p in files}
        index = STAGES.index(stage)
        receipt = digest({"batch": batch_id, "stage": stage, "files": bindings})
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
            if row is None or row["status"] in ("ready", "quarantined"):
                raise ValueError("unknown or terminal batch")
            previous = self.db.execute("SELECT receipt_sha FROM stage_receipts WHERE batch_id=? AND stage=?", (batch_id,index)).fetchone()
            if previous:
                if previous[0] != receipt:
                    raise ValueError("existing stage output changed")
                return
            if row["stage"] != index:
                raise ValueError("cannot skip a processing stage")
            self.db.execute("INSERT INTO stage_receipts VALUES (?,?,?,?)",
                            (batch_id,index,receipt,json.dumps(bindings,sort_keys=True)))
            self.db.execute("UPDATE batches SET stage=stage+1,status='queued',reason=NULL WHERE batch_id=?", (batch_id,))

    def hold(self, batch_id: str, reason: str, *, quarantine: bool = False) -> None:
        with self.db:
            row = self.db.execute("SELECT status FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
            if row is None or row[0] == "ready":
                raise ValueError("unknown or released batch")
            self.db.execute("UPDATE batches SET status=?,reason=? WHERE batch_id=?",
                            ("quarantined" if quarantine else "waiting_gate", reason, batch_id))

    def release(self, batch_id: str, receipt_path: Path) -> dict:
        receipt = json.loads(receipt_path.read_text())
        receipt_sha = file_sha(receipt_path)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            old = self.db.execute("SELECT receipt_sha FROM releases WHERE batch_id=?", (batch_id,)).fetchone()
            if old:
                if old[0] != receipt_sha:
                    raise ValueError("released receipt changed")
                return self.status()
            batch = self.db.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
            if batch is None or batch["stage"] != len(STAGES) or batch["status"] == "quarantined":
                raise ValueError("complete measured stage chain required")
            for stage in self.db.execute("SELECT files_json FROM stage_receipts WHERE batch_id=?", (batch_id,)):
                for name, expected in json.loads(stage[0]).items():
                    if file_sha(Path(name)) != expected:
                        raise ValueError("stage evidence changed")
            if (receipt.get("status") != "READY_INTERNAL" or receipt.get("batch_id") != batch_id or
                    receipt.get("recipe_sha256") != self.recipe_sha or
                    receipt.get("tokenizer_sha256") != self.recipe["tokenizer_sha256"] or
                    receipt.get("input_binding_sha256") != digest(json.loads(batch["input_binding"]))):
                raise ValueError("READY receipt identity mismatch")
            checks = receipt.get("checks", [])
            if (len(checks) != len(self.recipe["required_release_checks"]) or
                    {c["name"] for c in checks} != set(self.recipe["required_release_checks"])):
                raise ValueError("required release checks missing or duplicated")
            for check in checks:
                if (check.get("status") != "passed" or not check.get("checker_revision") or not check.get("evidence_files")
                        or check.get("batch_id") != batch_id or check.get("recipe_sha256") != self.recipe_sha
                        or check.get("input_binding_sha256") != receipt["input_binding_sha256"]):
                    raise ValueError("measured check evidence required")
                for evidence in check["evidence_files"]:
                    if file_sha(Path(evidence["path"])) != evidence["sha256"]:
                        raise ValueError("check evidence changed")
            index = receipt["document_index"]
            if file_sha(Path(index["path"])) != index["sha256"]:
                raise ValueError("document index changed")
            pack = receipt["pack_manifest"]
            if file_sha(Path(pack["path"])) != pack["sha256"]:
                raise ValueError("pack manifest changed")
            packed = json.loads(Path(pack["path"]).read_text())
            if packed.get("tokenizer_sha256") != self.recipe["tokenizer_sha256"]:
                raise ValueError("packed tokenizer differs")
            if type(receipt.get("unique_train_body_tokens")) is not int:
                raise ValueError("integer body counter required")
            for s in packed["shards"]:
                if (type(s.get("tokens")) is not int or type(s.get("documents")) is not int or
                        not 0 <= s["documents"] <= s["tokens"]):
                    raise ValueError("invalid pack counts")
            pack_body = sum(s["tokens"] - s["documents"] for s in packed["shards"] if s["split"] == "train")
            if packed.get("eos_token_id") != 2 or pack_body != receipt.get("unique_train_body_tokens"):
                raise ValueError("physical pack and body counter differ")
            totals = Counter()
            with Path(index["path"]).open() as f:
                for line in f:
                    doc = json.loads(line)
                    if (doc["domain"] != batch["domain"] or doc["split"] not in ("train","validation","test") or
                            type(doc["body_tokens"]) is not int or doc["body_tokens"] <= 0 or
                            len(doc["body_sha256"]) != 64 or any(c not in '0123456789abcdef' for c in doc["body_sha256"])):
                        raise ValueError("invalid released document")
                    # Global unique key also forbids body reuse across held-out splits.
                    if doc.get("family_id"):
                        old_family = self.db.execute("SELECT split FROM legacy_families WHERE family_id=?",
                                                     (doc["family_id"],)).fetchone()
                        if old_family and old_family[0] != doc["split"]:
                            raise ValueError("released family overlaps an inventory held-out split")
                        # The table name is historical; it now protects ALL released families.
                        # Insert inside the release transaction: restart/concurrent releases
                        # cannot omit or race the held-out-family reference update.
                        self.db.execute("INSERT OR IGNORE INTO legacy_families VALUES (?,?)",
                                        (doc["family_id"],doc["split"]))
                    self.db.execute("INSERT INTO documents VALUES (?,?,?,?,?)",
                                    (doc["body_sha256"],batch_id,doc["split"],doc["domain"],doc["body_tokens"]))
                    totals[doc["split"]] += doc["body_tokens"]
            if totals["train"] != receipt.get("unique_train_body_tokens") or totals["train"] <= 0:
                raise ValueError("train counter must agree with indexed unique bodies")
            self.db.execute("INSERT INTO releases VALUES (?,?,?)", (batch_id,receipt_sha,totals["train"]))
            self.db.execute("UPDATE batches SET status='ready',reason=NULL WHERE batch_id=?", (batch_id,))
        return self.status()
