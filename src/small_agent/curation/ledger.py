"""Campaign-wide CNY reservations; ambiguous requests are not silently re-sent."""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
from pathlib import Path
import sqlite3

from .core import canonical, fingerprint


def micros(cny) -> int:
    value = Decimal(str(cny))
    if not value.is_finite() or value < 0:
        raise ValueError("finite nonnegative CNY amount required")
    return int((value * 1_000_000).to_integral_value(rounding=ROUND_CEILING))


class Ledger:
    def __init__(self, path: Path, *, budget_cny, account_sha256: str, pricing: dict):
        if len(account_sha256) != 64:
            raise ValueError("account fingerprint required; never store the key")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests(job_key TEXT PRIMARY KEY, reserved INTEGER NOT NULL,
                counted INTEGER NOT NULL, state TEXT NOT NULL, result_json TEXT);
        """)
        self.cap = micros(budget_cny)
        identity = fingerprint({"currency": "CNY", "cap_micros": self.cap,
                                "account_sha256": account_sha256, "pricing": pricing})
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            found = self.db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
            if found and found[0] != identity:
                raise ValueError("campaign budget/account/pricing changed; do not reset the ledger")
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('identity',?)", (identity,))

    def reserve(self, job_key: str, bound_cny) -> bool:
        """True: caller may send once. False: an attempt already exists; don't resend."""
        bound = micros(bound_cny)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.db.execute("SELECT 1 FROM requests WHERE job_key=?", (job_key,)).fetchone():
                return False
            counted = self.db.execute("SELECT COALESCE(SUM(counted),0) FROM requests").fetchone()[0]
            if counted + bound > self.cap:
                raise ValueError("CNY budget reservation cap reached")
            self.db.execute("INSERT INTO requests VALUES (?,?,?,'inflight',NULL)", (job_key, bound, bound))
        return True

    def finish(self, job_key: str, result: dict, *, observed_cost_cny=None) -> None:
        """Usage-derived cost is an estimate, not a settled bill.

        Missing usage retains the full reservation. If the assumed bound/pricing
        was wrong, record the larger observed estimate and block future requests;
        no software can retroactively guarantee an already-incurred charge.
        """
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute("SELECT * FROM requests WHERE job_key=?", (job_key,)).fetchone()
            if not row or row["state"] != "inflight":
                raise ValueError("missing/already-completed reservation")
            counted = row["reserved"] if observed_cost_cny is None else micros(observed_cost_cny)
            self.db.execute("UPDATE requests SET counted=?,state=?,result_json=? WHERE job_key=?", (
                counted, "usage_estimated" if observed_cost_cny is not None else "uncertain_cost_reserved", canonical(result), job_key))

    def result(self, job_key: str):
        import json
        row = self.db.execute("SELECT result_json FROM requests WHERE job_key=?", (job_key,)).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def report(self) -> dict:
        total = self.db.execute("SELECT COALESCE(SUM(counted),0) FROM requests").fetchone()[0]
        return {"currency": "CNY", "cap_cny": self.cap / 1_000_000,
                "estimated_or_reserved_cny": total / 1_000_000,
                "headroom_cny": max(0, self.cap - total) / 1_000_000,
                "cap_exceeded": total > self.cap,
                "requests_by_state": dict(self.db.execute("SELECT state,COUNT(*) FROM requests GROUP BY state").fetchall()),
                "not_a_settled_bill": True}

    def close(self):
        self.db.close()
