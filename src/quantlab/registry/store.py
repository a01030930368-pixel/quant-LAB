"""Research registry: append-only SQLite store of hypotheses, policies,
datasets, experiments, verdicts, holdout access and strategy state changes.

Original records are immutable (03 §3.1 F, §7.2): UPDATE and DELETE are
rejected by triggers. A wrong experiment is never overwritten; it gets an
INVALID verdict and a new experiment row points back to it via `supersedes`.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quantlab.core.hashing import canonical_json, stable_hash

IMMUTABLE = ("hypotheses", "policies", "datasets", "experiments", "verdicts", "holdout_ledger", "state_changes")

SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS policies (
    policy_hash TEXT PRIMARY KEY, policy_id TEXT NOT NULL, version INTEGER NOT NULL,
    created_at TEXT NOT NULL, body TEXT NOT NULL, UNIQUE(policy_id, version));
CREATE TABLE IF NOT EXISTS datasets (
    dataset_version TEXT PRIMARY KEY, created_at TEXT NOT NULL, manifest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('exploratory', 'formal')),
    hypothesis_id TEXT,
    strategy_id TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    configuration_version TEXT NOT NULL,
    validation_policy_version TEXT,
    code_commit TEXT NOT NULL,
    random_seed INTEGER,
    execution_timestamp TEXT NOT NULL,
    config TEXT NOT NULL,
    metrics TEXT NOT NULL,
    result_hash TEXT NOT NULL,
    failure_reason TEXT,
    supersedes TEXT REFERENCES experiments(experiment_id));
CREATE TABLE IF NOT EXISTS verdicts (
    verdict_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    gate TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('PASS', 'HOLD', 'FAIL', 'INVALID', 'NOT_RUN')),
    reasons TEXT NOT NULL,
    created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS holdout_ledger (
    access_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    strategy_family TEXT NOT NULL,
    strategy_config_hash TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    experiment_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS state_changes (
    change_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT NOT NULL,
    reason TEXT NOT NULL,
    experiment_id TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ImmutableRecordError(RuntimeError):
    pass


class Registry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        for table in IMMUTABLE:
            for op in ("UPDATE", "DELETE"):
                self.conn.execute(
                    f"CREATE TRIGGER IF NOT EXISTS no_{op.lower()}_{table} BEFORE {op} ON {table} "
                    f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END"
                )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _insert(self, table: str, row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        try:
            with self.conn:
                self.conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))
        except sqlite3.IntegrityError as e:
            raise ImmutableRecordError(f"{table}: {e}") from e

    # --- registration -------------------------------------------------
    def register_hypothesis(self, hypothesis_id: str, body: dict) -> None:
        required = {"strategy_id", "market", "economic_rationale", "rules", "benchmark",
                    "risks", "falsification", "validation_plan"}
        missing = required - set(body)
        if missing:
            raise ValueError(f"hypothesis missing required fields (protocol 제3조): {sorted(missing)}")
        self._insert("hypotheses", {"hypothesis_id": hypothesis_id, "created_at": _now(), "body": canonical_json(body)})

    def register_policy(self, policy: dict) -> str:
        h = stable_hash(policy)
        existing = self.conn.execute("SELECT policy_hash FROM policies WHERE policy_hash=?", (h,)).fetchone()
        if existing:
            return h
        self._insert("policies", {"policy_hash": h, "policy_id": policy["policy_id"], "version": int(policy["version"]),
                                  "created_at": _now(), "body": canonical_json(policy)})
        return h

    def get_policy(self, policy_hash: str) -> dict:
        row = self.conn.execute("SELECT body FROM policies WHERE policy_hash=?", (policy_hash,)).fetchone()
        if row is None:
            raise KeyError(policy_hash)
        return json.loads(row["body"])

    def register_dataset(self, dataset_version: str, manifest: dict) -> None:
        if self.conn.execute("SELECT 1 FROM datasets WHERE dataset_version=?", (dataset_version,)).fetchone():
            return
        self._insert("datasets", {"dataset_version": dataset_version, "created_at": _now(),
                                  "manifest": canonical_json(manifest)})

    # --- experiments ----------------------------------------------------
    def record_experiment(self, *, kind: str, strategy_id: str, strategy_version: str, dataset_version: str,
                          config: dict, metrics: dict, code_commit: str, hypothesis_id: str | None = None,
                          validation_policy_version: str | None = None, random_seed: int | None = None,
                          failure_reason: str | None = None, supersedes: str | None = None) -> str:
        ts = _now()
        exp_id = f"EXP-{ts[:10].replace('-', '')}-{uuid.uuid4().hex[:10]}"
        self._insert("experiments", {
            "experiment_id": exp_id, "created_at": ts, "kind": kind, "hypothesis_id": hypothesis_id,
            "strategy_id": strategy_id, "strategy_version": strategy_version,
            "dataset_version": dataset_version, "configuration_version": stable_hash(config),
            "validation_policy_version": validation_policy_version, "code_commit": code_commit,
            "random_seed": random_seed, "execution_timestamp": ts, "config": canonical_json(config),
            "metrics": canonical_json(metrics), "result_hash": stable_hash(metrics),
            "failure_reason": failure_reason, "supersedes": supersedes,
        })
        return exp_id

    def record_verdict(self, experiment_id: str, gate: str, verdict: str, reasons: list[str]) -> None:
        self._insert("verdicts", {"experiment_id": experiment_id, "gate": gate, "verdict": verdict,
                                  "reasons": canonical_json(reasons), "created_at": _now()})

    def record_state_change(self, strategy_id: str, to_state: str, reason: str, experiment_id: str | None = None) -> None:
        self._insert("state_changes", {"created_at": _now(), "strategy_id": strategy_id,
                                       "from_state": self.current_state(strategy_id), "to_state": to_state,
                                       "reason": reason, "experiment_id": experiment_id})

    def current_state(self, strategy_id: str) -> str | None:
        row = self.conn.execute("SELECT to_state FROM state_changes WHERE strategy_id=? ORDER BY change_id DESC LIMIT 1",
                                (strategy_id,)).fetchone()
        return row["to_state"] if row else None

    # --- holdout ledger -------------------------------------------------
    def holdout_accesses(self, strategy_family: str, policy_hash: str) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM holdout_ledger WHERE strategy_family=? AND policy_hash=?",
                                (strategy_family, policy_hash)).fetchone()
        return int(row["n"])

    def log_holdout_access(self, strategy_family: str, strategy_config_hash: str, policy_hash: str, experiment_id: str) -> None:
        self._insert("holdout_ledger", {"created_at": _now(), "strategy_family": strategy_family,
                                        "strategy_config_hash": strategy_config_hash, "policy_hash": policy_hash,
                                        "experiment_id": experiment_id})

    # --- queries ----------------------------------------------------------
    def experiment(self, experiment_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
        if row is None:
            raise KeyError(experiment_id)
        d = dict(row)
        d["config"], d["metrics"] = json.loads(d["config"]), json.loads(d["metrics"])
        return d

    def verdicts(self, experiment_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM verdicts WHERE experiment_id=? ORDER BY verdict_id", (experiment_id,))
        return [dict(r) | {"reasons": json.loads(r["reasons"])} for r in rows]

    def count_trials(self, strategy_id: str) -> int:
        """Experiments run for a strategy family (feeds the DSR trial count)."""
        row = self.conn.execute("SELECT COUNT(*) AS n FROM experiments WHERE strategy_id=?", (strategy_id,)).fetchone()
        return int(row["n"])

    def history(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT e.experiment_id, e.created_at, e.kind, e.strategy_id, e.strategy_version, e.failure_reason, "
            "(SELECT verdict FROM verdicts v WHERE v.experiment_id=e.experiment_id AND v.gate='OVERALL' "
            " ORDER BY verdict_id DESC LIMIT 1) AS overall FROM experiments e ORDER BY e.created_at")
        return [dict(r) for r in rows]
