"""Registry immutability and bookkeeping (M6)."""

import sqlite3

import pytest

from quantlab.registry.store import ImmutableRecordError, Registry

HYP = {"strategy_id": "S", "market": "m", "economic_rationale": "r", "rules": "x", "benchmark": "b",
       "risks": "k", "falsification": "f", "validation_plan": "p"}


@pytest.fixture
def reg(tmp_path):
    r = Registry(tmp_path / "reg.sqlite")
    yield r
    r.close()


def _exp(reg, **kw):
    args = dict(kind="formal", strategy_id="S", strategy_version="1", dataset_version="d1",
                config={"a": 1}, metrics={"sharpe": 0.5})
    args.update(kw)
    return reg.record_experiment(**args)


def test_experiments_and_verdicts_cannot_be_changed_or_deleted(reg):
    eid = _exp(reg)
    reg.record_verdict(eid, "OVERALL", "FAIL", ["x"])
    for sql in ("UPDATE experiments SET metrics='{}'", "DELETE FROM experiments",
                "UPDATE verdicts SET verdict='PASS'", "DELETE FROM verdicts"):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            with reg.conn:
                reg.conn.execute(sql)
    assert reg.verdicts(eid)[0]["verdict"] == "FAIL"


def test_policy_versions_are_immutable(reg):
    p1 = {"policy_id": "pol", "version": 1, "threshold": 0.1}
    h1 = reg.register_policy(p1)
    assert reg.register_policy(dict(p1)) == h1  # idempotent for identical content
    with pytest.raises(ImmutableRecordError):
        reg.register_policy({"policy_id": "pol", "version": 1, "threshold": 0.2})  # same version, new content
    h2 = reg.register_policy({"policy_id": "pol", "version": 2, "threshold": 0.2})
    assert h2 != h1 and reg.get_policy(h1)["threshold"] == 0.1


def test_hypothesis_requires_preregistration_fields(reg):
    with pytest.raises(ValueError):
        reg.register_hypothesis("H-1", {"strategy_id": "S"})
    reg.register_hypothesis("H-1", HYP)
    with pytest.raises(ImmutableRecordError):
        reg.register_hypothesis("H-1", HYP)


def test_invalid_experiment_is_superseded_not_overwritten(reg):
    bad = _exp(reg, failure_reason="look-ahead found")
    reg.record_verdict(bad, "OVERALL", "INVALID", ["leak"])
    good = _exp(reg, supersedes=bad)
    assert reg.experiment(good)["supersedes"] == bad
    assert reg.experiment(bad)["failure_reason"] == "look-ahead found"
    assert [h["overall"] for h in reg.history()] == ["INVALID", None]


def test_holdout_ledger_counts_and_state_history(reg):
    assert reg.holdout_accesses("S", "p") == 0
    reg.log_holdout_access("S", "cfg", "p", "E1")
    assert reg.holdout_accesses("S", "p") == 1
    reg.record_state_change("S", "IDEA", "registered")
    reg.record_state_change("S", "RESEARCH", "started")
    assert reg.current_state("S") == "RESEARCH"
    assert reg.count_trials("S") == 0
    _exp(reg)
    assert reg.count_trials("S") == 1


def test_app_identity_recorded(reg):
    from quantlab.core import app_info
    eid = _exp(reg, strategy_hash="h1", spec_version=1, strategy_origin="claude", batch_id=None)
    e = reg.experiment(eid)
    assert e["app_version"] == app_info.app_version()
    assert e["app_code_hash"] == app_info.app_code_hash() and len(e["app_code_hash"]) == 64
    assert (e["strategy_hash"], e["spec_version"], e["strategy_origin"]) == ("h1", 1, "claude")


def test_strategies_lineage_and_holdout_sharing(reg):
    reg.register_strategy(strategy_hash="a", strategy_id="S", version="1", origin="user", body={})
    reg.register_strategy(strategy_hash="b", strategy_id="S2", version="1", origin="claude", body={},
                          parent="S@1", derived_after_holdout_view=True)
    assert not reg.register_strategy(strategy_hash="a", strategy_id="S", version="1", origin="user", body={})
    assert [r["strategy_id"] for r in reg.lineage("S2", "1")] == ["S2", "S"]
    assert not reg.holdout_shared("S")
    reg.record_export("S", True, "exports/x.zip", ["report.md"])
    assert reg.holdout_shared("S")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        with reg.conn:
            reg.conn.execute("DELETE FROM exports")


def test_trial_events_and_family_trial_count(reg):
    reg.register_strategy(strategy_hash="a", strategy_id="S", version="1", origin="user", body={})
    reg.create_batch("B1", "a", "d1", {"grid": [1, 2, 3]})
    for i in range(3):
        reg.trial_event("B1", i, "registered", {"p": i})
    reg.trial_event("B1", 0, "done", experiment_id="E0")
    reg.trial_event("B1", 1, "failed", detail="boom")
    assert reg.trial_status("B1") == {0: "done", 1: "failed", 2: "registered"}
    assert reg.count_trials_for_family("S") == 3


def test_workspace_layout(tmp_path):
    from quantlab.core.workspace import SUBDIRS, Workspace
    ws = Workspace.open(tmp_path / "QL")
    assert all((ws.root / s).is_dir() for s in SUBDIRS)
    assert ws.registry_path.name == "registry.sqlite"
