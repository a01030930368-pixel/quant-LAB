"""Batch experiments (M11, 03 v2.0.0 §8.1)."""

import pandas as pd
import pytest

from quantlab.services.lab import Lab, builtin_path
from quantlab.services.selftest import write_synthetic_csvs


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("batch")
    lab = Lab(tmp / "QL")
    for sym, p in write_synthetic_csvs(tmp).items():
        lab.import_csv(sym, p)
    lab.version, _ = lab.build_dataset(["SPY", "BIL"])
    lab.imp = lab.import_strategy(builtin_path("specs", "TT-SPY-SMA.qls.yaml"))
    yield lab
    lab.close()


def test_all_trials_recorded_and_counted(lab):
    out = lab.search(lab.imp.strategy_hash, lab.version, workers=1)
    assert out["total"] == 5 and out["done"] == 5 and out["failed"] == 0
    assert out["family_trials"] >= 5 and out["best_dsr"] is not None
    assert all(r["experiment_id"] for r in out["rows"])
    exps = [e for e in lab.registry.experiments() if e["batch_id"] == out["batch_id"]]
    assert sorted(e["trial_index"] for e in exps) == list(range(5))
    assert all(pd.Timestamp(e["metrics"]["window"][1]) <= pd.Timestamp("2018-12-31") for e in exps)
    assert lab.registry.holdout_ledger() == []
    assert lab.registry.confirmation(lab.imp.strategy_hash) is None  # nothing adopted automatically


def test_parallel_matches_serial(lab):
    space = {"lookback_months": [8, 12]}
    a = lab.search(lab.imp.strategy_hash, lab.version, space=space, workers=1)
    b = lab.search(lab.imp.strategy_hash, lab.version, space=space, workers=2)
    assert [r["sharpe"] for r in a["rows"]] == [r["sharpe"] for r in b["rows"]]


def test_cancel_then_resume_no_duplicates_or_gaps(lab):
    calls = {"n": 0}

    def stop_after_two():
        calls["n"] += 1
        return calls["n"] > 2

    space = {"lookback_months": [6, 7, 8, 9, 10, 11]}
    first = lab.search(lab.imp.strategy_hash, lab.version, space=space, workers=1, should_stop=stop_after_two)
    assert first["done"] == 2 and first["cancelled"] == 4
    second = lab.search(lab.imp.strategy_hash, lab.version, workers=1, resume=first["batch_id"])
    assert second["done"] == 6 and second["cancelled"] == 0
    exps = [e for e in lab.registry.experiments() if e["batch_id"] == first["batch_id"]]
    assert sorted(e["trial_index"] for e in exps) == list(range(6))


def test_workers_never_receive_holdout_dates(lab, monkeypatch):
    import quantlab.services.batch as batch

    seen = {}
    orig = batch._init_worker

    def spy(spec_path, h, policy, dev, qc):
        seen["last"] = dev.calendar.max()
        return orig(spec_path, h, policy, dev, qc)

    monkeypatch.setattr(batch, "_init_worker", spy)
    lab.search(lab.imp.strategy_hash, lab.version, space={"lookback_months": [10]}, workers=1)
    assert seen["last"] <= pd.Timestamp("2018-12-31")
