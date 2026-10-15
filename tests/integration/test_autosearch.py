"""Automatic strategy search (D-020): pre-registration, holdout isolation, resume, fixed order,
trial counting and the filters."""

import itertools

import numpy as np
import pandas as pd
import pytest

from quantlab.data.quality import exchange_sessions
from quantlab.services import autosearch as A
from quantlab.services.lab import Lab
from quantlab.strategies import combo
from synthetic import frame

GROUP = "AUTO-REAL"
SYMBOLS = ["TLT", "IEF", "LQD", "GLD", "SLV", "DBC", "VNQ", "SPY", "BIL"]


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("auto")
    lab = Lab(tmp / "QL")
    sessions = exchange_sessions("2007-06-01", "2020-12-31")
    rng = np.random.default_rng(5)
    for k, sym in enumerate(SYMBOLS):
        # SPY falls with large swings so the final-evaluation test gets past Gate 2 (lower drawdown)
        drift, vol = {"BIL": (0.00005, 0.0005), "SPY": (-0.0004, 0.02)}.get(sym, (0.0003, 0.012))
        close = 50 * np.exp(np.cumsum(rng.normal(drift, vol, len(sessions))))
        opn = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, vol / 4, len(sessions)))
        df = frame(opn, close)
        df.index = sessions
        p = tmp / f"{sym}.csv"
        df.rename_axis("date").reset_index().to_csv(p, index=False, date_format="%Y-%m-%d")
        lab.import_csv(sym, p)
    lab.auto = lab.auto_search()
    yield lab
    lab.close()


def test_run_registers_everything_first_and_never_reads_holdout(lab, monkeypatch):
    seen = {}
    real = A.trial_metrics

    def spy(factory, params, dev, policy):
        seen["last"] = max(seen.get("last", dev.calendar[-1]), dev.calendar[-1])
        return real(factory, params, dev, policy)

    monkeypatch.setattr(A, "trial_metrics", spy)
    registered_before_first_run = {}
    real_run = A._run_trial

    def watch(task):
        if not registered_before_first_run:
            registered_before_first_run["n"] = lab.registry.count_trials_for_family(GROUP)
        return real_run(task)

    monkeypatch.setattr(A, "_run_trial", watch)
    out = lab.auto.run(GROUP, workers=1, limit=8)
    assert registered_before_first_run["n"] >= 8  # all planned trials registered before the first ran
    assert out["done"] >= 8 and out["failed"] == 0 and out["pending"] == 0
    assert seen["last"] <= pd.Timestamp("2016-12-31")
    assert lab.registry.holdout_ledger() == []
    assert lab.registry.confirmation(out["strategy_hash"]) is None  # nothing adopted automatically
    assert out["family_trials"] == lab.registry.count_trials_for_family(GROUP) == out["tried"]
    assert out["beat_cagr"] is not None and sum(f["count"] for f in out["funnel"]) + len(out["candidates"]) == out["done"]


def test_fixed_order_and_no_repeats(lab):
    _, loaded = lab.auto.loaded(GROUP)
    _, policy = lab.policy_for(loaded)
    order = lab.auto.order(loaded, policy)
    assert len(order) == 34 * 6 * 7 * 4 * 2 * 2
    before = set(lab.auto.trials(GROUP))
    lab.auto.run(GROUP, workers=1, limit=4)
    after = lab.auto.trials(GROUP)
    new = [k for k in after if k not in before]
    planned = [A.key(p) for p in order if A.key(p) not in before][:4]
    assert set(planned) <= set(new)  # the next untried combinations, in the pre-registered order


def test_stop_then_resume_finishes_the_same_trials(lab):
    calls = {"n": 0}

    def stop_after_two():
        calls["n"] += 1
        return calls["n"] > 2

    first = lab.auto.run(GROUP, workers=1, limit=6, should_stop=stop_after_two)
    assert first["stopped"] and first["pending"] == 4
    left = {A.key(v["params"]): v["id"] for v in lab.auto.trials(GROUP).values() if v["status"] == "cancelled"}
    lab.auto.run(GROUP, workers=1, limit=1)  # resumes the interrupted trials; no new combinations first
    now = lab.auto.trials(GROUP)
    assert all(now[k]["status"] == "done" and now[k]["id"] == tid for k, tid in left.items())
    keys = [A.key(v["params"]) for v in lab.auto.trials(GROUP).values()]
    assert len(keys) == len(set(keys))


def test_parallel_workers_give_the_same_numbers(lab):
    params = dict(entry="z20_2.0", filter="none", exit="above5", max_hold=10, stop=0, take=0)
    strategy_hash, loaded = lab.auto.loaded(GROUP)
    _, policy = lab.policy_for(loaded)
    version = lab.ensure_dataset(loaded.spec["universe"]["symbols"], policy["splits"]["train"][0])
    data, _ = lab._strategy_symbols_data(loaded, version)
    from quantlab.validation.engine import dev_data

    dev = dev_data(data, policy)
    a = A.trial_metrics(loaded.factory, {**loaded.default_params, **params}, dev, policy)
    b = A.trial_metrics(loaded.factory, {**loaded.default_params, **params}, dev, policy)
    assert a == b


def test_final_evaluation_reads_group_holdout_once_and_second_cannot_pass(lab):
    p1 = dict(entry="z20_2.0", filter="none", exit="above5", max_hold=10, stop=0, take=0)
    p2 = dict(p1, entry="z20_2.5")
    out1 = lab.auto.final_evaluation(GROUP, p1)
    if out1.overall == "INVALID" or not any(g.gate == "GATE3" for g in out1.gates):
        pytest.skip(f"synthetic data stopped before the holdout: {out1.overall} {[(g.gate, g.verdict, g.reasons) for g in out1.gates]}")
    assert len([r for r in lab.registry.holdout_ledger() if r["strategy_family"] == GROUP]) == 1
    out2 = lab.auto.final_evaluation(GROUP, p2)
    g3 = [g for g in out2.gates if g.gate == "GATE3"]
    assert not g3 or g3[0].verdict != "PASS"
    assert out2.overall != "PASS"


def test_filters_in_order_and_required_t():
    auto = {"min_trades_train": 40, "min_trades_validation": 20, "min_gross_per_trade": 0.012,
            "max_drawdown_floor": -0.35}
    good = dict(n_train=50, n_validation=30, gross=0.02, ret_train=0.1, ret_validation=0.05, cagr_2x=0.03,
                cagr=0.09, mdd=-0.2, net_ex_top10=0.004, cagr_ex_best_year=0.05, t=5.0)
    assert A.first_failure(good, auto, beat_cagr=0.07, t_req=4.4) is None
    assert A.first_failure(dict(good, n_validation=10), auto, 0.07, 4.4) == "trades"
    assert A.first_failure(dict(good, gross=0.011), auto, 0.07, 4.4) == "gross"
    assert A.first_failure(dict(good, ret_validation=-0.01), auto, 0.07, 4.4) == "splits"
    assert A.first_failure(dict(good, cagr_2x=None), auto, 0.07, 4.4) == "cost2x"
    assert A.first_failure(good, auto, beat_cagr=0.10, t_req=4.4) == "spy"
    assert A.first_failure(dict(good, mdd=-0.4), auto, 0.07, 4.4) == "mdd"
    assert A.first_failure(dict(good, net_ex_top10=-0.001), auto, 0.07, 4.4) == "top10"
    assert A.first_failure(dict(good, cagr_ex_best_year=-0.01), auto, 0.07, 4.4) == "bestyear"
    assert A.first_failure(good, auto, 0.07, t_req=5.1) == "tvalue"
    assert A.required_t(1, 0.05) == pytest.approx(1.645, abs=1e-3)
    assert A.required_t(100, 0.05) == pytest.approx(3.29, abs=1e-2)
    assert A.required_t(10_000, 0.05) == pytest.approx(4.42, abs=1e-2)


def test_neighbours_are_one_step_away():
    p = dict(entry="z20_2.0", filter="up200", exit="above5", max_hold=10, stop=0, take=0)
    nbs = combo.neighbours(p)
    assert {n["entry"] for n in nbs if n["entry"] != "z20_2.0"} == {"z20_1.5", "z20_2.5"}
    assert {n["max_hold"] for n in nbs if n["max_hold"] != 10} == {5, 20}
    assert len(combo.neighbours(dict(p, entry="low20", max_hold=5))) == 1


def _loaded_policy(lab, auto):
    _, loaded = auto.loaded("AUTO-LEV")
    return loaded, lab.policy_for(loaded)[1]


def test_late_listed_leveraged_group_runs(tmp_path):
    """AUTO-LEV: the ETFs list between 2008 and 2014; rows before a listing are empty, not an error."""
    lab = Lab(tmp_path / "QL")
    try:
        rng = np.random.default_rng(9)
        starts = {"EDC": "2008-12-17", "YINN": "2009-12-03", "KORU": "2013-04-11", "EURL": "2014-01-23",
                  "SPY": "2007-06-01", "BIL": "2007-06-01"}
        for sym, start in starts.items():
            sessions = exchange_sessions(start, "2022-12-30")
            drift, vol = {"BIL": (0.00005, 0.0005), "SPY": (0.0003, 0.011)}.get(sym, (0.0, 0.035))
            close = 30 * np.exp(np.cumsum(rng.normal(drift, vol, len(sessions))))
            opn = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, vol / 4, len(sessions)))
            df = frame(opn, close)
            df.index = sessions
            p = tmp_path / f"{sym}.csv"
            df.rename_axis("date").reset_index().to_csv(p, index=False, date_format="%Y-%m-%d")
            lab.import_csv(sym, p)
        auto = lab.auto_search()
        out = auto.run("AUTO-LEV", workers=1, limit=6)
        assert out["done"] >= 6 and out["failed"] == 0 and out["pending"] == 0
        assert lab.registry.holdout_ledger() == []
        assert sum(f["count"] for f in out["funnel"]) + len(out["candidates"]) == out["done"]
        assert len(auto.order(*_loaded_policy(lab, auto))) == 342_000
        _, loaded = auto.loaded("AUTO-LEV")
        _, policy = lab.policy_for(loaded)
        version = lab.ensure_dataset(loaded.spec["universe"]["symbols"], policy["splits"]["train"][0])
        data, _ = lab._strategy_symbols_data(loaded, version)
        from quantlab.validation.engine import dev_data

        dev = dev_data(data, policy)
        base = dict(entry="rsi2_10", exit="above5", max_hold=10, stop=0, take=0)
        up = A.trial_metrics(loaded.factory, dict(base, filter="spy_up200"), dev, policy)
        down = A.trial_metrics(loaded.factory, dict(base, filter="spy_down200"), dev, policy)
        both = A.trial_metrics(loaded.factory, dict(base, filter="none"), dev, policy)
        assert up["n"] > 0 and down["n"] > 0 and up["n"] + down["n"] >= both["n"] * 0.8
    finally:
        lab.close()


def test_catalog_2_extends_catalog_1_without_changing_it():
    one, two = combo.full_space(1), combo.full_space(2)
    assert all(set(one[k]) <= set(two[k]) for k in combo.PARAMS)
    assert [k for k in two["entry"] if k in one["entry"]] == one["entry"]  # same relative order
    assert len(list(itertools.product(*two.values()))) == 342_000
    p = dict(entry="rsi2_5", filter="spy_up200", exit="high5", max_hold=3, stop=0.2, take=0.3)
    assert {n["entry"] for n in combo.neighbours(p, 2) if n["entry"] != "rsi2_5"} == {"rsi2_2", "rsi2_10"}
    assert {n["max_hold"] for n in combo.neighbours(p, 2) if n["max_hold"] != 3} == {5}
    assert {n["entry"] for n in combo.neighbours(dict(p, max_hold=5), 1) if n["entry"] != "rsi2_5"} == {"rsi2_10"}
    rules = combo.build_rules({"catalog": 2, "trade": ["EDC"], "slots": 4}, p)["entry_exit"]["legs"][0]
    assert {"indicator": "close_vs_sma", "op": ">", "value": 0, "days": 200, "of": "SPY"} in rules["entry"]
    with pytest.raises(ValueError):
        combo.build_rules({"catalog": 1, "trade": ["EDC"], "slots": 4}, p)
    assert combo.validate_combo({"catalog": 2, "trade": ["EDC"], "slots": 4}, ["EDC"])  # SPY missing


def test_trial_states_read_incrementally_match_a_fresh_read(lab):
    before = A.AutoSearch(lab).trials(GROUP)
    lab.auto.run(GROUP, workers=1, limit=3)
    cached = A.AutoSearch(lab).trials(GROUP)
    assert len(cached) == len(before) + 3
    A._STATE.clear()
    fresh = A.AutoSearch(lab).trials(GROUP)
    strip = lambda d: {k: {f: v[f] for f in ("id", "params", "status", "metrics", "detail")} for k, v in d.items()}
    assert strip(cached) == strip(fresh)
