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
    assert A.first_failure(good, auto, beat=0.07, t_req=4.4) is None
    assert A.first_failure(dict(good, n_validation=10), auto, 0.07, 4.4) == "trades"
    assert A.first_failure(dict(good, gross=0.011), auto, 0.07, 4.4) == "gross"
    assert A.first_failure(dict(good, ret_validation=-0.01), auto, 0.07, 4.4) == "splits"
    assert A.first_failure(dict(good, cagr_2x=None), auto, 0.07, 4.4) == "cost2x"
    assert A.first_failure(good, auto, beat=0.10, t_req=4.4) == "spy"
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
        assert out["beat_metric"] == "sharpe" and out["beat_value"] == auto.beat_value("AUTO-LEV", "sharpe")
        assert next(f["label"] for f in out["funnel"] if f["code"] == "spy") == A.SPY_SHARPE_LABEL

        # the same combinations with idle equity parked in SPY
        park = auto.run("AUTO-LEVP", workers=1, limit=4)
        assert park["done"] >= 4 and park["failed"] == 0 and park["beat_metric"] == "cagr"
        assert park["family_trials"] == 4 + 342_000  # AUTO-LEV's grid counts as prior trials
        _, lp = auto.loaded("AUTO-LEVP")
        _, pp = lab.policy_for(lp)
        mp = A.trial_metrics(lp.factory, dict(base, filter="none"), dev_data(data, pp), pp)
        assert mp["n"] == both["n"] and mp["gross"] == pytest.approx(both["gross"])  # SPY is not a trade
        assert mp["cagr"] != pytest.approx(both["cagr"])
        # one final evaluation across the two groups: they share the holdout
        monkey = [{"strategy_family": "AUTO-LEV"}]
        lab.registry.holdout_ledger = lambda: monkey
        assert auto.holdout_reads("AUTO-LEVP") == 1
        with pytest.raises(RuntimeError):
            auto.final_evaluation("AUTO-LEVP", dict(base, filter="none"))
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
    params = {k: {"default": v[0], "search": v} for k, v in two.items()}
    assert combo.validate_params(params, 2, ["EDC"])  # the SPY market filters are listed but SPY is missing


def test_trial_states_read_incrementally_match_a_fresh_read(lab):
    before = A.AutoSearch(lab).trials(GROUP)
    lab.auto.run(GROUP, workers=1, limit=3)
    cached = A.AutoSearch(lab).trials(GROUP)
    assert len(cached) == len(before) + 3
    A._STATE.clear()
    fresh = A.AutoSearch(lab).trials(GROUP)
    strip = lambda d: {k: {f: v[f] for f in ("id", "params", "status", "metrics", "detail")} for k, v in d.items()}
    assert strip(cached) == strip(fresh)


def test_parked_equity_sits_in_spy_and_is_not_a_trade():
    from quantlab.backtesting.engine import BacktestConfig, run_backtest
    from quantlab.analysis.trades import episodes
    from quantlab.strategies.entry_exit import EntryExitStrategy
    from synthetic import random_market

    d = random_market(4, n=700, symbols=("A", "B", "SPY"), split_prob=0.005)
    p = dict(entry="rsi2_10", filter="none", exit="above5", max_hold=10, stop=0, take=0)
    runs = {}
    for park in (None, "SPY"):
        c = {"catalog": 2, "trade": ["A", "B"], "slots": 4, **({"park": park} if park else {})}
        st = EntryExitStrategy(spec_id="X", spec_version_str="1", symbols=["A", "B", "SPY"],
                               rules=combo.build_rules(c, p), fast=True)
        runs[park] = run_backtest(st, d, BacktestConfig())
    plain, parked = runs[None], runs["SPY"]
    assert (plain.cash / plain.equity).mean() > 0.5
    # from the first fill on, idle equity is in SPY; never borrowed
    assert (parked.cash / parked.equity).iloc[1:].max() < 0.02 and parked.cash.min() > -1e-6
    ep_plain, ep_parked = episodes(plain, d), episodes(parked, d)
    for sym in ("A", "B"):  # the rule's own trades are the same days
        assert list(ep_plain[ep_plain.symbol == sym]["entry"]) == list(ep_parked[ep_parked.symbol == sym]["entry"])
    held = parked.shares["SPY"] > 0
    assert held.iloc[1:].all()  # bought on the first session and never sold out completely
    assert combo.validate_combo({"catalog": 2, "trade": ["A"], "slots": 4, "park": "A"}, ["A", "SPY"])


def test_beat_metric_sharpe():
    auto = {"min_trades_train": 40, "min_trades_validation": 20, "min_gross_per_trade": 0.012,
            "max_drawdown_floor": -0.35, "beat_metric": "sharpe"}
    m = dict(n_train=50, n_validation=30, gross=0.02, ret_train=0.1, ret_validation=0.05, cagr_2x=0.03,
             cagr=0.05, sharpe=0.9, mdd=-0.2, net_ex_top10=0.004, cagr_ex_best_year=0.05, t=6.0)
    assert A.first_failure(m, auto, beat=0.8, t_req=5.0) is None  # CAGR below SPY's is fine here
    assert A.first_failure(m, auto, beat=0.95, t_req=5.0) == "spy"
    assert A.first_failure(dict(m, sharpe=None), auto, beat=0.5, t_req=5.0) == "spy"


def test_group_loads_after_its_spec_got_a_new_version(tmp_path):
    """A PC that registered AUTO-LEV@1.0.0 (build-24) loads 1.1.0 and keeps counting the family's trials."""
    from quantlab.services.lab import Lab

    lab = Lab(tmp_path / "QL")
    try:
        cur = A.AutoSearch(lab).spec_path("AUTO-LEV").read_text(encoding="utf-8")
        old = tmp_path / "AUTO-LEV.qls.yaml"
        old.write_text(cur.replace("version: 1.1.0", "version: 1.0.0", 1).replace("auto_lev_v2", "auto_lev_v1"),
                       encoding="utf-8")
        imp = lab.import_strategy(old)
        assert imp.loaded.version == "1.0.0"
        assert imp.strategy_hash == "d4d7896e1ae9d0859f117cfac031cc63f0548c83e55937f3320dde2516609c64"
        lab.registry.create_batch("A-old", imp.strategy_hash, "v", {"auto": True})
        lab.registry.register_trials("A-old", [(0, {"entry": "z20_1.5", "filter": "none", "exit": "none",
                                                    "max_hold": 5, "stop": 0, "take": 0})])
        auto = A.AutoSearch(lab)
        strategy_hash, loaded = auto.loaded("AUTO-LEV")
        assert loaded.version == "1.1.0" and strategy_hash != imp.strategy_hash
        assert auto.n_trials("AUTO-LEV", lab.policy_for(loaded)[1]) == 1 + 342_000
        assert len(auto.trials("AUTO-LEV")) == 1
    finally:
        lab.close()


def test_near_misses_are_listed_best_first(lab, monkeypatch):
    """Trials past the four cheap filters that fail at SPY or later are kept for review, best first."""
    lab.auto.run(GROUP, workers=1, limit=6)
    monkeypatch.setattr(A, "_STATE", {})  # judge the stored trials afresh with the stubbed filter
    early = {}

    def judged(m, auto, beat, t_req):
        k = id(m)
        early.setdefault(k, ["trades", "spy", "mdd"][len(early) % 3])
        return early[k]

    monkeypatch.setattr(A, "first_failure", judged)
    out = lab.auto.summary(GROUP)
    near = out["near"]
    by_fail = {f["code"]: f["count"] for f in out["funnel"]}
    assert len(near) == by_fail["spy"] + by_fail["mdd"] > 0 and by_fail["trades"] > 0
    assert all(r["fail"] in ("spy", "mdd") and r["rule"] and r["params"] for r in near)
    assert {r["status"] for r in near} <= {A.FILTER_LABEL["spy"], A.FILTER_LABEL["mdd"]}
    vals = [r["cagr"] for r in near]
    assert vals == sorted(vals, reverse=True)


def test_preregistered_near_miss_comes_first(lab, monkeypatch):
    lab.auto.run(GROUP, workers=1, limit=6)
    monkeypatch.setattr(A, "_STATE", {})
    monkeypatch.setattr(A, "first_failure", lambda m, auto, beat, t_req: "tvalue")
    params = list(lab.auto.trials(GROUP).values())[-1]["params"]
    monkeypatch.setattr(A, "PREREGISTERED", {GROUP: params})
    out = lab.auto.summary(GROUP)
    assert out["near"][0]["params"] == params and out["near"][0]["preregistered"]
    assert out["near"][0]["status"] == A.PREREGISTERED_LABEL
    assert sum(r["preregistered"] for r in out["near"]) == 1
    assert out["holdout_start"].startswith("20")


def test_levp_preregistered_rule_is_in_its_catalog():
    p = A.PREREGISTERED["AUTO-LEVP"]
    space = combo.full_space(2)
    assert set(p) == set(space) and all(p[k] in space[k] for k in space)
