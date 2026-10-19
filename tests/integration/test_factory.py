"""D-033: the research factory — stock catalog 3, the affine search order, the factory loop and its
paper-trading picks."""

import json
import xml.etree.ElementTree as ET
from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.engine import run_backtest
from quantlab.services import autosearch as A
from quantlab.services import factory as F
from quantlab.services.batch import grid
from quantlab.services.lab import Lab, builtin_path
from quantlab.strategies import stock_combo as sc
from quantlab.strategies.spec import load_spec
from quantlab.validation.engine import leak_test
from test_stock_combo import market

SPEC = builtin_path("specs", "AUTO-SP500X.qls.yaml")
RULES3 = {"stock_combo": {"catalog": 3, "park": "SPY", "park_band": 2}}
NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}


def strat3(entry, hold=5, stop=0, take=0, slots=2):
    return sc.StockComboStrategy(spec_id="T", spec_version_str="1", rules=RULES3,
                                 entry=entry, hold=hold, stop=stop, take=take, slots=slots)


def buy_days(data, entry, **kw):
    t = run_backtest(strat3(entry, **kw), data).trades
    t = t[(t["symbol"] != "SPY") & (t["qty"] > 0)]
    return list(zip(t["date"].dt.strftime("%Y-%m-%d"), t["symbol"]))


# --- catalog 3 --------------------------------------------------------------------------------
def test_catalog_3_widens_without_changing_catalogs_1_and_2():
    old = load_spec(builtin_path("specs", "AUTO-SP500.qls.yaml"))
    assert sc.entries(1) == sc.entries(2) == list(old.search_space["entry"])  # same items, same order
    assert len(sc.entries(3)) == 3337
    space = sc.full_space(3)
    assert np.prod([len(v) for v in space.values()]) == 540594
    loaded = load_spec(SPEC)
    assert {k: list(v) for k, v in loaded.search_space.items()} == space
    # AUTO-SP500 ran exactly the catalog-1 part of this grid: those are skipped, not run again
    skipped = sum(1 for p in grid({k: v for k, v in space.items()}) if sc.already_tried(p, 3))
    assert skipped == 41850
    assert not sc.already_tried({"entry": "earn5", "hold": 3, "stop": 0, "take": 0, "slots": 10}, 3)
    assert not sc.already_tried({"entry": "earn5", "hold": 20, "stop": 0, "take": 0, "slots": 5}, 3)
    assert not sc.already_tried({"entry": "earn5", "hold": 20, "stop": 0, "take": 0, "slots": 10}, 1)
    assert sc.validate_params(loaded.spec["parameters"], 3) == []
    assert sc.validate_params(loaded.spec["parameters"], 1)  # catalog-3 items are not catalog 1


def test_new_conditions_fire_on_their_session():
    n = 60
    x = np.full(n, 100.0)
    x[20] = 94.0                  # one-day drop of 6%
    x[21:] = 94.0
    xo = x.copy()
    xo[40] = 90.0                 # opens 4.3% below the previous close
    y = 100 - np.r_[np.zeros(30), np.arange(1, 7), np.full(n - 36, 6)].astype(float)  # 6 down closes in a row
    data = market({"X": x, "Y": y}, opens={"X": xo, "Y": y}, n=n)
    idx = data.calendar
    d = lambda i: str(idx[i].date())  # noqa: E731
    assert buy_days(data, "drop1_4") == [(d(21), "X")]
    assert buy_days(data, "drop1_7") == []
    # the drop on session 20 is also a gap (the synthetic open is the close); session 40 gaps down 4.3%
    assert buy_days(data, "gapdn3") == [(d(21), "X"), (d(41), "X")]
    assert buy_days(data, "gapdn5") == [(d(21), "X")]
    assert buy_days(data, "dn3") == [(d(33), "Y")]
    assert buy_days(data, "dn5") == [(d(35), "Y")]


def test_negative_earnings_and_one_large_insider_purchase():
    n = 40
    x = np.full(n, 100.0)
    x[10:] = 90.0                 # -10% over SPY on the first close after the filing (01-15)
    ev = [("2020-01-14", "X", "earnings", None, None, None, None),
          ("2020-01-20", "Y", "insider_buy", "a", 100_000, True, False),
          ("2020-02-03", "Z", "insider_buy", "b", 300_000, False, True)]
    data = market({"X": x}, ev, n=n)
    assert buy_days(data, "earnneg8") == [("2020-01-16", "X")]
    assert buy_days(data, "earnneg5") == [("2020-01-16", "X")]
    assert buy_days(data, "earn3") == []
    assert buy_days(data, "insbig") == [("2020-02-04", "Z")]
    assert {s for _, s in buy_days(data, "ins1")} == {"Y", "Z"}


def test_catalog_3_panels_use_only_the_past():
    rng = np.random.default_rng(4)
    n = 420
    closes = {s: 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n))) for s in ("SPY", "X", "Y", "Z")}
    idx = pd.bdate_range("2020-01-01", periods=n)
    ev = [(str(idx[i].date()), s, "earnings", None, None, None, None) for i in range(10, n, 30) for s in ("X", "Y")]
    ev += [(str(idx[i].date()), "Z", "insider_buy", f"o{i}", 400000, True, False) for i in range(5, n, 17)]
    data = market(closes, ev, n=n)
    full = sc.build_panels(data)
    cut = 300
    part = sc.build_panels(data.slice(None, idx[cut]))
    for k, v in full.items():
        if k != "symbols":
            assert np.array_equal(v[: cut + 1], part[k], equal_nan=True), k
    for entry in ("earnneg5+low252", "insbig+spy_down200", "gapdn3+dn3", "drop20_15+squeeze", "drop1_4+spy_drop5"):
        assert leak_test(lambda **p: strat3(**p), {"entry": entry, "hold": 3, "stop": 0.1, "take": 0.2, "slots": 5},
                         data, 4, 7) == [], entry


# --- the affine order -------------------------------------------------------------------------
def test_affine_order_visits_every_combination_once_in_a_fixed_order():
    space = {"entry": ["earn5", "dn3", "low252", "squeeze"], "hold": [3, 5, 10], "stop": [0, 0.1], "take": [0],
             "slots": [5]}
    g = A.LazyGrid(space, 7)
    walk = [p for _, p in g]
    assert len(walk) == 24 and sorted(map(A.canonical_json, walk)) == sorted(map(A.canonical_json, grid(space)))
    assert walk == [p for _, p in A.LazyGrid(space, 7)]
    skip = A.LazyGrid(space, 7, skip=lambda p: p["stop"] == 0.1)
    assert skip.size() == 12 and all(p["stop"] == 0 for _, p in skip)


# --- the factory --------------------------------------------------------------------------------
def test_factory_text_and_report():
    p = {"enabled": True, "running": True, "finished": False, "done": 3000, "grid": 498744, "candidates": 0,
         "watch": 1, "error": None}
    from quantlab.gui.autopage import factory_text

    assert "498,744개 중 3,000개 시험 (0.6%)" in factory_text(p) and "돌고 있어요" in factory_text(p)
    assert "꺼져" in factory_text(p | {"enabled": False, "running": False})
    assert "모두 시험" in factory_text(p | {"finished": True})
    assert "충전기" in factory_text(p | {"running": False, "battery": True})
    rep = {"kind": "finished", "at": "2026-11-01 09:00", "grid": 498744, "done": 498744, "family_trials": 540597,
           "t_required": 5.3, "candidates": 0, "funnel": [("거래 수 부족", 10), ("이웃 조합이 버티지 못함", 0)],
           "best_near": [{"rule": "규칙", "t": 4.1, "cagr": 0.15, "mdd": -0.3}], "watch": 5, "began": "2026-10-18 09:00",
           "hours": 300.0}
    text = F.report_text(rep)
    assert "498,744개" in text and "거래 수 부족 10" in text and "이웃" not in text and "t 4.10" in text
    cand = {"kind": "candidate", "at": "x", "id": "W01", "rule": "규칙", "t": 5.5, "cagr": 0.2, "mdd": -0.2,
            "start": "2026-10-20"}
    assert "2026-10-20부터" in F.report_text(cand)


def test_factory_task_starts_at_sign_in_and_hourly_with_no_time_limit():
    from quantlab.services import scheduler

    xml = scheduler.factory_xml(r"C:\Q & L\QuantLab.exe", start=date(2026, 10, 18))
    root = ET.fromstring(xml.split("\n", 1)[1])
    assert root.find(".//t:LogonTrigger", NS) is not None
    assert root.find(".//t:Repetition/t:Interval", NS).text == "PT1H"
    assert root.find(".//t:ExecutionTimeLimit", NS).text == "PT0S"
    assert root.find(".//t:MultipleInstancesPolicy", NS).text == "IgnoreNew"
    assert root.find(".//t:Exec/t:Arguments", NS).text == "--factory"
    # charger only: never started on battery, started again when the power source changes
    assert root.find(".//t:DisallowStartIfOnBatteries", NS).text == "true"
    assert "EventID=105" in root.find(".//t:EventTrigger/t:Subscription", NS).text
    assert root.find(".//t:Exec/t:Command", NS).text == r"C:\Q & L\QuantLab.exe"


def test_factory_watch_list_news_and_switch(tmp_path, monkeypatch):
    lab = Lab(tmp_path / "QL")
    try:
        f = lab.factory()
        assert not f.enabled() and f.progress()["done"] == 0
        f.set_enabled(True)
        assert f.enabled()
        row = lambda e, t: {"params": {"entry": e, "hold": 3, "stop": 0, "take": 0, "slots": 5},  # noqa: E731
                            "rule": e, "t": t, "cagr": 0.2, "sharpe": 1.0, "mdd": -0.2, "n": 100}
        out = {"candidates": [row("earnneg5", 5.6) | {"status": "후보"}, row("dn3", 5.7) | {"status": "이웃 확인 대기"}],
               "near": [row("gapdn3", 4.0) | {"fail": "tvalue"}, row("low252", 4.5) | {"fail": "tvalue"},
                        row("squeeze", 9.0) | {"fail": "mdd"}]}
        f._update_watch(out, today=date(2026, 10, 18))
        w = f.watch()
        assert [(x["id"], x["kind"], x["rule"]) for x in w] == [("W01", "candidate", "earnneg5"), ("W02", "near", "low252"),
                                                               ("W03", "near", "gapdn3")]
        assert all(x["start"] == "2026-10-19" for x in w)
        f._update_watch(out, today=date(2026, 10, 19))  # nothing is added twice
        assert [x["id"] for x in f.watch()] == ["W01", "W02", "W03"] and f.watch()[2]["start"] == "2026-10-19"
        news = f.unseen()
        assert [n["kind"] for n in news] == ["candidate"] and news[0]["id"] == "W01"
        f.mark_seen()
        assert f.unseen() == []
        # paper trading lists the picks with their own parameters, start and folder
        rows = [r for r in lab.paper().candidates() if r.get("watch")]
        assert [r["id"] for r in rows] == ["AUTO-SP500X-W01", "AUTO-SP500X-W02", "AUTO-SP500X-W03"]
        assert rows[0]["params"]["entry"] == "earnneg5" and rows[0]["start"] == "2026-10-19"
        seen = {}
        monkeypatch.setattr(lab.paper().__class__, "refresh",
                            lambda self, path, say=None, today=None, **kw: seen.update(kw) or {"asof": "x", "orders": []})
        lab.paper().run_auto(today="20261020")  # the factory's picks are refreshed without a switch
        assert seen["paper_id"] in {r["id"] for r in rows}
        # a second factory process does not start while one runs; a killed one's lock goes stale
        with f._lock():
            with pytest.raises(F.FactoryBusy):
                with f._lock():
                    pass
            assert f.running()
        assert not f.running()
        f.record_error(ValueError("bad\nline"))
        assert f.progress()["error"].endswith("ValueError: bad line")
    finally:
        lab.close()


def test_factory_stops_when_switched_off(tmp_path, monkeypatch):
    lab = Lab(tmp_path / "QL")
    try:
        f = lab.factory()
        calls = []

        class FakeAuto:
            def loaded(self, group):
                return "h", None

            def trials(self, group):
                return {}

            def run(self, group, workers=0, should_stop=None, limit=None):
                calls.append(group)
                f.set_enabled(False)  # the user switches it off during the run
                return {"done": 3, "grid": 10, "candidates": [], "near": [], "t_required": 5.0}

        monkeypatch.setattr(lab, "auto_search", lambda: FakeAuto())
        monkeypatch.setattr(F, "low_priority", lambda: None)
        monkeypatch.setattr(F, "CHECK_SECONDS", 0)
        assert f.run() == "stopped" and calls == []  # off before it starts: nothing runs
        f.set_enabled(True)
        assert f.run() == "stopped" and calls == ["AUTO-SP500X"]
        log = f.log()
        assert len(log) == 1 and log[0]["tried"] == 3 and log[0]["done"] == 3
        assert not f.running()
    finally:
        lab.close()


def test_factory_reports_once_when_the_grid_is_done(tmp_path, monkeypatch):
    lab = Lab(tmp_path / "QL")
    try:
        f = lab.factory()
        f.set_enabled(True)
        summary = {"done": 10, "grid": 10, "candidates": [], "near": [], "t_required": 5.0, "family_trials": 41863,
                   "funnel": [{"label": "거래 수 부족", "count": 10}]}

        class FakeAuto:
            def loaded(self, group):
                return "hash-1", None

            def trials(self, group):
                return {}

            def run(self, group, workers=0, should_stop=None, limit=None):
                return summary | {"note": "이 묶음의 조합을 모두 시험했습니다."}

        monkeypatch.setattr(lab, "auto_search", lambda: FakeAuto())
        monkeypatch.setattr(F, "low_priority", lambda: None)
        assert f.run() == "finished"
        assert f.finished() and f.run() == "finished" and len(f.log()) == 1  # later starts do nothing
        news = f.unseen()
        assert len(news) == 1 and news[0]["kind"] == "finished"
        assert "10개" in F.report_text(news[0])
        assert json.loads(f.path("finished.json").read_text(encoding="utf-8"))["strategy_hash"] == "hash-1"
    finally:
        lab.close()


def test_factory_stops_when_the_charger_is_unplugged(tmp_path, monkeypatch):
    lab = Lab(tmp_path / "QL")
    try:
        f = lab.factory()
        f.set_enabled(True)
        power = {"battery": False}
        calls = []

        class FakeAuto:
            def loaded(self, group):
                return "h", None

            def trials(self, group):
                return {}

            def run(self, group, workers=0, should_stop=None, limit=None):
                calls.append(1)
                power["battery"] = True  # unplugged during the run
                assert should_stop()
                return {"done": 3, "grid": 10, "candidates": [], "near": [], "t_required": 5.0, "stopped": True}

        monkeypatch.setattr(lab, "auto_search", lambda: FakeAuto())
        monkeypatch.setattr(F, "low_priority", lambda: None)
        monkeypatch.setattr(F, "on_battery", lambda: power["battery"])
        monkeypatch.setattr(F, "CHECK_SECONDS", 0)
        assert f.run() == "stopped" and calls == [1]
        assert f.run() == "stopped" and calls == [1]  # on battery: does not start
        assert f.enabled()                              # the switch stays on; it resumes on the charger
        power["battery"] = False
        assert f.run(max_runs=1) in ("stopped", "max_runs") and len(calls) == 2
    finally:
        lab.close()


def test_watch_keeps_the_strongest_near_misses_and_replaces_weaker_ones(tmp_path, monkeypatch):
    """Round 1 (2026-10-19): the list filled with the first runs' best; it now keeps the top five so far."""
    monkeypatch.setattr(F, "NEAR_IN_WATCH", 2)
    lab = Lab(tmp_path / "QL")
    try:
        f = lab.factory()
        near = lambda e, t: {"params": {"entry": e, "hold": 3, "stop": 0, "take": 0, "slots": 5},  # noqa: E731
                             "rule": e, "t": t, "fail": "tvalue"}
        f._update_watch({"candidates": [], "near": [near("dn3", 2.0), near("dn5", 2.5)]}, today=date(2026, 10, 18))
        assert [w["rule"] for w in f.watch()] == ["dn5", "dn3"]  # strongest first
        f._update_watch({"candidates": [], "near": [near("dn3", 2.0), near("dn5", 2.5), near("low252", 4.0)]},
                        today=date(2026, 10, 20))
        assert [(w["id"], w["rule"]) for w in f.watch()] == [("W01", "dn5"), ("W03", "low252")]
        assert f.watch()[1]["start"] == "2026-10-21"
        assert [w["rule"] for w in f.watch(all_=True) if w.get("dropped")] == ["dn3"]
        assert [r["id"] for r in lab.paper().candidates() if r.get("watch")] == ["AUTO-SP500X-W01", "AUTO-SP500X-W03"]
    finally:
        lab.close()


def test_a_grid_finished_by_build_54_gets_its_final_top_near_misses_once(tmp_path, monkeypatch):
    lab = Lab(tmp_path / "QL")
    try:
        f = lab.factory()
        f.set_enabled(True)
        f._write("finished.json", {"strategy_hash": "h", "at": "x"})  # written before watch_rev existed
        reads = []

        class FakeAuto:
            def loaded(self, group):
                return "h", None

            def summary(self, group):
                reads.append(1)
                return {"candidates": [], "near": [{"params": {"entry": "dn3", "hold": 3, "stop": 0, "take": 0, "slots": 5},
                                                    "rule": "dn3", "t": 4.5, "fail": "tvalue"}]}

        monkeypatch.setattr(lab, "auto_search", lambda: FakeAuto())
        assert f.run() == "finished" and [w["rule"] for w in f.watch()] == ["dn3"]
        assert f.run() == "finished" and reads == [1]  # only once
    finally:
        lab.close()
