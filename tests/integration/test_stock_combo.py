"""D-028: automatic search on S&P 500 stocks (rules block `stock_combo`)."""

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.engine import run_backtest
from quantlab.core.market_data import MarketData
from quantlab.services import autosearch as A
from quantlab.services.lab import builtin_path
from quantlab.strategies import stock_combo as sc
from quantlab.strategies.spec import load_spec
from quantlab.validation.engine import leak_test
from synthetic import frame

SPEC = builtin_path("specs", "AUTO-SP500.qls.yaml")
RULES = {"stock_combo": {"catalog": 1, "park": "SPY", "park_band": 2}}


def strat(entry, hold=5, stop=0, take=0, slots=2):
    return sc.StockComboStrategy(spec_id="T", spec_version_str="1", rules=RULES,
                                 entry=entry, hold=hold, stop=stop, take=take, slots=slots)


def market(closes: dict, events=None, opens=None, n=80, members_off=()):
    idx = pd.bdate_range("2020-01-01", periods=n)
    frames = {}
    for s in ("SPY", "X", "Y", "Z"):
        c = np.asarray(closes.get(s, np.full(n, 100.0)), float)
        o = np.asarray((opens or {}).get(s, c), float)
        f = frame(o, c)
        f.index = idx
        frames[s] = f
    members = pd.DataFrame(True, index=idx, columns=["X", "Y", "Z"])
    for s in members_off:
        members[s] = False
    ev = pd.DataFrame(events or [], columns=["known", "symbol", "kind", "owner", "value", "officer", "director"])
    ev["known"] = pd.to_datetime(ev["known"])
    return MarketData(idx, frames, members, ev.sort_values("known", kind="stable").reset_index(drop=True))


def buys(data, **kw):
    t = run_backtest(strat(**kw), data).trades
    t = t[t["symbol"] != "SPY"]
    return t


def days(t):
    return list(t["date"].dt.strftime("%Y-%m-%d"))


def test_catalog_size_and_spec():
    assert len(sc.entries()) == 465
    space = sc.full_space()
    assert np.prod([len(v) for v in space.values()]) == 41850
    loaded = load_spec(SPEC)
    assert set(loaded.search_space["entry"]) == set(sc.entries())
    assert loaded.default_params == {"entry": "earn5", "hold": 20, "stop": 0, "take": 0, "slots": 10}
    # every entry uses conditions from different groups; '+' joins them in catalog order
    for e in sc.entries():
        parts = e.split("+")
        assert len({sc.CONDITIONS[p][1] for p in parts}) == len(parts)
        assert parts == sorted(parts, key=list(sc.CONDITIONS).index)
    assert sc.describe({"entry": "earn5+ins2", "hold": 20, "stop": 0.1, "take": 0, "slots": 10}).count(" · ") == 4


def test_neighbours_step_one_threshold_or_the_hold():
    p = {"entry": "earn5+drop5", "hold": 20, "stop": 0, "take": 0, "slots": 10}
    got = {(n["entry"], n["hold"]) for n in sc.neighbours(p)}
    assert got == {("earn3+drop5", 20), ("earn8+drop5", 20), ("earn5+drop10", 20),
                   ("earn5+drop5", 10), ("earn5+drop5", 30)}
    assert all(n["entry"] in set(sc.entries()) for n in sc.neighbours(p))


def test_drop_signal_fires_on_the_first_session_and_exits_after_hold():
    c = np.full(80, 100.0)
    c[20:] = 92.0                      # -8% on session 20: ret5 <= -5% on sessions 20..24
    t = buys(market({"X": c}), entry="drop5", hold=5)
    idx = pd.bdate_range("2020-01-01", periods=80)
    assert days(t) == [str(idx[21].date()), str(idx[26].date())]   # buy next open, sell 5 sessions later
    assert list(np.sign(t["qty"])) == [1, -1]
    # not a member that day: no trade
    assert buys(market({"X": c}, members_off=("X",)), entry="drop5").empty


def test_take_profit_and_stop_loss_on_the_close():
    idx = pd.bdate_range("2020-01-01", periods=80)
    c = np.full(80, 100.0)
    c[20:] = 92.0
    c[23:] = 92.0 * 1.12               # +12% over the entry open three sessions in
    t = buys(market({"X": c}), entry="drop5", hold=30, take=0.1)
    assert days(t) == [str(idx[21].date()), str(idx[24].date())]
    c2 = np.full(80, 100.0)
    c2[20:] = 92.0
    c2[25:] = 92.0 * 0.85
    t = buys(market({"X": c2}), entry="drop5", hold=30, stop=0.1)
    assert days(t)[1] == str(idx[26].date())


def test_earnings_reaction_counts_for_five_sessions_and_needs_the_move_over_spy():
    idx = pd.bdate_range("2020-01-01", periods=80)
    c = np.full(80, 100.0)
    c[idx > pd.Timestamp("2020-01-07")] = 110.0
    ev = [("2020-01-07", "X", "earnings", None, None, None, None)]
    p = sc.build_panels(market({"X": c}, ev))
    j = p["symbols"].index("X")
    on = np.flatnonzero(p["earn8"][:, j])
    assert [str(idx[i].date()) for i in on] == ["2020-01-08", "2020-01-09", "2020-01-10", "2020-01-13", "2020-01-14"]
    # the market rose as much: no abnormal reaction
    p = sc.build_panels(market({"X": c, "SPY": c}, ev))
    assert not p["earn3"].any()


def test_insider_count_is_distinct_officers_or_directors_over_30_days():
    ev = [("2020-01-06", "Y", "insider_buy", "o1", 20000, True, False),
          ("2020-01-08", "Y", "insider_buy", "o1", 20000, True, False),     # same person again
          ("2020-01-10", "Y", "insider_buy", "o2", 5000, True, False),      # too small
          ("2020-01-13", "Y", "insider_buy", "o3", 20000, False, False),    # no role
          ("2020-01-20", "Y", "insider_buy", "o4", 20000, False, True)]
    idx = pd.bdate_range("2020-01-01", periods=80)
    p = sc.build_panels(market({}, ev))
    j = p["symbols"].index("Y")
    on1 = [str(idx[i].date()) for i in np.flatnonzero(p["ins1"][:, j])]
    on2 = [str(idx[i].date()) for i in np.flatnonzero(p["ins2"][:, j])]
    assert on1[0] == "2020-01-06" and on1[-1] == "2020-02-18"      # o4 filed 01-20: 30 days through 02-18
    assert on2 == [str(d.date()) for d in idx if pd.Timestamp("2020-01-20") <= d <= pd.Timestamp("2020-02-06")]


def test_combined_conditions_and_pick_order_is_fixed():
    c = np.full(80, 100.0)
    c[20:] = 92.0
    data = market({"X": c, "Y": c, "Z": c})
    t1 = buys(data, entry="drop5", slots=1)
    t2 = buys(data, entry="drop5", slots=1)
    assert days(t1) == days(t2) and list(t1["symbol"]) == list(t2["symbol"])
    assert t1["symbol"].nunique() == 1
    # drop alone fires; drop and a gap up on the same day does not
    assert buys(data, entry="drop5+gap3").empty


def test_panels_and_decisions_use_only_the_past():
    rng = np.random.default_rng(3)
    n = 400
    closes = {s: 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n))) for s in ("SPY", "X", "Y", "Z")}
    idx = pd.bdate_range("2020-01-01", periods=n)
    ev = [(str(idx[i].date()), s, "earnings", None, None, None, None) for i in range(10, n, 30) for s in ("X", "Y")]
    ev += [(str(idx[i].date()), "Z", "insider_buy", f"o{i}", 50000, True, False) for i in range(5, n, 17)]
    data = market(closes, ev, n=n)
    full = sc.build_panels(data)
    cut = 250
    part = sc.build_panels(data.slice(None, idx[cut]))
    for k, v in full.items():
        if k != "symbols":
            a, b = v[: cut + 1], part[k]
            assert np.array_equal(a, b, equal_nan=True), k
    for entry in ("earn3", "ins1+rsi14_30", "drop5+down200", "gap3+spy_up200", "high252"):
        assert leak_test(lambda **p: strat(**p), {"entry": entry, "hold": 10, "stop": 0.1, "take": 0.2, "slots": 2},
                         data, 4, 7) == [], entry


def test_search_reads_return_over_spy():
    m = {"n_train": 50, "n_validation": 30, "gross": 0.05, "excess": 0.005, "ret_train": 0.1, "ret_validation": 0.1}
    auto = {"min_trades_train": 40, "min_trades_validation": 20, "min_gross_per_trade": 0.012}
    assert A.first_failure(m, auto, 0.0, 0.0) == "cost2x"           # gross filter reads `gross` for ETFs
    assert A.first_failure(m, auto | {"versus": "SPY"}, 0.0, 0.0) == "gross"
    assert A.key({"entry": "earn5", "hold": 20, "stop": 0, "take": 0, "slots": 10}) != \
        A.key({"entry": "earn5", "hold": 20, "stop": 0, "take": 0, "slots": 20})
