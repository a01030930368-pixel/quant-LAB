"""Swing-style entry/exit rules (M12b): hand-computed cases, look-ahead and split handling."""

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.engine import BacktestConfig, CostModel, run_backtest
from quantlab.core.market_data import MarketData
from quantlab.strategies.base import MarketView
from quantlab.strategies.declarative import validate_rules
from quantlab.strategies.entry_exit import EntryExitStrategy, _wilder_rsi, resolve_params
from synthetic import frame, random_market

ZERO = BacktestConfig(initial_cash=10_000.0, costs=CostModel(0.0, 0.0), integer_shares=False)


def make(legs, symbols=("A",), sizing="equal_active", **params):
    rules = resolve_params({"entry_exit": {"sizing": sizing, "legs": legs}}, params)
    return EntryExitStrategy(spec_id="T", spec_version_str="1", symbols=list(symbols), rules=rules, **params)


def leg(entry, max_hold=5, any_=(), **exit_kw):
    return {"name": "L", "trade": "A", "entry": list(entry), "exit": {"any": list(any_), "max_hold_days": max_hold, **exit_kw}}


DROP = {"indicator": "return", "days": 1, "op": "<=", "value": -0.05}


def market(closes, opens=None, splits=None):
    closes = np.asarray(closes, float)
    opens = closes if opens is None else np.asarray(opens, float)
    return MarketData.from_frames({"A": frame(opens, closes, splits=splits)})


def test_entry_next_open_and_max_hold_exit():
    # day 3 drops 10% -> buy day 4 open; held days 4,5 (max_hold 2) -> sell day 6 open
    closes = [100, 100, 100, 90, 92, 95, 99, 99]
    opens = [100, 100, 100, 100, 91, 93, 97, 99]
    d = market(closes, opens)
    res = run_backtest(make([leg([DROP], max_hold=2)]), d, ZERO)
    t = res.trades
    assert list(t["date"]) == [d.calendar[4], d.calendar[6]]
    assert t["fill_price"].tolist() == [91, 97]
    assert res.equity.iloc[-1] == pytest.approx(10_000 * 97 / 91)


def test_exit_condition_and_stop_loss_close_based():
    closes = [100, 100, 100, 90, 80, 85, 86, 86]
    opens = [100, 100, 100, 100, 90, 82, 85, 86]
    d = market(closes, opens)
    # stop 10% vs fill 90: day 4 close 80 -> -11% -> sell at day 5 open 82
    res = run_backtest(make([leg([DROP], max_hold=10, stop_loss=0.10)]), d, ZERO)
    assert list(res.trades["date"]) == [d.calendar[4], d.calendar[5]]
    # exit on first up close: day 4 down, day 5 up (85 > 80) -> sell day 6 open 85
    up = {"indicator": "return", "days": 1, "op": ">", "value": 0}
    res = run_backtest(make([leg([DROP], max_hold=10, any_=[up])]), d, ZERO)
    assert list(res.trades["date"]) == [d.calendar[4], d.calendar[6]]


def test_take_profit_restated_for_split_during_hold():
    # buy at 100 (day 4 open), 2:1 split on day 5: raw close 55 == 110 pre-split -> +10% target hit
    closes = [100, 100, 100, 90, 100, 55, 56, 56]
    opens = [100, 100, 100, 100, 100, 50, 55, 56]
    splits = [1, 1, 1, 1, 1, 2, 1, 1]
    d = market(closes, opens, splits)
    res = run_backtest(make([leg([DROP], max_hold=10, take_profit=0.10)]), d, ZERO)
    assert list(res.trades["date"]) == [d.calendar[4], d.calendar[6]]
    # without the split restatement 55/100 would look like a 45% loss and never take profit


def test_split_does_not_fake_a_drop_signal():
    closes = [100, 100, 100, 50, 50, 50]
    d = market(closes, closes, [1, 1, 1, 2, 1, 1])
    res = run_backtest(make([leg([DROP])]), d, ZERO)
    assert res.trades.empty


def test_equal_active_sizing_and_slots():
    d = MarketData.from_frames({s: frame([100] * 3 + [90] * 5, [100] * 3 + [90] * 5) for s in ("A", "B")})
    legs = [dict(leg([DROP], max_hold=3), name="a"), dict(leg([DROP], max_hold=3), name="b", trade="B")]
    s = make(legs, symbols=("A", "B"))
    run_backtest(s, d, ZERO)
    assert s._weights() == {"A": 0.0, "B": 0.0}
    s2 = make(legs, symbols=("A", "B"))
    w = None
    for t in range(len(d.calendar)):
        out = s2.on_bar(MarketView(d, t))
        if out is not None:
            w = out
            break
    assert w == {"A": 0.5, "B": 0.5}
    s3 = make(legs, symbols=("A", "B"), sizing={"slots": 4})
    for t in range(4):
        out = s3.on_bar(MarketView(d, t)) or out
    assert out == {"A": 0.25, "B": 0.25}


def test_wilder_rsi_matches_textbook_recursion():
    rng = np.random.default_rng(1)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60))))
    d = c.diff().dropna().to_numpy()
    n = 2
    au, ad = np.mean(np.clip(d[:n], 0, None)), np.mean(np.clip(-d[:n], 0, None))
    for x in d[n:]:
        au = (au * (n - 1) + max(x, 0)) / n
        ad = (ad * (n - 1) + max(-x, 0)) / n
    assert _wilder_rsi(c, n) == pytest.approx(100 - 100 / (1 + au / ad))


ALL_INDICATORS = [
    {"indicator": "close_vs_sma", "days": 20, "op": "<", "value": -0.01},
    {"indicator": "close_vs_min", "days": 5, "op": "<=", "value": 0},
    {"indicator": "bollinger_z", "days": 20, "op": "<", "value": -1.5},
    {"indicator": "rsi", "days": 2, "op": "<", "value": 15},
    {"indicator": "ibs", "op": "<", "value": 0.2},
    {"indicator": "down_streak", "op": ">=", "value": 2},
    {"indicator": "relative_return", "of": "A", "versus": "B", "days": 5, "op": "<", "value": -0.02},
]


@pytest.mark.parametrize("cond", ALL_INDICATORS, ids=lambda c: c["indicator"])
def test_decisions_do_not_change_when_future_prices_change(cond):
    data = random_market(5, n=260, symbols=("A", "B"), split_prob=0.01)
    k = 180
    other = {}
    for s, f in data.frames.items():
        g = f.copy()
        g.iloc[k + 1:, :4] *= 1.7
        other[s] = g
    data2 = MarketData.from_frames(other)

    def decisions(dd):
        s = make([leg([cond], max_hold=4, any_=[{"indicator": "close_vs_max", "days": 3, "op": ">=", "value": 0}],
                      stop_loss=0.05)], symbols=("A", "B"))
        return [s.on_bar(MarketView(dd, t)) for t in range(k + 1)]

    a, b = decisions(data), decisions(data2)
    assert a == b
    assert any(x is not None for x in a), "condition never fired; test would be vacuous"


def test_session_condition_and_validation():
    assert validate_rules({"entry_exit": {"legs": [leg([{"indicator": "session", "value": "first_of_week"}])]}},
                          ["A"], "daily") == []
    errs = validate_rules({"entry_exit": {"legs": [{"trade": "Z", "entry": [{"indicator": "nope"}], "exit": {}}]}},
                          ["A"], "monthly")
    text = " ".join(errs)
    for frag in ("rebalance must be daily", "trade must be", "indicator must be", "max_hold_days"):
        assert frag in text


def test_parameter_reference():
    s = make([leg([{"indicator": "return", "days": 1, "op": "<=", "value": "$drop"}], max_hold="$hold")], drop=-0.05,
             hold=3)
    assert s.legs[0].spec["exit"]["max_hold_days"] == 3
    with pytest.raises(ValueError):
        make([leg([DROP], max_hold="$missing")])


def test_n4_daily_approximation_imports(tmp_path):
    from pathlib import Path

    from quantlab.registry.store import Registry
    from quantlab.strategies.importer import import_spec

    reg = Registry(tmp_path / "r.sqlite")
    res = import_spec(Path(__file__).parents[1] / "fixtures" / "specs" / "N4-DAILY-APPROX.qls.yaml", reg)
    assert res.state == "RESEARCH", res.problems


def test_episode_stats_hand_case():
    from quantlab.analysis.trades import episodes, trade_stats

    closes = [100, 100, 100, 90, 92, 95, 99, 99]
    opens = [100, 100, 100, 100, 91, 93, 97, 99]
    d = market(closes, opens)
    cfg = BacktestConfig(initial_cash=10_000.0, costs=CostModel(commission_bps=25, slippage_bps=0),
                         integer_shares=False)
    res = run_backtest(make([leg([DROP], max_hold=2)]), d, cfg)
    ep = episodes(res, d)
    assert len(ep) == 1 and ep["sessions"][0] == 2
    assert ep["gross"][0] == pytest.approx(97 / 91 - 1)
    assert ep["net"][0] == pytest.approx(97 * (1 - 0.0025) / (91 * (1 + 0.0025)) - 1)
    st = trade_stats(ep)
    assert st["n"] == 1 and st["t_day_clustered"] is None
