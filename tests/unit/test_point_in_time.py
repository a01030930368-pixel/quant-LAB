"""Strategy interface and look-ahead controls (M2)."""

import pandas as pd
import pytest

from quantlab.backtesting.engine import BacktestConfig, CostModel, run_backtest
from quantlab.core.market_data import MarketData, total_return_index
from quantlab.strategies.base import LookaheadError, MarketView, Strategy
from quantlab.strategies.trend_timing import TrendTiming
from synthetic import frame, random_market


def test_view_never_returns_rows_after_now():
    data = random_market(0, n=50)
    for t in (0, 10, 49):
        v = MarketView(data, t)
        h = v.history("AAA")
        assert h.index.max() == data.calendar[t] and len(h) == t + 1


def test_explicit_future_access_raises():
    data = random_market(0, n=50)
    v = MarketView(data, 10)
    with pytest.raises(LookaheadError):
        v.at("AAA", data.calendar[11])
    assert v.at("AAA", data.calendar[10]) == data.frames["AAA"]["close"].iloc[10]


def test_total_return_index_is_not_revised_by_later_data():
    data = random_market(3, n=300, div_prob=0.05, split_prob=0.01)
    full = total_return_index(data.frames["AAA"])
    for t in (50, 150, 299):
        part = MarketView(data, t).total_return("AAA")
        pd.testing.assert_series_equal(part, full.iloc[: t + 1], check_names=False)


def test_signal_at_close_fills_at_next_open():
    class Once(Strategy):
        def on_bar(self, view):
            return {"A": 1.0} if view.now == view_dates[3] else None

    data = MarketData.from_frames({"A": frame(range(10, 20), range(10, 20))})
    view_dates = data.calendar
    res = run_backtest(Once(), data, BacktestConfig(costs=CostModel(slippage_bps=0)))
    assert res.signals[0][0] == view_dates[3]
    assert res.trades.iloc[0]["date"] == view_dates[4]
    assert res.trades.iloc[0]["open"] == 14


def test_strategy_decisions_unchanged_when_future_is_altered():
    """Leak detector used by Gate 1: perturbing data after t must not change the decision at t."""
    data = random_market(11, n=600, symbols=("SPY", "BIL"))
    strat = TrendTiming()
    cut = 400
    altered = {s: df.copy() for s, df in data.frames.items()}
    for df in altered.values():
        df.iloc[cut + 1 :, df.columns.get_indexer(["open", "high", "low", "close"])] *= 3.0
    data2 = MarketData(data.calendar, altered)
    for t in range(cut + 1):
        assert strat.on_bar(MarketView(data, t)) == strat.on_bar(MarketView(data2, t))


def test_trend_timing_rule_matches_hand_calculation():
    # monthly closes rising then a sharp drop in the last month
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    close = pd.Series(100.0, index=idx)
    month = idx.month
    close[:] = [100 + m for m in month]
    close[month == 12] = 50.0
    df = frame(close.values, close.values, start="2020-01-01")
    df.index = idx
    cash = frame([1.0] * len(idx), [1.0] * len(idx))
    cash.index = idx
    data = MarketData.from_frames({"SPY": df, "BIL": cash})
    strat = TrendTiming()
    last_nov = max(i for i, d in enumerate(idx) if d.month == 11)
    last_dec = len(idx) - 1
    # Nov: last 10 month-ends 102..111, mean 106.5 < 111 -> risk on
    assert strat.on_bar(MarketView(data, last_nov)) == {"SPY": 1.0, "BIL": 0.0}
    # Dec: 103..111 and 50 -> mean 98.7 > 50 -> risk off
    assert strat.on_bar(MarketView(data, last_dec)) == {"SPY": 0.0, "BIL": 1.0}
    # mid-month bars never trade
    assert strat.on_bar(MarketView(data, 5)) is None
