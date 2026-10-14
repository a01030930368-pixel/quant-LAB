"""Hand-computed golden cases for fills, costs and corporate actions (M3 test 1)."""

import pytest

from quantlab.backtesting.engine import BacktestConfig, CostModel, run_backtest
from quantlab.core.market_data import MarketData
from quantlab.strategies.base import Strategy
from synthetic import frame


class Schedule(Strategy):
    """Emits pre-set target weights on given bar numbers."""

    strategy_id = "TEST_SCHEDULE"

    def __init__(self, plan):
        super().__init__(plan=plan)
        self.plan, self.t = plan, -1

    def on_bar(self, view):
        self.t += 1
        return self.plan.get(self.t)


def md(**frames):
    return MarketData.from_frames(frames)


def test_buy_fills_next_open_with_slippage_and_commission():
    data = md(A=frame(opens=[10, 11, 12, 12], closes=[10, 11.5, 12, 13]))
    cfg = BacktestConfig(initial_cash=1000, costs=CostModel(commission_bps=10, slippage_bps=50, fixed_fee=1.0))
    res = run_backtest(Schedule({0: {"A": 1.0}}), data, cfg)
    # target = floor(1.0 * 1000 / 10) = 100 shares, at next open 11 * 1.005 = 11.055
    # budget 999 / (11.055 * 1.001) = 90.27 -> 90 shares (cash-limited)
    tr = res.trades.iloc[0]
    assert tr["date"] == data.calendar[1]
    assert tr["qty"] == 90
    assert tr["fill_price"] == pytest.approx(11.055)
    assert tr["commission"] == pytest.approx(90 * 11.055 * 0.001 + 1.0)
    cash = 1000 - 90 * 11.055 - (90 * 11.055 * 0.001 + 1.0)
    assert res.cash.iloc[1] == pytest.approx(cash)
    assert res.equity.iloc[1] == pytest.approx(cash + 90 * 11.5)
    assert res.equity.iloc[3] == pytest.approx(cash + 90 * 13)
    assert res.equity.iloc[0] == 1000  # nothing happens on the signal bar itself


def test_sell_receives_open_minus_slippage():
    data = md(A=frame(opens=[10, 10, 20, 20], closes=[10, 10, 20, 20]))
    cfg = BacktestConfig(initial_cash=1000, costs=CostModel(slippage_bps=100))
    res = run_backtest(Schedule({0: {"A": 1.0}, 1: {"A": 0.0}}), data, cfg)
    buy, sell = res.trades.iloc[0], res.trades.iloc[1]
    assert buy["qty"] == 99  # 1000 / (10 * 1.01) = 99.0099
    assert sell["qty"] == -99 and sell["fill_price"] == pytest.approx(19.8)
    assert res.cash.iloc[-1] == pytest.approx(1000 - 99 * 10.1 + 99 * 19.8)
    assert res.shares["A"].iloc[-1] == 0


def test_partial_rebalance_between_two_assets():
    data = md(A=frame([10, 10, 10, 10], [10, 10, 10, 10]), B=frame([20, 20, 25, 25], [20, 20, 25, 25]))
    cfg = BacktestConfig(initial_cash=1000, costs=CostModel(slippage_bps=0))
    res = run_backtest(Schedule({0: {"A": 0.5, "B": 0.5}, 2: {"A": 0.2, "B": 0.8}}), data, cfg)
    assert list(res.shares.iloc[1]) == [50, 25]
    # bar 2 close: equity = 500 + 25*25 = 1125 -> targets A floor(225/10)=22, B floor(900/25)=36
    assert list(res.shares.iloc[3]) == [22, 36]
    assert res.equity.iloc[3] == pytest.approx(1125)  # zero-cost rebalance keeps equity


def test_dividend_credited_on_ex_date():
    data = md(A=frame([10, 10, 10, 9.5], [10, 10, 10, 9.5], dividends=[0, 0, 0, 0.5]))
    cfg = BacktestConfig(initial_cash=1000, costs=CostModel(slippage_bps=0))
    res = run_backtest(Schedule({0: {"A": 1.0}}), data, cfg)
    assert res.shares["A"].iloc[-1] == 100
    assert res.cash.iloc[-1] == pytest.approx(50.0)
    assert res.equity.iloc[-1] == pytest.approx(1000.0)  # price drop offset by cash dividend
    assert (res.cash_flows["kind"] == "dividend").sum() == 1


def test_split_restates_shares_and_pending_orders():
    # 2-for-1 split effective on bar 2; signal on bar 1 targets 100 pre-split shares
    data = md(A=frame([10, 10, 5, 5], [10, 10, 5, 5], splits=[1, 1, 2, 1]))
    cfg = BacktestConfig(initial_cash=1000, costs=CostModel(slippage_bps=0))
    res = run_backtest(Schedule({1: {"A": 1.0}}), data, cfg)
    assert res.shares["A"].iloc[2] == 200
    assert res.equity.iloc[-1] == pytest.approx(1000.0)

    res2 = run_backtest(Schedule({0: {"A": 1.0}}), data, cfg)
    assert list(res2.shares["A"]) == [0, 100, 200, 200]
    assert res2.equity.iloc[-1] == pytest.approx(1000.0)


def test_untradable_day_carries_order():
    nan = float("nan")
    data = md(A=frame([10, nan, 12, 12], [10, nan, 12, 12]))
    cfg = BacktestConfig(initial_cash=1000, costs=CostModel(slippage_bps=0))
    res = run_backtest(Schedule({0: {"A": 1.0}}), data, cfg)
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["date"] == data.calendar[2]
    assert res.trades.iloc[0]["qty"] == 83  # cash-limited at the gap-up open: floor(1000/12)


def test_rejects_leverage_and_shorts():
    data = md(A=frame([10, 10], [10, 10]))
    with pytest.raises(ValueError):
        run_backtest(Schedule({0: {"A": 1.5}}), data)
    with pytest.raises(ValueError):
        run_backtest(Schedule({0: {"A": -0.5}}), data)
