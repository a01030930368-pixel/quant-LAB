"""Accounting invariants on random markets, and cross-check against the
independent reference implementation (M3 tests 2-4)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantlab.backtesting.engine import BacktestConfig, CostModel, run_backtest
from quantlab.backtesting.reference import reference_equity
from quantlab.core.market_data import total_return_index
from quantlab.strategies.base import Strategy
from synthetic import random_market


class RandomWeights(Strategy):
    strategy_id = "TEST_RANDOM_WEIGHTS"

    def __init__(self, seed, max_total=1.0, every=5):
        super().__init__(seed=seed, max_total=max_total, every=every)
        self.rng = np.random.default_rng(seed)
        self.max_total, self.every, self.t = max_total, every, -1

    def on_bar(self, view):
        self.t += 1
        if self.t % self.every:
            return None
        w = self.rng.dirichlet(np.ones(len(view.symbols) + 1))[:-1] * self.max_total
        return dict(zip(view.symbols, w))


class BuyHold(Strategy):
    strategy_id = "TEST_BUY_HOLD"

    def __init__(self, symbol):
        super().__init__(symbol=symbol)
        self.symbol, self.done = symbol, False

    def on_bar(self, view):
        if self.done:
            return None
        self.done = True
        return {self.symbol: 1.0}


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10_000), comm=st.floats(0, 20), slip=st.floats(0, 30), fee=st.floats(0, 5))
def test_accounting_invariants_hold_under_random_trading(seed, comm, slip, fee):
    data = random_market(seed, n=200)
    cfg = BacktestConfig(initial_cash=50_000, costs=CostModel(commission_bps=comm, slippage_bps=slip, fixed_fee=fee))
    res = run_backtest(RandomWeights(seed), data, cfg)  # run_backtest itself raises on any violation
    assert (res.cash >= -1e-6).all()
    if len(res.trades):
        assert (res.trades["commission"] >= 0).all() and (res.trades["slippage_cost"] >= 0).all()
    # P&L attribution: equity change = price P&L + dividends - commissions - slippage
    closes = data.field("close").ffill()
    flows = res.cash_flows
    divs = flows.loc[flows.kind == "dividend", "amount"].sum()
    comms = -flows.loc[flows.kind == "commission", "amount"].sum()
    traded = -flows.loc[flows.kind == "trade", "amount"].sum()  # net cash spent on shares
    final_pos = float((res.shares.iloc[-1] * closes.iloc[-1]).sum())
    assert res.equity.iloc[-1] - cfg.initial_cash == pytest.approx(final_pos - traded + divs - comms, abs=1e-6)


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_engine_matches_independent_reference(seed):
    data = random_market(seed, n=250, vol=0.005)
    cfg = BacktestConfig(initial_cash=1_000_000, costs=CostModel(slippage_bps=0), integer_shares=False)
    eng = run_backtest(RandomWeights(seed, max_total=0.9), data, cfg).equity
    ref = reference_equity(RandomWeights(seed, max_total=0.9), data, cfg.initial_cash)
    np.testing.assert_allclose(eng.to_numpy(), ref.to_numpy(), rtol=1e-9)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_buy_and_hold_equals_total_return_index(seed):
    # no dividends (engine keeps them as cash rather than reinvesting), splits included,
    # first fill at a price equal to the signal close so the position is exactly fully invested
    data = random_market(seed, n=400, div_prob=0.0, split_prob=0.01)
    data.frames["AAA"].iloc[1, 0] = data.frames["AAA"]["close"].iloc[0]
    cfg = BacktestConfig(initial_cash=1.0, costs=CostModel(slippage_bps=0), integer_shares=False)
    res = run_backtest(BuyHold("AAA"), data, cfg)
    tr = total_return_index(data.frames["AAA"])
    np.testing.assert_allclose(res.equity.to_numpy(), (tr / tr.iloc[0]).to_numpy(), rtol=1e-9)


def test_same_inputs_same_outputs():
    data = random_market(7)
    a = run_backtest(RandomWeights(7), data).equity
    b = run_backtest(RandomWeights(7), data).equity
    assert a.equals(b)
