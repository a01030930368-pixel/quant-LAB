"""Event-driven daily backtester: cash, share and cost accounting.

Timeline of one trading day t (protocol 제8조: observe → signal → order → fill → update):

  open  : 1. corporate actions (split adjusts shares, dividend credits cash)
          2. orders generated at the previous close fill at today's open
             (sells first, then buys limited by available cash)
  close : 3. mark to market
          4. strategy sees data up to today's close and may return target
             weights; resulting orders wait for the NEXT open

Fill price = open * (1 ± slippage). Commission = |notional| * commission_bps
+ fixed fee. Nothing here can make a fill better than the open price.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from quantlab.core.market_data import MarketData
from quantlab.strategies.base import MarketView, PartialWeights, Strategy

BPS = 1e-4
_EPS = 1e-9


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.0
    slippage_bps: float = 5.0
    fixed_fee: float = 0.0

    def fill_price(self, open_px: float, qty: float) -> float:
        side = 1.0 if qty > 0 else -1.0
        return open_px * (1.0 + side * self.slippage_bps * BPS)

    def commission(self, notional: float) -> float:
        return abs(notional) * self.commission_bps * BPS + (self.fixed_fee if notional else 0.0)


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    costs: CostModel = field(default_factory=CostModel)
    integer_shares: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BacktestResult:
    equity: pd.Series
    cash: pd.Series
    shares: pd.DataFrame
    trades: pd.DataFrame
    cash_flows: pd.DataFrame
    signals: list[tuple[pd.Timestamp, dict]]
    config: BacktestConfig

    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)


class AccountingError(RuntimeError):
    pass


def _validate_weights(weights, symbols) -> dict[str, float]:
    """Full targets for every symbol; PartialWeights keep only the symbols they list."""
    w = {s: float(weights.get(s, 0.0)) for s in symbols if not isinstance(weights, PartialWeights) or s in weights}
    unknown = set(weights) - set(symbols)
    if unknown:
        raise ValueError(f"unknown symbols in target weights: {sorted(unknown)}")
    if any(v < -_EPS or not math.isfinite(v) for v in w.values()):
        raise ValueError(f"MVP engine is long-only; got {w}")
    if sum(w.values()) > 1.0 + 1e-9:
        raise ValueError(f"weights sum above 1 (no leverage in MVP): {w}")
    return w


def run_backtest(strategy: Strategy, data: MarketData, config: BacktestConfig | None = None) -> BacktestResult:
    config = config or BacktestConfig()
    costs = config.costs
    syms = data.symbols
    cal = data.calendar
    n = len(cal)
    opens = data.field("open").to_numpy()
    closes_raw = data.field("close")
    closes = closes_raw.ffill().to_numpy()
    divs = data.field("dividend").to_numpy()
    splits = data.field("split").to_numpy()

    cash = float(config.initial_cash)
    shares = np.zeros(len(syms))
    pending: dict[int, float] = {}

    eq_hist = np.empty(n)
    cash_hist = np.empty(n)
    sh_hist = np.empty((n, len(syms)))
    trades: list[dict] = []
    flows: list[dict] = [{"date": cal[0], "kind": "initial", "symbol": None, "amount": cash}]
    signals: list[tuple[pd.Timestamp, dict]] = []

    for t in range(n):
        date = cal[t]
        # 1. corporate actions at the open
        for i in range(len(syms)):
            if splits[t, i] != 1.0:
                shares[i] *= splits[t, i]
                if i in pending:  # brokers restate open orders for splits
                    pending[i] *= splits[t, i]
            if divs[t, i] != 0.0 and shares[i] != 0.0:
                amt = shares[i] * divs[t, i]
                cash += amt
                flows.append({"date": date, "kind": "dividend", "symbol": syms[i], "amount": amt})

        # 2. fills at the open (sells first so their cash funds buys)
        if pending:
            # pending holds TARGET share counts; sells first so their cash funds buys
            order = sorted(pending.items(), key=lambda kv: kv[1] - shares[kv[0]])
            still_pending: dict[int, float] = {}
            for i, tgt in order:
                px_open = opens[t, i]
                if not np.isfinite(px_open):
                    still_pending[i] = tgt  # not tradable today: carry the order
                    continue
                qty = tgt - shares[i]
                if qty < 0:
                    qty = -min(-qty, shares[i])
                if qty > 0:
                    px = costs.fill_price(px_open, qty)
                    unit = px * (1.0 + costs.commission_bps * BPS)
                    budget = cash - costs.fixed_fee
                    affordable = budget / unit if budget > 0 else 0.0
                    if config.integer_shares:
                        affordable = math.floor(affordable + _EPS)
                    qty = min(qty, affordable)
                if abs(qty) < _EPS:
                    continue
                px = costs.fill_price(px_open, qty)
                notional = qty * px
                comm = costs.commission(notional)
                cash -= notional + comm
                shares[i] += qty
                slip_cost = abs(qty) * abs(px - px_open)
                trades.append({"date": date, "symbol": syms[i], "qty": qty, "open": px_open,
                               "fill_price": px, "notional": notional, "commission": comm,
                               "slippage_cost": slip_cost})
                flows.append({"date": date, "kind": "trade", "symbol": syms[i], "amount": -notional})
                if comm:
                    flows.append({"date": date, "kind": "commission", "symbol": syms[i], "amount": -comm})
                if cash < -1e-6:
                    raise AccountingError(f"negative cash {cash} on {date.date()}")
            pending = still_pending

        # 3. mark to market at the close
        pos_val = float(np.nansum(shares * closes[t]))
        equity = cash + pos_val
        eq_hist[t], cash_hist[t], sh_hist[t] = equity, cash, shares

        # 4. strategy decision at the close; fills at the next open
        if t < n - 1:
            target = strategy.on_bar(MarketView(data, t))
            if target is not None:
                w = _validate_weights(target, syms)
                signals.append((date, w))
                if isinstance(target, PartialWeights):  # unlisted symbols keep their orders and shares
                    pending = {i: q for i, q in pending.items() if syms[i] not in w}
                else:
                    pending = {}
                for i, s in enumerate(syms):
                    if s not in w:
                        continue
                    px = closes[t, i]
                    if not np.isfinite(px) or px <= 0:
                        continue
                    tgt = w[s] * equity / px
                    if config.integer_shares:
                        tgt = math.floor(tgt + _EPS)
                    if abs(tgt - shares[i]) > _EPS:
                        pending[i] = tgt

    flows_df = pd.DataFrame(flows)
    result = BacktestResult(
        equity=pd.Series(eq_hist, index=cal, name="equity"),
        cash=pd.Series(cash_hist, index=cal, name="cash"),
        shares=pd.DataFrame(sh_hist, index=cal, columns=syms),
        trades=pd.DataFrame(trades, columns=["date", "symbol", "qty", "open", "fill_price", "notional",
                                             "commission", "slippage_cost"]),
        cash_flows=flows_df,
        signals=signals,
        config=config,
    )
    check_accounting(result, data)
    return result


def check_accounting(result: BacktestResult, data: MarketData, tol: float = 1e-6) -> None:
    """Runtime invariants; any violation makes the experiment INVALID (Gate 1)."""
    final_cash = result.cash.iloc[-1]
    ledger_cash = result.cash_flows["amount"].sum()
    if abs(final_cash - ledger_cash) > tol * max(1.0, abs(final_cash)):
        raise AccountingError(f"cash ledger mismatch: {final_cash} vs {ledger_cash}")
    if (result.cash < -1e-6).any():
        raise AccountingError("cash went negative")
    if (result.shares < -_EPS).to_numpy().any():
        raise AccountingError("short position in long-only engine")
    closes = data.field("close").ffill().fillna(0.0)
    mtm = result.cash + (result.shares * closes).sum(axis=1)
    if not np.allclose(mtm.to_numpy(), result.equity.to_numpy(), rtol=0, atol=tol * result.equity.abs().max()):
        raise AccountingError("equity != cash + marked positions")
