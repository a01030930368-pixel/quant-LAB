"""Independent reference implementation used only to cross-check the engine.

It tracks position VALUES compounded by price relatives instead of share
counts, and applies corporate actions as returns instead of share/cash
adjustments. Scope is deliberately narrow: zero costs, fractional shares,
target weights that never require more cash than is available. Under those
conditions its equity curve must match `run_backtest` to floating-point
precision (03_SYSTEM_ARCHITECTURE 2.1: in-house engine must pass an
independent accuracy test).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.core.market_data import MarketData
from quantlab.strategies.base import MarketView, Strategy


def reference_equity(strategy: Strategy, data: MarketData, initial_cash: float) -> pd.Series:
    syms = data.symbols
    o = data.field("open").to_numpy()
    c = data.field("close").ffill().to_numpy()
    d = data.field("dividend").to_numpy()
    k = data.field("split").to_numpy()
    n, m = c.shape
    value = np.zeros(m)          # value of each position at the last mark
    last_px = np.full(m, np.nan)  # price at which `value` was last marked
    cash = initial_cash
    target_units: np.ndarray | None = None
    out = np.empty(n)

    for t in range(n):
        # overnight: previous close -> today's open, as a return on value
        if t > 0:
            for i in range(m):
                if value[i] != 0.0:
                    rel = o[t, i] * k[t, i] / c[t - 1, i] if np.isfinite(o[t, i]) else c[t, i] * k[t, i] / c[t - 1, i]
                    cash += value[i] * d[t, i] * k[t, i] / c[t - 1, i]
                    value[i] *= rel
            last_px = np.where(np.isfinite(o[t]), o[t], c[t])
        if target_units is not None and t > 0:
            target_units = target_units * k[t]
            new_value = target_units * o[t]
            cash += float(np.sum(value - new_value))
            value = new_value
            target_units = None
        # intraday: open -> close
        if t > 0:
            value = value * np.where(np.isfinite(o[t]), c[t] / last_px, 1.0)
        out[t] = cash + value.sum()
        if t < n - 1:
            w = strategy.on_bar(MarketView(data, t))
            if w is not None:
                target_units = np.array([w.get(s, 0.0) * out[t] / c[t, i] for i, s in enumerate(syms)])
    return pd.Series(out, index=data.calendar, name="reference_equity")
