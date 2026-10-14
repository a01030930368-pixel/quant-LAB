"""Baseline strategy: monthly moving-average trend filter (Faber 2007).

At the last session of each month: if the risk asset's total-return index is
above its simple moving average of the last `lookback_months` month-end values,
hold 100% risk asset; otherwise hold 100% cash asset. Parameters are fixed
from the literature before any data is examined (MVP plan §2, §5).
"""

from __future__ import annotations

from quantlab.strategies.base import MarketView, Strategy


class TrendTiming(Strategy):
    strategy_id = "TT-SPY-SMA"
    version = "1.0.0"

    def __init__(self, risk: str = "SPY", cash: str = "BIL", lookback_months: int = 10) -> None:
        super().__init__(risk=risk, cash=cash, lookback_months=lookback_months)
        self.risk, self.cash, self.lookback_months = risk, cash, lookback_months

    def on_bar(self, view: MarketView):
        if not view.is_last_session_of_month():
            return None
        tr = view.total_return(self.risk)
        month_end = tr.groupby([tr.index.year, tr.index.month]).tail(1)
        if len(month_end) < self.lookback_months:
            return None
        window = month_end.iloc[-self.lookback_months :]
        if window.iloc[-1] > window.mean():
            return {self.risk: 1.0, self.cash: 0.0}
        return {self.risk: 0.0, self.cash: 1.0}
