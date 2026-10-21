"""Low-volatility stocks of a point-in-time index, held for a quarter (rules block `lowvol`, D-035).

    universe:
      market: US_STOCK
      index: SP500
      symbols: [SPY, BIL]
    rules:
      kind: declarative
      lowvol:
        months: [3, 6, 9, 12]   # picked at the close of the last session of these months
        park: SPY               # held until the first pick (not enough history yet)
    parameters:
      lookback: {default: 252}  # sessions of daily returns behind each stock's volatility (126 or 252)
      top_n: {default: 50}      # how many of the calmest members are held, equal weight

At the close of the last session of each listed month (and at the first session with enough history),
the index members with a full `lookback` of returns are ranked by the standard deviation of their
daily returns; the `top_n` lowest are held at 1/top_n each from the next open, and everything else is
sold. A stock that leaves the index is kept until the next pick. Volatility at row t uses closes up to
t only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantlab.strategies.base import MarketView, PartialWeights, Strategy
from quantlab.strategies.stock_combo import _adjusted

KEYS = ("months", "park")
LOOKBACKS = (126, 252)


def validate_lowvol(cfg: Any, symbols: list[str]) -> list[str]:
    if not isinstance(cfg, dict):
        return ["rules.lowvol must be a mapping"]
    errs = []
    missing = [k for k in KEYS if k not in cfg]
    if missing:
        errs.append(f"rules.lowvol needs {missing}")
    extra = set(cfg) - set(KEYS)
    if extra:
        errs.append(f"rules.lowvol: unknown keys {sorted(extra)}")
    m = cfg.get("months")
    if not isinstance(m, list) or not m or not all(isinstance(x, int) and 1 <= x <= 12 for x in m):
        errs.append("rules.lowvol.months must be a non-empty list of months 1-12")
    if cfg.get("park") is not None and cfg.get("park") not in symbols:
        errs.append("rules.lowvol.park must be one of universe.symbols")
    return errs


def build_panels(data) -> dict:
    """Daily-return volatility (sessions x index stocks) for each lookback, membership and close."""
    syms = [s for s in data.members.columns if s in data.frames] if data.members is not None else []
    T, N = len(data.calendar), len(syms)
    C = np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        C[:, j] = _adjusted(data.values(s))[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = pd.DataFrame(np.log(C[1:] / C[:-1]))
    p: dict[str, Any] = {"symbols": tuple(syms), "close": C}
    for n in LOOKBACKS:
        v = np.full((T, N), np.nan)
        v[1:] = r.rolling(n, min_periods=n).std().to_numpy()
        p[f"vol{n}"] = v
    member = data.members.reindex(columns=syms).fillna(False).to_numpy(dtype=bool) if N else np.zeros((T, 0), bool)
    p["member"] = member
    return p


class LowVolStrategy(Strategy):
    def __init__(self, *, spec_id: str, spec_version_str: str, rules: dict, **params: Any) -> None:
        super().__init__(**params)
        self.strategy_id, self.version = spec_id, spec_version_str
        self.cfg = dict(rules["lowvol"])
        self.lookback, self.top_n = int(params.get("lookback", 252)), int(params.get("top_n", 50))
        if self.lookback not in LOOKBACKS:
            raise ValueError(f"lowvol lookback must be one of {LOOKBACKS}")
        if self.top_n < 1:
            raise ValueError("lowvol top_n must be >= 1")
        self.park_symbol = self.cfg.get("park")  # held only before the first pick
        self.held: set[str] = set()
        self.started = False
        self.picked = False

    def describe(self) -> dict:
        return {"strategy_id": self.strategy_id, "version": self.version, "params": self.params,
                "rules": {"lowvol": self.cfg}}

    def _pick(self, view: MarketView) -> list[str]:
        syms = view.panel_columns(build_panels)
        vol = view.panel(build_panels, f"vol{self.lookback}")
        ok = view.panel(build_panels, "member") & np.isfinite(vol) & np.isfinite(view.panel(build_panels, "close"))
        idx = np.flatnonzero(ok)
        if len(idx) < self.top_n:
            return []
        order = np.lexsort((np.array([syms[j] for j in idx]), vol[idx]))  # calmest first, then symbol
        return [syms[j] for j in idx[order[:self.top_n]]]

    def on_bar(self, view: MarketView):
        w = PartialWeights()
        park = self.park_symbol
        if not self.started and park:
            w[park] = 1.0
        self.started = True
        due = (view.is_last_session_of_month() and view.now.month in self.cfg["months"]) or not self.picked
        if not due:
            return w or None
        picks = self._pick(view)
        if not picks:
            return w or None
        if not self.picked and park:
            w[park] = 0.0
        self.picked = True
        for s in self.held - set(picks):
            w[s] = 0.0
        for s in picks:
            w[s] = 1.0 / self.top_n
        self.held = set(picks)
        return w
