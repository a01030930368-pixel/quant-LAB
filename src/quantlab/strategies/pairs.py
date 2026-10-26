"""Sector-against-market pairs held long only (rules block `pairs`, D-041).

    rules:
      kind: declarative
      pairs:
        trade: ["091160", "091170", ...]  # sector funds
        base: "069500"                    # the market fund the spread is measured against
        hedge: "114800"                   # 1x inverse of the market, bought instead of selling short
        leg_weight: 0.25                  # each leg's share of equity while a pair is open
        slots: 2                          # pairs open at once
    parameters:
      lookback: {default: 60}   # sessions behind the spread's mean and standard deviation (20 or 60)
      entry_z: {default: 2.0}   # open when the spread is this many standard deviations below its mean
      max_hold: {default: 20}   # close after this many sessions if the spread has not come back

The spread of a sector fund is log(fund close) - log(base close); z is its distance from the mean of
the last `lookback` sessions (today included) in standard deviations. At a close, a fund whose z is at
or below -entry_z opens a pair from the next open: `leg_weight` of equity in the fund and the same in
the hedge, so the pair earns roughly fund minus market (Gatev, Goetzmann & Rouwenhorst 2006, the
distance method, with the market fund as every pair's partner). Most negative z first, then symbol.
A pair closes when z is back at 0 or above, or `max_hold` sessions after the close that opened it.
Equity no pair uses stays in cash. Values at row t use closes up to t only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantlab.strategies.base import MarketView, PartialWeights, Strategy
from quantlab.strategies.stock_combo import _adjusted

KEYS = ("trade", "base", "hedge", "leg_weight", "slots")
LOOKBACKS = (20, 60)


def validate_pairs(cfg: Any, symbols: list[str]) -> list[str]:
    if not isinstance(cfg, dict):
        return ["rules.pairs must be a mapping"]
    errs = []
    missing = [k for k in KEYS if k not in cfg]
    if missing:
        errs.append(f"rules.pairs needs {missing}")
    extra = set(cfg) - set(KEYS)
    if extra:
        errs.append(f"rules.pairs: unknown keys {sorted(extra)}")
    trade = cfg.get("trade")
    if not isinstance(trade, list) or not trade or set(trade) - set(symbols):
        errs.append("rules.pairs.trade must be a non-empty list of universe.symbols")
    for k in ("base", "hedge"):
        if k in cfg and (cfg[k] not in symbols or (isinstance(trade, list) and cfg[k] in trade)):
            errs.append(f"rules.pairs.{k} must be one of universe.symbols and not a traded fund")
    if cfg.get("base") is not None and cfg.get("base") == cfg.get("hedge"):
        errs.append("rules.pairs.base and hedge must differ")
    slots = cfg.get("slots")
    if "slots" in cfg and (not isinstance(slots, int) or isinstance(slots, bool) or slots < 1):
        errs.append("rules.pairs.slots must be a whole number >= 1")
    v = cfg.get("leg_weight")
    if "leg_weight" in cfg and (not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0):
        errs.append("rules.pairs.leg_weight must be a positive number")
    elif isinstance(slots, int) and isinstance(v, (int, float)) and 2 * slots * v > 1 + 1e-9:
        errs.append("rules.pairs: 2 * slots * leg_weight must be at most 1")
    return errs


def build_panels(data, trade: tuple[str, ...], base: str) -> dict:
    """z of each sector fund's spread against `base` (sessions x funds) for each lookback."""
    T = len(data.calendar)
    C = np.full((T, len(trade)), np.nan)
    for j, s in enumerate(trade):
        if s in data.frames:
            C[:, j] = _adjusted(data.values(s))[1]
    b = _adjusted(data.values(base))[1] if base in data.frames else np.full(T, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = pd.DataFrame(np.log(C) - np.log(b)[:, None])
    p: dict[str, Any] = {"symbols": tuple(trade), "close": C}
    for n in LOOKBACKS:
        roll = spread.rolling(n, min_periods=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = ((spread - roll.mean()) / roll.std()).to_numpy()
        p[f"z{n}"] = np.where(np.isfinite(z), z, np.nan)
    return p


class PairsStrategy(Strategy):
    def __init__(self, *, spec_id: str, spec_version_str: str, rules: dict, **params: Any) -> None:
        super().__init__(**params)
        self.strategy_id, self.version = spec_id, spec_version_str
        self.cfg = dict(rules["pairs"])
        self.trade = tuple(self.cfg["trade"])
        self.lookback = int(params.get("lookback", 60))
        self.entry_z = float(params.get("entry_z", 2.0))
        self.max_hold = int(params.get("max_hold", 20))
        if self.lookback not in LOOKBACKS:
            raise ValueError(f"pairs lookback must be one of {LOOKBACKS}")
        if self.entry_z <= 0 or self.max_hold < 1:
            raise ValueError("pairs entry_z must be > 0 and max_hold >= 1")
        self.held: dict[str, int] = {}  # fund -> sessions since the close that opened its pair

        trade, base = self.trade, self.cfg["base"]

        def panels(data):
            return build_panels(data, trade, base)

        panels.__qualname__ = f"pairs_panels_{base}_{'_'.join(trade)}"  # one cache entry per fund list
        self._panels = panels

    def describe(self) -> dict:
        return {"strategy_id": self.strategy_id, "version": self.version, "params": self.params,
                "rules": {"pairs": self.cfg}}

    def on_bar(self, view: MarketView):
        z = view.panel(self._panels, f"z{self.lookback}")
        if z is None:
            return None
        col = {s: j for j, s in enumerate(self.trade)}
        hedge = self.cfg["hedge"]
        leg = float(self.cfg["leg_weight"])
        before = len(self.held)
        w = PartialWeights()
        for s in sorted(self.held):
            self.held[s] += 1
            zs = z[col[s]]
            if (np.isfinite(zs) and zs >= 0) or self.held[s] >= self.max_hold or not np.isfinite(zs):
                w[s] = 0.0
                del self.held[s]
        free = int(self.cfg["slots"]) - len(self.held)
        hp = view.tail_values(hedge, 1)
        hedge_ok = len(hp) and np.isfinite(hp[-1, 3]) and hp[-1, 3] > 0
        if free > 0 and hedge_ok:
            cand = [(z[j], s) for s, j in col.items()
                    if s not in self.held and s not in w and np.isfinite(z[j]) and z[j] <= -self.entry_z]
            for _, s in sorted(cand)[:free]:
                w[s] = leg
                self.held[s] = 0
        if len(self.held) != before or w:
            w[hedge] = leg * len(self.held)
        return w or None
