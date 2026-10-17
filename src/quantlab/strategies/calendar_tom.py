"""Turn-of-the-month calendar rule (rules block `calendar`, D-032).

    rules:
      kind: declarative
      calendar:
        trade: [SOXL, TNA, FAS, ERX, TECL, DRN, CURE]
        leg_weight: 0.0714   # each fund's share of equity inside the window
        enter_before_end: 1  # hold from the open of the last session of the month ...
        exit_session: 4      # ... to the open of the 4th session of the next month
        park: BIL            # held all the time with the equity the window never uses

The window covers the last session of a month and the first three of the next (Lakonishok & Smidt
1988; McConnell & Xu 2008). Decisions are read at a close and fill at the next open, so the buy is
decided at the close of the second-to-last session and the sale at the close of the third. Whether a
coming session is the last of its month comes from the NYSE schedule, which is published in advance.

Every fund with a price that session gets `leg_weight`; a fund that has not listed yet is skipped and
its share stays in cash. `park` holds 1 - len(trade) * leg_weight from the first session and is never
traded again, so the monthly window does not pay costs on it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from quantlab.strategies.base import MarketView, PartialWeights, Strategy

KEYS = ("trade", "leg_weight", "enter_before_end", "exit_session", "park")


@lru_cache(maxsize=512)
def _month_sessions(year: int, month: int) -> tuple[pd.Timestamp, ...]:
    from quantlab.data.quality import exchange_sessions

    start = pd.Timestamp(year=year, month=month, day=1)
    return tuple(exchange_sessions(start, start + pd.offsets.MonthEnd(0)))


def window_position(day: pd.Timestamp) -> tuple[int, int]:
    """(n, k) for a session: its number in the month counted from the start (1 = first) and from the
    end (1 = last), on the NYSE schedule. A date the schedule does not list (synthetic test data)
    counts as the next listed session."""
    sessions = _month_sessions(day.year, day.month)
    i = int(np.searchsorted(np.array(sessions, dtype="datetime64[ns]"), np.datetime64(pd.Timestamp(day).normalize(), "ns")))
    return i + 1, len(sessions) - i


def validate_calendar(cfg: Any, symbols: list[str]) -> list[str]:
    if not isinstance(cfg, dict):
        return ["rules.calendar must be a mapping"]
    errs = []
    missing = [k for k in KEYS if k not in cfg and k != "park"]
    if missing:
        errs.append(f"rules.calendar needs {missing}")
    extra = set(cfg) - set(KEYS)
    if extra:
        errs.append(f"rules.calendar: unknown keys {sorted(extra)}")
    trade = cfg.get("trade")
    if not isinstance(trade, list) or not trade or set(trade) - set(symbols):
        errs.append("rules.calendar.trade must be a non-empty list of universe.symbols")
    for k in ("enter_before_end", "exit_session"):
        v = cfg.get(k)
        if k in cfg and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
            errs.append(f"rules.calendar.{k} must be a whole number >= 1")
    v = cfg.get("leg_weight")
    if "leg_weight" in cfg and (not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0):
        errs.append("rules.calendar.leg_weight must be a positive number")
    elif isinstance(trade, list) and isinstance(v, (int, float)) and len(trade) * v > 1 + 1e-9:
        errs.append("rules.calendar: len(trade) * leg_weight must be at most 1")
    park = cfg.get("park")
    if park is not None and (park not in symbols or (isinstance(trade, list) and park in trade)):
        errs.append("rules.calendar.park must be one of universe.symbols and not a traded fund")
    return errs


class CalendarStrategy(Strategy):
    def __init__(self, *, spec_id: str, spec_version_str: str, rules: dict, **params: Any) -> None:
        super().__init__(**params)
        self.strategy_id, self.version = spec_id, spec_version_str
        self.cfg = dict(rules["calendar"])
        self.trade = list(self.cfg["trade"])
        self.park_symbol = self.cfg.get("park")  # idle equity, not a trade of the rule
        self.inside = False
        self.parked = False

    def describe(self) -> dict:
        return {"strategy_id": self.strategy_id, "version": self.version, "params": self.params,
                "rules": {"calendar": self.cfg}}

    def _next_sessions(self, now: pd.Timestamp, n: int) -> list[pd.Timestamp]:
        out: list[pd.Timestamp] = []
        y, m = now.year, now.month
        while len(out) < n:
            out += [s for s in _month_sessions(y, m) if s > now]
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return out[:n]

    def on_bar(self, view: MarketView):
        now = pd.Timestamp(view.now).normalize()
        nxt = self._next_sessions(now, 1)[0]
        _, from_end = window_position(nxt)
        n_now, _ = window_position(now)
        w = PartialWeights()
        if self.park_symbol and not self.parked:
            w[self.park_symbol] = max(0.0, 1.0 - len(self.trade) * float(self.cfg["leg_weight"]))
            self.parked = True
        enter = from_end == int(self.cfg["enter_before_end"])
        leave = window_position(nxt)[0] == int(self.cfg["exit_session"])
        if self.inside and leave:
            for s in self.trade:
                w[s] = 0.0
            self.inside = False
        elif not self.inside and enter:
            for s in self.trade:
                px = view.tail_values(s, 1)
                if len(px) and np.isfinite(px[-1, 3]) and px[-1, 3] > 0:
                    w[s] = float(self.cfg["leg_weight"])
            self.inside = True
        return w or None
