"""Post-earnings drift on a point-in-time stock index (rules block `earnings`, D-027).

    rules:
      kind: declarative
      earnings:
        min_abnormal: 0.05   # stock return minus SPY return over the reaction window, at least this
        hold_days: 30        # sessions from the entry fill to the exit fill
        slots: 20            # each position holds 1/slots of equity
        park: SPY            # equity not in a position is held here
        park_band: 2         # SPY is bought back only once this many slots' worth of cash is idle

Events of kind "earnings" carry `known` = the filing date of an 8-K with item 2.02. The reaction
window runs from the close of the last session before the filing date to the close of the first
session after it, so it holds the market's response whether the release came before the open,
during the session or after the close. The signal is read at that close and fills at the next open.
Only index members that day, not already held; with more signals than free slots, the larger
abnormal return goes first (then the symbol).

SPY parking: a new position is funded by selling SPY only when idle cash does not cover it; cash
from exits is put back into SPY only once `park_band` slots' worth is idle. This keeps the number of
SPY trades (each one paying full costs) well below one per position opened or closed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantlab.strategies.base import MarketView, PartialWeights, Strategy

KEYS = ("min_abnormal", "hold_days", "slots", "park", "park_band")
BENCHMARK = "SPY"


def validate_earnings(cfg: Any, symbols: list[str]) -> list[str]:
    if not isinstance(cfg, dict):
        return ["rules.earnings must be a mapping"]
    errs = []
    missing = [k for k in KEYS if k not in cfg]
    if missing:
        errs.append(f"rules.earnings needs {missing}")
    extra = set(cfg) - set(KEYS)
    if extra:
        errs.append(f"rules.earnings: unknown keys {sorted(extra)}")
    for k in ("hold_days", "slots", "park_band"):
        v = cfg.get(k)
        if k in cfg and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
            errs.append(f"rules.earnings.{k} must be a whole number >= 1")
    v = cfg.get("min_abnormal")
    if "min_abnormal" in cfg and (not isinstance(v, (int, float)) or isinstance(v, bool)):
        errs.append("rules.earnings.min_abnormal must be a number")
    if cfg.get("park") is not None and cfg.get("park") not in symbols:
        errs.append("rules.earnings.park must be one of universe.symbols")
    if BENCHMARK not in symbols:
        errs.append(f"earnings rules measure the reaction against {BENCHMARK}: add it to universe.symbols")
    return errs


class EarningsDriftStrategy(Strategy):
    def __init__(self, *, spec_id: str, spec_version_str: str, rules: dict, **params: Any) -> None:
        super().__init__(**params)
        self.strategy_id, self.version = spec_id, spec_version_str
        self.cfg = dict(rules["earnings"])
        self.park_symbol = self.cfg.get("park")  # idle equity, not a trade of the rule
        self.excess_benchmark = BENCHMARK        # each trade is also compared with SPY held for the same days
        self.held: dict[str, int] = {}
        self.park_w: float | None = None

    def describe(self) -> dict:
        return {"strategy_id": self.strategy_id, "version": self.version, "params": self.params,
                "rules": {"earnings": self.cfg}}

    @staticmethod
    def _window_return(view: MarketView, sym: str, back: int) -> float | None:
        px = view.tail_values(sym, back + 1)
        if len(px) < back + 1:
            return None
        a, b = px[0, 3], px[-1, 3]
        return float(b / a - 1) if np.isfinite(a) and np.isfinite(b) and a > 0 else None

    def _signals(self, view: MarketView) -> list[tuple[float, str]]:
        prev, now = view.previous_session(), view.now
        if prev is None or view.session(2) is None:
            return []
        ev = view.event_columns((now - prev).days + 1, ("symbol", "kind"))
        if not len(ev["known"]) or "kind" not in ev:
            return []
        known = ev["known"]
        take = (ev["kind"] == "earnings") & (known >= np.datetime64(prev, "ns")) & (known < np.datetime64(now, "ns"))
        out = []
        for k in np.flatnonzero(take):
            sym = ev["symbol"][k]
            if sym in self.held or sym not in view.symbols or not view.is_member(sym):
                continue
            back = 2 if known[k] == np.datetime64(prev, "ns") else 1  # base: last close before the filing date
            r, m = self._window_return(view, sym, back), self._window_return(view, BENCHMARK, back)
            if r is None or m is None:
                continue
            if r - m >= float(self.cfg["min_abnormal"]):
                out.append((r - m, sym))
        return sorted(set(out), key=lambda x: (-x[0], x[1]))

    def on_bar(self, view: MarketView):
        c = self.cfg
        slots, park = int(c["slots"]), c.get("park")
        exits = []
        for s in list(self.held):
            self.held[s] += 1
            if self.held[s] >= int(c["hold_days"]):
                exits.append(s)
                del self.held[s]
        picks = self._signals(view)[:max(slots - len(self.held), 0)]
        w = PartialWeights()
        for s in exits:
            w[s] = 0.0
        for _, s in picks:
            w[s] = 1.0 / slots
            self.held[s] = 0
        if park:
            room = 1.0 - len(self.held) / slots  # what SPY may hold with every position funded
            if self.park_w is None:
                self.park_w = room
                w[park] = room
            elif self.park_w > room + 1e-9:
                self.park_w = room
                w[park] = room
            elif room - self.park_w >= int(c["park_band"]) / slots - 1e-9:
                self.park_w = room
                w[park] = room
        return w or None
