"""Follow institutions' 5% holding reports on a point-in-time stock index (rules block `block`, D-047).

    universe:
      market: KR_STOCK
      index: KRTOP300            # needs the 5% report events (kr_dataset block=True)
      symbols: ["069500"]
    rules:
      kind: declarative
      block:
        kinds: [kr_block_new]    # event kinds that count (quantlab.data.dart_block.KINDS)
        min_change: 0.0          # the report's increase in holding, %p, at least this
        slots: 10                # each position holds 1/slots of equity
        park: "069500"           # equity not in a position is held here
    parameters:
      hold_days: {default: 20}   # sessions from the entry fill to the exit fill

A signal fires at the close of the session on which a counting report became public (received after
the previous session), for a stock that is an index member that day, has a close and is not already
held; it fills at the next open. When more stocks signal than there are free slots, the larger increase
goes first (then the symbol, for a fixed order). A stock whose exit day brings a new counting report
is kept for another `hold_days` (the exit and the new entry would cancel out). Events are read through `MarketView.events`, which
shows nothing dated after the bar.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quantlab.strategies.base import MarketView, PartialWeights, Strategy

KEYS = ("kinds", "min_change", "slots", "park")
KINDS = ("kr_block_buy", "kr_block_new")


def validate_block(cfg: Any, symbols: list[str]) -> list[str]:
    if not isinstance(cfg, dict):
        return ["rules.block must be a mapping"]
    errs = []
    missing = [k for k in KEYS if k not in cfg]
    if missing:
        errs.append(f"rules.block needs {missing}")
    extra = set(cfg) - set(KEYS)
    if extra:
        errs.append(f"rules.block: unknown keys {sorted(extra)}")
    kinds = cfg.get("kinds")
    if not isinstance(kinds, list) or not kinds or not set(kinds) <= set(KINDS):
        errs.append(f"rules.block.kinds must be a non-empty subset of {list(KINDS)}")
    v = cfg.get("slots")
    if "slots" in cfg and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
        errs.append("rules.block.slots must be a whole number >= 1")
    v = cfg.get("min_change")
    if "min_change" in cfg and (not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0):
        errs.append("rules.block.min_change must be a number >= 0")
    if cfg.get("park") is not None and cfg.get("park") not in symbols:
        errs.append("rules.block.park must be one of universe.symbols")
    return errs


class BlockHoldingStrategy(Strategy):
    def __init__(self, *, spec_id: str, spec_version_str: str, rules: dict, **params: Any) -> None:
        super().__init__(**params)
        self.strategy_id, self.version = spec_id, spec_version_str
        self.cfg = dict(rules["block"])
        self.hold_days = int(params.get("hold_days", 20))
        if self.hold_days < 1:
            raise ValueError("block hold_days must be >= 1")
        self.park_symbol = self.cfg.get("park")
        self.excess_benchmark = self.park_symbol   # each trade is also compared with the park held as long
        self.held: dict[str, int] = {}
        self.started = False

    def describe(self) -> dict:
        return {"strategy_id": self.strategy_id, "version": self.version, "params": self.params,
                "rules": {"block": self.cfg}}

    def _signals(self, view: MarketView) -> list[tuple[float, str]]:
        ev = view.event_columns(10, ("symbol", "kind", "value"))
        if not len(ev["known"]) or not {"symbol", "kind", "value"} <= set(ev):
            return []
        prev = view.previous_session()
        today = ev["known"] > np.datetime64(prev, "ns") if prev is not None else np.ones(len(ev["known"]), bool)
        value = ev["value"].astype(float)
        ok = today & np.isin(ev["kind"], list(self.cfg["kinds"])) & (value >= float(self.cfg["min_change"]))
        best: dict[str, float] = {}
        for s, v in zip(ev["symbol"][ok], value[ok]):
            best[s] = max(best.get(s, -np.inf), v)
        out = []
        for s, v in best.items():
            if s in self.held or s not in view.symbols or not view.is_member(s):
                continue
            px = view.tail_values(s, 1)
            if len(px) and np.isfinite(px[-1, 3]) and px[-1, 3] > 0:
                out.append((v, s))
        return out

    def on_bar(self, view: MarketView):
        slots, park = int(self.cfg["slots"]), self.cfg.get("park")
        exits = []
        for s in list(self.held):
            self.held[s] += 1
            if self.held[s] >= self.hold_days:
                exits.append(s)
                del self.held[s]
        free = slots - len(self.held)
        picks = sorted(self._signals(view), key=lambda x: (-x[0], x[1]))[:max(free, 0)]
        w = PartialWeights()
        for s in exits:
            w[s] = 0.0
        for _, s in picks:
            w[s] = 1.0 / slots
            self.held[s] = 0
        if park and (exits or picks or not self.started):
            w[park] = max(0.0, 1.0 - len(self.held) / slots)
        self.started = True
        return w or None
