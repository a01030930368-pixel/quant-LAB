"""Insider purchase clusters on a point-in-time stock index (rules block `insider`, D-026).

    universe:
      market: US_STOCK
      index: SP500            # stock symbols, membership and events come from the index dataset
      symbols: [SPY, BIL]     # non-index symbols the rule or the policy needs
    rules:
      kind: declarative
      insider:
        roles: [officer, director]   # who counts as an insider
        min_value: 10000             # an insider's open-market purchases in one filing, USD
        min_insiders: 2              # distinct insiders ...
        window_days: 30              # ... whose filings became public within this many calendar days
        hold_days: 20                # sessions from the entry fill to the exit fill
        slots: 10                    # each position holds 1/slots of equity
        park: SPY                    # equity not in a position is held here

A signal fires at the close of the session on which the count of distinct insiders reaches
`min_insiders` (it was below it before that session's filings), for a stock that is an index member
that day and is not already held. It fills at the next open. When more stocks signal than there are
free slots, the larger total purchase value in the window goes first (then the symbol, for a fixed
order). Events are read through `MarketView.events`, which shows nothing dated after the bar.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quantlab.strategies.base import MarketView, PartialWeights, Strategy

ROLES = ("officer", "director")
KEYS = ("roles", "min_value", "min_insiders", "window_days", "hold_days", "slots", "park")
INDEXES = ("SP500",)


def validate_insider(cfg: Any, symbols: list[str]) -> list[str]:
    if not isinstance(cfg, dict):
        return ["rules.insider must be a mapping"]
    errs = []
    missing = [k for k in KEYS if k not in cfg]
    if missing:
        errs.append(f"rules.insider needs {missing}")
    extra = set(cfg) - set(KEYS)
    if extra:
        errs.append(f"rules.insider: unknown keys {sorted(extra)}")
    roles = cfg.get("roles")
    if not isinstance(roles, list) or not roles or not set(roles) <= set(ROLES):
        errs.append(f"rules.insider.roles must be a non-empty subset of {list(ROLES)}")
    for k in ("min_insiders", "window_days", "hold_days", "slots"):
        v = cfg.get(k)
        if k in cfg and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
            errs.append(f"rules.insider.{k} must be a whole number >= 1")
    v = cfg.get("min_value")
    if "min_value" in cfg and (not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0):
        errs.append("rules.insider.min_value must be a number >= 0")
    if cfg.get("park") is not None and cfg.get("park") not in symbols:
        errs.append("rules.insider.park must be one of universe.symbols")
    return errs


class InsiderClusterStrategy(Strategy):
    def __init__(self, *, spec_id: str, spec_version_str: str, rules: dict, **params: Any) -> None:
        super().__init__(**params)
        self.strategy_id, self.version = spec_id, spec_version_str
        self.cfg = dict(rules["insider"])
        self.park_symbol = self.cfg.get("park")  # idle equity, not a trade of the rule
        self.held: dict[str, int] = {}  # symbol -> sessions since its entry signal
        self.started = False

    def describe(self) -> dict:
        return {"strategy_id": self.strategy_id, "version": self.version, "params": self.params,
                "rules": {"insider": self.cfg}}

    def _signals(self, view: MarketView) -> list[tuple[float, str]]:
        c = self.cfg
        ev = view.event_columns(int(c["window_days"]), ("symbol", "owner", "value", "kind") + tuple(c["roles"]))
        if not len(ev["known"]) or not {"symbol", "owner", "value"} <= set(ev):
            return []
        prev = view.previous_session()
        today = ev["known"] > np.datetime64(prev, "ns") if prev is not None else np.ones(len(ev["known"]), bool)
        if not today.any():
            return []
        ok = ev["value"].astype(float) >= float(c["min_value"])
        role = np.zeros(len(ok), dtype=bool)
        for r in c["roles"]:
            if r in ev:
                role |= ev[r].astype(bool)
        ok &= role
        if "kind" in ev:
            ok &= ev["kind"] == "insider_buy"
        if not (ok & today).any():
            return []
        k = int(c["min_insiders"])
        sym, owner, value = ev["symbol"], ev["owner"], ev["value"].astype(float)
        out = []
        for s in sorted(set(sym[ok & today])):
            if s in self.held or s not in view.symbols or not view.is_member(s):
                continue
            mine = ok & (sym == s)
            if len(set(owner[mine])) >= k and len(set(owner[mine & ~today])) < k:
                px = view.tail_values(s, 1)
                if len(px) and np.isfinite(px[-1, 3]) and px[-1, 3] > 0:
                    out.append((float(value[mine].sum()), s))
        return out

    def on_bar(self, view: MarketView):
        c = self.cfg
        slots, park = int(c["slots"]), c.get("park")
        exits = []
        for s in list(self.held):
            self.held[s] += 1
            if self.held[s] >= int(c["hold_days"]):
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
