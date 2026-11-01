"""The largest stocks of a point-in-time index by market value, equal weight (rules block `topcap`, D-044).

    universe:
      market: KR_STOCK
      index: KRTOP300           # needs the market values (events `kr_mktcap`, kr_dataset caps=True)
      symbols: ["069500"]
    rules:
      kind: declarative
      topcap:
        months: [1, 2, ..., 12] # picked at the close of the last session of these months
        park: "069500"          # held until the first pick
    parameters:
      top_n: {default: 10}      # how many of the largest members are held, 1/top_n each

At the close of the last session of each listed month (and at the first session with market values),
the index members that traded that day are ranked by their market value at that close; the `top_n`
largest are held at 1/top_n each from the next open (holdings are brought back to equal weight), and
everything else is sold. Market values at row t are those of closes up to t only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quantlab.strategies.base import MarketView, PartialWeights, Strategy
from quantlab.strategies.stock_combo import _adjusted

KEYS = ("months", "park")
KIND = "kr_mktcap"


def validate_topcap(cfg: Any, symbols: list[str]) -> list[str]:
    if not isinstance(cfg, dict):
        return ["rules.topcap must be a mapping"]
    errs = []
    missing = [k for k in KEYS if k not in cfg]
    if missing:
        errs.append(f"rules.topcap needs {missing}")
    extra = set(cfg) - set(KEYS)
    if extra:
        errs.append(f"rules.topcap: unknown keys {sorted(extra)}")
    m = cfg.get("months")
    if not isinstance(m, list) or not m or not all(isinstance(x, int) and 1 <= x <= 12 for x in m):
        errs.append("rules.topcap.months must be a non-empty list of months 1-12")
    if cfg.get("park") is not None and cfg.get("park") not in symbols:
        errs.append("rules.topcap.park must be one of universe.symbols")
    return errs


def build_panels(data) -> dict:
    """Market value on the sessions it was recorded (sessions x index stocks), membership and close."""
    syms = [s for s in data.members.columns if s in data.frames] if data.members is not None else []
    T, N = len(data.calendar), len(syms)
    C = np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        C[:, j] = _adjusted(data.values(s))[1]
    cap = np.full((T, N), np.nan)
    ev = data.events
    if ev is not None and len(ev) and N:
        e = ev[ev["kind"] == KIND]
        col = {s: j for j, s in enumerate(syms)}
        rows = data.calendar.get_indexer(e["known"])
        for i, s, v in zip(rows, e["symbol"], e["value"]):
            if i >= 0 and s in col:
                cap[i, col[s]] = v
    member = data.members.reindex(columns=syms).fillna(False).to_numpy(dtype=bool) if N else np.zeros((T, 0), bool)
    return {"symbols": tuple(syms), "close": C, "cap": cap, "member": member}


class TopCapStrategy(Strategy):
    def __init__(self, *, spec_id: str, spec_version_str: str, rules: dict, **params: Any) -> None:
        super().__init__(**params)
        self.strategy_id, self.version = spec_id, spec_version_str
        self.cfg = dict(rules["topcap"])
        self.top_n = int(params.get("top_n", 10))
        if self.top_n < 1:
            raise ValueError("topcap top_n must be >= 1")
        self.park_symbol = self.cfg.get("park")
        self.held: set[str] = set()
        self.started = False
        self.picked = False

    def describe(self) -> dict:
        return {"strategy_id": self.strategy_id, "version": self.version, "params": self.params,
                "rules": {"topcap": self.cfg}}

    def _pick(self, view: MarketView) -> list[str]:
        syms = view.panel_columns(build_panels)
        cap = view.panel(build_panels, "cap")
        ok = view.panel(build_panels, "member") & np.isfinite(cap) & np.isfinite(view.panel(build_panels, "close"))
        idx = np.flatnonzero(ok)
        if len(idx) < self.top_n:
            return []
        order = np.lexsort((np.array([syms[j] for j in idx]), -cap[idx]))  # largest first, then symbol
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
