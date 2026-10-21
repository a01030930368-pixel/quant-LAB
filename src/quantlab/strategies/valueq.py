"""Cheap and profitable stocks of a point-in-time index, held for a quarter (rules block `value`, D-037).

    universe:
      market: US_STOCK
      index: SP500
      symbols: [SPY, BIL]
    rules:
      kind: declarative
      value:
        months: [3, 6, 9, 12]   # picked at the close of the last session of these months
        park: SPY               # held until the first pick
    parameters:
      top_n: {default: 50}      # how many of the best-scored members are held, equal weight

Score (Novy-Marx 2013, value plus profitability): the percentile rank of book-to-market plus the
percentile rank of gross profit to assets, among index members that have both on the pick date.
    book-to-market  = stockholders' equity / market value
    market value    = public float on its date x the stock's price change since then
    gross profit    = last fiscal year's; assets and equity = latest balance sheet
Statement values come from SEC filings (quantlab.data.sec_facts) known by the pick date; a value whose
period ended more than STALE_DAYS before the pick is not used. Members with negative equity, no gross
profit line (banks, insurers) or no public float are not ranked.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quantlab.strategies.base import MarketView
from quantlab.strategies.lowvol import LowVolStrategy
from quantlab.strategies.stock_combo import _adjusted

STALE_DAYS = 500
FACT_KINDS = ("fact_assets", "fact_equity", "fact_gross_profit", "fact_float")


def validate_value(cfg: Any, symbols: list[str]) -> list[str]:
    from quantlab.strategies.lowvol import validate_lowvol

    return [e.replace("rules.lowvol", "rules.value") for e in validate_lowvol(cfg, symbols)]


def _latest(ev, kind: str, col: dict, cal64: np.ndarray, T: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Per session and stock: the value of `kind` with the latest period end among those known by that
    session (a later filing for the same period wins), and that period end."""
    val = np.full((T, N), np.nan)
    end = np.full((T, N), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    e = ev[ev["kind"] == kind]
    if not len(e):
        return val, end
    known = e["known"].to_numpy(dtype="datetime64[ns]")
    ends = e["period_end"].to_numpy(dtype="datetime64[ns]")
    vals = e["value"].to_numpy(dtype=float)
    syms = e["symbol"].to_numpy()
    rows = np.searchsorted(cal64, known, side="left")  # first session on or after the known day
    cur: dict[int, tuple] = {}
    changes: dict[int, list] = {}
    for i in np.lexsort((ends, known)):
        j = col.get(syms[i])
        if j is None or rows[i] >= T:
            continue
        old = cur.get(j)
        if old is None or ends[i] >= old[0]:
            cur[j] = (ends[i], vals[i])
            changes.setdefault(j, []).append((rows[i], ends[i], vals[i]))
    for j, ch in changes.items():
        for k, (t0, en, v) in enumerate(ch):
            t1 = ch[k + 1][0] if k + 1 < len(ch) else T
            val[t0:t1, j] = v
            end[t0:t1, j] = en
    return val, end


def build_panels(data) -> dict:
    """Book-to-market and gross profit to assets (sessions x index stocks), membership and close.
    Row t uses prices up to t and filings known by t only."""
    cal = data.calendar
    syms = [s for s in data.members.columns if s in data.frames] if data.members is not None else []
    T, N = len(cal), len(syms)
    cal64 = cal.to_numpy(dtype="datetime64[ns]")
    col = {s: j for j, s in enumerate(syms)}
    C = np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        C[:, j] = _adjusted(data.values(s))[1]
    ev = data.events
    ev = ev[ev["kind"].isin(FACT_KINDS)] if ev is not None and "kind" in ev and "period_end" in ev else None
    p: dict[str, Any] = {"symbols": tuple(syms), "close": C}
    p["member"] = data.members.reindex(columns=syms).fillna(False).to_numpy(dtype=bool) if N else np.zeros((T, 0), bool)
    if ev is None or not len(ev):
        p["bm"] = p["gpa"] = np.full((T, N), np.nan)
        return p
    fresh = cal64[:, None] - np.timedelta64(STALE_DAYS, "D")
    got = {}
    for k in FACT_KINDS:
        v, e = _latest(ev, k, col, cal64, T, N)
        v[~(e >= fresh)] = np.nan
        got[k] = (v, e)
    fl, fl_end = got["fact_float"]
    # price at the float date: the last session on or before it (no price before the data starts)
    t_end = np.searchsorted(cal64, fl_end.ravel(), side="right").reshape(T, N) - 1
    ok = np.isfinite(fl) & (t_end >= 0)
    base = np.full((T, N), np.nan)
    tt, jj = np.nonzero(ok)
    base[tt, jj] = C[t_end[tt, jj], jj]
    with np.errstate(invalid="ignore", divide="ignore"):
        mv = fl * C / base
        eq, assets, gp = got["fact_equity"][0], got["fact_assets"][0], got["fact_gross_profit"][0]
        bm = np.where((eq > 0) & (mv > 0), eq / mv, np.nan)
        gpa = np.where(assets > 0, gp / assets, np.nan)
    p["bm"], p["gpa"] = bm, gpa
    return p


def _pct(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


class ValueQualityStrategy(LowVolStrategy):
    def __init__(self, *, spec_id: str, spec_version_str: str, rules: dict, **params: Any) -> None:
        super().__init__(spec_id=spec_id, spec_version_str=spec_version_str, rules={"lowvol": rules["value"]},
                         **({"lookback": 252} | params))
        self.cfg = dict(rules["value"])
        self.params = dict(params)

    def describe(self) -> dict:
        return {"strategy_id": self.strategy_id, "version": self.version, "params": self.params,
                "rules": {"value": self.cfg}}

    def _pick(self, view: MarketView) -> list[str]:
        syms = view.panel_columns(build_panels)
        bm, gpa = view.panel(build_panels, "bm"), view.panel(build_panels, "gpa")
        ok = view.panel(build_panels, "member") & np.isfinite(bm) & np.isfinite(gpa) & np.isfinite(view.panel(build_panels, "close"))
        idx = np.flatnonzero(ok)
        if len(idx) < self.top_n:
            return []
        score = _pct(bm[idx]) + _pct(gpa[idx])
        order = np.lexsort((np.array([syms[j] for j in idx]), -score))  # best first, then symbol
        return [syms[j] for j in idx[order[:self.top_n]]]

