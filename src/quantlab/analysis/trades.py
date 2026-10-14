"""Trade-level checks (descriptive; not gate criteria in policy v1).

The prior project's lessons (handoff 2026-10-14 §6-4) ask for these on every result:
gross (pre-cost) mean per trade, median, mean without the 10 best trades, a t-value with
trades entered on the same day merged into one observation, and performance without the
best calendar year (e.g. 2020 for leveraged rebound strategies).

A trade here is a holding episode of one symbol: from the fill that opens a position until
the fill that brings it back to zero. Size changes inside an episode stay in the episode.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quantlab.backtesting.engine import BacktestResult
from quantlab.core.market_data import MarketData


def episodes(res: BacktestResult, data: MarketData | None = None, start=None, end=None) -> pd.DataFrame:
    """One row per episode entered within [start, end]: dates, sessions held, gross and net return.

    Episodes still open at the end are marked at the last close (needs `data`; otherwise NaN).
    """
    sh = res.shares
    cal = sh.index
    tr = res.trades
    cf = res.cash_flows
    divs = cf[cf["kind"] == "dividend"] if len(cf) else cf
    rows = []
    for sym in sh.columns:
        held = sh[sym].to_numpy() > 1e-12
        t = 0
        while t < len(cal):
            if not held[t]:
                t += 1
                continue
            s = t
            while t < len(cal) and held[t]:
                t += 1
            e = t  # first flat session (exit fill day), or len(cal) if still open
            first, last = cal[s], cal[e] if e < len(cal) else None
            in_ep = (tr["symbol"] == sym) & (tr["date"] >= first) & ((tr["date"] <= last) if last is not None else True)
            ft = tr.loc[in_ep]
            buys, sells = ft[ft["qty"] > 0], ft[ft["qty"] < 0]
            gross_in = float((buys["qty"] * buys["open"]).sum())
            gross_out = float((-sells["qty"] * sells["open"]).sum())
            net_in = float(buys["notional"].sum() + buys["commission"].sum())
            net_out = float(-sells["notional"].sum() - sells["commission"].sum())
            d = divs[(divs["symbol"] == sym) & (divs["date"] >= first) &
                     ((divs["date"] < last) if last is not None else True)]["amount"].sum() if len(divs) else 0.0
            mark = 0.0
            if last is None:  # still open: mark at the last close
                c = cal[-1]
                px = data.frames[sym]["close"].ffill().loc[c] if data is not None else float("nan")
                mark = float(sh.loc[c, sym] * px)
            if gross_in <= 0:
                continue
            rows.append({"symbol": sym, "entry": first, "exit": last, "sessions": e - s,
                         "gross": (gross_out + d + mark) / gross_in - 1, "net": (net_out + d + mark) / net_in - 1,
                         "open_at_end": last is None})
    out = pd.DataFrame(rows, columns=["symbol", "entry", "exit", "sessions", "gross", "net", "open_at_end"])
    if start is not None:
        out = out[out["entry"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["entry"] <= pd.Timestamp(end)]
    return out.sort_values("entry").reset_index(drop=True)


def trade_stats(ep: pd.DataFrame) -> dict:
    ep = ep.dropna(subset=["gross"])
    n = len(ep)
    if n == 0:
        return {"n": 0}
    g = ep["gross"].to_numpy()
    by_day = ep.groupby("entry")["gross"].mean().to_numpy()
    k = len(by_day)
    sd = float(np.std(by_day, ddof=1)) if k > 1 else float("nan")
    top = np.sort(g)[::-1]
    return {
        "n": int(n),
        "entry_days": int(k),
        "mean_gross": float(g.mean()),
        "median_gross": float(np.median(g)),
        "mean_gross_ex_top10": float(top[10:].mean()) if n > 10 else None,
        "mean_net": float(ep["net"].mean()),
        "win_rate_net": float((ep["net"] > 0).mean()),
        "t_day_clustered": float(by_day.mean() / (sd / math.sqrt(k))) if k > 1 and sd > 0 else None,
        "mean_sessions": float(ep["sessions"].mean()),
        "best": float(g.max()),
        "worst": float(g.min()),
    }


def without_best_year(equity: pd.Series) -> dict:
    """CAGR with and without the single best calendar year (by simple return)."""
    r = equity.pct_change().dropna()
    if r.empty:
        return {}
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    best = int(yearly.idxmax())
    rest = r[r.index.year != best]

    def ann(x):
        return float((1 + x).prod() ** (252 / len(x)) - 1) if len(x) else None

    return {"best_year": best, "best_year_return": float(yearly.max()), "cagr_all": ann(r), "cagr_ex_best_year": ann(rest)}
