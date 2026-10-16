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


def not_trades(strategy) -> tuple:
    """Symbols a strategy declares as idle-equity parking (`park_symbol`), excluded from trade stats."""
    park = getattr(strategy, "park_symbol", None)
    return (park,) if park else ()


def episodes(res: BacktestResult, data: MarketData | None = None, start=None, end=None,
             exclude=()) -> pd.DataFrame:
    """One row per episode entered within [start, end]: dates, sessions held, gross and net return.

    `exclude`: symbols whose holdings are not trades of the rule (idle equity parked in SPY).

    Episodes still open at the end are marked at the last close (needs `data`; otherwise NaN).
    """
    sh = res.shares
    cal = sh.index
    tr = res.trades
    cf = res.cash_flows
    divs = cf[cf["kind"] == "dividend"] if len(cf) else cf
    rows = []
    for sym in sh.columns:
        if sym in exclude:
            continue
        held = sh[sym].to_numpy() > 1e-12
        if not held.any():
            continue
        ft_all = tr[tr["symbol"] == sym]
        f_date = ft_all["date"].to_numpy()
        f_qty = ft_all["qty"].to_numpy(float)
        f_open = ft_all["open"].to_numpy(float)
        f_notional = ft_all["notional"].to_numpy(float)
        f_comm = ft_all["commission"].to_numpy(float)
        dv = divs[divs["symbol"] == sym] if len(divs) else divs
        d_date = dv["date"].to_numpy() if len(dv) else np.array([], dtype="datetime64[ns]")
        d_amt = dv["amount"].to_numpy(float) if len(dv) else np.array([])
        edges = np.flatnonzero(np.diff(np.r_[0, held.astype(np.int8), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            # e: first flat session (exit fill day), or len(cal) if still open
            first, last = cal[s], cal[e] if e < len(cal) else None
            lo = np.searchsorted(f_date, first.to_datetime64(), "left")
            hi = np.searchsorted(f_date, last.to_datetime64(), "right") if last is not None else len(f_date)
            q, o, nt, cm = f_qty[lo:hi], f_open[lo:hi], f_notional[lo:hi], f_comm[lo:hi]
            buy, sell = q > 0, q < 0
            gross_in = float((q[buy] * o[buy]).sum())
            gross_out = float((-q[sell] * o[sell]).sum())
            net_in = float(nt[buy].sum() + cm[buy].sum())
            net_out = float(-nt[sell].sum() - cm[sell].sum())
            d = 0.0
            if len(d_amt):
                dlo = np.searchsorted(d_date, first.to_datetime64(), "left")
                dhi = np.searchsorted(d_date, last.to_datetime64(), "left") if last is not None else len(d_date)
                d = float(d_amt[dlo:dhi].sum())
            mark = 0.0
            if last is None:  # still open: mark at the last close
                c = cal[-1]
                px = data.frames[sym]["close"].ffill().loc[c] if data is not None else float("nan")
                mark = float(sh.loc[c, sym] * px)
            if gross_in <= 0:
                continue
            rows.append({"symbol": sym, "entry": first, "exit": last, "sessions": int(e - s),
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
