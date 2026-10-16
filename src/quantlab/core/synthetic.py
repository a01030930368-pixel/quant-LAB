"""Deterministic synthetic markets.

Used by the spec importer's smoke and leak tests, by the app's `--selftest`, and by the
test suite. Synthetic data is never used as research evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.core.market_data import MarketData


def frame(opens, closes, dividends=None, splits=None, start="2020-01-01") -> pd.DataFrame:
    n = len(closes)
    idx = pd.bdate_range(start, periods=n)
    o = np.asarray(opens, float)
    c = np.asarray(closes, float)
    return pd.DataFrame(
        {
            "open": o,
            "high": np.fmax(o, c),
            "low": np.fmin(o, c),
            "close": c,
            "dividend": np.zeros(n) if dividends is None else np.asarray(dividends, float),
            "split": np.ones(n) if splits is None else np.asarray(splits, float),
        },
        index=idx,
    )


def random_market(seed: int, n: int = 300, symbols=("AAA", "BBB"), start="2015-01-01",
                  drift=0.0003, vol=0.01, div_prob=0.01, split_prob=0.002) -> MarketData:
    rng = np.random.default_rng(seed)
    frames = {}
    for sym in symbols:
        ret = rng.normal(drift, vol, n)
        close = 50 * np.exp(np.cumsum(ret))
        gap = rng.normal(0, vol / 3, n)
        opn = np.r_[close[0], close[:-1]] * np.exp(gap)
        div = np.where(rng.random(n) < div_prob, close * 0.005, 0.0)
        split = np.where(rng.random(n) < split_prob, 2.0, 1.0)
        factor = np.cumprod(split)  # a split divides traded prices from its date onwards
        close, opn, div = close / factor, opn / factor, div / factor
        frames[sym] = frame(opn, close, div, split, start=start)
    return MarketData.from_frames(frames)


def regime_market(seed: int, start="2000-01-03", end="2023-12-29", bull=(0.015, 0.03), bear=(-0.08, 0.06),
                  stay=(0.97, 0.93), risk="SPY", cash="BIL", index: pd.DatetimeIndex | None = None) -> MarketData:
    """Monthly Markov regimes (persistent bull/bear) on daily bars: trend following has a
    genuine built-in edge here. Used as the PASS control. `index` overrides the business-day
    calendar (e.g. real exchange sessions so data-quality checks pass)."""
    rng = np.random.default_rng(seed)
    idx = index if index is not None else pd.bdate_range(start, end)
    months = pd.PeriodIndex(idx, freq="M")
    state, states = 0, {}
    for mth in months.unique():
        states[mth] = state
        if rng.random() > stay[state]:
            state = 1 - state
    mu = np.array([(bull, bear)[states[m]][0] for m in months]) / 21
    sd = np.array([(bull, bear)[states[m]][1] for m in months]) / np.sqrt(21)
    close = 100 * np.exp(np.cumsum(rng.normal(mu, sd)))
    opn = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.002, len(idx)))
    spy = frame(opn, close, start=start)
    spy.index = idx
    c = 50 * np.exp(np.cumsum(np.full(len(idx), 0.0001)))
    bil = frame(c, c, start=start)
    bil.index = idx
    return MarketData.from_frames({risk: spy, cash: bil})


def universe_market(symbols, seed: int = 0, n: int = 1500, start="2010-01-01") -> MarketData:
    """Random market for an arbitrary symbol list (spec smoke tests)."""
    return random_market(seed, n=n, symbols=tuple(symbols), start=start, div_prob=0.005, split_prob=0.001)


def index_market(symbols, seed: int = 0, n: int = 1500, n_stocks: int = 30, start="2010-01-01") -> MarketData:
    """Random stock index with membership, insider-purchase and earnings events (smoke tests of index specs).

    Stocks are named S00, S01, ...; each is a member for one random span. Events are random filings
    by a few insiders per stock, officers or directors, some below any sensible value threshold."""
    rng = np.random.default_rng(seed)
    stocks = tuple(f"S{i:02d}" for i in range(n_stocks))
    data = random_market(seed, n=n, symbols=stocks + tuple(s for s in symbols if s not in stocks), start=start,
                         div_prob=0.005, split_prob=0.001)
    cal = data.calendar
    members = pd.DataFrame(False, index=cal, columns=list(stocks))
    for s in stocks:
        a = int(rng.integers(0, n // 2))
        b = int(rng.integers(a + n // 4, n + 1))
        members.iloc[a:b, members.columns.get_loc(s)] = True
    rows = []
    for _ in range(n_stocks * 12):
        s = stocks[int(rng.integers(n_stocks))]
        day = cal[int(rng.integers(n))] + pd.Timedelta(days=int(rng.integers(0, 3)))
        for _ in range(int(rng.integers(1, 4))):
            rows.append({"known": day + pd.Timedelta(days=int(rng.integers(0, 10))), "symbol": s,
                         "owner": f"{s}-{int(rng.integers(5))}", "value": float(rng.choice([5e3, 5e4, 5e5])),
                         "officer": bool(rng.random() < 0.6), "director": bool(rng.random() < 0.6),
                         "kind": "insider_buy"})
    for s in stocks:  # quarterly earnings filings
        for day in cal[int(rng.integers(0, 60))::63]:
            rows.append({"known": day, "symbol": s, "owner": None, "value": float("nan"),
                         "officer": False, "director": False, "kind": "earnings"})
    events = pd.DataFrame(rows).sort_values(["known", "symbol"], kind="stable").reset_index(drop=True)
    return data.with_tables(members, events)
