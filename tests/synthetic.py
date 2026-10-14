"""Synthetic market builders for tests (no network, fully deterministic)."""

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
        # a split divides traded prices from its date onwards
        factor = np.cumprod(split)
        close, opn, div = close / factor, opn / factor, div / factor
        frames[sym] = frame(opn, close, div, split, start=start)
    return MarketData.from_frames(frames)


def regime_market(seed: int, start="2000-01-03", end="2023-12-29", bull=(0.015, 0.03), bear=(-0.08, 0.06),
                  stay=(0.97, 0.93), risk="SPY", cash="BIL") -> MarketData:
    """Monthly Markov regimes (persistent bull/bear) on daily bars: a market where
    trend following has a genuine, built-in edge. Used as the PASS control."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    months = pd.PeriodIndex(idx, freq="M")
    uniq = months.unique()
    state, states = 0, {}
    for mth in uniq:
        states[mth] = state
        if rng.random() > stay[state]:
            state = 1 - state
    mu = np.array([(bull, bear)[states[mth]][0] for mth in months]) / 21
    sd = np.array([(bull, bear)[states[mth]][1] for mth in months]) / np.sqrt(21)
    ret = rng.normal(mu, sd)
    close = 100 * np.exp(np.cumsum(ret))
    opn = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.002, len(idx)))
    spy = frame(opn, close, start=start)
    spy.index = idx
    c = 50 * np.exp(np.cumsum(np.full(len(idx), 0.0001)))
    bil = frame(c, c, start=start)
    bil.index = idx
    return MarketData.from_frames({risk: spy, cash: bil})
