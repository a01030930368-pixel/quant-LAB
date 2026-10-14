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
