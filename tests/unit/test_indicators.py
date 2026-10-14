"""Whole-series indicators: causal (truncation-invariant, also across splits), equal to the windowed
rule values where both exist, and hand-checked crossovers."""

import numpy as np
import pandas as pd
import pytest

from quantlab.core.market_data import MarketData
from quantlab.strategies.base import MarketView
from quantlab.strategies.entry_exit import _value
from quantlab.strategies.indicators import series
from synthetic import frame, random_market

KEYS = [("close_vs_sma", 20), ("close_vs_min", 10), ("close_vs_max", 10), ("bollinger_z", 20), ("return", 5),
        ("rsi", 2), ("rsi", 14), ("ibs",), ("down_streak",), ("sma_ratio", 5, 20), ("sma_cross", 5, 20),
        ("close_cross_sma", 20), ("macd", 12, 26, 9), ("macd_hist", 12, 26, 9), ("macd_cross", 12, 26, 9)]


@pytest.mark.parametrize("key", KEYS, ids=lambda k: "_".join(map(str, k)))
def test_causal_under_truncation_with_splits_and_gaps(key):
    d = random_market(7, n=420, symbols=("A",), split_prob=0.01)
    v = d.values("A").copy()
    v[50:53, :4] = np.nan  # a few non-trading rows
    full = series(v, key)
    for cut in (60, 140, 233, 300, 419):
        part = series(v[: cut + 1], key)
        np.testing.assert_allclose(part, full[: cut + 1], rtol=1e-9, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("cond", [
    {"indicator": "close_vs_sma", "days": 20}, {"indicator": "close_vs_min", "days": 10},
    {"indicator": "close_vs_max", "days": 10}, {"indicator": "bollinger_z", "days": 20},
    {"indicator": "return", "days": 5}, {"indicator": "ibs"}, {"indicator": "down_streak"}],
    ids=lambda c: c["indicator"])
def test_matches_windowed_rule_values(cond):
    d = random_market(3, n=300, symbols=("A",), split_prob=0.01)
    from quantlab.strategies.indicators import key_for

    s = series(d.values("A"), key_for(cond))
    for t in range(80, 300, 7):
        a = _value(MarketView(d, t), cond, "A", {})
        b = s[t]
        if a is None:
            assert not np.isfinite(b) or cond["indicator"] == "down_streak"
        else:
            assert b == pytest.approx(a, rel=1e-9, abs=1e-12)


def test_crossovers_by_hand():
    c = np.array([10, 10, 10, 10, 9, 8, 7, 9, 12, 13], float)
    d = MarketData.from_frames({"A": frame(c, c)})
    up = series(d.values("A"), ("close_cross_sma", 3))
    # SMA3 from row 2; close - sma: r3 0, r4 -0.67, r5 -1, r6 -1, r7 +1, r8 +2.67, r9 +1.67
    assert list(up[3:]) == [0, -1, 0, 0, 1, 0, 0]
    assert np.isnan(up[:3]).all()
    cross = series(d.values("A"), ("sma_cross", 2, 3))
    # SMA2 - SMA3: r3 0, r4 -0.17, r5..r6 -0.5, r7 0, r8 +1.17 -> down-cross r4, up-cross r8 (not r7: 0 is not above)
    assert list(cross[3:]) == [0, -1, 0, 0, 0, 1, 0]
