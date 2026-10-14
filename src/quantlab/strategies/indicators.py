"""Whole-series indicators for entry/exit rules, computed once per symbol and read one bar at a time.

`MarketView.indicator(symbol, key)` returns the value at the current bar only. Each function here
must be causal: the value at bar t may depend on bars <= t and nothing later. The unit tests check
this by truncating the data at many points (and the final evaluation's leak test perturbs every
price after a cut), so a non-causal function cannot reach a verdict unnoticed.

Prices are split-adjusted with the split factors of the whole series. Before bar t that scales all
earlier prices by one common constant (the splits after t), and every indicator below is scale-free
(a ratio of prices, of averages, or of averages to prices), so the value at t equals the one
computed from data up to t alone.

Rows where the symbol did not trade (NaN close) are skipped for the calculation and return NaN.

Keys (tuples):
    ("close_vs_sma", d)     close / SMA(d) - 1
    ("close_vs_min", d)     close / min(close, last d) - 1
    ("close_vs_max", d)     close / max(close, last d) - 1
    ("bollinger_z", d)      (close - SMA(d)) / std(d)   (population std, as in the windowed rule)
    ("return", d)           close / close d sessions ago - 1
    ("rsi", d)              Wilder RSI(d), smoothed from the first trading row
    ("ibs",)                (close - low) / (high - low)
    ("down_streak",)        consecutive lower closes ending today
    ("sma_ratio", f, s)     SMA(f) / SMA(s) - 1             (> 0: short average above long)
    ("sma_cross", f, s)     +1 on the day SMA(f) crosses above SMA(s), -1 below, else 0
    ("close_cross_sma", d)  +1 on the day close crosses above SMA(d), -1 below, else 0
    ("macd", f, s, g)       (EMA(f) - EMA(s)) / close
    ("macd_hist", f, s, g)  (MACD - signal EMA(g)) / close
    ("macd_cross", f, s, g) +1 on the day MACD crosses above its signal line, -1 below, else 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MACD_DEFAULT = (12, 26, 9)


def key_for(cond: dict) -> tuple:
    """Cache key for one rule condition (the indicator and its lengths)."""
    ind = cond["indicator"]
    if ind in ("ibs", "down_streak"):
        return (ind,)
    if ind in ("sma_ratio", "sma_cross"):
        return (ind, int(cond["fast"]), int(cond["slow"]))
    if ind in ("macd", "macd_hist", "macd_cross"):
        return (ind, int(cond.get("fast", MACD_DEFAULT[0])), int(cond.get("slow", MACD_DEFAULT[1])),
                int(cond.get("signal", MACD_DEFAULT[2])))
    return (ind, int(cond["days"]))


def _adjusted(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trading-row positions and split-adjusted high, low, close on those rows."""
    close = values[:, 3]
    rows = np.flatnonzero(np.isfinite(close))
    k = np.nan_to_num(values[:, 5], nan=1.0)
    later = np.concatenate([np.cumprod(k[::-1])[::-1][1:], [1.0]])
    return rows, values[rows, 1] / later[rows], values[rows, 2] / later[rows], close[rows] / later[rows]


def _cross(diff: np.ndarray) -> np.ndarray:
    prev = np.r_[np.nan, diff[:-1]]
    out = np.where((diff > 0) & (prev <= 0), 1.0, np.where((diff < 0) & (prev >= 0), -1.0, 0.0))
    out[~(np.isfinite(diff) & np.isfinite(prev))] = np.nan
    return out


def _wilder(c: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(c), np.nan)
    d = np.diff(c)
    if len(d) < n:
        return out
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = up[:n].mean(), dn[:n].mean()
    for i in range(n, len(d) + 1):
        if i > n:
            au = (au * (n - 1) + up[i - 1]) / n
            ad = (ad * (n - 1) + dn[i - 1]) / n
        out[i] = (100.0 if au > 0 else 50.0) if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


def _on_rows(key: tuple, h: np.ndarray, lo: np.ndarray, c: np.ndarray) -> np.ndarray:
    ind = key[0]
    s = pd.Series(c)

    def sma(d: int) -> pd.Series:
        return s.rolling(d, min_periods=d).mean()

    if ind == "close_vs_sma":
        return (s / sma(key[1]) - 1).to_numpy()
    if ind == "close_vs_min":
        return (s / s.rolling(key[1], min_periods=key[1]).min() - 1).to_numpy()
    if ind == "close_vs_max":
        return (s / s.rolling(key[1], min_periods=key[1]).max() - 1).to_numpy()
    if ind == "bollinger_z":
        d = key[1]
        sd = s.rolling(d, min_periods=d).std(ddof=0)
        return ((s - sma(d)) / sd.where(sd > 0)).to_numpy()
    if ind == "return":
        return (s / s.shift(key[1]) - 1).to_numpy()
    if ind == "rsi":
        return _wilder(c, key[1])
    if ind == "ibs":
        rng = h - lo
        return np.where(rng > 0, (c - lo) / np.where(rng > 0, rng, 1.0), np.nan)
    if ind == "down_streak":
        out = np.full(len(c), np.nan)
        n = 0
        for i in range(1, len(c)):
            n = n + 1 if c[i] < c[i - 1] else 0
            out[i] = n
        return out
    if ind == "sma_ratio":
        return (sma(key[1]) / sma(key[2]) - 1).to_numpy()
    if ind == "sma_cross":
        return _cross((sma(key[1]) - sma(key[2])).to_numpy())
    if ind == "close_cross_sma":
        return _cross((s - sma(key[1])).to_numpy())
    if ind in ("macd", "macd_hist", "macd_cross"):
        f, sl, g = key[1:]
        line = s.ewm(span=f, adjust=False).mean() - s.ewm(span=sl, adjust=False).mean()
        line[: sl - 1] = np.nan  # slow average still warming up
        signal = line.ewm(span=g, adjust=False, ignore_na=True).mean()
        signal[line.isna()] = np.nan
        signal.iloc[: sl - 1 + g - 1] = np.nan
        if ind == "macd":
            return (line / s).to_numpy()
        if ind == "macd_hist":
            return ((line - signal) / s).to_numpy()
        return _cross((line - signal).to_numpy())
    raise ValueError(f"unknown indicator {key}")


def series(values: np.ndarray, key: tuple) -> np.ndarray:
    """Indicator value on every calendar row of one symbol (NaN where unavailable)."""
    rows, h, lo, c = _adjusted(values)
    out = np.full(len(values), np.nan)
    if len(rows):
        out[rows] = _on_rows(key, h, lo, c)
    return out
