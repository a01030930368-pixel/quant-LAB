"""Performance, risk and statistical-confidence metrics (protocol 제18~20조).

Conventions: daily simple returns, 252 sessions per year, Sharpe uses returns in
excess of the risk-free proxy. PSR/DSR follow Bailey & López de Prado
(2012, 2014) on non-annualized per-period Sharpe ratios.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

PERIODS = 252
EULER_GAMMA = 0.5772156649015329


def cagr(equity: pd.Series) -> float:
    years = (len(equity) - 1) / PERIODS
    if years <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def ann_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * math.sqrt(PERIODS))


def sharpe(returns: pd.Series, rf: pd.Series | None = None, annualize: bool = True) -> float:
    ex = returns - (rf.reindex(returns.index).fillna(0.0) if rf is not None else 0.0)
    sd = ex.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    sr = ex.mean() / sd
    return float(sr * math.sqrt(PERIODS)) if annualize else float(sr)


def sortino(returns: pd.Series, rf: pd.Series | None = None) -> float:
    ex = returns - (rf.reindex(returns.index).fillna(0.0) if rf is not None else 0.0)
    downside = np.sqrt((np.minimum(ex, 0.0) ** 2).mean())
    if downside == 0:
        return float("nan")
    return float(ex.mean() / downside * math.sqrt(PERIODS))


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest run of sessions spent below a previous peak (peak to recovery)."""
    under = (drawdown(equity) < 0).to_numpy()
    longest = run = 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return int(longest)


def calmar(equity: pd.Series) -> float:
    mdd = max_drawdown(equity)
    return float(cagr(equity) / abs(mdd)) if mdd < 0 else float("nan")


def turnover(trades: pd.DataFrame, equity: pd.Series) -> float:
    """Annual traded notional as a multiple of average equity."""
    if trades.empty:
        return 0.0
    years = max((len(equity) - 1) / PERIODS, 1e-9)
    return float(trades["notional"].abs().sum() / equity.mean() / years)


def top_period_share(returns: pd.Series, frac: float = 0.05) -> float:
    """Share of total log-return contributed by the best `frac` of sessions."""
    lr = np.log1p(returns.to_numpy())
    total = lr.sum()
    if total <= 0:
        return float("nan")
    k = max(1, int(len(lr) * frac))
    return float(np.sort(lr)[-k:].sum() / total)


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0, rf: pd.Series | None = None) -> float:
    """P(true per-period SR > sr_benchmark) given sample length, skew and kurtosis."""
    ex = returns - (rf.reindex(returns.index).fillna(0.0) if rf is not None else 0.0)
    n = len(ex)
    sr = sharpe(ex, annualize=False)
    if n < 3 or not np.isfinite(sr):
        return float("nan")
    g3 = float(stats.skew(ex, bias=False))
    g4 = float(stats.kurtosis(ex, fisher=False, bias=False))
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period SR among n_trials unskilled strategies."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns: pd.Series, n_trials: int, sr_variance: float, rf: pd.Series | None = None) -> float:
    return probabilistic_sharpe(returns, expected_max_sharpe(n_trials, sr_variance), rf)


def block_bootstrap_sharpe_diff(strat: pd.Series, bench: pd.Series, rf: pd.Series | None,
                                block: int = 63, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Moving-block bootstrap of annualized Sharpe(strat) - Sharpe(bench).

    Paired resampling keeps the cross-correlation between the two series.
    p_value is the one-sided probability that the difference is <= 0, from the
    bootstrap distribution centred on zero (null of no difference).
    """
    idx = strat.index.intersection(bench.index)
    rfv = rf.reindex(idx).fillna(0.0).to_numpy() if rf is not None else np.zeros(len(idx))
    a = strat.reindex(idx).to_numpy() - rfv
    b = bench.reindex(idx).to_numpy() - rfv
    n = len(idx)

    def sr(x):
        s = x.std(ddof=1)
        return x.mean() / s * math.sqrt(PERIODS) if s > 0 else np.nan

    observed = sr(a) - sr(b)
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    offs = np.arange(block)
    diffs = np.empty(n_boot)
    for j in range(n_boot):
        ii = (starts[j][:, None] + offs).ravel()[:n]
        diffs[j] = sr(a[ii]) - sr(b[ii])
    centred = diffs - np.nanmean(diffs)
    p = float(np.mean(centred >= observed))
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {"observed": float(observed), "p_value": p, "ci95": [float(lo), float(hi)],
            "block": block, "n_boot": n_boot, "seed": seed}


def summarize(equity: pd.Series, trades: pd.DataFrame, rf: pd.Series | None = None) -> dict:
    r = equity.pct_change().dropna()
    return {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "sessions": int(len(equity)),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "cagr": cagr(equity),
        "ann_vol": ann_vol(r),
        "sharpe": sharpe(r, rf),
        "sortino": sortino(r, rf),
        "max_drawdown": max_drawdown(equity),
        "max_drawdown_sessions": max_drawdown_duration(equity),
        "calmar": calmar(equity),
        "n_trades": int(len(trades)),
        "turnover": turnover(trades, equity),
        "top5pct_sessions_share": top_period_share(r),
        "psr_vs_0": probabilistic_sharpe(r, 0.0, rf),
    }
