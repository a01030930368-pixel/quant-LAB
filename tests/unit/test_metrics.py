"""Metric correctness against hand-computed values and analytic identities (M4)."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantlab.validation import metrics as m


def eq(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)), dtype=float)


def test_drawdown_hand_example():
    e = eq([100, 120, 90, 95, 130, 117])
    assert m.max_drawdown(e) == pytest.approx(90 / 120 - 1)
    assert m.max_drawdown_duration(e) == 2  # sessions at 90 and 95
    assert m.drawdown(e).iloc[-1] == pytest.approx(117 / 130 - 1)


def test_cagr_exact_for_constant_growth():
    daily = 1.0003
    e = eq(100 * daily ** np.arange(253))  # exactly one year of 252 returns
    assert m.cagr(e) == pytest.approx(daily**252 - 1, rel=1e-12)


def test_sharpe_and_sortino_hand_example():
    r = pd.Series([0.01, -0.02, 0.015, 0.005])
    mu, sd = r.mean(), r.std(ddof=1)
    assert m.sharpe(r) == pytest.approx(mu / sd * math.sqrt(252))
    down = math.sqrt((0.02**2) / 4)
    assert m.sortino(r) == pytest.approx(mu / down * math.sqrt(252))
    rf = pd.Series([0.001] * 4)
    assert m.sharpe(r, rf) == pytest.approx((mu - 0.001) / sd * math.sqrt(252))


def test_psr_reduces_to_normal_formula_for_gaussian_moments():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000))
    sr = r.mean() / r.std(ddof=1)
    g3 = stats.skew(r, bias=False)
    g4 = stats.kurtosis(r, fisher=False, bias=False)
    expected = stats.norm.cdf(sr * math.sqrt(len(r) - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    assert m.probabilistic_sharpe(r) == pytest.approx(expected)
    assert 0.5 < m.probabilistic_sharpe(r) < 1.0


def test_dsr_identities():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0004, 0.01, 1500))
    assert m.deflated_sharpe(r, 1, 0.001) == pytest.approx(m.probabilistic_sharpe(r, 0.0))
    d = [m.deflated_sharpe(r, n, 0.0004) for n in (1, 10, 100, 1000)]
    assert all(a > b for a, b in zip(d, d[1:]))  # more trials -> less confidence


def test_expected_max_sharpe_matches_simulation():
    rng = np.random.default_rng(2)
    n_trials, var = 50, 0.0025
    sims = rng.normal(0, math.sqrt(var), size=(20000, n_trials)).max(axis=1).mean()
    assert m.expected_max_sharpe(n_trials, var) == pytest.approx(sims, rel=0.03)


def test_bootstrap_detects_real_difference_and_not_noise():
    rng = np.random.default_rng(3)
    n = 2520
    bench = pd.Series(rng.normal(0.0003, 0.01, n))
    better = bench * 0.5 + pd.Series(rng.normal(0.0006, 0.004, n))
    same = bench.copy()
    res = m.block_bootstrap_sharpe_diff(better, bench, None, n_boot=1000, seed=0)
    assert res["observed"] > 0 and res["p_value"] < 0.05
    noise = m.block_bootstrap_sharpe_diff(bench + rng.normal(0, 1e-4, n), same, None, n_boot=1000, seed=0)
    assert noise["p_value"] > 0.05
    again = m.block_bootstrap_sharpe_diff(better, bench, None, n_boot=1000, seed=0)
    assert again == res  # seed fixes the result


def test_top_period_share_and_turnover():
    r = pd.Series([0.1] + [0.0] * 99)
    assert m.top_period_share(r) == pytest.approx(1.0)
    trades = pd.DataFrame({"notional": [1000.0, -1000.0]})
    e = eq([1000.0] * 253)
    assert m.turnover(trades, e) == pytest.approx(2.0)
