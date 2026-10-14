"""Control experiments for the validation engine (M5): each must get the verdict
it is built to get. If any of these fail, no real-data verdict is trustworthy."""

import copy

import numpy as np
import pytest

from quantlab.registry.store import Registry
from quantlab.strategies.base import Strategy
from quantlab.strategies.trend_timing import TrendTiming
from quantlab.validation.engine import validate
from synthetic import regime_market

POLICY = {
    "policy_id": "test_trend", "version": 1, "strategy_family": "TT-SPY-SMA",
    "universe": ["SPY", "BIL"], "benchmark_symbol": "SPY", "risk_free_symbol": "BIL", "initial_cash": 100000,
    "splits": {"train": ["2000-01-03", "2009-12-31"], "validation": ["2010-01-01", "2015-12-31"],
               "holdout": ["2016-01-01", None]},
    "costs": {"commission_bps": 0.0, "slippage_bps": 5.0, "fixed_fee": 0.0},
    "gate1": {"leak_test_cutpoints": 3, "leak_test_seed": 1},
    "gate2": {"min_sharpe_diff": -0.10, "max_mdd_ratio": 0.90},
    "gate3": {"min_sharpe_diff": 0.0, "bootstrap": {"block": 63, "n_boot": 2000, "seed": 7, "max_p_value": 0.10},
              "max_mdd_ratio": 0.80, "min_position_changes": 6, "min_sessions": 504},
    "gate4_partial": {"stress_cost_multiplier": 2.0, "stress_min_sharpe_diff": 0.0,
                      "neighbor_params": {"lookback_months": [8, 12]}, "neighbor_min_sharpe_diff": 0.0},
}


class RandomTiming(Strategy):
    """No edge: coin-flip risk-on/off each month. Same family id so the same policy applies."""
    strategy_id = "TT-SPY-SMA"
    version = "random"

    def __init__(self, seed=0, lookback_months=None, risk="SPY", cash="BIL"):
        super().__init__(seed=seed)
        self.rng = np.random.default_rng(seed)

    def on_bar(self, view):
        if not view.is_last_session_of_month():
            return None
        on = self.rng.random() < 0.5
        return {"SPY": float(on), "BIL": float(not on)}


def make_leaky(data):
    class Leaky(Strategy):
        """Reads next month's return from the full dataset captured in a closure."""
        strategy_id = "TT-SPY-SMA"
        version = "leaky"

        def __init__(self, lookback_months=None, risk="SPY", cash="BIL"):
            super().__init__()

        def on_bar(self, view):
            if not view.is_last_session_of_month():
                return None
            close = data.frames["SPY"]["close"]
            i = close.index.get_loc(view.now)
            fwd = close.iloc[min(i + 21, len(close) - 1)] / close.iloc[i] - 1
            return {"SPY": float(fwd > 0), "BIL": float(fwd <= 0)}
    return Leaky


class PeeksPrivateData(Strategy):
    """Bypasses the view through its private attribute; caught by the perturbation test."""
    strategy_id = "TT-SPY-SMA"
    version = "peek"

    def on_bar(self, view):
        if not view.is_last_session_of_month():
            return None
        full = getattr(view, "_" + "data")  # spelled indirectly so only the dynamic test can catch it
        close = full.frames["SPY"]["close"]
        i = close.index.get_loc(view.now)
        fwd = close.iloc[min(i + 21, len(close) - 1)] / close.iloc[i] - 1
        return {"SPY": float(fwd > 0), "BIL": float(fwd <= 0)}


def test_view_bypass_is_invalid(reg):
    out = run(reg, PeeksPrivateData, regime_market(0))
    assert out.overall == "INVALID"
    assert "look-ahead" in " ".join(out.gates[0].reasons)


@pytest.fixture
def reg(tmp_path):
    r = Registry(tmp_path / "reg.sqlite")
    yield r
    r.close()


def run(reg, make, data, policy=POLICY, params=None):
    h = reg.register_policy(policy)
    return validate(make, params or {}, data, h, reg, dataset_version="synthetic")


def test_real_edge_passes():
    reg = Registry(":memory:")
    out = run(reg, TrendTiming, regime_market(0))
    assert out.overall == "PASS", [(g.gate, g.verdict, g.reasons) for g in out.gates]
    assert reg.current_state("TT-SPY-SMA") == "VALIDATION"


def test_no_edge_fails(reg):
    out = run(reg, RandomTiming, regime_market(0), params={"seed": 3})
    assert out.overall == "FAIL", [(g.gate, g.verdict, g.reasons) for g in out.gates]
    assert reg.current_state("TT-SPY-SMA") == "REJECTED"


def test_lookahead_is_invalid_and_does_not_touch_holdout(reg):
    data = regime_market(0)
    out = run(reg, make_leaky(data), data)
    assert out.overall == "INVALID"
    assert "outside MarketView" in " ".join(out.gates[0].reasons)
    assert reg.holdout_accesses("TT-SPY-SMA", reg.register_policy(POLICY)) == 0
    assert reg.current_state("TT-SPY-SMA") is None  # INVALID is not a strategy failure


def test_second_holdout_read_cannot_pass(reg):
    data = regime_market(0)
    first = run(reg, TrendTiming, data)
    assert first.overall == "PASS"
    second = run(reg, TrendTiming, data)
    assert second.overall == "HOLD"
    assert any("already read" in r for g in second.gates for r in g.reasons)
    assert reg.verdicts(first.experiment_id)[-1]["verdict"] == "PASS"  # original record untouched


def test_insufficient_evidence_holds(reg):
    pol = copy.deepcopy(POLICY)
    pol["policy_id"] = "test_trend_strict_n"
    pol["gate3"]["min_position_changes"] = 500
    out = run(reg, TrendTiming, regime_market(0), policy=pol)
    assert out.overall == "HOLD"


def test_qc_fatal_is_invalid(reg):
    h = reg.register_policy(POLICY)
    out = validate(TrendTiming, {}, regime_market(0), h, reg, dataset_version="synthetic",
                   qc_fatal=["duplicate timestamps in SPY"])
    assert out.overall == "INVALID"


def test_policy_mismatch_rejected(reg):
    class Other(TrendTiming):
        strategy_id = "OTHER"
    with pytest.raises(ValueError):
        run(reg, Other, regime_market(0))


def test_derived_after_holdout_view_cannot_pass(reg):
    reg.register_strategy(strategy_hash="h", strategy_id="TT-SPY-SMA", version="1.0.0", origin="claude",
                          body={}, derived_after_holdout_view=True)
    out = run(reg, TrendTiming, regime_market(0))
    assert out.overall == "HOLD"
    assert any("derived after holdout" in r for g in out.gates for r in g.reasons)
