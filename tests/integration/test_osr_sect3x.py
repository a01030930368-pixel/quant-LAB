"""D-025: the pre-fixed oversold-rebound rule on unseen US sector 3x ETFs (10% legs, idle equity in SPY)."""

import numpy as np
import pytest

from quantlab.data.quality import exchange_sessions
from quantlab.services.lab import Lab, builtin_path
from quantlab.services.autosearch import PREREGISTERED
from quantlab.strategies.spec import load_spec
from synthetic import frame

SPEC = builtin_path("specs", "OSR-SECT3X.qls.yaml")
SYMBOLS = ["SOXL", "TNA", "FAS", "ERX", "TECL", "DRN", "CURE", "SPY", "BIL"]


def test_rule_is_the_preregistered_one_unchanged():
    loaded = load_spec(SPEC)
    assert loaded.default_params == PREREGISTERED["AUTO-LEVP"]
    combo = loaded.spec["rules"]["combo"]
    assert combo["slots"] == 10 and combo["park"] == "SPY" and len(combo["trade"]) == 7
    assert loaded.spec["strategy"]["derived_after_holdout_view"] is True


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("osr")
    lab = Lab(tmp / "QL")
    sessions = exchange_sessions("2009-06-01", "2021-12-31")
    rng = np.random.default_rng(7)
    for sym in SYMBOLS:
        vol = {"BIL": 0.0005, "SPY": 0.011}.get(sym, 0.04)  # 3x funds swing hard enough to hit -15% in 5 days
        close = 50 * np.exp(np.cumsum(rng.normal(0.0002, vol, len(sessions))))
        df = frame(close, close)
        df.index = sessions
        p = tmp / f"{sym}.csv"
        df.rename_axis("date").reset_index().to_csv(p, index=False, date_format="%Y-%m-%d")
        lab.import_csv(sym, p)
    yield lab
    lab.close()


def test_backtest_runs_on_dev_only_and_holdout_is_not_spent(lab):
    assert any(p.name == "OSR-SECT3X.qls.yaml" for p in lab.strategy_files())
    bt = lab.quick_backtest(next(p for p in lab.strategy_files() if p.name == "OSR-SECT3X.qls.yaml"))
    e = lab.registry.experiment(bt.experiment_id)
    assert e["kind"] == "exploratory" and e["metrics"]["trades"]["n"] > 0
    assert e["metrics"]["window"][1] == "2019-12-31"  # TRAIN+VALIDATION only
    with pytest.raises(RuntimeError, match="전진 보류 구간"):
        lab.quick_final(SPEC)
    assert lab.registry.holdout_ledger() == []
