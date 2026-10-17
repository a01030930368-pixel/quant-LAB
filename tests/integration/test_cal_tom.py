"""D-032: the pre-fixed turn-of-the-month rule on US sector 3x ETFs (half the equity, the rest in BIL)."""

import numpy as np
import pandas as pd
import pytest

from quantlab.data.quality import exchange_sessions
from quantlab.services.lab import Lab, builtin_path
from quantlab.strategies.calendar_tom import validate_calendar, window_position
from quantlab.strategies.spec import load_spec
from synthetic import frame

SPEC = builtin_path("specs", "CAL-TOM-SECT3X.qls.yaml")
FUNDS = ["SOXL", "TNA", "FAS", "ERX", "TECL", "DRN", "CURE"]
SYMBOLS = FUNDS + ["SPY", "BIL"]


def test_rule_is_fixed_and_half_the_equity():
    loaded = load_spec(SPEC)
    cal = loaded.spec["rules"]["calendar"]
    assert loaded.default_params == {} and cal["trade"] == FUNDS and cal["park"] == "BIL"
    assert len(cal["trade"]) * cal["leg_weight"] == pytest.approx(0.5, abs=0.001)
    assert (cal["enter_before_end"], cal["exit_session"]) == (1, 4)
    assert loaded.spec["strategy"]["derived_after_holdout_view"] is False


def test_validation_rejects_bad_blocks():
    ok = {"trade": ["A"], "leg_weight": 0.5, "enter_before_end": 1, "exit_session": 4, "park": "B"}
    assert validate_calendar(ok, ["A", "B"]) == []
    assert validate_calendar(dict(ok, leg_weight=0.6, trade=["A", "B"], park=None), ["A", "B"])  # over 100%
    assert validate_calendar(dict(ok, park="A"), ["A", "B"])
    assert validate_calendar(dict(ok, exit_session=0), ["A", "B"])
    assert validate_calendar(dict(ok, trade=["C"]), ["A", "B"])


def test_session_numbers_follow_the_nyse_schedule():
    assert window_position(pd.Timestamp("2019-12-31")) == (21, 1)
    assert window_position(pd.Timestamp("2020-01-02"))[0] == 1   # 2020-01-01 is a holiday
    assert window_position(pd.Timestamp("2012-10-31"))[0] == 21  # 10-29/30 closed (Hurricane Sandy)


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tom")
    lab = Lab(tmp / "QL")
    sessions = exchange_sessions("2009-06-01", "2021-12-31")
    rng = np.random.default_rng(11)
    for sym in SYMBOLS:
        vol = {"BIL": 0.0005, "SPY": 0.011}.get(sym, 0.035)
        close = 50 * np.exp(np.cumsum(rng.normal(0.0002, vol, len(sessions))))
        df = frame(close, close)
        df.index = sessions
        if sym == "CURE":  # lists 2011-06, as the real fund
            df = df[df.index >= "2011-06-15"]
        p = tmp / f"{sym}.csv"
        df.rename_axis("date").reset_index().to_csv(p, index=False, date_format="%Y-%m-%d")
        lab.import_csv(sym, p)
    yield lab
    lab.close()


def test_backtest_trades_only_the_window_on_dev_only(lab):
    bt = lab.quick_backtest(next(p for p in lab.strategy_files() if p.name == "CAL-TOM-SECT3X.qls.yaml"))
    e = lab.registry.experiment(bt.experiment_id)
    assert e["kind"] == "exploratory" and e["metrics"]["window"][1] == "2019-12-31"
    tr = bt.result.trades
    funds = tr[tr["symbol"].isin(FUNDS)]
    buys, sells = funds[funds["qty"] > 0], funds[funds["qty"] < 0]
    assert len(buys) and len(sells)
    assert all(window_position(d)[1] == 1 for d in pd.to_datetime(buys["date"]))   # last session of the month
    assert all(window_position(d)[0] == 4 for d in pd.to_datetime(sells["date"]))  # 4th session
    months = pd.to_datetime(buys["date"]).dt.to_period("M").nunique()
    assert months >= 115                                                           # about every month of 2010-2019
    assert not len(buys[(buys["symbol"] == "CURE") & (pd.to_datetime(buys["date"]) < "2011-06-30")])
    bil = tr[tr["symbol"] == "BIL"]
    assert len(bil) <= 2   # bought once (whole-share rounding may add one top-up), never traded with the window
    assert lab.registry.holdout_ledger() == []


def test_final_evaluation_passes_the_integrity_checks(lab):
    # synthetic prices: only Gate 1 (look-ahead, isolation, accounting) is meaningful here
    out = lab.quick_final(SPEC)
    gates = {v["gate"]: v["verdict"] for v in lab.registry.verdicts(out.experiment_id)}
    assert gates["GATE1"] == "PASS"
