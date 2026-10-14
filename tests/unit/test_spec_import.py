"""Strategy spec import (M8, 03 v2.0.0 §8.1 import tests)."""

from pathlib import Path

import pytest
import yaml

from quantlab.registry.store import Registry
from quantlab.strategies.base import MarketView
from quantlab.strategies.declarative import DeclarativeStrategy
from quantlab.strategies.importer import import_spec
from quantlab.strategies.spec import SpecError, check_code, load_spec
from quantlab.strategies.trend_timing import TrendTiming
from synthetic import random_market, regime_market

REPO = Path(__file__).resolve().parents[2]
BASELINE = REPO / "strategies" / "specs" / "TT-SPY-SMA.qls.yaml"


@pytest.fixture
def reg():
    r = Registry(":memory:")
    yield r
    r.close()


def write(tmp_path, spec: dict, code: str | None = None, name="s.qls.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    if code is not None:
        (tmp_path / "strat.py").write_text(code, encoding="utf-8")
    return p


def base_spec() -> dict:
    return yaml.safe_load(BASELINE.read_text(encoding="utf-8"))


def test_baseline_spec_matches_python_class_every_session():
    loaded = load_spec(BASELINE)
    for seed in (0, 1):
        data = regime_market(seed, end="2010-12-31")
        a, b = loaded.factory(**loaded.default_params), TrendTiming()
        for t in range(len(data.calendar)):
            assert a.on_bar(MarketView(data, t)) == b.on_bar(MarketView(data, t)), data.calendar[t]


def test_baseline_imports_to_research(reg, tmp_path):
    res = import_spec(BASELINE, reg, tmp_path / "strategies")
    assert res.new and res.state == "RESEARCH", res.problems
    assert reg.current_state("TT-SPY-SMA") == "RESEARCH"
    assert any(tmp_path.joinpath("strategies").rglob("*.qls.yaml"))
    again = import_spec(BASELINE, reg)
    assert not again.new and again.strategy_hash == res.strategy_hash


@pytest.mark.parametrize("path,value,msg", [
    ("hypothesis.falsification", None, "falsification"),
    ("strategy.author", "gpt", "author"),
    ("execution.fill", "same_close", "next_open"),
    ("rules.signal.source", "QQQ.total_return", "universe"),
    ("rules.long", {"SPY": 1.5}, "sum to at most 1"),
    ("spec_version", 9, "spec_version"),
])
def test_malformed_specs_rejected(tmp_path, reg, path, value, msg):
    spec = base_spec()
    cur = spec
    keys = path.split(".")
    for k in keys[:-1]:
        cur = cur[k]
    if value is None:
        del cur[keys[-1]]
    else:
        cur[keys[-1]] = value
    with pytest.raises(SpecError, match=msg):
        import_spec(write(tmp_path, spec), reg)
    assert reg.strategies() == []


def test_same_content_same_hash_one_char_new_hash(tmp_path):
    s1 = base_spec()
    h1 = load_spec(write(tmp_path, s1, name="a.qls.yaml")).strategy_hash
    h1b = load_spec(write(tmp_path, base_spec(), name="b.qls.yaml")).strategy_hash
    s2 = base_spec()
    s2["hypothesis"]["falsification"] += "."
    h2 = load_spec(write(tmp_path, s2, name="c.qls.yaml")).strategy_hash
    assert h1 == h1b and h1 != h2


def test_changed_content_needs_new_version(tmp_path, reg):
    import_spec(BASELINE, reg)
    s = base_spec()
    s["rules"]["long"] = {"SPY": 0.9}
    with pytest.raises(SpecError, match="bump strategy.version"):
        import_spec(write(tmp_path, s), reg)
    s["strategy"]["version"] = "1.0.1"
    assert import_spec(write(tmp_path, s), reg).state == "RESEARCH"


PY_SPEC_RULES = {"kind": "python", "entry": "strat.py:Mine"}

GOOD_CODE = '''
import numpy as np
from quantlab.strategies.base import Strategy

class Mine(Strategy):
    def __init__(self, lookback_months=10):
        super().__init__(lookback_months=lookback_months)

    def on_bar(self, view):
        if not view.is_last_session_of_month():
            return None
        r = view.total_return("SPY")
        return {"SPY": 1.0, "BIL": 0.0} if r.iloc[-1] >= r.iloc[max(0, len(r) - 200)] else {"SPY": 0.0, "BIL": 1.0}
'''


def test_python_spec_imports(tmp_path, reg):
    s = base_spec()
    s["rules"] = PY_SPEC_RULES
    res = import_spec(write(tmp_path, s, GOOD_CODE), reg)
    assert res.state == "RESEARCH", res.problems


@pytest.mark.parametrize("snippet", [
    "import os", "import subprocess", "from pathlib import Path", "x = open('f')",
    "y = eval('1')", "import requests", "z = view._data", "pd.read_csv('prices.csv')",
])
def test_forbidden_code_rejected(tmp_path, reg, snippet):
    assert check_code(snippet)
    s = base_spec()
    s["rules"] = PY_SPEC_RULES
    with pytest.raises(SpecError):
        import_spec(write(tmp_path, s, GOOD_CODE + "\n" + snippet + "\n"), reg)


def test_leaky_python_strategy_stays_idea(tmp_path, reg):
    # leak through a module global holding market data built at import time is impossible
    # (no file access), so the realistic leak left is shared mutable state across instances
    code = '''
from quantlab.strategies.base import Strategy

FUTURE = {}

class Mine(Strategy):
    def __init__(self, lookback_months=10):
        super().__init__(lookback_months=lookback_months)

    def on_bar(self, view):
        r = view.total_return("SPY")
        FUTURE.update(r.to_dict())          # first run stores the whole path it has seen
        if not view.is_last_session_of_month():
            return None
        later = [d for d in FUTURE if d > view.now]
        if later:                           # a later run peeks at values from an earlier, longer run
            up = FUTURE[max(later)] > r.iloc[-1]
            return {"SPY": float(up), "BIL": float(not up)}
        return {"SPY": 1.0, "BIL": 0.0}
'''
    s = base_spec()
    s["rules"] = PY_SPEC_RULES
    with pytest.raises(SpecError, match="shared mutable state"):
        import_spec(write(tmp_path, s, code), reg)
    assert check_code("class A:\n    cache = {}\n")
    assert check_code("def f(x, memo=[]):\n    return x\n")
    assert not check_code("LOOKBACK = 10\nNAMES = ('SPY', 'BIL')\n")


def test_crashing_strategy_stays_idea(tmp_path, reg):
    code = GOOD_CODE.replace('r = view.total_return("SPY")', 'r = view.total_return("NOPE")')
    s = base_spec()
    s["rules"] = PY_SPEC_RULES
    res = import_spec(write(tmp_path, s, code), reg)
    assert res.state == "IDEA" and "crashed" in res.problems[0]
    assert reg.current_state("TT-SPY-SMA") == "IDEA"


def test_rank_rules_hold_top_n_with_cash_filter():
    data = random_market(4, n=800, symbols=("A", "B", "C", "BIL"), drift=0.0)
    rules = {"rank": {"symbols": ["A", "B", "C"], "score": "momentum", "lookback_months": 3, "top_n": 2,
                      "min_score": 0.0}, "cash": {"BIL": 1.0}}
    s = DeclarativeStrategy(spec_id="R", spec_version_str="1", symbols=["A", "B", "C", "BIL"], rules=rules,
                            rebalance="monthly")
    seen = 0
    for t in range(len(data.calendar)):
        w = s.on_bar(MarketView(data, t))
        if w is None:
            continue
        seen += 1
        assert abs(sum(w.values()) - 1.0) < 1e-12
        assert sum(1 for k in ("A", "B", "C") if w[k] > 0) <= 2
    assert seen > 20
