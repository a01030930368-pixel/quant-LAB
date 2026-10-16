"""Spec import pipeline (03 v2.0.0 §4.3):

file → schema check → code static check (python kind) → content hash, register (IDEA)
→ synthetic smoke test → leak checks → RESEARCH, or stay IDEA with reasons.

Imported strategies get the same checks whoever wrote them; origin is recorded, never trusted.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from quantlab.backtesting.engine import run_backtest
from quantlab.core.synthetic import index_market, universe_market
from quantlab.registry.store import Registry
from quantlab.strategies.spec import LoadedSpec, SpecError, load_spec
from quantlab.validation.engine import isolation_check, leak_test


@dataclass
class ImportResult:
    strategy_hash: str
    strategy_id: str
    version: str
    state: str
    new: bool
    problems: list[str] = field(default_factory=list)
    loaded: LoadedSpec | None = None


def preflight(loaded: LoadedSpec) -> list[str]:
    """Smoke test and leak checks on synthetic data (no market data needed)."""
    problems: list[str] = []
    symbols = loaded.spec["universe"]["symbols"]
    data = index_market(symbols, seed=1) if "index" in loaded.spec["universe"] else universe_market(symbols, seed=1)
    params = loaded.default_params
    try:
        probe = loaded.factory(**params)
        problems += isolation_check(type(probe))
        res = run_backtest(loaded.factory(**params), data)
        if not res.signals:
            problems.append("smoke test: strategy produced no target weights on 6 years of synthetic data")
    except Exception as e:  # any crash is a reason to keep the strategy at IDEA
        problems.append(f"smoke test crashed: {type(e).__name__}: {e}")
        return problems
    problems += leak_test(loaded.factory, params, data, cutpoints=3, seed=7)
    return problems


def import_spec(path: str | Path, registry: Registry, strategies_dir: str | Path | None = None) -> ImportResult:
    loaded = load_spec(path)  # raises SpecError: nothing is registered for malformed specs
    sid, ver = loaded.strategy_id, loaded.version
    for row in registry.strategies():
        if row["strategy_id"] == sid and row["version"] == ver and row["strategy_hash"] != loaded.strategy_hash:
            raise SpecError([f"{sid}@{ver} is already registered with different content; bump strategy.version"])
    st = loaded.spec["strategy"]
    new = registry.register_strategy(
        strategy_hash=loaded.strategy_hash, strategy_id=sid, version=ver, origin=st["author"],
        body={"spec": loaded.spec, "code": loaded.code}, spec_version=loaded.spec["spec_version"],
        parent=st.get("parent"), derived_after_holdout_view=st["derived_after_holdout_view"])
    if strategies_dir is not None:
        dest = Path(strategies_dir) / loaded.strategy_hash[:16]
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(loaded.path, dest / loaded.path.name)
        if loaded.code is not None:
            fname = loaded.spec["rules"]["entry"].split(":", 1)[0]
            (dest / fname).write_text(loaded.code, encoding="utf-8")
    if not new:
        return ImportResult(loaded.strategy_hash, sid, ver, registry.current_state(sid) or "IDEA", False, [], loaded)
    registry.record_state_change(sid, "IDEA", f"imported {sid}@{ver} ({st['author']}), hash {loaded.strategy_hash[:12]}")
    problems = preflight(loaded)
    if problems:
        registry.record_state_change(sid, "IDEA", "import checks failed: " + "; ".join(problems))
        return ImportResult(loaded.strategy_hash, sid, ver, "IDEA", True, problems, loaded)
    registry.record_state_change(sid, "RESEARCH", "import checks passed (schema, smoke test, leak checks)")
    return ImportResult(loaded.strategy_hash, sid, ver, "RESEARCH", True, [], loaded)
