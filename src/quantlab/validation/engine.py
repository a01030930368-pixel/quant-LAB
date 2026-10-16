"""Validation engine: fixed-policy gates and automatic verdicts.

Responsibilities are separate from research code (04 §4, 03 §4.2):
  * the policy is registered (hashed) before evaluation and read back from the registry;
  * exploratory runs can only see data up to the end of the validation split;
  * only `validate()` reads the holdout, and every read is written to the ledger;
  * verdict rows are append-only.

Gate order: 1 integrity → 2 exploratory value (dev data) → 3 holdout → 4 (partial).
A Gate 2 failure stops before the holdout is touched.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from quantlab.analysis.trades import episodes, excess_benchmark, not_trades, trade_stats, without_best_year
from quantlab.backtesting.engine import AccountingError, BacktestConfig, CostModel, run_backtest
from quantlab.core.market_data import MarketData, total_return_index
from quantlab.registry.store import Registry
from quantlab.strategies.base import MarketView, Strategy
from quantlab.validation import metrics as M

StrategyFactory = Callable[..., Strategy]
RANK = {"PASS": 0, "HOLD": 1, "FAIL": 2, "INVALID": 3}
STATE_FOR = {"FAIL": "REJECTED", "HOLD": "HOLD", "PASS": "VALIDATION"}  # INVALID: no state change


@dataclass
class GateResult:
    gate: str
    verdict: str
    reasons: list[str]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationOutcome:
    experiment_id: str
    overall: str
    gates: list[GateResult]
    metrics: dict[str, Any]
    results: dict[str, Any] = field(default_factory=dict)  # BacktestResults for charts/artifacts


class BuyAndHold(Strategy):
    strategy_id = "BENCHMARK_BUY_HOLD"
    version = "1"

    def __init__(self, symbol: str) -> None:
        super().__init__(symbol=symbol)
        self.symbol, self.done = symbol, False

    def on_bar(self, view):
        if self.done:
            return None
        self.done = True
        return {self.symbol: 1.0}


def _cfg(policy: dict, cost_multiplier: float = 1.0) -> BacktestConfig:
    c = dict(policy["costs"])
    c["commission_bps"] = float(c["commission_bps"]) * cost_multiplier
    c["slippage_bps"] = float(c["slippage_bps"]) * cost_multiplier
    return BacktestConfig(initial_cash=float(policy["initial_cash"]), costs=CostModel(**c))


def dev_end(policy: dict) -> pd.Timestamp:
    return pd.Timestamp(policy["splits"]["validation"][1])


def dev_data(data: MarketData, policy: dict) -> MarketData:
    """The only slice research code may use (TRAIN + VALIDATION)."""
    return data.slice(policy["splits"]["train"][0], dev_end(policy))


def _rf(data: MarketData, policy: dict) -> pd.Series:
    return total_return_index(data.frames[policy["risk_free_symbol"]]).pct_change().fillna(0.0)


def _window(s: pd.Series, start, end) -> pd.Series:
    s = s.loc[pd.Timestamp(start):]
    return s.loc[: pd.Timestamp(end)] if end else s


def _position_changes(signals, start, end) -> int:
    n, prev = 0, None
    for date, w in signals:
        key = tuple(sorted((k, round(v, 9)) for k, v in w.items()))
        if prev is not None and key != prev and pd.Timestamp(start) <= date <= (pd.Timestamp(end) if end else date):
            n += 1
        prev = key
    return n


def _compare(strat_eq, bench_eq, rf, trades, start, end) -> dict:
    s, b = _window(strat_eq, start, end), _window(bench_eq, start, end)
    rs, rb = s.pct_change().dropna(), b.pct_change().dropna()
    sm, bm = M.summarize(s, trades.loc[trades["date"] >= pd.Timestamp(start)] if len(trades) else trades, rf), M.summarize(b, trades.iloc[0:0], rf)
    out = {
        "strategy": sm, "benchmark": bm,
        "sharpe_diff": sm["sharpe"] - bm["sharpe"],
        "mdd_ratio": (sm["max_drawdown"] / bm["max_drawdown"]) if bm["max_drawdown"] < 0 else float("nan"),
        "excess_cagr": sm["cagr"] - bm["cagr"],
    }
    out["_returns"] = (rs, rb)
    return out


# ------------------------------------------------------------------ Gate 1
_DATA_TYPES = (MarketData, pd.DataFrame, pd.Series, np.ndarray)
_FORBIDDEN_ATTRS = {"_data", "read_csv", "read_parquet", "read_sql", "read_json", "urlopen"}


def isolation_check(cls: type) -> list[str]:
    """Strategy code may reach market data only through MarketView.

    Perturbing the view's data cannot reveal a strategy that reads prices from a
    closure, a module global or a file, so those routes are checked statically:
    closures/globals holding data objects, and names that bypass the view or do I/O.
    Not a security boundary (03 §4.2); OS-level isolation comes in Phase 4.
    """
    problems = []
    for klass in cls.__mro__:
        if klass in (Strategy, object):
            continue
        for name, fn in vars(klass).items():
            code = getattr(fn, "__code__", None)
            if code is None:
                continue
            for var, cell in zip(code.co_freevars, fn.__closure__ or ()):
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue
                if isinstance(val, _DATA_TYPES):
                    problems.append(f"{klass.__name__}.{name} captures market data '{var}' outside MarketView")
            for ref in code.co_names:
                if ref in _FORBIDDEN_ATTRS:
                    problems.append(f"{klass.__name__}.{name} uses '{ref}' (bypasses MarketView or does I/O)")
                elif isinstance(fn.__globals__.get(ref), _DATA_TYPES):
                    problems.append(f"{klass.__name__}.{name} reads global market data '{ref}'")
    return problems


def leak_test(make: StrategyFactory, params: dict, data: MarketData, cutpoints: int, seed: int) -> list[str]:
    """Perturb all prices (and point-in-time tables) after a cut; decisions up to the cut must not change."""
    rng = np.random.default_rng(seed)
    n = len(data.calendar)
    problems = []
    cuts = sorted(set(np.linspace(n // 4, n - 2, cutpoints).astype(int)))
    for cut in cuts:
        altered = {}
        for sym, df in data.frames.items():
            d = df.copy()
            scale = rng.uniform(0.5, 1.5, size=n - cut - 1)
            for f in ("open", "high", "low", "close"):
                d.iloc[cut + 1:, d.columns.get_loc(f)] = d[f].to_numpy()[cut + 1:] * scale
            altered[sym] = d
        # point-in-time tables: membership after the cut is inverted, later events are removed
        members = events = None
        if data.members is not None:
            members = data.members.copy()
            members.iloc[cut + 1:] = ~members.iloc[cut + 1:].to_numpy(dtype=bool)
        if data.events is not None:
            events = data.events[data.events["known"] <= data.calendar[cut]].reset_index(drop=True)
        perturbed = MarketData(data.calendar, altered, members, events)
        a, b = make(**params), make(**params)
        for t in range(cut + 1):
            if a.on_bar(MarketView(data, t)) != b.on_bar(MarketView(perturbed, t)):
                problems.append(f"decision at {data.calendar[t].date()} changed when data after "
                                f"{data.calendar[cut].date()} was perturbed (look-ahead)")
                break
    return problems


def lineage_independence(registry: Registry, strategy_id: str, version: str) -> list[str]:
    """03 v2.0.0 §4.4: a strategy derived after its line's holdout results were seen or
    shared cannot use that holdout as independent evidence."""
    problems = []
    for row in registry.lineage(strategy_id, version):
        if row["derived_after_holdout_view"]:
            problems.append(f"{row['strategy_id']}@{row['version']} was derived after holdout results were seen")
        if (row["strategy_id"], row["version"]) != (strategy_id, version) and registry.holdout_shared(row["strategy_id"]):
            problems.append(f"ancestor {row['strategy_id']} holdout results were exported; use the forward holdout")
    return problems


def validate(make: StrategyFactory, params: dict, data: MarketData, policy_hash: str, registry: Registry, *,
             dataset_version: str, hypothesis_id: str | None = None,
             qc_fatal: list[str] | None = None, supersedes: str | None = None,
             strategy_hash: str | None = None, spec_version: int | None = None,
             origin: str | None = None) -> ValidationOutcome:
    policy = registry.get_policy(policy_hash)
    family = policy["strategy_family"]
    probe = make(**params)
    if probe.strategy_id != family:
        raise ValueError(f"policy covers {family}, strategy is {probe.strategy_id}")
    sp = policy["splits"]
    data = data.slice(sp["train"][0], sp["holdout"][1])
    gates: list[GateResult] = []
    metrics: dict[str, Any] = {"policy_id": policy["policy_id"], "policy_version": policy["version"]}

    # Gate 1 — integrity
    g1 = list(qc_fatal or [])
    g1 += isolation_check(type(probe))
    g1 += leak_test(make, params, dev_data(data, policy), policy["gate1"]["leak_test_cutpoints"],
                    policy["gate1"]["leak_test_seed"])
    dev = dev_data(data, policy)
    try:
        dev_s = run_backtest(make(**params), dev, _cfg(policy))
        dev_b = run_backtest(BuyAndHold(policy["benchmark_symbol"]), dev, _cfg(policy))
    except AccountingError as e:
        g1.append(f"accounting invariant violated: {e}")
    gates.append(GateResult("GATE1", "INVALID" if g1 else "PASS", g1 or ["data, look-ahead and accounting checks passed"]))

    if not g1:
        # Gate 2 — exploratory value on TRAIN+VALIDATION
        rf_dev = _rf(dev, policy)
        g2p = policy["gate2"]
        comp = {k: _compare(dev_s.equity, dev_b.equity, rf_dev, dev_s.trades, *sp[k]) for k in ("train", "validation")}
        both = _compare(dev_s.equity, dev_b.equity, rf_dev, dev_s.trades, sp["train"][0], sp["validation"][1])
        for k, v in comp.items():
            v.pop("_returns")
            metrics[f"dev_{k}"] = v
        both.pop("_returns")
        both["trades"] = trade_stats(episodes(dev_s, dev, sp["train"][0], sp["validation"][1],
                                                    exclude=not_trades(probe),
                                                    excess_vs=excess_benchmark(probe)))
        both["without_best_year"] = without_best_year(_window(dev_s.equity, sp["train"][0], sp["validation"][1]))
        metrics["dev"] = both
        ok = both["sharpe_diff"] >= g2p["min_sharpe_diff"] or both["mdd_ratio"] <= g2p["max_mdd_ratio"]
        reasons = [f"dev sharpe_diff={both['sharpe_diff']:.3f} (min {g2p['min_sharpe_diff']}), "
                   f"mdd_ratio={both['mdd_ratio']:.3f} (max {g2p['max_mdd_ratio']})"]
        gates.append(GateResult("GATE2", "PASS" if ok else "FAIL", reasons))

    if gates[-1].gate == "GATE2" and gates[-1].verdict == "PASS":
        # Gate 3 — independent holdout evaluation (the only holdout read)
        prior = registry.holdout_accesses(family, policy_hash)
        # logged before the read, so an exception mid-evaluation still leaves a trace
        attempt = f"ATTEMPT-{uuid.uuid4().hex[:12]}"
        registry.log_holdout_access(family, probe.config_hash(), policy_hash, attempt)
        metrics["holdout_attempt"] = attempt
        g3p = policy["gate3"]
        hs, he = sp["holdout"]
        rf = _rf(data, policy)
        full_s = run_backtest(make(**params), data, _cfg(policy))
        full_b = run_backtest(BuyAndHold(policy["benchmark_symbol"]), data, _cfg(policy))
        ho = _compare(full_s.equity, full_b.equity, rf, full_s.trades, hs, he)
        rs, rb = ho.pop("_returns")
        ho["trades"] = trade_stats(episodes(full_s, data, hs, he, exclude=not_trades(probe),
                                                    excess_vs=excess_benchmark(probe)))
        bp = g3p["bootstrap"]
        boot = M.block_bootstrap_sharpe_diff(rs, rb, rf, bp["block"], bp["n_boot"], bp["seed"])
        changes = _position_changes(full_s.signals, hs, he)
        n_trials = registry.count_trials_for_family(family) + 1
        # DSR needs the cross-trial variance of Sharpe ratios; with a single trial it equals PSR(0).
        # Multi-trial DSR is reported once families run parameter searches (not used for verdicts in v1).
        ho.update({"bootstrap": boot, "position_changes": changes, "prior_holdout_accesses": prior,
                   "dsr_trials": n_trials,
                   "dsr": ho["strategy"]["psr_vs_0"] if n_trials == 1 else None})
        metrics["holdout"] = ho
        fails, holds = [], []
        if ho["sharpe_diff"] <= g3p["min_sharpe_diff"]:
            fails.append(f"holdout sharpe_diff={ho['sharpe_diff']:.3f} not above {g3p['min_sharpe_diff']}")
        if boot["p_value"] >= bp["max_p_value"]:
            fails.append(f"bootstrap p={boot['p_value']:.3f} not below {bp['max_p_value']}")
        if not ho["mdd_ratio"] <= g3p["max_mdd_ratio"]:
            fails.append(f"holdout mdd_ratio={ho['mdd_ratio']:.3f} above {g3p['max_mdd_ratio']}")
        if changes < g3p["min_position_changes"]:
            holds.append(f"only {changes} position changes in holdout (min {g3p['min_position_changes']})")
        if ho["strategy"]["sessions"] < g3p["min_sessions"]:
            holds.append(f"holdout has {ho['strategy']['sessions']} sessions (min {g3p['min_sessions']})")
        if prior:
            holds.append(f"holdout already read {prior} time(s) for {family} under this policy; not independent")
        holds += lineage_independence(registry, probe.strategy_id, probe.version)
        v = "FAIL" if fails else ("HOLD" if holds else "PASS")
        gates.append(GateResult("GATE3", v, fails + holds or ["all pre-registered holdout conditions met"]))

        if v == "PASS":
            g4p = policy["gate4_partial"]
            g4 = []
            mult = float(g4p["stress_cost_multiplier"])
            stress = run_backtest(make(**params), data, _cfg(policy, mult))
            stress_b = run_backtest(BuyAndHold(policy["benchmark_symbol"]), data, _cfg(policy, mult))
            st = _compare(stress.equity, stress_b.equity, rf, stress.trades, hs, he)
            st.pop("_returns")
            metrics["gate4_cost_stress"] = {"cost_multiplier": mult, "sharpe_diff": st["sharpe_diff"]}
            if st["sharpe_diff"] < g4p["stress_min_sharpe_diff"]:
                g4.append(f"sharpe_diff {st['sharpe_diff']:.3f} at {mult:g}x costs")
            neigh = {}
            for name, values in g4p["neighbor_params"].items():
                for val in values:
                    p2 = dict(params, **{name: val})
                    r2 = run_backtest(make(**p2), data, _cfg(policy))
                    c2 = _compare(r2.equity, full_b.equity, rf, r2.trades, hs, he)
                    neigh[f"{name}={val}"] = c2["sharpe_diff"]
                    if c2["sharpe_diff"] < g4p["neighbor_min_sharpe_diff"]:
                        g4.append(f"neighbor {name}={val} sharpe_diff {c2['sharpe_diff']:.3f}")
            metrics["gate4_neighbors"] = neigh
            gates.append(GateResult("GATE4_PARTIAL", "HOLD" if g4 else "PASS", g4 or ["cost stress and neighbors held"]))

    overall = max((g.verdict for g in gates), key=RANK.__getitem__)
    exp_id = registry.record_experiment(
        kind="formal", strategy_id=family, strategy_version=probe.version, dataset_version=dataset_version,
        config={"strategy": probe.describe(), "policy_hash": policy_hash}, metrics=metrics,
        hypothesis_id=hypothesis_id, validation_policy_version=policy_hash,
        strategy_hash=strategy_hash, spec_version=spec_version, strategy_origin=origin,
        random_seed=policy["gate3"]["bootstrap"]["seed"],
        failure_reason=None if overall == "PASS" else "; ".join(r for g in gates if g.verdict != "PASS" for r in g.reasons),
        supersedes=supersedes)
    for g in gates:
        registry.record_verdict(exp_id, g.gate, g.verdict, g.reasons)
    registry.record_verdict(exp_id, "OVERALL", overall, [f"{g.gate}={g.verdict}" for g in gates])
    if overall in STATE_FOR:
        registry.record_state_change(family, STATE_FOR[overall], f"formal validation {overall}", exp_id)
    results = {}
    if "full_s" in locals():
        results = {"strategy": full_s, "benchmark": full_b}
    elif "dev_s" in locals():
        results = {"strategy": dev_s, "benchmark": dev_b}
    return ValidationOutcome(exp_id, overall, gates, metrics, results)
