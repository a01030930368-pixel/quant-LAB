"""Declarative strategy rules (QUANT LAB Strategy Spec v1, `rules.kind: declarative`).

Built-in components only, so imported specs execute no code. Two rule shapes:

conditional — one signal decides between two weight sets:
    signal: {indicator: sma, source: SPY.total_return, lookback_months: 10}
    long:   {SPY: 1.0}
    else:   {BIL: 1.0}

rank — score a universe, hold the best N (equal weight), optional absolute filter:
    rank: {symbols: [SPY, EFA, TLT], score: momentum, source_field: total_return,
           lookback_months: 12, skip_months: 0, top_n: 2, min_score: 0.0}
    cash: {BIL: 1.0}            # weight left over by filtered-out slots goes here

Sources are `SYMBOL.field` with field one of:
    total_return  forward-built total return index (dividends and splits)
    close         forward-built split-adjusted close (no dividends)
Both are built only from data up to the current bar.

Indicators (all evaluated on month-end or session values depending on `lookback_months`/`lookback_days`):
    sma        source > simple moving average of the last N values  (signal true)
    momentum   source return over N periods > threshold (default 0)
    volatility annualized daily vol over N days < threshold

Rebalance timing (`execution.rebalance`): monthly (last session of month), weekly
(last session of week) or daily. Components are added only when a real spec needs them.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from quantlab.core.market_data import total_return_index
from quantlab.strategies.base import MarketView, Strategy

INDICATORS = ("sma", "momentum", "volatility")
FIELDS = ("total_return", "close")
REBALANCE = ("monthly", "weekly", "daily")


class RuleError(ValueError):
    pass


def split_adjusted_close(df: pd.DataFrame) -> pd.Series:
    close = df["close"].ffill()
    prev = close.shift(1)
    gross = (close * df["split"] / prev).where(df["close"].notna() & prev.notna(), 1.0).fillna(1.0)
    return close.iloc[0] * gross.cumprod() if len(close) and np.isfinite(close.iloc[0]) else gross.cumprod()


def _series(view: MarketView, source: str) -> pd.Series:
    sym, _, field = source.partition(".")
    if field not in FIELDS:
        raise RuleError(f"unknown source field '{field}' (use one of {FIELDS})")
    if sym not in view.symbols:
        raise RuleError(f"source symbol '{sym}' not in universe")
    frame = view.frame(sym)
    return total_return_index(frame) if field == "total_return" else split_adjusted_close(frame)


def _sampled(s: pd.Series, spec: dict) -> tuple[pd.Series, int]:
    if "lookback_months" in spec:
        return s.groupby([s.index.year, s.index.month]).tail(1), int(spec["lookback_months"])
    if "lookback_days" in spec:
        return s, int(spec["lookback_days"])
    raise RuleError("indicator needs lookback_months or lookback_days")


def _score(view: MarketView, spec: dict, source: str) -> float | None:
    ind = spec.get("indicator", spec.get("score"))
    s = _series(view, source)
    if ind == "volatility":
        n = int(spec.get("lookback_days", 63))
        r = s.pct_change().dropna()
        return float(r.iloc[-n:].std(ddof=1) * math.sqrt(252)) if len(r) >= n else None
    vals, n = _sampled(s, spec)
    skip = int(spec.get("skip_months", spec.get("skip", 0)))
    if ind == "sma":
        if len(vals) < n:
            return None
        window = vals.iloc[-n:]
        return float(window.iloc[-1] / window.mean() - 1.0)  # > 0 means above its average
    if ind == "momentum":
        if len(vals) < n + 1 + skip:
            return None
        end = vals.iloc[-1 - skip]
        start = vals.iloc[-1 - skip - n]
        return float(end / start - 1.0)
    raise RuleError(f"unknown indicator '{ind}' (use one of {INDICATORS})")


def _signal_true(view: MarketView, sig: dict) -> bool | None:
    score = _score(view, sig, sig["source"])
    if score is None:
        return None
    ind = sig["indicator"]
    threshold = float(sig.get("threshold", 0.0))
    return score < threshold if ind == "volatility" else score > threshold


def _is_rebalance(view: MarketView, when: str) -> bool:
    if when == "daily":
        return True
    if when == "monthly":
        return view.is_last_session_of_month()
    if when == "weekly":
        return view.is_last_session_of_week()
    raise RuleError(f"unknown rebalance '{when}' (use one of {REBALANCE})")


def _weights(symbols: list[str], w: dict) -> dict[str, float]:
    out = {s: 0.0 for s in symbols}
    for k, v in (w or {}).items():
        if k not in out:
            raise RuleError(f"weight for '{k}' which is not in the universe")
        out[k] = float(v)
    return out


def validate_rules(rules: dict, symbols: list[str], rebalance: str) -> list[str]:
    errs = []
    if rebalance not in REBALANCE:
        errs.append(f"execution.rebalance must be one of {REBALANCE}")
    if "signal" in rules:
        sig = rules["signal"]
        if not isinstance(sig, dict) or sig.get("indicator") not in INDICATORS:
            errs.append(f"rules.signal.indicator must be one of {INDICATORS}")
        elif str(sig.get("source", "")).partition(".")[0] not in symbols:
            errs.append("rules.signal.source must be SYMBOL.field with SYMBOL in universe.symbols")
        elif str(sig.get("source", "")).partition(".")[2] not in FIELDS:
            errs.append(f"rules.signal.source field must be one of {FIELDS}")
        for key in ("long", "else"):
            w = rules.get(key)
            if not isinstance(w, dict):
                errs.append(f"rules.{key} weights are required")
            elif set(w) - set(symbols):
                errs.append(f"rules.{key} uses symbols outside the universe: {sorted(set(w) - set(symbols))}")
            elif sum(w.values()) > 1 + 1e-9 or any(v < 0 for v in w.values()):
                errs.append(f"rules.{key} weights must be non-negative and sum to at most 1")
    elif "entry_exit" in rules:
        from quantlab.strategies.entry_exit import validate_entry_exit

        if rebalance != "daily":
            errs.append("entry_exit rules are judged every session: execution.rebalance must be daily")
        errs += validate_entry_exit(rules["entry_exit"], symbols)
    elif "rank" in rules:
        rk = rules["rank"]
        if rk.get("score") not in ("momentum", "sma", "volatility"):
            errs.append("rules.rank.score must be momentum, sma or volatility")
        if not set(rk.get("symbols", [])) <= set(symbols) or not rk.get("symbols"):
            errs.append("rules.rank.symbols must be a non-empty subset of universe.symbols")
        if int(rk.get("top_n", 0)) < 1:
            errs.append("rules.rank.top_n must be >= 1")
        if set(rules.get("cash", {})) - set(symbols):
            errs.append("rules.cash uses symbols outside the universe")
    else:
        errs.append("declarative rules need 'signal' (conditional), 'rank' or 'entry_exit'")
    return errs


class DeclarativeStrategy(Strategy):
    """Executes a validated declarative rule set. Parameter values override rule keys of the same name."""

    def __init__(self, *, spec_id: str, spec_version_str: str, symbols: list[str], rules: dict,
                 rebalance: str, **params: Any) -> None:
        super().__init__(**params)
        self.strategy_id, self.version = spec_id, spec_version_str
        self.symbols, self.rebalance = list(symbols), rebalance
        self.rules = _apply_params(rules, params)

    def describe(self) -> dict:
        return {"strategy_id": self.strategy_id, "version": self.version, "params": self.params,
                "rules": self.rules, "rebalance": self.rebalance}

    def on_bar(self, view: MarketView):
        if not _is_rebalance(view, self.rebalance):
            return None
        r = self.rules
        if "signal" in r:
            on = _signal_true(view, r["signal"])
            if on is None:
                return None
            return _weights(self.symbols, r["long"] if on else r["else"])
        rk = r["rank"]
        source_field = rk.get("source_field", "total_return")
        scores = {}
        for s in rk["symbols"]:
            sc = _score(view, rk, f"{s}.{source_field}")
            if sc is None:
                return None
            scores[s] = sc
        ascending = rk["score"] == "volatility"
        ranked = sorted(scores, key=lambda s: scores[s], reverse=not ascending)[: int(rk["top_n"])]
        slot = 1.0 / int(rk["top_n"])
        w = {s: 0.0 for s in self.symbols}
        cash_share = 0.0
        min_score = rk.get("min_score")
        for s in ranked:
            if min_score is not None and scores[s] <= float(min_score):
                cash_share += slot
            else:
                w[s] += slot
        for s, cw in (r.get("cash") or {}).items():
            w[s] += cash_share * float(cw)
        return w


def _apply_params(rules: Any, params: dict) -> Any:
    if isinstance(rules, dict):
        return {k: (params[k] if k in params and not isinstance(v, (dict, list)) else _apply_params(v, params))
                for k, v in rules.items()}
    if isinstance(rules, list):
        return [_apply_params(v, params) for v in rules]
    return rules
