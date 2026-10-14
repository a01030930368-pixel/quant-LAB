"""Swing-style entry/exit rules (QUANT LAB Strategy Spec v1, `rules.kind: declarative`, `entry_exit`).

Each leg trades one symbol, one position at a time:

    entry_exit:
      sizing: equal_active          # or {slots: K}: each open leg holds 1/K
      legs:
        - name: S2
          trade: SPXL               # symbol bought
          entry:                    # ALL conditions true at the close -> buy at next open
            - {indicator: close_vs_min, of: SPY, days: 5, op: "<=", value: 0}
            - {indicator: close_vs_sma, of: SPY, days: 200, op: ">", value: 0}
          exit:                     # ANY condition true at a close -> sell at next open
            any:
              - {indicator: close_vs_max, of: SPY, days: 5, op: ">=", value: 0}
            max_hold_days: 15       # required; sessions held, counted from the fill day
            stop_loss: 0.10         # optional, close-based, vs the fill price (split-adjusted)
            take_profit: 0.20       # optional, close-based

Timing is conservative and identical for every rule: decide at the close, fill at the next
open. Stops and targets are checked on closes, not intraday, so a gap through a stop is taken
at the next open. The previous project's 15:55-decide / 15:59-fill daily variant cannot be
reproduced from daily bars and is not attempted.

Indicators (on split-adjusted prices of `of`, default the traded symbol):
    close_vs_sma   close / SMA(days) - 1
    close_vs_min   close / min(close, last `days` sessions incl. today) - 1   (<= 0: new low)
    close_vs_max   close / max(close, last `days` sessions incl. today) - 1   (>= 0: new high)
    bollinger_z    (close - SMA(days)) / std(days)
    rsi            Wilder RSI(days) on closes (0..100)
    ibs            (close - low) / (high - low) today
    return         close / close `days` sessions ago - 1
    down_streak    number of consecutive lower closes ending today
    relative_return  return(days) of `of` minus return(days) of `versus`
    session        value: first_of_week | last_of_week | last_of_month  (op ignored)

Any rule value may be written "$name" to take a spec parameter; plain keys whose name matches
a parameter are also overridden (same as the other declarative rules).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantlab.strategies.base import MarketView, Strategy

INDICATORS = ("close_vs_sma", "close_vs_min", "close_vs_max", "bollinger_z", "rsi", "ibs", "return",
              "down_streak", "relative_return", "session")
OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}
SESSIONS = ("first_of_week", "last_of_week", "last_of_month")
MAX_HOLD_LIMIT = 252


class RuleError(ValueError):
    pass


def _lookback(cond: dict) -> int:
    ind = cond.get("indicator")
    d = int(cond.get("days", 0) or 0)
    if ind == "rsi":
        return max(10 * d, 50)  # Wilder smoothing needs a warm-up
    if ind in ("return", "relative_return"):
        return d + 1
    if ind == "down_streak":
        return 60
    return max(d, 2)


def validate_entry_exit(ee: Any, symbols: list[str]) -> list[str]:
    errs: list[str] = []
    if not isinstance(ee, dict) or not isinstance(ee.get("legs"), list) or not ee["legs"]:
        return ["rules.entry_exit.legs must be a non-empty list"]
    sizing = ee.get("sizing", "equal_active")
    if not (sizing == "equal_active" or (isinstance(sizing, dict) and int(sizing.get("slots", 0)) >= 1)):
        errs.append("rules.entry_exit.sizing must be equal_active or {slots: K}")
    names = set()
    for k, leg in enumerate(ee["legs"]):
        where = f"rules.entry_exit.legs[{k}]"
        if not isinstance(leg, dict):
            errs.append(f"{where} must be a mapping")
            continue
        name = leg.get("name", str(k))
        if name in names:
            errs.append(f"{where}.name '{name}' repeated")
        names.add(name)
        if leg.get("trade") not in symbols:
            errs.append(f"{where}.trade must be one of universe.symbols")
        entry = leg.get("entry")
        if not isinstance(entry, list) or not entry:
            errs.append(f"{where}.entry must be a non-empty list of conditions")
            entry = []
        ex = leg.get("exit")
        if not isinstance(ex, dict):
            errs.append(f"{where}.exit is required")
            ex = {}
        mh = ex.get("max_hold_days")
        if not (isinstance(mh, str) and mh.startswith("$")) and not (isinstance(mh, int) and 1 <= mh <= MAX_HOLD_LIMIT):
            errs.append(f"{where}.exit.max_hold_days is required (1..{MAX_HOLD_LIMIT})")
        for key in ("stop_loss", "take_profit"):
            v = ex.get(key)
            if v is not None and not (isinstance(v, str) and v.startswith("$")) and not (0 < float(v) < 10):
                errs.append(f"{where}.exit.{key} must be a positive fraction (0.1 = 10%)")
        for j, c in enumerate(list(entry) + list(ex.get("any") or [])):
            errs += _check_cond(c, symbols, f"{where} condition {j}")
    return errs


def _check_cond(c: Any, symbols: list[str], where: str) -> list[str]:
    if not isinstance(c, dict) or c.get("indicator") not in INDICATORS:
        return [f"{where}: indicator must be one of {INDICATORS}"]
    errs = []
    ind = c["indicator"]
    for key in ("of", "versus"):
        if key in c and c[key] not in symbols:
            errs.append(f"{where}: '{key}: {c[key]}' is not in universe.symbols")
    if ind == "session":
        if c.get("value") not in SESSIONS:
            errs.append(f"{where}: session value must be one of {SESSIONS}")
        return errs
    if c.get("op") not in OPS:
        errs.append(f"{where}: op must be one of {tuple(OPS)}")
    if "value" not in c:
        errs.append(f"{where}: value is required")
    if ind not in ("ibs", "down_streak") and "days" not in c:
        errs.append(f"{where}: days is required for {ind}")
    if ind == "relative_return" and "versus" not in c:
        errs.append(f"{where}: relative_return needs versus")
    return errs


def _adjusted(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split-adjusted (high, low, close) within a window, scaled so the last close is raw."""
    k = np.nan_to_num(v[:, 5], nan=1.0)
    # price p_j in the window is divided by the product of later split factors
    later = np.concatenate([np.cumprod(k[::-1])[::-1][1:], [1.0]])
    return v[:, 1] / later, v[:, 2] / later, v[:, 3] / later


def _wilder_rsi(close, n: int) -> float | None:
    c = np.asarray(close, float)
    d = np.diff(c[np.isfinite(c)])
    if len(d) < n:
        return None
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = up[:n].mean(), dn[:n].mean()
    for u, v in zip(up[n:], dn[n:]):
        au = (au * (n - 1) + u) / n
        ad = (ad * (n - 1) + v) / n
    if ad == 0:
        return 100.0 if au > 0 else 50.0
    return float(100 - 100 / (1 + au / ad))


def _value(view: MarketView, c: dict, trade: str, cache: dict) -> float | None:
    ind = c["indicator"]
    sym = c.get("of", trade)
    days = int(c.get("days", 0) or 0)

    def px(symbol: str):
        key = (symbol, _lookback(c))
        if key not in cache:
            cache[key] = _adjusted(view.tail_values(symbol, _lookback(c) + 1))
        return cache[key]

    high, low, raw_close = px(sym)
    if len(raw_close) == 0 or not np.isfinite(raw_close[-1]):
        return None  # no trade today: no decision on this symbol
    close = raw_close[np.isfinite(raw_close)]
    last = float(close[-1])
    if ind == "close_vs_sma":
        return float(last / close[-days:].mean() - 1) if len(close) >= days else None
    if ind == "close_vs_min":
        return float(last / close[-days:].min() - 1) if len(close) >= days else None
    if ind == "close_vs_max":
        return float(last / close[-days:].max() - 1) if len(close) >= days else None
    if ind == "bollinger_z":
        if len(close) < days:
            return None
        w = close[-days:]
        sd = float(w.std())
        return float((last - w.mean()) / sd) if sd > 0 else None
    if ind == "rsi":
        return _wilder_rsi(close, days)
    if ind == "ibs":
        hi, lo = float(high[-1]), float(low[-1])
        return (last - lo) / (hi - lo) if hi > lo else None
    if ind == "return":
        return float(last / close[-1 - days] - 1) if len(close) > days else None
    if ind == "down_streak":
        diffs = np.diff(close)[::-1]
        n = 0
        for x in diffs:
            if x < 0:
                n += 1
            else:
                break
        return float(n) if n < len(diffs) else None
    if ind == "relative_return":
        _, _, oc = px(c["versus"])
        if len(oc) == 0 or not np.isfinite(oc[-1]):
            return None
        other = oc[np.isfinite(oc)]
        if len(close) <= days or len(other) <= days:
            return None
        return float((last / close[-1 - days] - 1) - (other[-1] / other[-1 - days] - 1))
    raise RuleError(f"unknown indicator {ind}")


def _true(view: MarketView, c: dict, trade: str, cache: dict) -> bool:
    if c["indicator"] == "session":
        v = c["value"]
        if v == "last_of_week":
            return view.is_last_session_of_week()
        if v == "last_of_month":
            return view.is_last_session_of_month()
        return view.is_first_session_of_week()
    x = _value(view, c, trade, cache)
    return x is not None and bool(OPS[c["op"]](x, float(c["value"])))


class _Leg:
    __slots__ = ("spec", "decided", "last_seen", "held", "held_since", "fill", "split_after")

    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.reset()

    def reset(self) -> None:
        self.decided = None       # close at which the entry was decided
        self.last_seen = None     # last bar judged
        self.held = 0             # sessions held, fill day = 1
        self.held_since = None    # fill session
        self.fill = None          # raw fill price (open of the fill day)
        self.split_after = 1.0    # product of split factors after the fill day


class EntryExitStrategy(Strategy):
    """Runs validated entry/exit legs. State lives on the instance and is rebuilt every run."""

    def __init__(self, *, spec_id: str, spec_version_str: str, symbols: list[str], rules: dict, **params: Any):
        super().__init__(**params)
        self.strategy_id, self.version = spec_id, spec_version_str
        self.symbols = list(symbols)
        self.rules = rules
        ee = rules["entry_exit"]
        self.sizing = ee.get("sizing", "equal_active")
        self.legs = [_Leg(leg) for leg in ee["legs"]]
        self.trades: list[dict] = []

    def describe(self) -> dict:
        return {"strategy_id": self.strategy_id, "version": self.version, "params": self.params, "rules": self.rules}

    def on_bar(self, view: MarketView):
        changed = False
        cache: dict = {}
        now = view.now
        for leg in self.legs:
            s = leg.spec
            trade = s["trade"]
            if leg.decided is not None:
                if now <= leg.last_seen:
                    continue  # same bar again: nothing new to judge
                leg.last_seen = now
                today = view.tail_values(trade, 1)[0]  # open, high, low, close, dividend, split
                leg.held += 1
                if leg.held == 1:  # fill day: bought at today's open
                    leg.held_since, leg.fill, leg.split_after = now, float(today[0]), 1.0
                else:  # splits after the fill day restate the fill price onto today's share basis
                    leg.split_after *= float(today[5]) if np.isfinite(today[5]) else 1.0
                ex = s["exit"]
                last_raw = float(today[3])
                ratio = last_raw * leg.split_after / leg.fill if np.isfinite(last_raw) else 1.0
                hit = leg.held >= int(ex["max_hold_days"])
                if ex.get("stop_loss") is not None and ratio - 1 <= -float(ex["stop_loss"]):
                    hit = True
                if ex.get("take_profit") is not None and ratio - 1 >= float(ex["take_profit"]):
                    hit = True
                if not hit:
                    hit = any(_true(view, c, trade, cache) for c in ex.get("any") or [])
                if hit:
                    self.trades.append({"leg": s.get("name"), "symbol": trade, "decided": leg.decided,
                                        "filled": leg.held_since, "exit_decided": now})
                    leg.reset()
                    changed = True
            elif all(_true(view, c, trade, cache) for c in s["entry"]):
                leg.decided = leg.last_seen = now
                changed = True
        if not changed:
            return None
        return self._weights()

    def _weights(self) -> dict[str, float]:
        w = {s: 0.0 for s in self.symbols}
        active = [leg for leg in self.legs if leg.decided is not None]
        if not active:
            return w
        share = 1.0 / len(active) if self.sizing == "equal_active" else 1.0 / int(self.sizing["slots"])
        share = min(share, 1.0 / len(active))
        for leg in active:
            w[leg.spec["trade"]] += share
        return w


def resolve_params(rules: Any, params: dict) -> Any:
    """Replace "$name" values with parameter values."""
    if isinstance(rules, dict):
        return {k: resolve_params(v, params) for k, v in rules.items()}
    if isinstance(rules, list):
        return [resolve_params(v, params) for v in rules]
    if isinstance(rules, str) and rules.startswith("$"):
        name = rules[1:]
        if name not in params:
            raise RuleError(f"rule refers to unknown parameter '{name}'")
        return params[name]
    return rules

