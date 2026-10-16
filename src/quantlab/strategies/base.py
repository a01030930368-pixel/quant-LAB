"""Strategy interface and the point-in-time market view.

A strategy only ever sees a `MarketView` positioned at the close of day t.
It cannot read prices after t, cannot place fills, and cannot see metrics or
validation settings. It returns target portfolio weights; the backtester turns
those into orders that fill at the NEXT trading day's open (protocol 제8조).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np
import pandas as pd

from quantlab.core.hashing import stable_hash
from quantlab.core.market_data import MarketData, total_return_index


class PartialWeights(dict):
    """Target weights for the listed symbols only: the engine leaves every other position as it is.

    A plain dict re-targets every symbol (unlisted ones to 0), which also trims or tops up
    holdings whose weight drifted. Swing strategies that size each position once at entry
    return this instead, so a new entry does not trade the positions already held.
    """


class LookaheadError(RuntimeError):
    """Raised when strategy code asks for information dated after the current bar."""


class MarketView:
    __slots__ = ("_data", "_t")

    def __init__(self, data: MarketData, t: int) -> None:
        self._data = data
        self._t = t

    @property
    def now(self) -> pd.Timestamp:
        return self._data.calendar[self._t]

    @property
    def symbols(self) -> list[str]:
        return self._data.symbols

    def _frame(self, symbol: str) -> pd.DataFrame:
        return self._data.frames[symbol].iloc[: self._t + 1]

    def history(self, symbol: str, field: str = "close") -> pd.Series:
        return self._frame(symbol)[field].copy()

    def frame(self, symbol: str) -> pd.DataFrame:
        """All fields up to and including now (a copy; edits do not reach the data)."""
        return self._frame(symbol).copy()

    def tail(self, symbol: str, n: int) -> pd.DataFrame:
        """The last n rows up to and including now (a copy); cheaper than frame() for indicators."""
        lo = max(0, self._t + 1 - n)
        return self._data.frames[symbol].iloc[lo: self._t + 1].copy()

    def tail_values(self, symbol: str, n: int) -> np.ndarray:
        """Last n rows up to and including now as an array (columns: open, high, low, close,
        dividend, split). A copy; fast path for indicator code."""
        lo = max(0, self._t + 1 - n)
        return self._data.values(symbol)[lo: self._t + 1].copy()

    def indicator(self, symbol: str, key: tuple) -> float | None:
        """One of the app's causal indicators (`quantlab.strategies.indicators`) at the current bar.

        The whole series is computed once per dataset and cached; only today's value is returned."""
        from quantlab.strategies.indicators import series

        cache = self._data.__dict__.setdefault("_indicators", {})
        arr = cache.get((symbol, key))
        if arr is None:
            arr = series(self._data.values(symbol), key)
            arr.setflags(write=False)
            cache[(symbol, key)] = arr
        v = arr[self._t]
        return float(v) if np.isfinite(v) else None

    def _panels(self, builder) -> dict:
        cache = self._data.__dict__.setdefault("_panels", {})
        name = f"{builder.__module__}.{builder.__qualname__}"
        if name not in cache:
            cache[name] = builder(self._data)
        return cache[name]

    def panel(self, builder, name: str, back: int = 0) -> np.ndarray | None:
        """Row `back` sessions before now of a whole-dataset table (sessions x columns) built by app
        code: `builder(data)` returns {name: array}, built once per dataset from data up to each row
        only (as `indicator`). Read-only; None before the first session."""
        t = self._t - back
        if t < 0:
            return None
        return self._panels(builder)[name][t]

    def panel_columns(self, builder) -> tuple:
        """Column names of `builder`'s tables (its 'symbols' entry)."""
        return self._panels(builder)["symbols"]

    def is_member(self, symbol: str) -> bool:
        """Index membership today (True when the data carries no membership table)."""
        m = self._data.members
        if m is None:
            return True
        return bool(m[symbol].iat[self._t]) if symbol in m.columns else False

    def events(self, days: int) -> pd.DataFrame:
        """Events that became public in the last `days` calendar days, today included (a copy).

        `known` is a calendar date; a fact dated on a weekend or holiday shows up at the next bar."""
        ev = self._data.events
        if ev is None:
            return pd.DataFrame(columns=["known", "symbol"])
        known = self._data.__dict__.get("_events_known")
        if known is None:
            known = ev["known"].to_numpy(dtype="datetime64[ns]")
            self._data.__dict__["_events_known"] = known
        now = self.now
        lo = np.searchsorted(known, np.datetime64(now - pd.Timedelta(days=days - 1), "ns"), side="left")
        hi = np.searchsorted(known, np.datetime64(now, "ns"), side="right")
        return ev.iloc[lo:hi].copy()

    def event_columns(self, days: int, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
        """Same rows as `events(days)`, as read-only arrays of the named columns that exist (fast path)."""
        ev = self._data.events
        if ev is None:
            return {"known": np.array([], dtype="datetime64[ns]")}
        columns = ("known",) + tuple(c for c in columns if c in ev.columns and c != "known")
        cache = self._data.__dict__.setdefault("_event_arrays", {})
        for c in columns:
            if c not in cache:
                arr = ev[c].to_numpy(dtype="datetime64[ns]" if c == "known" else None)
                arr = arr.copy()
                arr.setflags(write=False)
                cache[c] = arr
        known = cache["known"]
        now = self.now
        lo = np.searchsorted(known, np.datetime64(now - pd.Timedelta(days=days - 1), "ns"), side="left")
        hi = np.searchsorted(known, np.datetime64(now, "ns"), side="right")
        return {c: cache[c][lo:hi] for c in columns}

    def session(self, back: int) -> pd.Timestamp | None:
        """Date of the session `back` sessions before now (0 = now); the calendar is public."""
        return self._data.calendar[self._t - back] if 0 <= back <= self._t else None

    def previous_session(self) -> pd.Timestamp | None:
        return self._data.calendar[self._t - 1] if self._t > 0 else None

    def total_return(self, symbol: str) -> pd.Series:
        return total_return_index(self._frame(symbol))

    def at(self, symbol: str, date, field: str = "close") -> float:
        date = pd.Timestamp(date)
        if date > self.now:
            raise LookaheadError(f"requested {symbol} {field} on {date.date()} at bar {self.now.date()}")
        return float(self._frame(symbol).loc[date, field])

    # The exchange schedule is published in advance, so knowing whether the
    # next session falls in a new month is not look-ahead.
    def is_last_session_of_month(self) -> bool:
        cal = self._data.calendar
        if self._t + 1 >= len(cal):
            return True
        return cal[self._t + 1].month != cal[self._t].month

    def is_first_session_of_week(self) -> bool:
        cal = self._data.calendar
        if self._t == 0:
            return True
        return cal[self._t - 1].isocalendar()[:2] != cal[self._t].isocalendar()[:2]

    def is_last_session_of_week(self) -> bool:
        cal = self._data.calendar
        if self._t + 1 >= len(cal):
            return True
        return cal[self._t + 1].isocalendar()[:2] != cal[self._t].isocalendar()[:2]


class Strategy(ABC):
    strategy_id: str = "UNSET"
    version: str = "0"

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    @abstractmethod
    def on_bar(self, view: MarketView) -> Mapping[str, float] | None:
        """Return target weights (long-only, sum <= 1) or None to keep current holdings."""

    def describe(self) -> dict[str, Any]:
        return {"strategy_id": self.strategy_id, "version": self.version, "params": self.params}

    def config_hash(self) -> str:
        return stable_hash(self.describe())