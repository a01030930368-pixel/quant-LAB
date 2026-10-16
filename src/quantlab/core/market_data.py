"""In-memory market data container used by strategies and the backtester.

Prices are UNADJUSTED; corporate actions are explicit columns so that the
backtester can do cash and share accounting directly:

    open, high, low, close : raw traded prices (NaN on days the asset did not trade)
    dividend               : cash dividend per share, credited on its ex-date
    split                  : split factor effective at the open of that date (1.0 = none);
                             2.0 means one old share becomes two new shares

All symbols share one trading calendar index (the exchange schedule, which is
public in advance and therefore not look-ahead information).

Two optional point-in-time tables serve stock-universe strategies:

    members : bool frame (calendar x symbols), True on the sessions a symbol was an index member
    events  : one row per dated fact (e.g. an insider purchase filing), sorted by `known`, the
              calendar date it became public; columns `known`, `symbol` and kind-specific fields

Strategies read both only through `MarketView`, which never shows a row dated after the bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "dividend", "split")


@dataclass(frozen=True)
class MarketData:
    calendar: pd.DatetimeIndex
    frames: dict[str, pd.DataFrame]
    members: pd.DataFrame | None = field(default=None, compare=False)
    events: pd.DataFrame | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not self.calendar.is_monotonic_increasing or self.calendar.has_duplicates:
            raise ValueError("calendar must be strictly increasing")
        for sym, df in self.frames.items():
            missing = [f for f in FIELDS if f not in df.columns]
            if missing:
                raise ValueError(f"{sym}: missing fields {missing}")
            if not df.index.equals(self.calendar):
                raise ValueError(f"{sym}: index must equal the shared calendar")
        if self.members is not None and not self.members.index.equals(self.calendar):
            raise ValueError("members: index must equal the shared calendar")
        if self.events is not None:
            if not {"known", "symbol"} <= set(self.events.columns):
                raise ValueError("events need columns 'known' and 'symbol'")
            if not self.events["known"].is_monotonic_increasing:
                raise ValueError("events must be sorted by 'known'")

    @classmethod
    def from_frames(cls, frames: dict[str, pd.DataFrame]) -> "MarketData":
        cal = None
        for df in frames.values():
            cal = df.index if cal is None else cal.union(df.index)
        cal = pd.DatetimeIndex(cal).sort_values()
        out = {}
        for sym, df in frames.items():
            d = df.reindex(cal).copy()
            for f in ("dividend",):
                d[f] = d[f].fillna(0.0) if f in d else 0.0
            d["split"] = d["split"].fillna(1.0) if "split" in d else 1.0
            out[sym] = d[list(FIELDS)].astype(float)
        return cls(cal, out)

    def values(self, symbol: str) -> np.ndarray:
        """Read-only (n x 6) array of FIELDS for one symbol, cached per instance."""
        cache = self.__dict__.setdefault("_values", {})
        if symbol not in cache:
            arr = self.frames[symbol][list(FIELDS)].to_numpy(dtype=float, copy=True)
            arr.setflags(write=False)
            cache[symbol] = arr
        return cache[symbol]

    @property
    def symbols(self) -> list[str]:
        return list(self.frames)

    def field(self, name: str) -> pd.DataFrame:
        return pd.DataFrame({s: df[name] for s, df in self.frames.items()}, index=self.calendar)

    def slice(self, start=None, end=None) -> "MarketData":
        cal = self.calendar
        mask = np.ones(len(cal), dtype=bool)
        if start is not None:
            mask &= cal >= pd.Timestamp(start)
        if end is not None:
            mask &= cal <= pd.Timestamp(end)
        new_cal = cal[mask]
        members = self.members.loc[new_cal] if self.members is not None else None
        events = self.events
        if events is not None and end is not None:
            events = events[events["known"] <= pd.Timestamp(end)].reset_index(drop=True)
        return MarketData(new_cal, {s: df.loc[new_cal] for s, df in self.frames.items()}, members, events)

    def with_tables(self, members: pd.DataFrame | None, events: pd.DataFrame | None) -> "MarketData":
        return MarketData(self.calendar, self.frames, members, events)


def total_return_index(df: pd.DataFrame) -> pd.Series:
    """Forward-built total return index (start = 1.0).

    Built only from data up to each date, so truncating the input never changes
    earlier values (unlike back-adjusted prices, which embed future dividends).
    """
    close = df["close"].ffill()
    prev = close.shift(1)
    gross = (close + df["dividend"]) * df["split"] / prev
    gross = gross.where(df["close"].notna() & prev.notna(), 1.0)
    return gross.fillna(1.0).cumprod()
