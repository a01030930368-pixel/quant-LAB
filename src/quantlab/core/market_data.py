"""In-memory market data container used by strategies and the backtester.

Prices are UNADJUSTED; corporate actions are explicit columns so that the
backtester can do cash and share accounting directly:

    open, high, low, close : raw traded prices (NaN on days the asset did not trade)
    dividend               : cash dividend per share, credited on its ex-date
    split                  : split factor effective at the open of that date (1.0 = none);
                             2.0 means one old share becomes two new shares

All symbols share one trading calendar index (the exchange schedule, which is
public in advance and therefore not look-ahead information).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "dividend", "split")


@dataclass(frozen=True)
class MarketData:
    calendar: pd.DatetimeIndex
    frames: dict[str, pd.DataFrame]

    def __post_init__(self) -> None:
        if not self.calendar.is_monotonic_increasing or self.calendar.has_duplicates:
            raise ValueError("calendar must be strictly increasing")
        for sym, df in self.frames.items():
            missing = [f for f in FIELDS if f not in df.columns]
            if missing:
                raise ValueError(f"{sym}: missing fields {missing}")
            if not df.index.equals(self.calendar):
                raise ValueError(f"{sym}: index must equal the shared calendar")

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
        return MarketData(new_cal, {s: df.loc[new_cal] for s, df in self.frames.items()})


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
