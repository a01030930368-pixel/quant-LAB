"""Data quality checks (protocol 제6조, 03 v2.0.0 §3.3 B).

Each check returns ok / warn / fatal with a human-readable detail. Any fatal
finding makes experiments on that dataset INVALID (Gate 1).

Input frame (one symbol): DatetimeIndex, columns open/high/low/close (raw traded
prices), dividend, split, optional adj_close (provider's adjusted close).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

MISSING_SESSION_WARN_FRACTION = 0.005   # up to 0.5% missing sessions → warn, above → fatal
OHLC_FATAL_FRACTION = 0.001
JUMP_WARN_ABS_RETURN = 0.15
ADJ_TOLERANCE = 1e-4                    # per-day total-return difference vs provider adjusted close
ADJ_FATAL_FRACTION = 0.005


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fatal
    detail: str
    dates: list[str] = field(default_factory=list)


@dataclass
class QCReport:
    symbol: str
    checks: list[Check]

    @property
    def fatal(self) -> list[str]:
        return [f"{self.symbol}: {c.name}: {c.detail}" for c in self.checks if c.status == "fatal"]

    @property
    def warnings(self) -> list[str]:
        return [f"{self.symbol}: {c.name}: {c.detail}" for c in self.checks if c.status == "warn"]

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "checks": [asdict(c) for c in self.checks]}


def _dates(idx, limit=20) -> list[str]:
    return [str(pd.Timestamp(d).date()) for d in list(idx)[:limit]]


def exchange_sessions(start, end, exchange: str = "XNYS") -> pd.DatetimeIndex:
    import exchange_calendars as xc

    cal = xc.get_calendar(exchange, start=pd.Timestamp(start) - pd.Timedelta(days=7))
    return pd.DatetimeIndex(cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))).tz_localize(None)


def check_calendar(df: pd.DataFrame, sessions: pd.DatetimeIndex) -> list[Check]:
    idx = df.index[df["close"].notna()]
    extra = idx.difference(sessions)
    missing = sessions.difference(idx)
    out = []
    out.append(Check("non-session dates", "fatal" if len(extra) else "ok",
                     f"{len(extra)} rows dated on exchange holidays/weekends", _dates(extra)))
    frac = len(missing) / max(len(sessions), 1)
    status = "ok" if not len(missing) else ("warn" if frac <= MISSING_SESSION_WARN_FRACTION else "fatal")
    out.append(Check("missing sessions", status, f"{len(missing)} of {len(sessions)} sessions have no price",
                     _dates(missing)))
    return out


def check_values(df: pd.DataFrame) -> list[Check]:
    out = []
    dup = df.index[df.index.duplicated()]
    out.append(Check("duplicate dates", "fatal" if len(dup) else "ok", f"{len(dup)} duplicated dates", _dates(dup)))
    px = df[["open", "high", "low", "close"]]
    bad = df.index[(px <= 0).any(axis=1)]
    out.append(Check("non-positive prices", "fatal" if len(bad) else "ok", f"{len(bad)} rows", _dates(bad)))
    nan_part = df.index[px.isna().any(axis=1) & px.notna().any(axis=1)]
    out.append(Check("partial OHLC", "warn" if len(nan_part) else "ok", f"{len(nan_part)} rows with some fields missing",
                     _dates(nan_part)))
    viol = df.index[(df["low"] > df[["open", "close"]].min(axis=1) + 1e-9)
                    | (df["high"] < df[["open", "close"]].max(axis=1) - 1e-9)
                    | (df["low"] > df["high"] + 1e-9)]
    frac = len(viol) / max(len(df), 1)
    out.append(Check("OHLC consistency", "ok" if not len(viol) else ("fatal" if frac > OHLC_FATAL_FRACTION else "warn"),
                     f"{len(viol)} rows where low/high do not bracket open/close", _dates(viol)))
    neg = df.index[(df["dividend"] < 0) | (df["split"] <= 0)]
    out.append(Check("corporate action values", "fatal" if len(neg) else "ok",
                     f"{len(neg)} negative dividends or non-positive split factors", _dates(neg)))
    return out


def daily_total_return(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    return (close + df["dividend"]) * df["split"] / close.shift(1) - 1


def check_jumps(df: pd.DataFrame) -> list[Check]:
    r = daily_total_return(df)
    big = r.index[r.abs() > JUMP_WARN_ABS_RETURN]
    return [Check("large daily moves", "warn" if len(big) else "ok",
                  f"{len(big)} days with |total return| > {JUMP_WARN_ABS_RETURN:.0%} (check for missing splits)",
                  _dates(big))]


def check_adjusted(df: pd.DataFrame) -> list[Check]:
    if "adj_close" not in df or df["adj_close"].isna().all():
        return [Check("adjusted-price reconciliation", "ok", "provider gave no adjusted close; skipped")]
    ours = daily_total_return(df)
    theirs = df["adj_close"] / df["adj_close"].shift(1) - 1
    diff = (ours - theirs).abs()
    bad = diff.index[diff > ADJ_TOLERANCE]
    frac = len(bad) / max(diff.notna().sum(), 1)
    status = "ok" if not len(bad) else ("fatal" if frac > ADJ_FATAL_FRACTION else "warn")
    return [Check("adjusted-price reconciliation", status,
                  f"{len(bad)} days where raw prices + dividends + splits disagree with the provider's adjusted "
                  f"close by more than {ADJ_TOLERANCE:g}", _dates(bad))]


def check_revisions(new: pd.DataFrame, old: pd.DataFrame | None) -> list[Check]:
    """Provider retroactively changed values already downloaded (protocol 제6조 10항)."""
    if old is None or old.empty:
        return [Check("provider revisions", "ok", "no earlier download to compare")]
    common = new.index.intersection(old.index)
    cols = [c for c in ("open", "high", "low", "close", "dividend", "split") if c in old and c in new]
    a, b = new.loc[common, cols], old.loc[common, cols]
    changed = common[((a - b).abs() > 1e-9 * b.abs().clip(lower=1)).any(axis=1)]
    return [Check("provider revisions", "warn" if len(changed) else "ok",
                  f"{len(changed)} previously downloaded days changed value", _dates(changed))]


def run_qc(symbol: str, df: pd.DataFrame, sessions: pd.DatetimeIndex | None = None,
           previous: pd.DataFrame | None = None) -> QCReport:
    if sessions is None:
        sessions = exchange_sessions(df.index.min(), df.index.max())
    checks = check_values(df)
    clean = df[~df.index.duplicated()]
    checks += check_calendar(clean, sessions) + check_jumps(clean) + check_adjusted(clean)
    checks += check_revisions(clean, previous)
    return QCReport(symbol, checks)
