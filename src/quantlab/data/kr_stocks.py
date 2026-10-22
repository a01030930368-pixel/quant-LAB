"""Korean stock universe KRTOP300: the 300 largest common stocks of KOSPI and KOSDAQ, point in time,
with adjusted daily prices (D-038).

Input: the exchange's daily snapshots (`quantlab.data.krx_open`), traded prices.

Adjusted prices. The exchange quotes each day's change against its base price (기준가), which it
resets for splits, reverse splits, bonus and rights issues; it does not reset it for cash dividends.
So close / (close - change) - 1 is the day's price return across every corporate action, and the
adjusted close is the product of those returns. Open, high and low are scaled by the same factor as
the close of their day. Dividends are left out (price returns; Korean yields are about 1-2% a year).
A day without a trade (volume 0: a halt) keeps its close for marking positions and has no open, so
orders wait until trading resumes.

Membership, fixed at the close of each month's last session and held for the next month:
    common stock (short code ends in 0), not a SPAC, REIT, infrastructure or ship fund, not under
    administrative issue (관리종목) or investment-caution environment (투자주의환기) that day;
    traded on at least 120 of the last 126 sessions (all of them in the first half year of data);
    median daily traded value over the last 20 sessions at least MIN_VALUE won;
    then the TOP_N largest by market capitalisation.
A member counts on a session only when it traded that session.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TOP_N = 300
MIN_VALUE = 2e9                      # won per day, median of 20 sessions
SEASON = (120, 126)                  # traded sessions out of the last ones
EXCLUDE_NAME = re.compile(r"스팩|기업인수목적|리츠|REIT|인프라|선박|투융자|유전|맥쿼리|\(Reg\.?S\)", re.I)
EXCLUDE_SECTION = re.compile(r"관리|환기|SPAC", re.I)
LIMIT_CHANGE = pd.Timestamp("2015-06-15")    # daily price limit ±15% before, ±30% from this day
SOURCE = {"provider": "krx-open-api", "universe": "KRTOP300"}


def common(codes: pd.Series, names: pd.Series) -> pd.Series:
    c = codes.astype(str)
    return (c.str.len() == 6) & c.str.endswith("0") & ~names.astype(str).str.contains(EXCLUDE_NAME)


def _wide(df: pd.DataFrame, col: str, sessions: pd.DatetimeIndex, codes) -> pd.DataFrame:
    return df.pivot_table(index="date", columns="code", values=col, aggfunc="last").reindex(
        index=sessions, columns=codes)


def adjusted(df: pd.DataFrame, sessions: pd.DatetimeIndex, codes) -> tuple[dict, dict]:
    """Per-code frames (open, high, low, close adjusted; dividend 0, split 1) on `sessions`, and counts."""
    d = df[df["code"].isin(set(codes)) & df["date"].isin(sessions)].copy()
    counts = {"rows": int(len(d)), "duplicates": 0, "bad_price": 0, "halted_rows": 0, "beyond_limit": 0}
    dup = d.duplicated(["code", "date"], keep="last")
    counts["duplicates"] = int(dup.sum())
    d = d[~dup]
    bad = ~(d["close"] > 0)
    counts["bad_price"] = int(bad.sum())
    d = d[~bad].sort_values(["code", "date"])
    base = d["close"] - d["change"].fillna(0.0)
    r = (d["close"] / base.where(base > 0)).fillna(1.0)
    first = ~d["code"].duplicated()
    r[first] = 1.0                                     # the listing day starts the series
    d["adj"] = r.groupby(d["code"]).cumprod()
    # anchored at each code's last close, so recent levels are the traded ones
    d["adj"] *= d.groupby("code")["close"].transform("last") / d.groupby("code")["adj"].transform("last")
    f = d["adj"] / d["close"]
    halted = ~(d["volume"] > 0) | ~(d["open"] > 0)
    counts["halted_rows"] = int(halted.sum())
    limit = np.where(d["date"] < LIMIT_CHANGE, 0.15, 0.30)
    counts["beyond_limit"] = int(((r - 1).abs() > limit + 0.01).sum())
    out = pd.DataFrame({"code": d["code"], "date": d["date"], "close": d["adj"]})
    for c in ("open", "high", "low"):
        out[c] = (d[c] * f).where(~halted)
    out["high"] = out[["high", "open", "close"]].max(axis=1).where(~halted)
    out["low"] = out[["low", "open", "close"]].min(axis=1).where(~halted)
    frames = {}
    for code, g in out.groupby("code", sort=True):
        fr = g.set_index("date")[["open", "high", "low", "close"]].reindex(sessions)
        fr["dividend"] = 0.0
        fr["split"] = 1.0
        fr.index.name = "date"
        frames[code] = fr.astype(float)
    return frames, counts


def membership(df: pd.DataFrame, sessions: pd.DatetimeIndex, top_n: int = TOP_N,
               min_value: float = MIN_VALUE) -> pd.DataFrame:
    """Bool frame (sessions x codes), see the module docstring."""
    d = df[df["date"].isin(sessions)]
    d = d[common(d["code"], d["name"])]
    codes = sorted(d["code"].unique())
    ok_day = d.assign(ok=~d["section"].astype(str).str.contains(EXCLUDE_SECTION) & (d["volume"] > 0))
    traded = _wide(ok_day.assign(t=(d["volume"] > 0).astype(float)), "t", sessions, codes).fillna(0.0)
    okw = _wide(ok_day.assign(o=ok_day["ok"].astype(float)), "o", sessions, codes).fillna(0.0) > 0
    value = _wide(d, "value", sessions, codes)
    cap = _wide(d, "mktcap", sessions, codes)
    n_traded = traded.rolling(SEASON[1], min_periods=1).sum()
    n_sessions = pd.Series(np.minimum(np.arange(1, len(sessions) + 1), SEASON[1]), index=sessions)
    need = np.where(n_sessions < SEASON[1], n_sessions, SEASON[0])
    seasoned = n_traded.ge(pd.Series(need, index=sessions), axis=0)
    med_value = value.where(traded > 0).rolling(20, min_periods=10).median()
    month_end = sessions[:-1][sessions[1:].month != sessions[:-1].month]
    picks = pd.DataFrame(False, index=sessions, columns=codes)
    for i, day in enumerate(month_end):
        elig = okw.loc[day] & seasoned.loc[day] & (med_value.loc[day] >= min_value) & (cap.loc[day] > 0)
        top = cap.loc[day][elig].sort_values(ascending=False, kind="stable").index[:top_n]
        lo = sessions.searchsorted(day, "right")
        hi = sessions.searchsorted(month_end[i + 1], "right") if i + 1 < len(month_end) else len(sessions)
        picks.iloc[lo:hi, picks.columns.get_indexer(top)] = True
    members = picks & okw
    return members.loc[:, members.any(axis=0)]


def coverage(members: pd.DataFrame, frames: dict) -> dict:
    have = pd.DataFrame({c: frames[c]["close"].reindex(members.index).notna() if c in frames else False
                         for c in members.columns}, index=members.index)
    total = members.to_numpy().sum()
    hit = (members & have).to_numpy().sum()
    per_day = members.sum(axis=1)
    started = per_day[per_day > 0]
    return {"share": float(hit / total) if total else 0.0, "member_sessions": int(total),
            "stocks": int(members.shape[1]), "first_member_day": str(started.index[0].date()) if len(started) else None,
            "members_per_day": {"min": int(started.min()) if len(started) else 0,
                                "median": float(started.median()) if len(started) else 0.0}}
