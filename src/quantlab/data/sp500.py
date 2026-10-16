"""S&P 500 history: daily prices of every index member since 1995, delisted names included, and
point-in-time membership (D-026).

Source: the public GitHub project Johnbrick123/sp500-data (release asset `data`), free and without
a key. It joins Yahoo, Tiingo, CRSP-derived and other public copies; it carries no licence, so the
files are for the user's own research only and stay in the work folder (never in the repository
or in result packets).

    prices.parquet  : date, ticker, open, high, low, close, volume, adj_close, dividends, stock_splits, source
                      `close` is split-adjusted; `adj_close` is split- and dividend-adjusted.
                      Delisted names carry a suffix, e.g. AAL-199702.
    members.parquet : date, ticker; one snapshot of the index on each date its membership changed

The app turns this into total-return prices: every price is scaled by adj_close / close, with
dividends 0 and split factor 1. Returns are then exact; price levels are not the traded ones, so
rules must not depend on price levels. Where a source has no open (the CRSP-derived rows), the
open is set to that day's close: an order then fills at the close of the day after the signal.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.core.hashing import file_sha256

RELEASE_URL = "https://github.com/Johnbrick123/sp500-data/releases/download/data/"
FILES = ("prices.parquet", "members.parquet")
SOURCE = {"provider": "github:Johnbrick123/sp500-data", "release": "data"}
MIN_MEMBER_COVERAGE = 0.98     # share of member-sessions with a price; below this the dataset is unusable
WARN_MEMBER_COVERAGE = 0.995


def bundle_dir(root: str | Path) -> Path:
    return Path(root) / "data" / "bulk" / "sp500-data"


def download_bundle(root: str | Path, session=None, say=None, timeout: float = 120.0) -> dict:
    """Download both files as received; returns the bundle record (also written as meta.json)."""
    import requests

    say = say or (lambda s: None)
    http = session or requests
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d = bundle_dir(root) / ts
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in FILES:
        say(f"S&P 500 자료 받는 중: {name}")
        resp = http.get(RELEASE_URL + name, timeout=timeout, stream=True)
        if resp.status_code != 200:
            raise RuntimeError(f"S&P 500 자료 {name}: HTTP {resp.status_code}")
        tmp = d / (name + ".part")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(d / name)
        files[name] = {"sha256": file_sha256(d / name), "bytes": (d / name).stat().st_size}
    meta = {"source": SOURCE, "url": RELEASE_URL, "fetched_at": ts, "dir": str(d), "files": files}
    (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def bundles(root: str | Path) -> list[dict]:
    out = []
    for p in sorted(bundle_dir(root).glob("*/meta.json")):
        meta = json.loads(p.read_text(encoding="utf-8"))
        meta["dir"] = str(p.parent)
        out.append(meta)
    return sorted(out, key=lambda m: m["fetched_at"])


def read_bundle(meta: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = Path(meta["dir"])
    for name, f in meta["files"].items():
        if file_sha256(d / name) != f["sha256"]:
            raise ValueError(f"S&P 500 자료 파일이 받은 뒤 바뀌었습니다: {d / name}")
    prices = pd.read_parquet(d / "prices.parquet")
    members = pd.read_parquet(d / "members.parquet")
    need_p = {"date", "ticker", "open", "high", "low", "close", "adj_close"}
    if need_p - set(prices.columns) or {"date", "ticker"} - set(members.columns):
        raise ValueError("S&P 500 자료의 열 구성이 예상과 다릅니다 (원본 형식이 바뀐 것 같습니다)")
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    members["date"] = pd.to_datetime(members["date"]).dt.normalize()
    return prices, members


def membership(members: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Bool frame (sessions x tickers): each snapshot holds from its date until the next snapshot."""
    snap = members.assign(v=True).pivot_table(index="date", columns="ticker", values="v", aggfunc="any",
                                              fill_value=False).sort_index()
    snap = snap.astype(bool)
    pos = snap.index.searchsorted(sessions, side="right") - 1
    out = np.zeros((len(sessions), snap.shape[1]), dtype=bool)
    ok = pos >= 0
    out[ok] = snap.to_numpy()[pos[ok]]
    frame = pd.DataFrame(out, index=sessions, columns=snap.columns)
    return frame.loc[:, frame.any(axis=0)]


def total_return_frames(prices: pd.DataFrame, tickers, sessions: pd.DatetimeIndex) -> tuple[dict, dict]:
    """Per-ticker frames on the session calendar (see module docstring) and counts for the report."""
    p = prices[prices["ticker"].isin(set(tickers))]
    p = p[(p["date"] >= sessions[0]) & (p["date"] <= sessions[-1])]
    counts = {"rows": int(len(p)), "off_calendar": 0, "duplicates": 0, "bad_price": 0, "open_from_close": 0}
    on_cal = p["date"].isin(sessions)
    counts["off_calendar"] = int((~on_cal).sum())
    p = p[on_cal]
    dup = p.duplicated(["ticker", "date"], keep="last")
    counts["duplicates"] = int(dup.sum())
    p = p[~dup]
    bad = ~((p["close"] > 0) & (p["adj_close"] > 0))
    counts["bad_price"] = int(bad.sum())
    p = p[~bad].copy()
    factor = p["adj_close"] / p["close"]
    no_open = ~(p["open"] > 0)
    counts["open_from_close"] = int(no_open.sum())
    p.loc[no_open, "open"] = p.loc[no_open, "close"]
    p["high"] = p["high"].where(p["high"] > 0, p[["open", "close"]].max(axis=1))
    p["low"] = p["low"].where(p["low"] > 0, p[["open", "close"]].min(axis=1))
    p["high"] = p[["high", "open", "close"]].max(axis=1)
    p["low"] = p[["low", "open", "close"]].min(axis=1)
    for c in ("open", "high", "low", "close"):
        p[c] = p[c] * factor
    frames = {}
    for t, g in p.groupby("ticker", sort=True):
        f = g.set_index("date")[["open", "high", "low", "close"]].reindex(sessions)
        f["dividend"] = 0.0
        f["split"] = 1.0
        f.index.name = "date"
        frames[t] = f.astype(float)
    return frames, counts


def coverage(members: pd.DataFrame, frames: dict, start, end=None) -> dict:
    """Share of member-sessions with a closing price, overall and per year; tickers with gaps."""
    m = members.loc[pd.Timestamp(start): (pd.Timestamp(end) if end else None)]
    have = pd.DataFrame({t: frames[t]["close"].reindex(m.index).notna() if t in frames else False
                         for t in m.columns}, index=m.index)
    hit = (m & have).to_numpy().sum()
    total = m.to_numpy().sum()
    by_year = {}
    for y, part in m.groupby(m.index.year):
        tot = part.to_numpy().sum()
        by_year[str(y)] = float((part & have.loc[part.index]).to_numpy().sum() / tot) if tot else None
    per_ticker = ((m & have).sum() / m.sum().replace(0, np.nan)).dropna()
    gaps = per_ticker[per_ticker < 0.9].sort_values()
    return {"share": float(hit / total) if total else 0.0, "member_sessions": int(total), "by_year": by_year,
            "tickers": int((m.sum() > 0).sum()), "tickers_without_prices": sorted(per_ticker[per_ticker == 0].index),
            "tickers_mostly_missing": {t: round(float(v), 3) for t, v in gaps.head(30).items()}}
