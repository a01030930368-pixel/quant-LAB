"""Market data sources: Tiingo EOD (US), Naver Finance daily charts (KRX ETFs, D-035) and CSV import.

Both return the same normalized frame: DatetimeIndex (naive, session date),
columns open, high, low, close (raw traded prices), volume, dividend, split,
adj_close (provider's adjusted close, NaN for CSV unless supplied).

API keys come from the OS credential store (Windows Credential Manager via
`keyring`) or the TIINGO_API_KEY environment variable. Keys are never written
to files, logs, the registry or result packets.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume", "dividend", "split", "adj_close"]
KEYRING_SERVICE = "QuantLab"
TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{symbol}/prices"


class ProviderError(RuntimeError):
    pass


# --- credentials ---------------------------------------------------------------
def get_api_key(provider: str = "tiingo") -> str | None:
    try:
        import keyring

        key = keyring.get_password(KEYRING_SERVICE, provider)
        if key:
            return key
    except Exception:  # no usable keyring backend (e.g. headless Linux)
        pass
    return os.environ.get(f"{provider.upper()}_API_KEY") or None


def set_api_key(key: str, provider: str = "tiingo") -> None:
    import keyring

    keyring.set_password(KEYRING_SERVICE, provider, key)


# --- Tiingo ----------------------------------------------------------------------
def parse_tiingo(rows: list[dict]) -> pd.DataFrame:
    if not isinstance(rows, list):
        raise ProviderError(f"unexpected Tiingo response: {str(rows)[:200]}")
    if not rows:
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="date"))
    df = pd.DataFrame(rows)
    missing = {"date", "open", "high", "low", "close", "divCash", "splitFactor"} - set(df.columns)
    if missing:
        raise ProviderError(f"Tiingo response lacks fields {sorted(missing)}")
    out = pd.DataFrame({
        "open": df["open"].astype(float), "high": df["high"].astype(float), "low": df["low"].astype(float),
        "close": df["close"].astype(float), "volume": df.get("volume", np.nan),
        "dividend": df["divCash"].astype(float), "split": df["splitFactor"].astype(float),
        "adj_close": df.get("adjClose", np.nan),
    })
    out.index = pd.DatetimeIndex(pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize(), name="date")
    return out[COLUMNS].astype(float)


def fetch_tiingo(symbol: str, start: str, end: str | None = None, api_key: str | None = None,
                 session=None, timeout: float = 30.0) -> tuple[pd.DataFrame, bytes]:
    """Returns the normalized frame and the raw response bytes (stored unmodified)."""
    import requests

    key = api_key or get_api_key("tiingo")
    if not key:
        raise ProviderError("no Tiingo API key: set it in the Data screen (stored in the OS credential manager)")
    params = {"startDate": start, "format": "json", "resampleFreq": "daily"}
    if end:
        params["endDate"] = end
    http = session or requests
    resp = http.get(TIINGO_URL.format(symbol=symbol.lower()), params=params,
                    headers={"Authorization": f"Token {key}", "Content-Type": "application/json"}, timeout=timeout)
    if resp.status_code != 200:
        raise ProviderError(f"Tiingo {symbol}: HTTP {resp.status_code}: {resp.text[:200]}")
    return parse_tiingo(resp.json()), resp.content


# --- Naver Finance (KRX) ----------------------------------------------------------
# Korean listings are 6-character codes (e.g. 122630 KODEX 레버리지); Tiingo does not carry them.
NAVER_URL = "https://fchart.stock.naver.com/sise.nhn"
KRX_NAMES = {"069500": "KODEX 200", "122630": "KODEX 레버리지", "233740": "KODEX 코스닥150레버리지",
             "252670": "KODEX 200선물인버스2X"}
NAVER_MAX_DAILY_MOVE = 0.35
NAVER_SPLIT_RATIOS = tuple(r for k in (2, 2.5, 4, 5, 10, 20, 25, 50, 100) for r in (k, 1 / k))
NAVER_SPLIT_TOLERANCE = 0.10
_NAVER_ITEM = re.compile(r'<item\s+data="([^"]*)"')


def is_krx(symbol: str) -> bool:
    """KRX codes are six characters starting with a digit; US tickers start with a letter."""
    s = str(symbol).strip()
    return len(s) == 6 and s[0].isdigit() and s.isalnum()


def display_name(symbol: str) -> str:
    return KRX_NAMES.get(str(symbol), str(symbol))


def parse_naver(text: str) -> pd.DataFrame:
    """Rows `<item data="YYYYMMDD|open|high|low|close|volume" />` of Naver's daily chart.

    The chart carries no dividends (dividend 0: returns are price returns) and has been seen to keep a
    split unadjusted (KODEX 레버리지, 2026-07-31), so splits are found by `naver_splits` and recorded
    as split factors on the raw prices. Rows from the early years sometimes have open/high/low 0; they
    take the close (the same repair as the S&P 500 bundle) and the quality check reports nothing extra."""
    rows = _NAVER_ITEM.findall(text)
    if not rows and "<chartdata" not in text:
        raise ProviderError(f"unexpected Naver response: {text[:200]}")
    recs = []
    for r in rows:
        f = r.split("|")
        if len(f) < 5:
            raise ProviderError(f"Naver row has {len(f)} fields: {r[:80]}")
        recs.append((f[0], *(float(x) if x not in ("", "null") else np.nan for x in f[1:6])))
    if not recs:
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="date"))
    df = pd.DataFrame(recs, columns=["date", "open", "high", "low", "close", "volume"][: len(recs[0])])
    out = pd.DataFrame({c: df[c].astype(float) for c in ("open", "high", "low", "close")})
    for c in ("open", "high", "low"):
        out[c] = out[c].where(out[c] > 0, out["close"])
    out["volume"] = df["volume"].astype(float) if "volume" in df else np.nan
    out["dividend"], out["split"], out["adj_close"] = 0.0, 1.0, np.nan
    out.index = pd.DatetimeIndex(pd.to_datetime(df["date"], format="%Y%m%d"), name="date")
    return out[COLUMNS].sort_index()


def naver_splits(frame: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Record unadjusted splits as split factors (shares after / shares before).

    KRX limits a day to ±30%, so a close-to-close step beyond NAVER_MAX_DAILY_MOVE is a split, never a
    real move. Its factor is the KRX-style ratio (2, 2.5, 4, 5, 10, 20, 25, 50, 100 or the reverse) nearest
    to previous close / open, which must be within NAVER_SPLIT_TOLERANCE of it; the day's return after
    the factor must be an ordinary move again. Anything else stops the download."""
    out = frame.copy()
    close = out["close"].to_numpy()
    for i in np.flatnonzero(np.abs(close[1:] / close[:-1] - 1) > NAVER_MAX_DAILY_MOVE) + 1:
        day = out.index[i].date()
        q = close[i - 1] / out["open"].iloc[i]
        k = min(NAVER_SPLIT_RATIOS, key=lambda r: abs(np.log(q / r)))
        if abs(np.log(q / k)) > np.log(1 + NAVER_SPLIT_TOLERANCE) or \
                abs(close[i] * k / close[i - 1] - 1) > NAVER_MAX_DAILY_MOVE:
            raise ProviderError(f"Naver {symbol}: close moved {close[i] / close[i - 1] - 1:+.0%} on {day}, "
                                f"which is no recognisable split (previous close / open = {q:.3f})")
        out.iloc[i, out.columns.get_loc("split")] = k
    return out


def fetch_naver(symbol: str, start: str, end: str | None = None, session=None,
                timeout: float = 30.0) -> tuple[pd.DataFrame, bytes]:
    """Whole daily history of a KRX code (no key needed), cut to [start, end]."""
    import requests

    http = session or requests
    resp = http.get(NAVER_URL, params={"symbol": symbol, "timeframe": "day", "count": 10000, "requestType": 0},
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    if resp.status_code != 200:
        raise ProviderError(f"Naver {symbol}: HTTP {resp.status_code}: {resp.text[:200]}")
    text = resp.content.decode(resp.encoding or "euc-kr", errors="replace")
    frame = parse_naver(text)
    if frame.empty:
        raise ProviderError(f"Naver {symbol}: no prices (check the code)")
    frame = naver_splits(frame, symbol)
    frame = frame.loc[pd.Timestamp(start):]
    return (frame.loc[: pd.Timestamp(end)] if end else frame), resp.content


# --- CSV -------------------------------------------------------------------------
CSV_FORMAT = """CSV import format (one file per symbol, UTF-8, comma separated):
  date,open,high,low,close[,volume][,dividend][,split][,adj_close]
  date as YYYY-MM-DD; prices raw (not adjusted); dividend = cash per share on the
  ex-date (default 0); split = shares-after / shares-before on the effective date (default 1)."""


def read_csv_prices(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"date", "open", "high", "low", "close"}
    if need - set(df.columns):
        raise ProviderError(f"CSV lacks columns {sorted(need - set(df.columns))}.\n{CSV_FORMAT}")
    out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(df["date"]).dt.normalize(), name="date"))
    for c in ("open", "high", "low", "close"):
        out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy()
    out["volume"] = pd.to_numeric(df["volume"], errors="coerce").to_numpy() if "volume" in df else np.nan
    out["dividend"] = pd.to_numeric(df["dividend"], errors="coerce").fillna(0.0).to_numpy() if "dividend" in df else 0.0
    out["split"] = pd.to_numeric(df["split"], errors="coerce").fillna(1.0).to_numpy() if "split" in df else 1.0
    out["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce").to_numpy() if "adj_close" in df else np.nan
    return out[COLUMNS].sort_index()
