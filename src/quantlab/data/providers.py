"""Market data sources: Tiingo EOD (one provider for the MVP) and CSV import.

Both return the same normalized frame: DatetimeIndex (naive, session date),
columns open, high, low, close (raw traded prices), volume, dividend, split,
adj_close (provider's adjusted close, NaN for CSV unless supplied).

API keys come from the OS credential store (Windows Credential Manager via
`keyring`) or the TIINGO_API_KEY environment variable. Keys are never written
to files, logs, the registry or result packets.
"""

from __future__ import annotations

import os
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
