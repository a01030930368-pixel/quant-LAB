"""Data module (M9): provider parsing, CSV import, quality checks, dataset versions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.data.providers import ProviderError, parse_tiingo, read_csv_prices
from quantlab.data.quality import run_qc
from quantlab.data.store import DataStore

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def clean_frame(start="2016-01-04", end="2021-06-30", seed=0):
    from quantlab.data.quality import exchange_sessions
    idx = exchange_sessions(start, end)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    opn = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({"open": opn, "high": np.maximum(opn, close) * 1.002, "low": np.minimum(opn, close) * 0.998,
                       "close": close, "volume": 1e6, "dividend": 0.0, "split": 1.0}, index=idx)
    tr = ((df["close"] + df["dividend"]) * df["split"] / df["close"].shift(1)).fillna(1.0).cumprod()
    df["adj_close"] = tr * df["close"].iloc[0]
    return df


def statuses(rep):
    return {c.name: c.status for c in rep.checks}


def test_parse_tiingo_fixture():
    df = parse_tiingo(json.loads((FIX / "tiingo_spy_sample.json").read_text()))
    assert list(df.index.strftime("%Y-%m-%d")) == ["2010-06-01", "2010-06-02", "2010-06-03"]
    assert df.loc["2010-06-02", "close"] == 110.0 and df["split"].eq(1.0).all()
    with pytest.raises(ProviderError):
        parse_tiingo({"detail": "Invalid token."})
    with pytest.raises(ProviderError):
        parse_tiingo([{"date": "2010-06-01", "close": 1.0}])


def test_clean_data_passes_all_checks():
    rep = run_qc("X", clean_frame())
    assert not rep.fatal and not rep.warnings, rep.to_dict()


@pytest.mark.parametrize("defect,check,status", [
    ("drop_one", "missing sessions", "warn"),
    ("drop_many", "missing sessions", "fatal"),
    ("holiday", "non-session dates", "fatal"),
    ("duplicate", "duplicate dates", "fatal"),
    ("negative", "non-positive prices", "fatal"),
    ("ohlc", "OHLC consistency", "fatal"),
    ("missed_split", "large daily moves", "warn"),
    ("missed_dividend", "adjusted-price reconciliation", "warn"),
])
def test_each_injected_defect_trips_its_check(defect, check, status):
    df = clean_frame()
    if defect == "drop_one":
        df = df.drop(df.index[10])
    elif defect == "drop_many":
        df = df.drop(df.index[10:60])
    elif defect == "holiday":
        extra = df.iloc[[5]].copy()
        extra.index = [pd.Timestamp("2021-04-02")]  # Good Friday 2021
        df = pd.concat([df, extra]).sort_index()
    elif defect == "duplicate":
        df = pd.concat([df, df.iloc[[7]]]).sort_index()
    elif defect == "negative":
        df.iloc[12, df.columns.get_loc("close")] = -1.0
    elif defect == "ohlc":
        df.iloc[3:30, df.columns.get_loc("high")] = df["low"].iloc[3:30] * 0.9
    elif defect == "missed_split":
        df.iloc[40:, [df.columns.get_loc(c) for c in ("open", "high", "low", "close")]] /= 2.0
        df["adj_close"] = df["adj_close"]  # provider adj series is right; ours lacks the split row
    elif defect == "missed_dividend":
        df.iloc[30:, df.columns.get_loc("adj_close")] *= 1.01  # provider saw a dividend we do not have
    rep = run_qc("X", df)
    st = statuses(rep)
    assert st[check] == status, rep.to_dict()


def test_revision_detection_and_dataset_version(tmp_path):
    store = DataStore(tmp_path)
    df = clean_frame()
    r1 = store.ingest("spy", df, json.dumps({"n": 1}).encode(), {"provider": "test"})
    assert r1.qc.fatal == []
    v1, man = store.build_dataset(["SPY"])
    v1b, _ = store.build_dataset(["SPY"])
    assert v1 == v1b
    data, man2 = store.load_dataset(v1)
    assert data.symbols == ["SPY"] and len(data.calendar) == len(df)
    revised = df.copy()
    revised.iloc[5, revised.columns.get_loc("close")] *= 1.01
    r2 = store.ingest("SPY", revised, json.dumps({"n": 2}).encode(), {"provider": "test"})
    assert {c.name: c.status for c in r2.qc.checks}["provider revisions"] == "warn"
    v2, _ = store.build_dataset(["SPY"])
    assert v2 != v1
    Path(r1.raw_path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed on disk"):
        store.load_dataset(v1)


def test_csv_import(tmp_path):
    df = clean_frame()
    p = tmp_path / "x.csv"
    df.drop(columns=["adj_close", "volume"]).rename_axis("date").reset_index().to_csv(p, index=False)
    got = read_csv_prices(p)
    np.testing.assert_allclose(got["close"].to_numpy(), df["close"].to_numpy())
    assert got["dividend"].eq(0).all() and got["split"].eq(1).all()
    bad = tmp_path / "bad.csv"
    bad.write_text("day,price\n2021-01-04,1\n")
    with pytest.raises(ProviderError, match="CSV import format"):
        read_csv_prices(bad)
    store = DataStore(tmp_path / "ws")
    rec = store.ingest("X", got, p, {"provider": "csv", "file": p.name})
    assert rec.raw_path.suffix == ".csv" and not rec.qc.fatal
