"""Index datasets: every stock of a point-in-time index plus its membership and insider events,
on the session calendar of the Tiingo SPY/BIL dataset (D-026).

Layout in data/processed/<version>/:
    manifest.json      kind "index"; inputs (raw hashes), coverage and checks
    stocks.parquet     long table date, ticker, open, high, low, close (total-return prices)
    members.parquet    bool frame, sessions x tickers
    events.parquet     insider purchase rows (see quantlab.data.sec_insider)
The SPY/BIL part is the ordinary dataset named in `base_dataset`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.core.hashing import file_sha256, stable_hash
from quantlab.core.market_data import MarketData
from quantlab.data import sec_insider, sp500

PROCESSING_VERSION = 1
MIN_LINKED_SHARE = 0.90      # members with at least one Form 4 linked; below this the mapping is suspect
MIN_BASKET_CORR = 0.90       # equal-weight member basket vs SPY, daily returns


def build(store, base_version: str, bundle: dict, sec_records: dict[str, dict], start: str,
          say=None) -> tuple[str, dict]:
    say = say or (lambda s: None)
    inputs = {"processing": PROCESSING_VERSION, "kind": "index", "index": "SP500", "start": str(start),
              "base_dataset": base_version, "bundle": {k: v["sha256"] for k, v in bundle["files"].items()},
              "sec": {q: m["sha256"] for q, m in sorted(sec_records.items())}}
    version = stable_hash(inputs)
    done = store.proc_dir / version[:16] / "manifest.json"
    if done.exists():  # same inputs, same dataset
        return version, json.loads(done.read_text(encoding="utf-8"))
    base, base_manifest = store.load_dataset(base_version)
    sessions = base.calendar[base.calendar >= pd.Timestamp(start)]
    say("S&P 500 자료 정리 중")
    prices, members_long = sp500.read_bundle(bundle)
    members = sp500.membership(members_long, sessions)
    frames, counts = sp500.total_return_frames(prices, members.columns, sessions)
    members = members.loc[:, [t for t in members.columns if t in frames]]
    frames = {t: frames[t] for t in members.columns}

    say("SEC 내부자 거래 자료 정리 중")
    purchases, filings, sec_counts = [], [], {}
    for q, meta in sorted(sec_records.items()):
        p, f, c = sec_insider.load_parsed(meta)
        purchases.append(p)
        filings.append(f)
        sec_counts[q] = c
    purchases = pd.concat(purchases, ignore_index=True)
    filings = pd.concat(filings, ignore_index=True)
    have = prices[prices["ticker"].isin(set(members.columns))]
    ranges = have.groupby("ticker")["date"].agg(first="min", last="max").reset_index()
    mapped, links = sec_insider.map_issuers(purchases, filings, ranges)
    events = sec_insider.insider_events(mapped)
    events = events[events["symbol"].isin(set(members.columns))
                    & (events["known"] >= sessions[0] - pd.Timedelta(days=60))
                    & (events["known"] <= sessions[-1])].reset_index(drop=True)

    say("품질 검사 중")
    fatal, warnings = list(base_manifest.get("qc_fatal", [])), list(base_manifest.get("qc_warnings", []))
    cov = sp500.coverage(members, frames, sessions[0])
    if cov["share"] < sp500.MIN_MEMBER_COVERAGE:
        fatal.append(f"S&P 500: only {cov['share']:.2%} of member-sessions have a price "
                     f"(min {sp500.MIN_MEMBER_COVERAGE:.0%})")
    elif cov["share"] < sp500.WARN_MEMBER_COVERAGE:
        warnings.append(f"S&P 500: {cov['share']:.2%} of member-sessions have a price")
    for k in ("off_calendar", "duplicates", "bad_price"):
        if counts[k]:
            warnings.append(f"S&P 500: {counts[k]} price rows dropped ({k})")
    linked = set(links["ticker"])
    share_linked = float(np.mean([t in linked for t in members.columns])) if len(members.columns) else 0.0
    if share_linked < MIN_LINKED_SHARE:
        warnings.append(f"SEC: only {share_linked:.1%} of index stocks are linked to Form 4 filings "
                        f"(expected at least {MIN_LINKED_SHARE:.0%})")
    bad_dates = sum(c.get("bad_dates", 0) for c in sec_counts.values())
    if bad_dates:
        warnings.append(f"SEC: {bad_dates} Form 4 filings with an unreadable filing date were left out")
    expected = sec_insider.quarter_list(min(sec_records) if sec_records else sec_insider.FIRST_QUARTER,
                                        sessions[-1].date())
    gaps = [q for q in expected[:-1] if q not in sec_records]
    if gaps:
        fatal.append(f"SEC: quarters missing {gaps}")
    corr = _basket_corr(frames, members, base, store_benchmark="SPY")
    if corr is not None and corr < MIN_BASKET_CORR:
        warnings.append(f"S&P 500 member basket vs SPY daily correlation {corr:.3f} (min {MIN_BASKET_CORR})")

    out = store.proc_dir / version[:16]
    out.mkdir(parents=True, exist_ok=True)
    stocks = pd.concat({t: f[["open", "high", "low", "close"]] for t, f in frames.items()}, names=["ticker", "date"])
    stocks = stocks.dropna(subset=["close"]).reset_index()
    stocks.to_parquet(out / "stocks.parquet", index=False)
    members.to_parquet(out / "members.parquet")
    events.to_parquet(out / "events.parquet", index=False)
    manifest = {"dataset_version": version, "processing_version": PROCESSING_VERSION, "kind": "index",
                "index": "SP500", "inputs": inputs, "symbols": base_manifest["symbols"],
                "base_dataset": base_version,
                "files": {n: file_sha256(out / n) for n in ("stocks.parquet", "members.parquet", "events.parquet")},
                "bundle_source": bundle.get("source"), "bundle_fetched_at": bundle.get("fetched_at"),
                "stocks": len(frames), "sessions": [str(sessions[0].date()), str(sessions[-1].date())],
                "coverage": cov, "price_counts": counts, "insider": {
                    "purchase_rows": int(len(events)), "linked_issuers": int(links["issuer_cik"].nunique()),
                    "index_stocks_linked": share_linked, "quarters": sorted(sec_records)},
                "basket_corr_spy": corr, "qc_fatal": fatal, "qc_warnings": warnings}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str),
                                       encoding="utf-8")
    if store.registry is not None:
        store.registry.register_dataset(version, json.loads(json.dumps(manifest, default=str)))
    return version, manifest


def _basket_corr(frames: dict, members: pd.DataFrame, base: MarketData, store_benchmark: str) -> float | None:
    if store_benchmark not in base.frames or not frames:
        return None
    close = pd.DataFrame({t: f["close"] for t, f in frames.items()})
    r = close.pct_change(fill_method=None).where(members.shift(1, fill_value=False))
    basket = r.mean(axis=1).dropna()
    from quantlab.core.market_data import total_return_index

    spy = total_return_index(base.frames[store_benchmark]).pct_change().reindex(basket.index)
    both = pd.concat([basket, spy], axis=1).dropna()
    return float(both.corr().iloc[0, 1]) if len(both) > 60 else None


def load(store, manifest: dict, d: Path) -> MarketData:
    for name, sha in manifest["files"].items():
        if file_sha256(d / name) != sha:
            raise ValueError(f"processed file changed on disk: {d / name}")
    base, _ = store.load_dataset(manifest["base_dataset"])
    cal = base.calendar[base.calendar >= pd.Timestamp(manifest["inputs"]["start"])]
    base = base.slice(cal[0], None)
    stocks = pd.read_parquet(d / "stocks.parquet")
    frames = dict(base.frames)
    for t, g in stocks.groupby("ticker", sort=True):
        f = g.set_index("date")[["open", "high", "low", "close"]].reindex(cal)
        f["dividend"] = 0.0
        f["split"] = 1.0
        frames[t] = f.astype(float)
    members = pd.read_parquet(d / "members.parquet").reindex(cal, fill_value=False).astype(bool)
    events = pd.read_parquet(d / "events.parquet")
    return MarketData(cal, frames, members, events)
