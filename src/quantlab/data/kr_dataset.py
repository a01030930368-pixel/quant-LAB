"""Index dataset KRTOP300: Korean stocks from the exchange's daily snapshots on the session calendar of
the KODEX 200 (069500) dataset (D-038). Same layout as the S&P 500 datasets (quantlab.data.index_dataset,
which also loads it): stocks.parquet, members.parquet, events.parquet (empty, or with `dart=True` the
OpenDART filings of the members, D-040).
"""

from __future__ import annotations

import json

import pandas as pd

from quantlab.core.hashing import file_sha256, stable_hash
from quantlab.data import dart_open, index_dataset, kr_stocks, krx_open

PROCESSING_VERSION = 1
INDEX = "KRTOP300"
BENCHMARK = "069500"
MAX_MISSING_SHARE = 0.005    # sessions without a snapshot (either market); above this the dataset is unusable
MIN_COVERAGE = 0.995         # member-sessions with a close
MIN_BASKET_CORR = 0.80       # equal-weight member basket vs KODEX 200, daily returns
MIN_DART_SHARE = 0.80        # members with a DART company code; fewer is a warning (delisted names may lack one)
EVENT_COLUMNS = dart_open.EVENT_COLUMNS


def build(store, base_version: str, root, start: str, say=None, dart: bool = False) -> tuple[str, dict]:
    say = say or (lambda s: None)
    base, base_manifest = store.load_dataset(base_version)
    sessions = base.calendar[base.calendar >= pd.Timestamp(start)]
    snaps, inputs_raw = krx_open.load(root, sessions[0], sessions[-1], say=say)
    if len(snaps):  # the KODEX 200 prices can run a session ahead of the exchange's published snapshots
        sessions = sessions[sessions <= snaps["date"].max()]
    inputs = {"processing": PROCESSING_VERSION, "kind": "index", "index": INDEX, "start": str(start),
              "base_dataset": base_version, "krx": inputs_raw, "top_n": kr_stocks.TOP_N,
              "min_value": kr_stocks.MIN_VALUE}
    if dart:
        inputs["dart"] = dart_open.inputs(root)
    version = stable_hash(inputs)
    done = store.proc_dir / version[:16] / "manifest.json"
    if done.exists():
        return version, json.loads(done.read_text(encoding="utf-8"))
    say("한국 주식 종목 정하는 중")
    missing = krx_open.missing_sessions(root, sessions[0], sessions[-1])
    have_days = set(snaps["date"].unique()) if len(snaps) else set()
    members = kr_stocks.membership(snaps, sessions)
    say("한국 주식 가격 정리 중")
    frames, counts = kr_stocks.adjusted(snaps, sessions, members.columns)
    members = members.loc[:, [c for c in members.columns if c in frames]]
    frames = {c: frames[c] for c in members.columns}
    names = snaps.drop_duplicates("code", keep="last").set_index("code")["name"].reindex(members.columns)

    say("품질 검사 중")
    fatal, warnings = list(base_manifest.get("qc_fatal", [])), list(base_manifest.get("qc_warnings", []))
    for m, days in missing.items():
        share = len(days) / max(len(sessions), 1)
        if share > MAX_MISSING_SHARE:
            fatal.append(f"KRX {m}: {len(days)} of {len(sessions)} sessions not downloaded yet "
                         f"(first {days[:3]}); run again to continue")
        elif days:
            warnings.append(f"KRX {m}: {len(days)} sessions without a snapshot ({days[:5]})")
    empty = [str(d.date()) for d in sessions if d not in have_days]
    if empty and len(empty) <= MAX_MISSING_SHARE * len(sessions):
        warnings.append(f"KRX: {len(empty)} sessions with no rows ({empty[:5]})")
    cov = kr_stocks.coverage(members, frames)
    if cov["share"] < MIN_COVERAGE:
        fatal.append(f"KRTOP300: only {cov['share']:.2%} of member-sessions have a price")
    if not len(members.columns):
        fatal.append("KRTOP300: no members (no snapshots?)")
    for k in ("duplicates", "bad_price"):
        if counts[k]:
            warnings.append(f"KRX: {counts[k]} price rows dropped ({k})")
    if counts["beyond_limit"]:
        warnings.append(f"KRX: {counts['beyond_limit']} daily returns beyond the price limit "
                        "(listing days, delisting trades or a base-price reset)")
    corr = index_dataset._basket_corr(frames, members, base, store_benchmark=BENCHMARK)
    if corr is not None and corr < MIN_BASKET_CORR:
        warnings.append(f"KRTOP300 member basket vs {BENCHMARK} daily correlation {corr:.3f} (min {MIN_BASKET_CORR})")

    out = store.proc_dir / version[:16]
    out.mkdir(parents=True, exist_ok=True)
    stocks = pd.concat({c: f[["open", "high", "low", "close"]] for c, f in frames.items()}, names=["ticker", "date"])
    stocks = stocks.dropna(subset=["close"]).reset_index()
    stocks.to_parquet(out / "stocks.parquet", index=False)
    members.to_parquet(out / "members.parquet")
    events = pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c == "known" else "object") for c in EVENT_COLUMNS})
    dart_counts = None
    if dart:
        say("DART 공시 정리 중")
        raw = snaps[snaps["code"].isin(set(members.columns))]
        closes = kr_stocks._wide(raw, "close", sessions, list(members.columns))
        caps = kr_stocks._wide(raw, "mktcap", sessions, list(members.columns))
        events, dart_counts = dart_open.events(root, list(members.columns), closes, caps)
        share = dart_counts["mapped"] / max(len(members.columns), 1)
        dart_counts["mapped_share"] = share
        if not len(events):
            fatal.append("DART: no filings for any member (download not finished?)")
        elif share < MIN_DART_SHARE:
            warnings.append(f"DART: only {share:.0%} of members have a company code (delisted names may lack one)")
        if dart_counts["no_date"]:
            warnings.append(f"DART: {dart_counts['no_date']} filings without a receipt date dropped")
    events.to_parquet(out / "events.parquet", index=False)
    manifest = {"dataset_version": version, "processing_version": PROCESSING_VERSION, "kind": "index",
                "index": INDEX, "inputs": inputs, "symbols": base_manifest["symbols"], "base_dataset": base_version,
                "files": {n: file_sha256(out / n) for n in ("stocks.parquet", "members.parquet", "events.parquet")},
                "bundle_source": kr_stocks.SOURCE, "stocks": len(frames),
                "sessions": [str(sessions[0].date()), str(sessions[-1].date())], "coverage": cov,
                "price_counts": counts, "names": {c: str(n) for c, n in names.items()},
                "missing_sessions": {m: len(v) for m, v in missing.items()},
                "basket_corr_benchmark": corr, "dart": dart_counts,
                "qc_fatal": fatal, "qc_warnings": warnings}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str),
                                       encoding="utf-8")
    if store.registry is not None:
        store.registry.register_dataset(version, json.loads(json.dumps(manifest, default=str)))
    return version, manifest


def latest_day(today=None) -> pd.Timestamp:
    """The last session whose snapshot should be out: the exchange publishes the previous session at
    about 08:00 KST."""
    now = pd.Timestamp.now(tz="Asia/Seoul") if today is None else pd.Timestamp(today).tz_localize("Asia/Seoul")
    end = now.normalize().tz_localize(None) - pd.Timedelta(days=1)
    days = krx_open.sessions(end - pd.Timedelta(days=14), end)
    return days[-1] if len(days) else end

