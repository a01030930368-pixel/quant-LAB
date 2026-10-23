"""D-040: Korean filings from OpenDART — download, events, stock catalog 6 and the research factory's
round 4 (AUTO-KRDART) end to end on synthetic snapshots and filings; and KODEX 200 prices older than
Naver's chart cap."""

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.engine import run_backtest
from quantlab.data import dart_open, kr_dataset, krx_open, providers
from quantlab.services import autosearch as A
from quantlab.services import factory as F
from quantlab.services.batch import grid
from quantlab.services.lab import Lab, builtin_path
from quantlab.strategies import stock_combo as sc
from quantlab.strategies.spec import load_spec
from kr_fixtures import STOCKS, FakeDart, FakeKrx, corp, dart_docs, snapshots
from test_kr2x import FakeResponse, naver_text
from test_stock_combo import market

END = "2021-12-30"


@pytest.fixture(scope="module")
def snaps():
    return snapshots(start="2014-01-02", end=END)[0]


@pytest.fixture(scope="module")
def docs(snaps):
    return dart_docs(snaps)


# --- KODEX 200 before Naver's chart cap ---------------------------------------------------------
def test_older_kodex_prices_come_from_the_dated_list(monkeypatch):
    full = naver_text("069500", 0)
    items = full.split("\n")[1:-1]
    capped = full.split("\n")[0] + "\n" + "\n".join(items[-providers.NAVER_CAP:]) + "\n</chartdata></protocol>"
    older = providers.parse_naver(full)
    rows = ",\n".join(f'["{d:%Y%m%d}", {r.open:.0f}, {r.high:.0f}, {r.low:.0f}, {r.close:.0f}, 1000, 0.0]'
                      for d, r in older.iterrows())
    dated = f"[['날짜', '시가', '고가', '저가', '종가', '거래량', '외국인소진율'],\n{rows}]"
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        return FakeResponse(capped if url == providers.NAVER_URL else dated)

    import requests

    monkeypatch.setattr(requests, "get", get)
    frame, raw = providers.fetch_naver("069500", "2010-01-01")
    assert calls == [providers.NAVER_URL, providers.NAVER_RANGE_URL]
    assert frame.index[0] == older.loc["2010-01-01":].index[0] and frame.index.is_unique
    assert frame["close"].equals(older.loc["2010-01-01":, "close"])
    # a start inside the chart's span asks nothing more
    calls.clear()
    providers.fetch_naver("069500", str(providers.parse_naver(capped).index[0].date()))
    assert calls == [providers.NAVER_URL]
    # the two lists must agree where they meet
    meet = providers.parse_naver(capped).index[0]
    bad = ",\n".join(f'["{d:%Y%m%d}", {r.open:.0f}, {r.high:.0f}, {r.low:.0f}, {r.close * (1.1 if d == meet else 1):.0f}, 1, 0]'
                      for d, r in older.iterrows())
    monkeypatch.setattr(requests, "get", lambda url, **k: FakeResponse(capped if url == providers.NAVER_URL else bad))
    with pytest.raises(providers.ProviderError, match="disagree"):
        providers.fetch_naver("069500", "2010-01-01")


# --- download -----------------------------------------------------------------------------------
def test_download_maps_companies_resumes_after_the_limit_and_splits_long_spans(tmp_path, docs):
    codes = sorted(STOCKS)
    http = FakeDart(docs, limit=10)
    out = dart_open.download(tmp_path, "dart-key", codes, today="2022-01-10", session=http)
    n = 3 * (len(STOCKS) - 1)
    assert out["fetched"] == 10 and out["left"] == n - 10 and "한도" in out["stopped"]
    assert out["unmapped"] == ["123450"]
    http2 = FakeDart(docs, long_span=True)
    out = dart_open.download(tmp_path, "dart-key", codes, today="2022-01-10", session=http2)
    assert out["left"] == 0 and out["stopped"] is None
    dated = [c for c in http2.calls if c[0] in dart_open.DATED]
    assert dated and len(dated) > 2 * len({c for _, c in dated})    # asked again one year at a time
    stored = dart_open._read(dart_open._path(tmp_path, "buyback", corp("005930")))
    assert len(stored) == len(docs[("buyback", corp("005930"))])
    again = FakeDart(docs)
    assert dart_open.download(tmp_path, "dart-key", codes, today="2022-01-10", session=again)["fetched"] == 0
    assert again.calls == []
    with pytest.raises(dart_open.DartError, match="인증키"):
        dart_open.download(tmp_path / "other", "bad", codes, today="2022-01-10", session=FakeDart(docs))


def test_events_from_holdings_reports_and_buybacks(tmp_path, snaps, docs):
    dart_open.download(tmp_path, "dart-key", sorted(STOCKS), today="2022-01-10", session=FakeDart(docs))
    store = tmp_path / "krx"
    from kr_fixtures import store_snapshots

    store_snapshots(store, snaps)
    table = krx_open.load(store)[0]
    from quantlab.data import kr_stocks

    days = krx_open.sessions("2014-01-02", END)
    closes = kr_stocks._wide(table, "close", days, sorted(STOCKS))
    caps = kr_stocks._wide(table, "mktcap", days, sorted(STOCKS))
    ev, counts = dart_open.events(tmp_path, sorted(STOCKS), closes, caps)
    assert counts["mapped"] == len(STOCKS) - 1 and counts["first_reports"] == len(STOCKS) - 1
    assert set(ev["kind"]) == {"kr_ins_buy", "kr_ins_sell", "kr_buyback"}
    sam = ev[ev["symbol"] == "005930"]
    rows = [r for r in docs[("elestock", corp("005930"))][1:]]
    assert (sam["kind"] != "kr_buyback").sum() == len(rows)
    r = rows[0]
    day = pd.Timestamp(r["rcept_dt"])
    hit = sam[(sam["known"] == day) & (sam["owner"] == r["repror"])].iloc[0]
    assert hit["value"] == pytest.approx(20_000 * closes.loc[day, "005930"])
    bb = sam[sam["owner"] == "company:buyback"]
    assert len(bb) == 3 and (bb["value"] - 3e9 / caps.loc[bb["known"], "005930"].to_numpy()).abs().max() < 1e-12
    assert sam[sam["owner"] == "company:trust"]["value"].isna().all()   # amount not stated


# --- catalog 6 ----------------------------------------------------------------------------------
RULES6 = {"stock_combo": {"catalog": 6, "park": "SPY", "park_band": 2}}


def buys6(data, entry, hold=5, slots=3):
    s = sc.StockComboStrategy(spec_id="T", spec_version_str="1", rules=RULES6, entry=entry, hold=hold, stop=0,
                              take=0, slots=slots)
    t = run_backtest(s, data).trades
    t = t[(t["symbol"] != "SPY") & (t["qty"] > 0)]
    return list(zip(t["date"].dt.strftime("%Y-%m-%d"), t["symbol"]))


def test_filing_conditions_fire_the_session_after_they_are_public():
    n = 140
    ev = [("2020-02-03", "X", "kr_ins_buy", "a", 50_000_000, True, True),
          ("2020-02-03", "Y", "kr_ins_buy", "a", 5_000_000, True, True),        # below 10 million won
          ("2020-02-05", "Y", "kr_ins_buy", "b", 400_000_000, False, True),     # a major shareholder, big
          ("2020-03-02", "Z", "kr_buyback", "company:buyback", 0.02, True, True),
          ("2020-04-01", "X", "kr_buyback", "company:trust", np.nan, True, True)]
    ev += [("2020-05-01", "Z", "kr_ins_sell", o, 200_000_000, True, False) for o in ("a", "b", "c")]
    data = market({}, ev, n=n)
    assert buys6(data, "kins1") == [("2020-02-04", "X"), ("2020-02-06", "Y")]
    assert buys6(data, "kinsbig") == [("2020-02-06", "Y")]
    assert buys6(data, "kins2") == []
    assert buys6(data, "bb") == [("2020-03-03", "Z"), ("2020-04-02", "X")]
    assert buys6(data, "bb1") == [("2020-03-03", "Z")]
    assert buys6(data, "ksell3") == [("2020-05-04", "Z")]


def test_catalog_6_adds_filings_and_skips_what_round_3_ran():
    keys = sc.conditions(6)
    assert set(sc.conditions(5)) < set(keys) and len(keys) == 30
    assert sc.entries(5) == [e for e in sc.entries(6) if all(sc._in_catalog(c, 5) for c in e.split("+"))]
    space = sc.full_space(6)
    assert np.prod([len(v) for v in space.values()]) == 636336
    fresh = [p for p in grid(space) if not sc.already_tried(p, 6)]
    assert len(fresh) == 360126
    assert all(any(sc._CONDITIONS[c][2] == 6 for c in p["entry"].split("+")) for p in fresh[:5000])
    loaded = load_spec(builtin_path("specs", "AUTO-KRDART.qls.yaml"))
    assert {k: list(v) for k, v in loaded.search_space.items()} == space
    assert sc.benchmark(6) == "069500"
    assert F.GROUPS[-1] == "AUTO-KRDART" and A.SHARED_HOLDOUT["AUTO-KRDART"] == ("AUTO-KRSTK", "AUTO-KRDART")


# --- round 4 ------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dart_lab(tmp_path_factory, snaps, docs):
    tmp = tmp_path_factory.mktemp("dartlab")
    text = naver_text("069500", 0)
    mp = pytest.MonkeyPatch()
    import requests

    mp.setattr(requests, "get", lambda url, params=None, headers=None, timeout=None: FakeResponse(text))
    keys = {"krx": "good-key", "dart": "dart-key"}
    mp.setattr(providers, "get_api_key", lambda provider="tiingo": keys.get(provider))
    mp.setattr(kr_dataset, "latest_day", lambda today=None: pd.Timestamp(END))
    krx, dart = FakeKrx(snaps), FakeDart(docs)
    real_krx, real_dart = krx_open.download, dart_open.download
    mp.setattr(krx_open, "download", lambda *a, **k: real_krx(*a, **(k | {"session": krx, "today": "2022-06-01"})))
    mp.setattr(dart_open, "download", lambda *a, **k: real_dart(*a, **(k | {"session": dart, "today": "2022-06-01"})))
    lab = Lab(tmp / "QL")
    lab.dart_http = dart
    yield lab
    lab.close()
    mp.undo()


def test_round_4_downloads_filings_builds_and_searches(dart_lab):
    auto = dart_lab.auto_search()
    out = auto.run("AUTO-KRDART", workers=1, limit=4)
    assert out["done"] >= 4 and out["failed"] == 0 and out["pending"] == 0
    assert out["family_trials"] == out["done"] + 276211
    assert dart_lab.registry.holdout_ledger() == []
    calls = len(dart_lab.dart_http.calls)
    _, loaded = auto.loaded("AUTO-KRDART")
    _, policy = dart_lab.policy_for(loaded)
    version = dart_lab.ensure_spec_dataset(loaded, policy)
    assert len(dart_lab.dart_http.calls) == calls                  # stored: not asked again
    data, manifest = dart_lab.data.load_dataset(version)
    assert manifest["qc_fatal"] == [] and manifest["dart"]["mapped"] == len(manifest["names"]) - 1  # 123450
    assert calls == 3 * manifest["dart"]["mapped"]                 # members only, once each
    assert data.calendar[0] >= pd.Timestamp("2014-08-01")
    assert {"kr_ins_buy", "kr_ins_sell", "kr_buyback"} <= set(data.events["kind"])
    assert set(data.events["symbol"]) <= set(data.members.columns)
    params = dict(loaded.default_params, entry="kins1", hold=10, slots=5)
    res = dart_lab.backtest(auto.loaded("AUTO-KRDART")[0], version, params).result
    assert len(res.trades[res.trades["symbol"] != "069500"])
    # on the user's PC the pre-fixed round 3 rule read the shared holdout (D-039): no final evaluation here
    mp = pytest.MonkeyPatch()
    mp.setattr(dart_lab.registry, "holdout_ledger", lambda: [{"strategy_family": "AUTO-KRSTK"}])
    try:
        with pytest.raises(RuntimeError, match="보류 구간"):
            auto.final_evaluation("AUTO-KRDART", params)
    finally:
        mp.undo()


def test_dart_key_is_asked_for(tmp_path, monkeypatch):
    monkeypatch.setattr(providers, "get_api_key", lambda provider="tiingo": None)
    lab = Lab(tmp_path / "QL")
    try:
        assert not lab.has_dart_key()
    finally:
        lab.close()
