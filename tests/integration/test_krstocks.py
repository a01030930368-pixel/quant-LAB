"""D-038: Korean stocks from the KRX Open API — download, adjusted prices, the KRTOP300 universe, the
transaction tax and the research factory's round 3 (AUTO-KRSTK), end to end on synthetic snapshots."""

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.engine import BacktestConfig, CostModel, run_backtest
from quantlab.core.market_data import MarketData
from quantlab.data import kr_dataset, kr_stocks, krx_open, providers
from quantlab.services.batch import grid
from quantlab.services.lab import Lab, builtin_path
from quantlab.strategies import stock_combo as sc
from quantlab.strategies.base import Strategy
from quantlab.strategies.spec import load_spec
from kr_fixtures import ADMIN, HALT, SPLIT, FakeKrx, snapshots, store_snapshots
from synthetic import frame
from test_kr2x import FakeResponse, naver_text

END = "2021-12-30"


@pytest.fixture(scope="module")
def snaps():
    return snapshots(end=END)[0]


def small(snaps, start, end):
    return {k: v for k, v in snaps.items() if pd.Timestamp(start) <= k[1] <= pd.Timestamp(end)}


# --- download -----------------------------------------------------------------------------------
def test_download_stores_each_session_once_and_resumes_after_the_daily_limit(tmp_path, snaps):
    part = small(snaps, "2010-01-04", "2010-03-31")
    http = FakeKrx(part, limit=50)
    out = krx_open.download(tmp_path, "good-key", "2010-03-31", session=http, today="2021-06-01")
    days = krx_open.sessions("2010-01-04", "2010-03-31")
    assert out["fetched"] == 50 and out["left"] == 2 * len(days) - 50 and "한도" in out["stopped"]
    http2 = FakeKrx(part)
    out = krx_open.download(tmp_path, "good-key", "2010-03-31", session=http2, today="2021-06-01")
    assert out == {"fetched": 2 * len(days) - 50, "left": 0, "stopped": None}
    assert len(http2.calls) == 2 * len(days) - 50                     # nothing asked twice
    assert krx_open.missing_sessions(tmp_path, "2010-01-04", "2010-03-31") == {"KOSPI": [], "KOSDAQ": []}
    with pytest.raises(krx_open.KrxError, match="키를 거절"):
        krx_open.download(tmp_path, "bad-key", "2010-04-30", session=FakeKrx(part), today="2021-06-01")


def test_an_empty_answer_close_to_today_is_asked_again(tmp_path):
    http = FakeKrx({})
    out = krx_open.download(tmp_path, "good-key", "2021-05-31", start="2021-05-24", session=http, today="2021-06-01")
    assert out["fetched"] == 0 and out["left"] > 0           # not published yet: nothing stored
    out = krx_open.download(tmp_path, "good-key", "2021-05-31", start="2021-05-24", session=http, today="2021-07-01")
    assert out["left"] == 0                                    # long past: an empty day is stored


def test_parse_reads_numbers_with_separators_and_short_codes(tmp_path, snaps):
    day = pd.Timestamp("2012-05-02")
    rows = [dict(r) for r in snaps[("KOSPI", day)]]
    rows[0]["ISU_CD"] = "KR7005930003"                       # an ISIN instead of the short code
    rows[1]["MKTCAP"] = "-"
    store_snapshots(tmp_path, {("KOSPI", day): rows})
    df = krx_open.parse_day(krx_open._path(tmp_path, "KOSPI", day), "KOSPI")
    assert df["code"].iloc[0] == "005930" and np.isnan(df["mktcap"].iloc[1])
    assert df["close"].iloc[0] == float(rows[0]["TDD_CLSPRC"].replace(",", ""))
    table, inputs = krx_open.load(tmp_path)
    assert len(table) == len(rows) and list(inputs) == ["KOSPI/2012"]


def test_an_empty_day_keeps_the_table_numeric(tmp_path, snaps):
    part = small(snaps, "2012-05-02", "2012-05-31")
    part[("KOSPI", pd.Timestamp("2012-05-10"))] = []          # the exchange answered with no rows
    part[("KOSDAQ", pd.Timestamp("2012-05-10"))] = []
    store_snapshots(tmp_path, part)
    store_snapshots(tmp_path, {("KOSPI", pd.Timestamp("2013-01-02")): []})   # a year with no rows at all
    table = krx_open.load(tmp_path)[0]
    assert table["close"].dtype == "float64" and table["mktcap"].dtype == "float64"
    assert pd.api.types.is_datetime64_any_dtype(table["date"])
    days = krx_open.sessions("2012-05-02", "2012-05-31")
    frames, _ = kr_stocks.adjusted(table, days, ["005930"])
    assert frames["005930"]["close"].notna().sum() == len(days) - 1


# --- prices and universe ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def table(tmp_path_factory, snaps):
    root = tmp_path_factory.mktemp("krx")
    store_snapshots(root, small(snaps, "2010-01-04", "2016-12-30"))
    return krx_open.load(root)[0]


def test_adjusted_prices_follow_the_exchange_base_price(table):
    days = krx_open.sessions("2010-01-04", "2016-12-30")
    frames, counts = kr_stocks.adjusted(table, days, ["051910", "068270", "123450"])
    lg = frames["051910"]["close"]
    raw = table[table["code"] == "051910"].set_index("date")
    t = pd.Timestamp(SPLIT[1])
    i = days.get_loc(t)
    # the raw close drops to a fifth; the adjusted close moves by the day's real return only
    assert raw.loc[t, "close"] < raw.loc[days[i - 1], "close"] / 4
    real = raw.loc[t, "close"] / (raw.loc[t, "close"] - raw.loc[t, "change"]) - 1
    assert lg.iloc[i] / lg.iloc[i - 1] - 1 == pytest.approx(real)
    assert lg.iloc[-1] == pytest.approx(raw["close"].iloc[-1])  # recent levels are the traded ones
    halt = frames["068270"].loc[HALT[1]:HALT[2]]
    assert halt["open"].isna().all() and halt["close"].notna().all() and counts["halted_rows"] == len(halt)
    dead = frames["123450"]["close"]
    assert dead.loc[:"2016-05-31"].notna().all() and dead.loc["2016-06-01":].isna().all()


def test_membership_is_point_in_time_and_fixed_for_a_month(table):
    days = krx_open.sessions("2010-01-04", "2016-12-30")
    m = kr_stocks.membership(table, days, top_n=5)
    assert "005935" not in m and "319400" not in m                     # preferred share, SPAC
    assert m.sum(axis=1).max() <= 5
    assert not m.loc[:"2010-01-29"].any().any()                        # first pick at January's last close
    jan, feb = m.loc["2010-02-01"], m.loc["2010-02-26"]
    assert jan.equals(feb)                                             # held through the month
    # members are the largest by market value at the month's last close
    cap = table[table["date"] == pd.Timestamp("2010-01-29")].set_index("code")["mktcap"]
    cap = cap[kr_stocks.common(cap.index.to_series(), table.drop_duplicates("code").set_index("code")["name"].reindex(cap.index))]
    assert set(m.columns[jan.to_numpy()]) == set(cap.sort_values(ascending=False).index[:5])
    full = kr_stocks.membership(table, days)
    assert not full.loc[ADMIN[1]:ADMIN[2], "222800"].any()             # 관리종목: not a member
    assert not full.loc[HALT[1]:HALT[2], "068270"].any()               # halted: not a member those days
    assert not full.loc[:"2011-08-31", "105560"].any()                 # listed 2011-03: needs 120 traded sessions
    assert full.loc["2011-10-04":"2011-10-31", "105560"].all()
    assert not full.loc["2016-06-01":, "123450"].any()


# --- costs --------------------------------------------------------------------------------------
class BuyThenSell(Strategy):
    def __init__(self):
        super().__init__()
        self.strategy_id, self.version = "T", "1"

    def describe(self):
        return {}

    def on_bar(self, view):
        k = len(view.history("AAA")) - 1
        if k == 1:
            return {"AAA": 0.45, "ETF": 0.45}
        if k == 3:
            return {"AAA": 0.0, "ETF": 0.0}
        return None


def test_sales_of_stocks_pay_the_transaction_tax_and_etfs_do_not():
    idx = pd.bdate_range("2020-01-01", periods=6)
    frames = {}
    for s in ("AAA", "ETF"):
        f = frame(np.full(6, 1000.0), np.full(6, 1000.0))
        f.index = idx
        frames[s] = f
    data = MarketData(idx, frames)
    costs = CostModel(commission_bps=1.5, slippage_bps=0.0, sell_tax_bps=20.0, tax_exempt=("ETF",))
    t = run_backtest(BuyThenSell(), data, BacktestConfig(initial_cash=1_000_000, costs=costs)).trades
    sells = t[t["qty"] < 0].set_index("symbol")
    assert sells.loc["AAA", "commission"] == pytest.approx(abs(sells.loc["AAA", "notional"]) * 21.5e-4)
    assert sells.loc["ETF", "commission"] == pytest.approx(abs(sells.loc["ETF", "notional"]) * 1.5e-4)
    buys = t[t["qty"] > 0].set_index("symbol")
    assert buys.loc["AAA", "commission"] == pytest.approx(buys.loc["AAA", "notional"] * 1.5e-4)


# --- catalog 5 and the factory group ------------------------------------------------------------
def test_catalog_5_is_the_price_items_on_kodex_200():
    keys = sc.conditions(5)
    assert not [k for k in keys if sc.CONDITIONS[k][1] in ("earn", "ins", "spy", "index", "sell", "pre")]
    assert {"mkt_up200", "mkt_down200", "mkt_drop5", "uplimit", "dnlimit", "drop5", "squeeze"} <= set(keys)
    assert len(sc.entries(5)) == 1705 and np.prod([len(v) for v in sc.full_space(5).values()]) == 276210
    assert all("mkt_" not in e and "limit" not in e for e in sc.entries(4))   # older catalogs unchanged
    loaded = load_spec(builtin_path("specs", "AUTO-KRSTK.qls.yaml"))
    assert {k: list(v) for k, v in loaded.search_space.items()} == sc.full_space(5)
    assert not any(sc.already_tried(p, 5) for p in list(grid(sc.full_space(5)))[:2000])
    assert sc.benchmark(5) == "069500" and sc.benchmark(4) == "SPY"


@pytest.fixture(scope="module")
def kr_lab(tmp_path_factory, snaps):
    tmp = tmp_path_factory.mktemp("krlab")
    text = naver_text("069500", 0)
    mp = pytest.MonkeyPatch()
    import requests

    mp.setattr(requests, "get", lambda url, params=None, headers=None, timeout=None: FakeResponse(text))
    mp.setattr(providers, "get_api_key", lambda provider="tiingo": "good-key" if provider == "krx" else None)
    mp.setattr(kr_dataset, "latest_day", lambda today=None: pd.Timestamp(END))
    http = FakeKrx(snaps)
    real = krx_open.download
    mp.setattr(krx_open, "download", lambda *a, **k: real(*a, **(k | {"session": http, "today": "2022-06-01"})))
    lab = Lab(tmp / "QL")
    lab.krx_http = http
    yield lab
    lab.close()
    mp.undo()


def test_round_3_downloads_builds_and_searches_korean_stocks(kr_lab):
    auto = kr_lab.auto_search()
    out = auto.run("AUTO-KRSTK", workers=1, limit=4)
    assert out["done"] >= 4 and out["failed"] == 0 and out["pending"] == 0
    assert kr_lab.registry.holdout_ledger() == []
    assert out["beat_symbol"] == "069500" and out["family_trials"] == out["done"]
    calls = len(kr_lab.krx_http.calls)
    assert calls == 2 * len(krx_open.sessions("2010-01-04", END))
    _, loaded = auto.loaded("AUTO-KRSTK")
    _, policy = kr_lab.policy_for(loaded)
    version = kr_lab.ensure_spec_dataset(loaded, policy)
    assert len(kr_lab.krx_http.calls) == calls                     # stored: not asked again
    manifest = kr_lab.data.load_dataset(version)[1]
    assert manifest["index"] == "KRTOP300" and manifest["qc_fatal"] == []
    assert manifest["names"]["005930"] == "삼성전자" and "005935" not in manifest["names"]
    params = dict(loaded.default_params, entry="drop1_4", hold=5, slots=5)
    res = kr_lab.backtest(auto.loaded("AUTO-KRSTK")[0], version, params).result
    sells = res.trades[(res.trades["qty"] < 0) & (res.trades["symbol"] != "069500")]
    assert len(sells) and (sells["commission"] > sells["notional"].abs() * 21e-4).all()   # tax paid
    assert set(res.trades["symbol"]) - {"069500"} <= set(manifest["names"])


def test_krx_key_is_asked_for_before_any_download(tmp_path, monkeypatch):
    monkeypatch.setattr(providers, "get_api_key", lambda provider="tiingo": None)
    lab = Lab(tmp_path / "QL")
    try:
        with pytest.raises(RuntimeError, match="KRX Open API 인증키"):
            lab.ensure_kr_dataset("whatever", "2010-01-01")
        assert not lab.has_krx_key()
    finally:
        lab.close()
