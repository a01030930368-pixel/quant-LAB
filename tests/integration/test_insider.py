"""D-026: S&P 500 history + SEC insider filings → INS-CLUSTER, end to end on synthetic sources."""

import numpy as np
import pandas as pd
import pytest

from quantlab.core.market_data import MarketData
from quantlab.data import sec_insider
from quantlab.data.quality import exchange_sessions
from quantlab.services.lab import Lab, builtin_path
from quantlab.strategies.insider import InsiderClusterStrategy
from quantlab.strategies.spec import load_spec
from quantlab.backtesting.engine import run_backtest
from insider_fixtures import bundle_frames, store_bundle, store_sec, write_quarter_zip
from synthetic import frame

SPEC = builtin_path("specs", "INS-CLUSTER.qls.yaml")
STOCKS = [f"A{i:02d}" for i in range(1, 12)]
CIK = {t: str(1000 + i) for i, t in enumerate(STOCKS)}
CIK.update({"META": "1326801", "DEAD-201506": "2001"})


def cluster(acc, day, cik, symbol, owners, value=50_000, doc="4", rel="Officer"):
    """One filing per owner, all on `day`."""
    out = []
    for j, o in enumerate(owners):
        out.append({"acc": f"{acc}-{j}", "date": day, "doc": doc, "cik": cik, "symbol": symbol,
                    "owners": [(o, rel)], "trans": [("P", "A", value / 10, 10)]})
    return out


def make_filings(sessions):
    rng = np.random.default_rng(11)
    filings = []
    n = 0
    days = sessions[(sessions >= "2010-02-01") & (sessions <= "2021-03-31")]
    for k in range(160):  # clusters: insider 1 on day d, insider 2 three days later
        d = days[int(rng.integers(len(days) - 5))]
        t = STOCKS[int(rng.integers(len(STOCKS)))]
        n += 1
        filings += cluster(f"C{n}", d, CIK[t], t, [f"{t}-o1"])
        filings += cluster(f"D{n}", d + pd.Timedelta(days=3), CIK[t], t, [f"{t}-o2"])
    # renamed issuer: filings under the old symbol FB reach the ticker META through the CIK
    filings += cluster("FB1", "2012-06-04", CIK["META"], "FB", ["fb-1", "fb-2"])
    filings += cluster("MT1", "2021-03-01", CIK["META"], "META", ["fb-3"])
    # delisted member, and an unrelated later company reusing its symbol (different CIK)
    filings += cluster("DD1", "2013-05-06", CIK["DEAD-201506"], "DEAD", ["d-1", "d-2"])
    filings += cluster("RC1", "2019-05-06", "9999", "DEAD", ["r-1", "r-2"])
    # noise: amendment, sales, tiny purchases
    filings += cluster("AM1", "2011-03-01", CIK["A01"], "A01", ["a-1", "a-2"], doc="4/A")
    filings.append({"acc": "S1", "date": "2011-04-01", "cik": CIK["A02"], "symbol": "A02",
                    "owners": [("s-1", "Director")], "trans": [("S", "D", 1000, 50)]})
    filings += cluster("TINY", "2011-05-02", CIK["A03"], "A03", ["t-1", "t-2"], value=500)
    return filings


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ins")
    lab = Lab(tmp / "QL")
    sessions = exchange_sessions("2009-06-01", "2021-12-31")
    rng = np.random.default_rng(5)
    for sym in ("SPY", "BIL"):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, {"SPY": 0.01, "BIL": 0.0003}[sym], len(sessions))))
        df = frame(close, close)
        df.index = sessions
        p = tmp / f"{sym}.csv"
        df.rename_axis("date").reset_index().to_csv(p, index=False, date_format="%Y-%m-%d")
        lab.import_csv(sym, p)
    spans = {t: ("2009-06-01", "2021-12-31") for t in STOCKS + ["META", "NOTM"]}
    spans["DEAD-201506"] = ("2009-06-01", "2015-06-30")
    spans["DEAD"] = ("2018-01-02", "2021-12-31")
    prices = bundle_frames(sessions, spans, crsp=("DEAD-201506",))
    first = STOCKS[:10] + ["META", "DEAD-201506"]
    members = pd.concat([pd.DataFrame({"date": pd.Timestamp("2009-12-31"), "ticker": first}),
                         pd.DataFrame({"date": pd.Timestamp("2015-07-01"),
                                       "ticker": STOCKS + ["META"]})], ignore_index=True)
    store_bundle(lab.ws.root, prices, members)
    store_sec(lab.ws.root, make_filings(sessions))
    lab.set_sec_email("someone@example.com")
    sec_insider_download = sec_insider.download
    sec_insider.download = lambda *a, **k: {"fetched": [], "missing": []}  # no network in tests
    yield lab
    sec_insider.download = sec_insider_download
    lab.close()


@pytest.fixture(scope="module")
def built(lab):
    loaded = load_spec(SPEC)
    _, policy = lab.policy_for(loaded)
    version = lab.ensure_spec_dataset(loaded, policy)
    data, manifest = lab.load_dataset(version)
    return version, data, manifest


def test_dataset_links_filings_to_tickers_point_in_time(built):
    version, data, m = built
    assert m["kind"] == "index" and m["qc_fatal"] == [], m["qc_fatal"]
    assert m["coverage"]["share"] > 0.999
    assert set(data.members.columns) == set(STOCKS + ["META", "DEAD-201506"])  # NOTM never a member
    ev = data.events
    assert ev["known"].is_monotonic_increasing
    # FB filings (2012) reach META through the issuer's CIK
    assert ((ev["symbol"] == "META") & (ev["known"] == "2012-06-04")).sum() == 2
    # the delisted member keeps its own filings; the later company reusing "DEAD" maps to nothing here
    dead = ev[ev["symbol"] == "DEAD-201506"]
    assert len(dead) == 2 and dead["known"].max() < pd.Timestamp("2015-07-01")
    assert not (ev["owner"].isin(["r-1", "r-2"])).any()
    # amendments and sales are not purchases; tiny purchases stay (the rule filters by value)
    assert not ev["owner"].isin(["a-1", "a-2", "s-1"]).any()
    assert ev.loc[ev["owner"] == "t-1", "value"].iloc[0] == 500
    # CRSP-style rows without an open trade at that day's close
    dead_px = data.frames["DEAD-201506"].dropna()
    assert (dead_px["open"] == dead_px["close"]).all()


def test_backtest_uses_dev_data_and_trades_only_members(lab, built):
    bt = lab.quick_backtest(SPEC)
    e = lab.registry.experiment(bt.experiment_id)
    assert e["kind"] == "exploratory" and e["metrics"]["window"][1] == "2019-12-31"
    assert e["metrics"]["trades"]["n"] > 20
    tr = bt.result.trades
    traded = set(tr["symbol"]) - {"SPY"}
    assert traded <= set(STOCKS + ["META", "DEAD-201506"])
    assert lab.registry.holdout_ledger() == []
    _, data, _ = built
    # every stock entry fills on the session after a filing that completed a cluster, while a member
    buys = tr[(tr["qty"] > 0) & (tr["symbol"] != "SPY")]
    ev = data.events
    for _, b in buys.head(30).iterrows():
        prev = data.calendar[data.calendar.get_loc(b["date"]) - 1]
        assert data.members.loc[prev, b["symbol"]]
        e_sym = ev[(ev["symbol"] == b["symbol"]) & (ev["known"] <= prev) & (ev["known"] > prev - pd.Timedelta(days=30))]
        assert e_sym["owner"].nunique() >= 2


def test_final_evaluation_reads_holdout_once(lab):
    out = lab.quick_final(SPEC)
    assert out.gates[0].verdict == "PASS", out.gates[0].reasons
    assert out.overall in ("PASS", "HOLD", "FAIL")
    assert len(lab.registry.holdout_ledger()) == (1 if any(g.gate == "GATE3" for g in out.gates) else 0)


def test_sec_email_is_required_and_checked(tmp_path):
    lab = Lab(tmp_path / "QL")
    try:
        assert lab.needs_sec_email(SPEC)
        with pytest.raises(ValueError):
            lab.set_sec_email("not an email")
        lab.set_sec_email("me@example.com")
        assert not lab.needs_sec_email(SPEC) and lab.sec_email() == "me@example.com"
        assert not lab.needs_sec_email(builtin_path("specs", "OSR-SECT3X.qls.yaml"))
    finally:
        lab.close()


# --- the rule itself ---------------------------------------------------------------------------
RULES = {"insider": {"roles": ["officer", "director"], "min_value": 10000, "min_insiders": 2, "window_days": 30,
                     "hold_days": 5, "slots": 2, "park": "SPY"}}


def small_market(events, members_off=()):
    idx = pd.bdate_range("2020-01-01", periods=60)
    frames = {}
    for s in ("SPY", "X", "Y", "Z"):
        c = np.linspace(100, 110, len(idx))
        f = frame(c, c)
        f.index = idx
        frames[s] = f
    members = pd.DataFrame(True, index=idx, columns=["X", "Y", "Z"])
    for s in members_off:
        members[s] = False
    ev = pd.DataFrame(events, columns=["known", "symbol", "owner", "value", "officer", "director"])
    ev["known"] = pd.to_datetime(ev["known"])
    ev["kind"] = "insider_buy"
    return MarketData(idx, frames, members, ev.sort_values("known", kind="stable").reset_index(drop=True))


def entries(data):
    res = run_backtest(InsiderClusterStrategy(spec_id="T", spec_version_str="1", rules=RULES), data)
    t = res.trades
    return t[(t["symbol"] != "SPY")]


def test_signal_fires_once_when_the_second_insider_files_and_exits_after_hold():
    data = small_market([("2020-01-06", "X", "o1", 20000, True, False),
                         ("2020-01-09", "X", "o2", 20000, False, True),
                         ("2020-01-13", "X", "o3", 20000, True, False)])   # third insider: no new signal
    t = entries(data)
    assert list(t["date"].dt.strftime("%Y-%m-%d")) == ["2020-01-10", "2020-01-17"]
    assert t["qty"].iloc[0] > 0 and t["qty"].iloc[1] < 0  # held 5 sessions open to open


def test_filters_value_role_membership_and_slots():
    data = small_market([("2020-01-06", "X", "o1", 5000, True, False),      # too small
                         ("2020-01-07", "X", "o2", 20000, True, False),
                         ("2020-01-06", "Y", "p1", 20000, False, False),     # neither officer nor director
                         ("2020-01-07", "Y", "p2", 20000, True, False),
                         ("2020-01-06", "Z", "q1", 20000, True, False),      # not a member
                         ("2020-01-07", "Z", "q2", 20000, True, False)], members_off=("Z",))
    assert entries(data).empty


def test_more_signals_than_slots_take_larger_value_first():
    data = small_market([("2020-01-06", s, f"{s}{i}", v, True, False)
                         for s, v in (("X", 20000), ("Y", 90000), ("Z", 50000)) for i in (1, 2)])
    buys = entries(data)
    assert set(buys.loc[buys["qty"] > 0, "symbol"]) == {"Y", "Z"}


def test_events_after_the_bar_are_invisible():
    from quantlab.validation.engine import leak_test

    data = small_market([("2020-01-06", "X", "o1", 20000, True, False), ("2020-01-09", "X", "o2", 20000, True, False),
                         ("2020-02-03", "Y", "p1", 20000, True, False), ("2020-02-04", "Y", "p2", 20000, True, False)])
    make = lambda **p: InsiderClusterStrategy(spec_id="T", spec_version_str="1", rules=RULES)  # noqa: E731
    assert leak_test(make, {}, data, cutpoints=4, seed=1) == []
