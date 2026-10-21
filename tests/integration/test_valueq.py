"""D-037: SEC financial statement facts and the value-plus-profitability rule (VALQ-SP500)."""

import json

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.engine import run_backtest
from quantlab.core.market_data import MarketData
from quantlab.data import sec_facts
from quantlab.services.lab import builtin_path
from quantlab.strategies import valueq as vq
from quantlab.strategies.spec import load_spec
from quantlab.validation.engine import leak_test
from insider_fixtures import facts_doc, store_facts
from synthetic import frame

STOCKS = ["A", "B", "C", "D", "E"]


def test_spec_and_policy_load():
    x = load_spec(builtin_path("specs", "VALQ-SP500.qls.yaml"))
    assert x.default_params == {"top_n": 50} and x.spec["rules"]["value"] == {"months": [3, 6, 9, 12], "park": "SPY"}


def test_parse_keeps_annual_values_point_in_time(tmp_path):
    docs = {
        "1": facts_doc("1", [2010, 2011], assets=lambda y: 1000.0 * (y - 2009), equity=lambda y: 400.0,
                       gross=lambda y: 300.0, float_=lambda y: 5000.0, amend={2010: ("2011-09-01", 1100.0)}),
        "2": facts_doc("2", [2010], assets=lambda y: 800.0, equity=lambda y: -5.0,
                       revenue_cost=lambda y: (1000.0, 650.0)),
    }
    store_facts(tmp_path, docs)
    rec = sec_facts.records(tmp_path)
    one = sec_facts.load_parsed(rec["1"])
    gp = one[one["fact"] == "gross_profit"]
    assert sorted(gp["value"]) == [300.0, 300.0]                       # the quarter in the 10-K is not annual
    a = one[(one["fact"] == "assets") & (one["end"] == "2010-12-31")]
    assert sorted(zip(a["filed"].dt.strftime("%Y-%m-%d"), a["value"])) == [("2011-02-20", 1000.0), ("2011-09-01", 1100.0)]
    fl = one[one["fact"] == "float"]
    assert list(fl["end"].dt.strftime("%Y-%m-%d")) == ["2010-06-30", "2011-06-30"]
    two = sec_facts.load_parsed(rec["2"])
    assert two.loc[two["fact"] == "gross_profit", "value"].tolist() == [350.0]   # revenue minus cost of revenue
    links = pd.DataFrame({"issuer_cik": ["1", "2"], "ticker": ["A", "B"],
                          "first_t": pd.Timestamp("2009-01-02"), "last_t": pd.Timestamp("2021-12-31")})
    ev = sec_facts.fact_events(pd.concat([one, two]), links)
    assert set(ev["kind"]) == {"fact_assets", "fact_equity", "fact_gross_profit", "fact_float"}
    assert ev[ev["symbol"] == "A"]["known"].min() == pd.Timestamp("2011-02-21")   # the day after filing


def test_download_remembers_companies_without_facts(tmp_path):
    got = []

    class Resp:
        def __init__(self, code, content=b""):
            self.status_code, self.content = code, content

    class Http:
        def get(self, url, headers=None, timeout=0):
            got.append(url)
            return Resp(404) if "0000000002" in url else Resp(200, json.dumps(facts_doc(
                "1", [2010], assets=lambda y: 1.0, equity=lambda y: 1.0)).encode())

    real = sec_facts.PAUSE_SECONDS
    try:
        import quantlab.data.sec_earnings as se
        se.PAUSE_SECONDS = 0
        out = sec_facts.download(tmp_path, "someone@example.com", ["1", "2"], session=Http())
        again = sec_facts.download(tmp_path, "someone@example.com", ["1", "2"], session=Http())
    finally:
        se.PAUSE_SECONDS = real
    assert out == {"fetched": ["1", "2"], "missing": ["2"]} and again == {"fetched": [], "missing": []}
    assert len(got) == 2 and got[0].endswith("/api/xbrl/companyfacts/CIK0000000001.json")
    rec = sec_facts.records(tmp_path)
    assert rec["2"]["empty"] and sec_facts.load_parsed(rec["2"]).empty


def market(n=560, events=None):
    """Five members, flat prices except C (doubles over the span); facts as given."""
    idx = pd.bdate_range("2010-01-01", periods=n)
    frames = {}
    for s in ["SPY"] + STOCKS:
        c = np.full(n, 100.0) if s != "C" else 100 * np.linspace(1, 2, n)
        f = frame(c, c)
        f.index = idx
        frames[s] = f
    members = pd.DataFrame(True, index=idx, columns=STOCKS)
    ev = pd.DataFrame(events or [], columns=["known", "symbol", "kind", "value", "period_end"])
    ev["known"] = pd.to_datetime(ev["known"])
    ev["period_end"] = pd.to_datetime(ev["period_end"])
    return MarketData(idx, frames, members, ev.sort_values("known", kind="stable").reset_index(drop=True))


def facts(sym, known, equity, gp, assets=1000.0, flt=1000.0, end="2010-06-30", float_end="2010-06-30"):
    return [(known, sym, "fact_equity", equity, end), (known, sym, "fact_gross_profit", gp, end),
            (known, sym, "fact_assets", assets, end), (known, sym, "fact_float", flt, float_end)]


def test_scores_value_plus_profitability_from_known_filings_only():
    ev = (facts("A", "2010-08-02", 900, 800) + facts("B", "2010-08-02", 100, 900)
          + facts("C", "2010-08-02", 500, 100) + facts("D", "2010-08-02", 50, 50)
          + facts("E", "2010-08-02", -10, 999))                         # negative equity: not ranked
    data = market(events=ev)
    p = vq.build_panels(data)
    t0 = data.calendar.get_loc(pd.Timestamp("2010-07-30"))
    t1 = data.calendar.get_loc(pd.Timestamp("2010-08-02"))
    assert np.isnan(p["bm"][t0]).all() and np.isnan(p["gpa"][t0]).all()          # before the filings
    col = {s: j for j, s in enumerate(p["symbols"])}
    assert p["bm"][t1, col["A"]] == pytest.approx(0.9) and p["gpa"][t1, col["B"]] == pytest.approx(0.9)
    assert p["gpa"][t1, col["A"]] == pytest.approx(0.8)
    assert np.isnan(p["bm"][t1, col["E"]])
    # C's price rose since the float date, so its market value did too and its book-to-market fell
    t2 = data.calendar.get_loc(pd.Timestamp("2010-12-31"))
    rise = data.frames["C"]["close"].iloc[t2] / data.frames["C"]["close"].loc["2010-06-30"]
    assert p["bm"][t2, col["C"]] == pytest.approx(0.5 / rise)
    s = vq.ValueQualityStrategy(spec_id="T", spec_version_str="1",
                                rules={"value": {"months": [3, 6, 9, 12], "park": "SPY"}}, top_n=2)
    t = run_backtest(s, data).trades
    first = t[(t["symbol"] != "SPY") & (t["qty"] > 0)]
    # first pick: the first session with 2+ ranked members (2010-08-02), filled next open.
    # ranks (0-3) book-to-market D<B<C<A, gross profit to assets D<C<A<B: A 5, B 4, C 3, D 0
    assert first["date"].min() == pd.Timestamp("2010-08-03")
    assert sorted(first[first["date"] == first["date"].min()]["symbol"]) == ["A", "B"]


def test_stale_statements_are_dropped_and_restatements_count_from_their_filing():
    ev = facts("A", "2010-03-01", 100, 100, end="2009-12-31", float_end="2009-06-30")
    ev += [("2010-09-01", "A", "fact_assets", 500.0, "2009-12-31")]      # restated, same period
    data = market(n=800, events=ev)
    p = vq.build_panels(data)
    col = {s: j for j, s in enumerate(p["symbols"])}
    t = lambda d: data.calendar.get_loc(pd.Timestamp(d))  # noqa: E731
    assert p["gpa"][t("2010-08-31"), col["A"]] == pytest.approx(0.1)
    assert p["gpa"][t("2010-09-01"), col["A"]] == pytest.approx(0.2)
    # the float's date (2009-06-30) is before the first price, so there is no market value yet
    assert np.isnan(p["bm"][t("2010-09-01"), col["A"]])
    assert np.isnan(p["gpa"][t("2011-05-16"), col["A"]])                  # period ended over 500 days ago


def test_panels_use_only_the_past():
    rng = np.random.default_rng(4)
    ev = []
    for y in (2010, 2011):
        for s in STOCKS:
            ev += facts(s, f"{y}-08-0{rng.integers(2, 6)}", float(rng.integers(50, 900)), float(rng.integers(50, 900)),
                        end=f"{y}-06-30", float_end=f"{y}-06-30")
    data = market(events=ev)
    full = vq.build_panels(data)
    cut = 400
    part = vq.build_panels(MarketData(data.calendar[:cut], {s: f.iloc[:cut] for s, f in data.frames.items()},
                                      data.members.iloc[:cut], data.events[data.events["known"] <= data.calendar[cut - 1]]))
    for k in ("bm", "gpa"):
        np.testing.assert_array_equal(np.isnan(full[k][:cut]), np.isnan(part[k]))
        np.testing.assert_allclose(full[k][:cut][np.isfinite(part[k])], part[k][np.isfinite(part[k])])
    s = lambda **p: vq.ValueQualityStrategy(spec_id="T", spec_version_str="1",  # noqa: E731
                                            rules={"value": {"months": [3, 6, 9, 12], "park": "SPY"}}, **p)
    assert leak_test(s, {"top_n": 2}, data, 4, 7) == []
