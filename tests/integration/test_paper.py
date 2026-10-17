"""D-030: paper trading (모의 투자) of a frozen rule on the forward holdout, on synthetic sources."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from quantlab.core.hashing import file_sha256
from quantlab.data import sec_earnings, sec_insider, sp500
from quantlab.data.quality import exchange_sessions
from quantlab.services import paper as P
from quantlab.services.lab import Lab, builtin_path
from insider_fixtures import bundle_frames, store_bundle, store_sec, store_submissions
from synthetic import frame
from test_insider import CIK, STOCKS, earnings_calendar, make_filings

START = "2021-03-01"


def lag(bundle_dir):
    p = f"{bundle_dir}/meta.json"
    meta = json.loads(open(p, encoding="utf-8").read())
    meta["last_date"] = "2021-12-30"
    open(p, "w", encoding="utf-8").write(json.dumps(meta))
EARN_DOWNLOAD = sec_earnings.download  # the module fixture replaces it while it runs


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("paper")
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
    spans = {t: ("2009-06-01", "2021-12-31") for t in STOCKS + ["META"]}
    spans["DEAD-201506"] = ("2009-06-01", "2015-06-30")
    prices = bundle_frames(sessions, spans, crsp=("DEAD-201506",))
    companies, jumps = earnings_calendar(sessions)
    for t, day in jumps:  # +12% from the first session after the filing date, gap included
        nxt = sessions[sessions > day][0]
        m = (prices["ticker"] == t) & (prices["date"] >= nxt)
        prices.loc[m, ["open", "high", "low", "close", "adj_close"]] *= 1.12
    first = STOCKS[:10] + ["META", "DEAD-201506"]
    members = pd.concat([pd.DataFrame({"date": pd.Timestamp("2009-12-31"), "ticker": first}),
                         pd.DataFrame({"date": pd.Timestamp("2015-07-01"), "ticker": STOCKS + ["META"]})],
                        ignore_index=True)
    store_bundle(lab.ws.root, prices, members)
    store_sec(lab.ws.root, make_filings(sessions))
    store_submissions(lab.ws.root, companies)
    lab.set_sec_email("someone@example.com")

    # the rule of GAP-EARN-SP500 under a test policy whose forward holdout starts inside the data
    policy = yaml.safe_load(builtin_path("policies", "gap_earn_v1.yaml").read_text(encoding="utf-8"))
    policy.update(policy_id="gap_paper_t", strategy_family="GAP-PAPER-T")
    policy["splits"]["forward_holdout_start"] = START
    pol = tmp / "gap_paper_t.yaml"
    pol.write_text(yaml.safe_dump(policy, allow_unicode=True), encoding="utf-8")
    lab.register_policy_file(pol)
    spec = yaml.safe_load(builtin_path("specs", "GAP-EARN-SP500.qls.yaml").read_text(encoding="utf-8"))
    spec["strategy"].update(id="GAP-PAPER-T", parent=None)
    spec["validation"]["policy"] = "gap_paper_t"
    spec_path = tmp / "GAP-PAPER-T.qls.yaml"
    spec_path.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")

    calls = {"bundle": 0, "earn": []}
    fixture_bundle = sp500.bundles(lab.ws.root)[0]
    lag(fixture_bundle["dir"])  # the stored bundle lacks the last SPY session: a refresh downloads it again

    def fake_bundle(root, session=None, say=None, timeout=0):  # "today's" download: the same files again
        calls["bundle"] += 1
        ts = f"20261016T0{calls['bundle']}0000000000Z"
        d = sp500.bundle_dir(root) / ts
        d.mkdir(parents=True)
        for n in sp500.FILES:
            shutil.copy(f"{fixture_bundle['dir']}/{n}", d / n)
        meta = {**fixture_bundle, "fetched_at": ts, "dir": str(d), "last_date": "2021-12-31"}
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return meta

    saved = sec_insider.download, sec_earnings.download, sp500.download_bundle
    sec_insider.download = lambda *a, **k: {"fetched": [], "missing": []}
    sec_earnings.download = lambda *a, **k: calls["earn"].append(k.get("refresh_before")) or {"fetched": [], "missing": []}
    sp500.download_bundle = fake_bundle
    yield lab, spec_path, calls
    sec_insider.download, sec_earnings.download, sp500.download_bundle = saved
    lab.close()


def test_refresh_trades_only_from_the_forward_start(env):
    lab, spec, calls = env
    seen = []
    real = P.FromDate.on_bar

    def spy(self, view):
        out = real(self, view)
        if view.now >= self.start:
            seen.append(view.now)
        return out

    P.FromDate.on_bar = spy
    try:
        snap = lab.paper().refresh(spec, today="20261017")
    finally:
        P.FromDate.on_bar = real
    assert snap["started"] and snap["start"] == START and snap["asof"] == "2021-12-31"
    assert min(seen) >= pd.Timestamp(START)
    assert snap["curve"][0][0] >= START and abs(snap["curve"][0][1] - snap["initial"]) < 1e-6
    fills = json.loads((lab.paper().folder("GAP-PAPER-T", "1.0.0") / "ledger.json").read_text())["fills"]
    assert fills and min(f["date"] for f in fills) > START
    buys = [f for f in fills if f["side"] == "buy" and f["symbol"] not in ("SPY", "BIL")]
    assert buys, "the synthetic earnings gaps should give at least one signal"
    held = [h for h in snap["holdings"] if not h["park"]]
    assert len(held) <= 5 and all(h["weight"] < 0.25 for h in held)
    assert {o["side"] for o in snap["orders"]} <= {"buy", "sell", "park"}
    assert snap["ledger"]["changed"] == []
    # today's data: the bundle again and the current members' earnings filings
    assert calls["bundle"] == 1 and "20261017" in calls["earn"]
    assert lab.paper().latest("GAP-PAPER-T", "1.0.0")["asof"] == "2021-12-31"


def test_a_second_refresh_keeps_the_record_and_reports_a_changed_fill(env):
    lab, spec, calls = env
    paper = lab.paper()
    led_path = paper.folder("GAP-PAPER-T", "1.0.0") / "ledger.json"
    led = json.loads(led_path.read_text())
    n = len(led["fills"])
    led["fills"][0]["price"] *= 1.01           # as if the price had since been revised
    led_path.write_text(json.dumps(led))
    snap = paper.refresh(spec, today="20261018")
    assert calls["bundle"] == 1        # the bundle has the last session now: not downloaded again
    assert [c["kind"] for c in snap["ledger"]["changed"]] == ["different"]
    after = json.loads(led_path.read_text())
    assert len(after["fills"]) == n and after["fills"][0]["price"] == led["fills"][0]["price"]  # first version kept


def test_old_bundles_from_refreshes_are_removed(env):
    lab, spec, calls = env
    lag(sp500.bundles(lab.ws.root)[-1]["dir"])
    lab.paper().refresh(spec, today="20261019")
    assert calls["bundle"] == 2
    dirs = [b["dir"] for b in sp500.bundles(lab.ws.root)]
    # the fixture's own bundle stays; of the refreshes' bundles only the newest is kept
    assert len(dirs) == 2 and dirs[0].endswith("20261016T000000000000Z") and dirs[1].endswith("T020000000000Z")
    made = json.loads(lab.ws.path("paper", "created.json").read_text())
    assert made["bundles"] == [dirs[1]]


def test_before_the_start_nothing_is_run(env, tmp_path):
    lab, spec, calls = env
    s = yaml.safe_load(spec.read_text(encoding="utf-8"))
    pol = yaml.safe_load(builtin_path("policies", "gap_earn_v1.yaml").read_text(encoding="utf-8"))
    pol.update(policy_id="gap_paper_late", strategy_family="GAP-PAPER-LATE")
    pol["splits"]["forward_holdout_start"] = "2022-03-01"
    p = tmp_path / "late.yaml"
    p.write_text(yaml.safe_dump(pol, allow_unicode=True), encoding="utf-8")
    lab.register_policy_file(p)
    s["strategy"]["id"] = "GAP-PAPER-LATE"
    s["validation"]["policy"] = "gap_paper_late"
    sp = tmp_path / "late.qls.yaml"
    sp.write_text(yaml.safe_dump(s, allow_unicode=True), encoding="utf-8")
    snap = lab.paper().refresh(sp, today="20261018")
    assert snap["started"] is False and "curve" not in snap
    assert not (lab.paper().folder("GAP-PAPER-LATE", "1.0.0") / "ledger.json").exists()


def test_session_without_stock_prices_is_not_used():
    from quantlab.core.market_data import MarketData

    idx = pd.bdate_range("2021-01-01", periods=5)
    frames = {}
    for s in ("SPY", "X", "Y"):
        f = frame(np.full(5, 10.0), np.full(5, 10.0))
        f.index = idx
        if s != "SPY":
            f.iloc[-1, :4] = np.nan
        frames[s] = f
    members = pd.DataFrame(True, index=idx, columns=["X", "Y"])
    data = MarketData(idx, frames, members, None)
    assert P.priced_until(data) == idx[-2]


def test_earnings_refresh_fetches_only_the_main_file(tmp_path):
    store_submissions(tmp_path, {"1001": [("2020-01-1%d" % i, "8-K", "2.02") for i in range(8)]}, page_size=3)
    got = []

    class Resp:
        status_code = 200

        def __init__(self, content):
            self.content = content

    main = {"cik": "1001", "filings": {"recent": {"accessionNumber": ["n1"], "filingDate": ["2021-02-01"],
                                                  "form": ["8-K"], "items": ["2.02"], "acceptanceDateTime": [""]},
                                       "files": [{"name": "CIK0000001001-submissions-000.json"},
                                                 {"name": "CIK0000001001-submissions-001.json"}]}}

    class Http:
        def get(self, url, headers=None, timeout=0):
            got.append(url)
            return Resp(json.dumps(main).encode())

    real = sec_earnings.PAUSE_SECONDS
    sec_earnings.PAUSE_SECONDS = 0
    try:
        EARN_DOWNLOAD(tmp_path, "someone@example.com", ["1001"], session=Http(), refresh_before="20261016")
        assert got == []                                   # fetched that day already
        EARN_DOWNLOAD(tmp_path, "someone@example.com", ["1001"], session=Http(), refresh_before="20261017")
    finally:
        sec_earnings.PAUSE_SECONDS = real
    assert len(got) == 1 and got[0].endswith("CIK0000001001.json")
    meta = sec_earnings.records(tmp_path)["1001"]
    df = sec_earnings.parse_company(meta["path"], "1001")
    assert len(df) == 1 + 5 and "n1" in set(df["accession"])   # new main file + the stored older pages
    assert meta["sha256"] == file_sha256(meta["path"])


def test_automatic_refresh_logs_one_line_per_rule_and_skips_while_locked(env):
    """D-031: what `QuantLab.exe --paper-refresh` runs."""
    lab, spec, calls = env
    paper = lab.paper()
    assert paper.run_auto() is True and paper.last_auto() is None   # nothing switched on: nothing run
    paper.set_auto("GAP-PAPER-T", True)
    assert paper.auto_ids() == ["GAP-PAPER-T"]
    assert paper.run_auto(today="20261020") is True, paper.last_auto()
    assert "GAP-PAPER-T 기준일 2021-12-31" in paper.last_auto()
    lock = lab.ws.path("paper", "refresh.lock")
    lock.write_text("")
    try:
        with pytest.raises(RuntimeError, match="진행 중"):
            paper.refresh(spec, today="20261020")
        assert paper.run_auto(today="20261020") is False and "실패" in paper.last_auto()
    finally:
        lock.unlink()
    old = lock.with_name("refresh.lock")
    old.write_text("")
    import os
    import time

    os.utime(old, (time.time() - 4 * 3600,) * 2)                  # left by a killed run
    paper.refresh(spec, today="20261020")
    assert not old.exists()
    paper.set_auto("GAP-PAPER-T", False)


def test_scheduled_task_runs_the_app_twice_a_day_and_catches_up():
    import xml.etree.ElementTree as ET
    from datetime import date

    from quantlab.services import scheduler

    xml = scheduler.task_xml(r"C:\Users\A & B\QuantLab\QuantLab.exe", start=date(2026, 10, 17))
    root = ET.fromstring(xml.split("\n", 1)[1])
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    starts = [e.text for e in root.findall(".//t:CalendarTrigger/t:StartBoundary", ns)]
    assert starts == ["2026-10-17T08:30:00", "2026-10-17T21:00:00"]
    assert root.find(".//t:StartWhenAvailable", ns).text == "true"
    assert root.find(".//t:Exec/t:Command", ns).text == r"C:\Users\A & B\QuantLab\QuantLab.exe"
    assert root.find(".//t:Exec/t:Arguments", ns).text == "--paper-refresh"
    assert root.find(".//t:LogonType", ns).text == "InteractiveToken"
