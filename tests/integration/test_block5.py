"""D-047: institutions' 5% holding reports from OpenDART (BLOCK5-KR) — download, events and the rule end
to end on synthetic exchange snapshots and reports."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from kr_fixtures import STOCKS, FakeKrx, Resp, corp, snapshots
from test_kr2x import FakeResponse, naver_text

from quantlab.data import dart_block, dart_open, kr_dataset, krx_open, providers
from quantlab.services.lab import Lab, builtin_path
from quantlab.strategies.block import validate_block
from quantlab.strategies.spec import load_spec

END = "2021-12-30"
SPEC = builtin_path("specs", "BLOCK5-KR.qls.yaml")


def report(day: str, form: str, held: float, change: float, who: str = "국민연금공단") -> dict:
    return {"rcept_no": f"{day.replace('-', '')}000123", "rcept_dt": day, "corp_code": "", "corp_name": "",
            "report_tp": form, "repror": who, "stkqy": "1,000,000", "stkqy_irds": "100,000",
            "stkrt": f"{held:.2f}", "stkrt_irds": f"{change:.2f}", "ctr_stkqy": "0", "ctr_stkrt": "0",
            "report_resn": "보유주식등의 수 증가"}


class FakeBlockDart:
    """Answers like OpenDART: corpCode.xml and majorstock.json from `reports` {corp_code: rows};
    `limit` majorstock calls, then status 020."""

    def __init__(self, reports: dict, key: str = "dart-key", limit: int | None = None) -> None:
        self.reports, self.key, self.limit, self.calls = reports, key, limit, []

    def get(self, url, params=None, headers=None, timeout=None):
        if params.get("crtfc_key") != self.key:
            return Resp(200, {"status": "010", "message": "등록되지 않은 키입니다."})
        if url.endswith("corpCode.xml"):
            items = "".join(f"<list><corp_code>{corp(c)}</corp_code><corp_name>{n}</corp_name>"
                            f"<stock_code>{c}</stock_code></list>" for c, (n, *_r) in STOCKS.items() if c != "123450")
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("CORPCODE.xml", f'<?xml version="1.0" encoding="UTF-8"?><result>{items}</result>')
            r = Resp(200, {})
            r.content = buf.getvalue()
            return r
        assert url.endswith(dart_block.ENDPOINT)
        if self.limit is not None and len(self.calls) >= self.limit:
            return Resp(200, {"status": "020", "message": "사용한도를 초과하였습니다."})
        self.calls.append(params["corp_code"])
        rows = self.reports.get(params["corp_code"], [])
        return Resp(200, {"status": "000", "message": "정상", "list": rows} if rows else
                    {"status": "013", "message": "조회된 데이타가 없습니다."})


def test_rule_is_fixed_before_any_run():
    loaded = load_spec(SPEC)
    assert loaded.default_params == {"hold_days": 20}
    assert loaded.spec["rules"]["block"] == {"kinds": ["kr_block_new"], "min_change": 0.0, "slots": 10,
                                             "park": "069500"}
    assert validate_block({"kinds": ["x"], "min_change": -1, "slots": 0, "park": "X"}, ["069500"])
    assert validate_block({"kinds": ["kr_block_buy"], "min_change": 0, "slots": 5, "park": "069500", "y": 1},
                          ["069500"])


def test_download_resumes_and_events_count_only_institutions_buying(tmp_path):
    samsung, hynix = corp("005930"), corp("000660")
    reports = {samsung: [report("2015-03-10", "약식", 5.30, 1.10),          # crossed 5%: buy and new
                         report("2016-04-11", "약식", 7.20, 1.20),          # more on top: buy only
                         report("2016-09-12", "약식", 4.10, -3.10),         # sold: nothing
                         report("2017-01-05", "일반", 6.00, 6.00, "행동주의펀드"),  # general form: nothing
                         report("", "약식", 6.0, 1.0) | {"rcept_no": "x"}],  # no date
               hynix: [report("2018-02-09", "약식(단순투자)", 5.00, 5.00, "블랙록")]}
    codes = sorted(STOCKS)
    out = dart_block.download(tmp_path, "dart-key", codes, today="2022-01-10", session=FakeBlockDart(reports, limit=3))
    assert out["fetched"] == 3 and out["left"] == len(STOCKS) - 1 - 3 and "한도" in out["stopped"]
    assert out["unmapped"] == ["123450"]
    out = dart_block.download(tmp_path, "dart-key", codes, today="2022-01-10", session=FakeBlockDart(reports))
    assert out["left"] == 0
    again = FakeBlockDart(reports)
    assert dart_block.download(tmp_path, "dart-key", codes, today="2022-01-10", session=again)["fetched"] == 0
    assert again.calls == []
    assert not list(dart_open.store_dir(tmp_path).glob("*/*.json.gz"))   # apart from the KRDART files
    ev, counts = dart_block.events(tmp_path, codes)
    assert counts["reports"] == 6 and counts["short_form"] == 5 and counts["general_form"] == 1
    assert counts["no_date"] == 1 and counts["up"] == 3 and counts["new"] == 2
    got = {(f"{r.known:%Y-%m-%d}", r.symbol, r.kind, r.value, r.owner) for r in ev.itertuples()}
    assert got == {("2015-03-10", "005930", "kr_block_buy", 1.1, "국민연금공단"),
                   ("2015-03-10", "005930", "kr_block_new", 1.1, "국민연금공단"),
                   ("2016-04-11", "005930", "kr_block_buy", 1.2, "국민연금공단"),
                   ("2018-02-09", "000660", "kr_block_buy", 5.0, "블랙록"),
                   ("2018-02-09", "000660", "kr_block_new", 5.0, "블랙록")}
    assert dart_block.inputs(tmp_path) != dart_block.inputs(tmp_path / "empty")


def synthetic_reports(seed: int = 8) -> dict:
    """Institutions crossing 5% in the members at random sessions 2011-2021, a few on the same day."""
    rng = np.random.default_rng(seed)
    days = krx_open.sessions("2011-01-03", "2021-11-30")
    out: dict[str, list] = {}
    for c in STOCKS:
        if c in ("123450", "005935", "319400"):
            continue
        for d in rng.choice(days, 14, replace=False):
            out.setdefault(corp(c), []).append(report(f"{pd.Timestamp(d):%Y-%m-%d}", "약식", 5.5,
                                                      float(rng.uniform(0.6, 2.0))))
    shared = days[600]
    for c in list(STOCKS)[:12]:
        if corp(c) in out:
            out[corp(c)].append(report(f"{shared:%Y-%m-%d}", "약식", 5.2, 0.5 + 0.01 * len(c)))
    return out


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    snaps = snapshots(end=END)[0]
    text = naver_text("069500", 0)
    mp = pytest.MonkeyPatch()
    import requests

    mp.setattr(requests, "get", lambda url, params=None, headers=None, timeout=None: FakeResponse(text))
    keys = {"krx": "good-key", "dart": "dart-key"}
    mp.setattr(providers, "get_api_key", lambda provider="tiingo": keys.get(provider))
    mp.setattr(kr_dataset, "latest_day", lambda today=None: pd.Timestamp(END))
    mp.setattr(krx_open, "PAUSE_SECONDS", 0.0)
    mp.setattr(dart_open, "PAUSE_SECONDS", 0.0)
    http = FakeKrx(snaps)
    real = krx_open.download
    mp.setattr(krx_open, "download", lambda *a, **k: real(*a, **(k | {"session": http, "today": "2022-06-01"})))
    dart = FakeBlockDart(synthetic_reports())
    real_block = dart_block.download
    mp.setattr(dart_block, "download", lambda *a, **k: real_block(*a, **(k | {"session": dart, "today": "2022-06-01"})))
    lab = Lab(tmp_path_factory.mktemp("block5") / "QL")
    yield lab
    lab.close()
    mp.undo()


def test_buys_after_each_new_report_and_holds_twenty_sessions(lab):
    bt = lab.quick_backtest(SPEC)
    e = lab.registry.experiment(bt.experiment_id)
    assert e["kind"] == "exploratory" and lab.registry.holdout_ledger() == []
    data, manifest = lab.load_dataset(e["dataset_version"])
    assert manifest["qc_fatal"] == [] and "block" in manifest["inputs"] and "dart" not in manifest["inputs"]
    assert manifest["block"]["new"] > 0 and manifest["block"]["short_form"] == manifest["block"]["reports"]
    new = data.events[data.events["kind"] == "kr_block_new"]
    assert len(new) and set(data.events["kind"]) == {"kr_block_buy", "kr_block_new"}
    cal = data.calendar
    tr = bt.result.trades.assign(date=lambda d: pd.to_datetime(d["date"]))
    stock = tr[tr["symbol"] != "069500"]
    buys, sells = stock[stock["qty"] > 0], stock[stock["qty"] < 0]
    assert len(buys) > 20
    known = {(s, cal.get_indexer([k])[0]) for s, k in zip(new["symbol"], new["known"])}
    for r in buys.itertuples():
        i = cal.get_indexer([r.date])[0]
        assert (r.symbol, i - 1) in known                     # the session after the report
        assert data.members.at[cal[i - 1], r.symbol]          # a member that day
        later = sells[(sells["symbol"] == r.symbol) & (sells["date"] > r.date)]
        if len(later) and i + 20 < len(cal):
            out = i + 20
            while (r.symbol, out - 1) in known:                # a new report on the exit day renews it
                out += 20
            assert cal.get_indexer([later["date"].iloc[0]])[0] == out
    held = (bt.result.shares.drop(columns="069500") > 0).sum(axis=1)
    assert held.max() <= 10
    # a day with more reports than free slots takes the larger increases first
    first = tr["date"].min()
    assert tr.loc[tr["symbol"] == "069500", "date"].min() == first   # parked until the first report
    # the same rule without 5% reports would need another dataset
    assert manifest["inputs"]["block"] == dart_block.inputs(lab.ws.root)


def test_dataset_versions_of_other_korean_rules_do_not_change(lab):
    loaded = load_spec(builtin_path("specs", "AUTO-KRSTK.qls.yaml"))
    assert "block" not in loaded.spec["rules"]
    _, policy = lab.policy_for(loaded)
    version = lab.ensure_spec_dataset(loaded, policy)
    _, manifest = lab.load_dataset(version)
    assert "block" not in manifest["inputs"] and manifest["block"] is None
