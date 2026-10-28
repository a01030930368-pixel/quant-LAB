"""D-042: Korean IPOs, the offer price from OpenDART's equity registration statements against the first
day's open and close in the exchange's snapshots, on synthetic snapshots and filings."""

import io
import zipfile

import pandas as pd
import pytest

from quantlab.data import dart_open, ipo, krx_open, providers
from quantlab.services.lab import Lab
from kr_fixtures import FakeKrx, Resp, store_snapshots

END = "2016-12-29"
TODAY = "2016-12-30"   # the exchange publishes the previous session: 2016-12-29 is the last one asked
# code -> name, market, first day, first open, first close
STOCKS = {
    "005930": ("삼성전자", "KOSPI", "2014-01-02", 10000, 10000),
    "123450": ("오래된회사", "KOSPI", "2014-03-03", 5000, 5100),        # listed in the warm-up year: not new
    "247540": ("이전회사", "KOSDAQ", "2014-01-02", 7000, 7000),         # moves to KOSPI in 2015: not new
    "111110": ("신규전자", "KOSDAQ", "2015-03-02", 18000, 15000),        # offer 10,000 after a correction
    "333330": ("대신밸런스스팩5호", "KOSDAQ", "2015-05-04", 2000, 2010),  # SPAC
    "555550": ("이름만회사", "KOSDAQ", "2015-09-01", 9000, 8000),        # no stock code at DART: by name
    "888880": ("유상증자", "KOSDAQ", "2015-11-02", 4000, 4100),          # only a rights issue filed
    "666660": ("분할신설", "KOSPI", "2016-03-02", 30000, 31000),         # a spin-off: nothing filed
    "777770": ("엉뚱제약", "KOSDAQ", "2016-05-02", 30000, 29000),        # a filing far from the price
    "222220": ("하락바이오", "KOSPI", "2016-07-01", 19000, 17000),       # offer 20,000: below on day one
    "222225": ("하락바이오우", "KOSPI", "2016-07-01", 15000, 15000),     # preferred
    "999990": ("이름없음", "KOSDAQ", "2016-09-01", 3000, 3000),          # no company at DART
}
MOVE = ("247540", "2015-07-01")
CORPS = {"005930": "00126380", "123450": "00100001", "247540": "00100002", "111110": "00111110",
         "333330": "00333330", "888880": "00888880", "666660": "00666660", "777770": "00777770",
         "222220": "00222220"}
NAMED = {"00555550": "이름만회사(주)"}       # listed company without its stock code in corpCode.xml
# corp -> filings: (rcept_no, 청약기일, [(stksen, slprc)], method)
FILINGS = {
    "00111110": [("20150120000001", "2015년 02월 16일 ~ 2015년 02월 17일", [("종류주", "99,999"), ("보통주", "9,000")], "일반공모"),
                 ("20150210000002", "2015년 02월 16일 ~ 2015년 02월 17일", [("종류주", "99,999"), ("보통주", "10,000")], "일반공모"),
                 ("20160801000003", "2016년 08월 20일 ~ 2016년 08월 21일", [("보통주", "50,000")], "주주배정후 실권주 일반공모")],
    "00555550": [("20150720000001", "2015년 08월 20일 ~ 2015년 08월 21일", [("보통주", "5,000")], "일반공모")],
    "00888880": [("20150920000001", "2015년 10월 10일 ~ 2015년 10월 11일", [("보통주", "3,000")], "주주배정후 실권주 일반공모")],
    "00777770": [("20160401000001", "2016년 04월 20일 ~ 2016년 04월 21일", [("보통주", "1,000")], "일반공모")],
    "00222220": [("20160601000001", "2016년 06월 20일 ~ 2016년 06월 21일", [("보통주", "20,000")], "일반공모")],
}


def _fmt(x: float) -> str:
    return f"{x:,.0f}"


def snapshots() -> dict:
    days = krx_open.sessions("2014-01-02", END)
    out = {(m, d): [] for m in krx_open.MARKETS for d in days}
    for code, (name, mkt, first, opn, close) in STOCKS.items():
        for i, d in enumerate(days[days >= pd.Timestamp(first)]):
            m = mkt
            if code == MOVE[0] and d >= pd.Timestamp(MOVE[1]):
                m = "KOSPI"
            o, c = (opn, close) if i == 0 else (close, close)
            out[(m, d)].append({
                "BAS_DD": f"{d:%Y%m%d}", "ISU_CD": code, "ISU_NM": name, "MKT_NM": m, "SECT_TP_NM": "",
                "TDD_CLSPRC": _fmt(c), "CMPPREVDD_PRC": "0", "FLUC_RT": "0.00", "TDD_OPNPRC": _fmt(o),
                "TDD_HGPRC": _fmt(max(o, c)), "TDD_LWPRC": _fmt(min(o, c)), "ACC_TRDVOL": "1000",
                "ACC_TRDVAL": str(1000 * c), "MKTCAP": str(1_000_000 * c), "LIST_SHRS": "1000000"})
    return out


def estk_doc(corp: str, bgn: str, end: str) -> dict:
    rows = [f for f in FILINGS.get(corp, []) if bgn <= f[0][:8] <= end]
    if not rows:
        return {"status": "013", "message": "조회된 데이타가 없습니다."}
    general = [{"rcept_no": no, "corp_cls": "K", "corp_code": corp, "corp_name": "x", "sbd": sbd, "pymd": "-"}
               for no, sbd, _, _ in rows]
    kinds = [{"rcept_no": no, "corp_code": corp, "stksen": k, "stkcnt": "1,000,000", "fv": "500", "slprc": p,
              "slta": "-", "slmthn": how} for no, _, ks, how in rows for k, p in ks]
    return {"status": "000", "message": "정상",
            "group": [{"title": "일반사항", "list": general}, {"title": "증권의종류", "list": kinds},
                      {"title": "인수인정보", "list": []}]}


class FakeIpoDart:
    def __init__(self, key: str = "dart-key", limit: int | None = None) -> None:
        self.key, self.limit, self.calls = key, limit, []

    def get(self, url, params=None, headers=None, timeout=None):
        if params.get("crtfc_key") != self.key:
            return Resp(200, {"status": "010", "message": "등록되지 않은 키입니다."})
        if url.endswith("corpCode.xml"):
            self.calls.append(("corpCode", None))
            names = {c: STOCKS[s][0] for s, c in CORPS.items()} | NAMED
            stock = {c: s for s, c in CORPS.items()}
            items = "".join(f"<list><corp_code>{c}</corp_code><corp_name>{n}</corp_name>"
                            f"<stock_code>{stock.get(c, ' ')}</stock_code></list>" for c, n in names.items())
            items += "<list><corp_code>99999999</corp_code><corp_name>비상장</corp_name><stock_code> </stock_code></list>"
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("CORPCODE.xml", f'<?xml version="1.0" encoding="UTF-8"?><result>{items}</result>')
            r = Resp(200, {})
            r.content = buf.getvalue()
            return r
        assert url == dart_open.API + ipo.ENDPOINT
        if self.limit is not None and len(self.calls) >= self.limit:
            return Resp(200, {"status": "020", "message": "사용한도를 초과하였습니다."})
        self.calls.append(("estkRs", params["corp_code"]))
        return Resp(200, estk_doc(params["corp_code"], params["bgn_de"], params["end_de"]))


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(krx_open, "PAUSE_SECONDS", 0.0)
    monkeypatch.setattr(dart_open, "PAUSE_SECONDS", 0.0)


def test_new_listings_are_first_appearances_of_common_shares(tmp_path):
    store_snapshots(tmp_path, snapshots())
    got, left_out = ipo.new_listings(tmp_path, END)
    assert list(got["code"]) == ["111110", "555550", "888880", "666660", "777770", "222220", "999990"]
    assert left_out == {"spac": 1, "not_common": 1}
    first = got.set_index("code").loc["111110"]
    assert (first["date"], first["open"], first["close"]) == (pd.Timestamp("2015-03-02"), 18000, 15000)


def test_the_offer_is_the_last_filing_whose_subscription_ended_before_listing():
    rows = ipo.offers(estk_doc("00111110", "20140101", "20161231")["group"])
    assert {r["rcept_no"]: r["offer"] for r in rows} == {"20150120000001": 9000, "20150210000002": 10000,
                                                        "20160801000003": 50000}
    assert rows[0]["sub_end"] == pd.Timestamp("2015-02-17")
    best = ipo.choose(rows, pd.Timestamp("2015-03-02"))
    assert (best["rcept_no"], best["offer"]) == ("20150210000002", 10000)
    assert ipo.choose(rows, pd.Timestamp("2016-09-01")) is None   # a rights issue is not an IPO
    assert ipo.choose(rows, pd.Timestamp("2015-06-01")) is None   # subscription too long before


def test_study_end_to_end_resumes_after_the_daily_limit(tmp_path, monkeypatch):
    keys = {"krx": "good-key", "dart": "dart-key"}
    monkeypatch.setattr(providers, "get_api_key", lambda provider="tiingo": keys.get(provider))
    krx = FakeKrx(snapshots())
    lab = Lab(tmp_path / "QL")
    try:
        dart = FakeIpoDart(limit=4)   # corpCode + 3 filings, then the daily limit
        with pytest.raises(RuntimeError, match="더 받아야"):
            lab.ipo_study(today=TODAY, krx_session=krx, dart_session=dart)
        dart = FakeIpoDart()
        folder, text = lab.ipo_study(today=TODAY, krx_session=krx, dart_session=dart)
        assert ("corpCode", None) not in dart.calls        # fetched once a day
        asked = [c for _, c in dart.calls]
        assert len(asked) == 6 - 3
        t = pd.read_csv(folder / "ipo_firstday.csv", dtype={"code": str, "corp_code": str, "rcept_no": str})
        status = dict(zip(t["code"], t["status"]))
        assert status == {"111110": "ok", "555550": "ok", "888880": "no_filing", "666660": "no_filing",
                          "777770": "price_check", "222220": "ok", "999990": "no_company"}
        row = t.set_index("code").loc["111110"]
        assert (row["offer"], row["rcept_no"], row["match"]) == (10000, "20150210000002", "code")
        assert row["ret_open"] == pytest.approx(0.8) and row["ret_close"] == pytest.approx(0.5)
        assert t.set_index("code").loc["555550", "match"] == "name"
        s = pd.read_csv(folder / "ipo_summary.csv").set_index("group")
        assert list(s.index) == ["2015", "2016", "KOSDAQ", "KOSPI", "전체"]
        assert s.loc["2015", "n"] == 2 and s.loc["2015", "open_mean"] == pytest.approx(0.8)
        assert s.loc["2015", "close_mean"] == pytest.approx(0.55) and s.loc["2015", "close_below"] == 0
        assert s.loc["2016", "close_mean"] == pytest.approx(-0.15) and s.loc["2016", "close_below"] == 1
        assert s.loc["전체", "close_worst"] == pytest.approx(-0.15)
        assert s.loc["전체", "close_net_mean"] == pytest.approx((0.5 + 0.6 - 0.15) / 3 - ipo.SELL_COST)
        assert s.loc["2016", "won_per_share_close"] == pytest.approx(-3000)
        assert "새로 상장한 보통주 7곳" in text and "스팩 1곳" in text and "가격 확인 필요 1" in text
        assert (folder / "ipo_summary.txt").read_text(encoding="utf-8") == text
        # a later run asks nothing more of DART
        dart = FakeIpoDart()
        lab.ipo_study(today=TODAY, krx_session=krx, dart_session=dart)
        assert dart.calls == []
    finally:
        lab.close()


def test_study_needs_both_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(providers, "get_api_key", lambda provider="tiingo": "k" if provider == "krx" else None)
    lab = Lab(tmp_path / "QL")
    try:
        with pytest.raises(RuntimeError, match="OpenDART"):
            lab.ipo_study(today=TODAY)
    finally:
        lab.close()
