"""Synthetic KRX Open API snapshots (D-038): every listed stock per session and market, in the layout
the API returns (strings, some with thousands separators)."""

import gzip
import json

import numpy as np
import pandas as pd

from quantlab.data import krx_open

# code -> (name, market, first day, last day or None, shares)
STOCKS = {
    "005930": ("삼성전자", "KOSPI", None, None, 1_000_000),
    "005935": ("삼성전자우", "KOSPI", None, None, 200_000),         # preferred: never a member
    "000660": ("SK하이닉스", "KOSPI", None, None, 700_000),
    "035420": ("NAVER", "KOSPI", None, None, 500_000),
    "051910": ("LG화학", "KOSPI", None, None, 300_000),
    "068270": ("셀트리온", "KOSPI", None, None, 400_000),
    "096770": ("SK이노베이션", "KOSPI", None, None, 300_000),
    "105560": ("KB금융", "KOSPI", "2011-03-02", None, 600_000),     # listed later
    "123450": ("사라진전자", "KOSPI", None, "2016-05-31", 400_000),  # delisted
    "091990": ("셀트리온헬스", "KOSDAQ", None, None, 500_000),
    "247540": ("에코프로비엠", "KOSDAQ", None, None, 300_000),
    "086520": ("에코프로", "KOSDAQ", None, None, 300_000),
    "293490": ("카카오게임즈", "KOSDAQ", None, None, 400_000),
    "263750": ("펄어비스", "KOSDAQ", None, None, 300_000),
    "222800": ("심텍", "KOSDAQ", None, None, 300_000),
    "196170": ("알테오젠", "KOSDAQ", None, None, 300_000),
    "319400": ("하나스팩1호", "KOSDAQ", None, None, 300_000),       # SPAC: never a member
    "112040": ("위메이드", "KOSDAQ", None, None, 300_000),
}
SPLIT = ("051910", "2014-06-02", 5)            # 1 share becomes 5: price / 5, base price / 5
HALT = ("068270", "2013-03-04", "2013-03-08")  # no trades: volume 0, open 0
ADMIN = ("222800", "2015-01-02", "2015-12-30")  # 관리종목 in this span


def _fmt(x: float) -> str:
    return f"{x:,.0f}"


def snapshots(start="2010-01-04", end="2021-12-30", seed=3):
    """{(market, day): rows} and the true daily returns {code: Series}."""
    days = krx_open.sessions(start, end)
    rng = np.random.default_rng(seed)
    out = {(m, d): [] for m in krx_open.MARKETS for d in days}
    true = {}
    for k, (code, (name, mkt, first, last, shares)) in enumerate(STOCKS.items()):
        live = days[(days >= pd.Timestamp(first or start)) & (days <= pd.Timestamp(last or end))]
        r = np.clip(rng.normal(0.0003, 0.02, len(live)), -0.14, 0.14)
        r[0] = 0.0
        if code == "247540":
            r[200] = 0.149 if live[200] < pd.Timestamp("2015-06-15") else 0.30  # an upper-limit close
        px = 10000.0 * (1 + 0.1 * k) * np.cumprod(1 + r)
        px = np.round(px)
        sh = float(shares)
        prev = None
        true[code] = pd.Series(r, index=live)
        for i, d in enumerate(live):
            close = px[i]
            base = prev if prev is not None else close
            if code == SPLIT[0] and d >= pd.Timestamp(SPLIT[1]):
                close = np.round(px[i] / SPLIT[2])
                sh = shares * SPLIT[2]
                if d == pd.Timestamp(SPLIT[1]):
                    base = np.round(prev / SPLIT[2])   # the exchange resets the base price
            halted = code == HALT[0] and pd.Timestamp(HALT[1]) <= d <= pd.Timestamp(HALT[2])
            if halted:
                close = base = prev
            change = close - base
            opn = 0.0 if halted else np.round(base * (1 + rng.normal(0, 0.003)))
            vol = 0 if halted else int(rng.integers(500_000, 2_000_000))
            section = ""
            if mkt == "KOSDAQ":
                section = "관리종목(소속부없음)" if code == ADMIN[0] and pd.Timestamp(ADMIN[1]) <= d <= pd.Timestamp(ADMIN[2]) else "우량기업부"
            out[(mkt, d)].append({
                "BAS_DD": f"{d:%Y%m%d}", "ISU_CD": code, "ISU_NM": name, "MKT_NM": mkt, "SECT_TP_NM": section,
                "TDD_CLSPRC": _fmt(close), "CMPPREVDD_PRC": _fmt(change), "FLUC_RT": f"{(close / base - 1) * 100:.2f}",
                "TDD_OPNPRC": _fmt(opn), "TDD_HGPRC": _fmt(0 if halted else max(opn, close) + 10),
                "TDD_LWPRC": _fmt(0 if halted else min(opn, close) - 10), "ACC_TRDVOL": str(vol),
                "ACC_TRDVAL": str(int(vol * close)), "MKTCAP": str(int(close * sh)), "LIST_SHRS": str(int(sh))})
            prev = close
    return out, true


class Resp:
    def __init__(self, code: int, body: dict | str) -> None:
        self.status_code = code
        self.content = (json.dumps(body, ensure_ascii=False) if isinstance(body, dict) else body).encode("utf-8")


class FakeKrx:
    """Answers like the KRX Open API from `snapshots`; `limit` calls, then the daily limit."""

    def __init__(self, snaps: dict, key: str = "good-key", limit: int | None = None) -> None:
        self.snaps, self.key, self.limit, self.calls = snaps, key, limit, []

    def get(self, url, params=None, headers=None, timeout=None):
        if headers.get("AUTH_KEY") != self.key:
            return Resp(401, {"respCode": "401", "respMsg": "Unauthorized Key"})
        if self.limit is not None and len(self.calls) >= self.limit:
            return Resp(429, {"respMsg": "daily call limit exceeded"})
        mkt = next(m for m, ep in krx_open.MARKETS.items() if url.endswith(ep))
        day = pd.Timestamp(params["basDd"])
        self.calls.append((mkt, day))
        return Resp(200, {"OutBlock_1": self.snaps.get((mkt, day), [])})


def store_snapshots(root, snaps: dict) -> None:
    """Write snapshots straight into the download store (as a finished download would)."""
    for (m, d), rows in snaps.items():
        p = krx_open._path(root, m, d)
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wb") as g:
            g.write(json.dumps({"OutBlock_1": rows}, ensure_ascii=False).encode("utf-8"))
