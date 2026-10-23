"""Korean company filings from the Financial Supervisory Service's OpenDART (D-040).

Source: https://opendart.fss.or.kr (free; the user's own key, about 20,000 calls a day). Per company
(8-digit `corp_code`, mapped from the 6-digit stock code by corpCode.xml):

    GET /api/elestock.json?crtfc_key=..&corp_code=..          임원ㆍ주요주주 소유보고 (every report)
    GET /api/tsstkAqDecsn.json?..&bgn_de=..&end_de=..         자기주식 취득 결정 (from 2015)
    GET /api/tsstkAqTrctrCnsDecsn.json?..&bgn_de=..&end_de=..  자기주식취득 신탁계약 체결 결정 (from 2015)
    -> {"status": "000", "message": "정상", "list": [{...}, ...]}; status 013 = no data, 020 = over the
       daily limit, 010/011 = key refused or locked

The layout was taken from public clients of the API (dart-fss, OpenDartReader); the first download on
the user's PC is the real format check. Each answer is stored gzip-compressed as received, one file
per company and endpoint; a later run asks again only for files older than `refresh_days`.

Events (the `events` table of the KRTOP300 dataset, known on the receipt date):
- `kr_ins_buy` / `kr_ins_sell`: an executive's or major shareholder's report whose holding went up /
  down (sp_stock_lmp_irds_cnt); a first report (no holding before) is not counted. The report does
  not say why the holding changed: open-market trades, but also option exercises, new shares,
  gifts and inheritance count. Value in won = changed shares x that day's close.
- `kr_buyback`: a decision to buy back shares, directly or through a trust contract; value = planned
  amount / market value that day when the filing states the amount (else NaN).
"""

from __future__ import annotations

import gzip
import io
import json
import re
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from quantlab.core.hashing import file_sha256, stable_hash

API = "https://opendart.fss.or.kr/api/"
ENDPOINTS = {"elestock": "elestock.json", "buyback": "tsstkAqDecsn.json", "trust": "tsstkAqTrctrCnsDecsn.json"}
DATED = ("buyback", "trust")       # these take a receipt-date range
FIRST_DAY = "20150101"             # the decisions are served from 2015
PAUSE_SECONDS = 0.05
PARSE_VERSION = 1
EVENT_COLUMNS = ["known", "symbol", "kind", "value", "officer", "director", "owner"]


class DartError(RuntimeError):
    pass


class DailyLimit(DartError):
    pass


def store_dir(root: str | Path) -> Path:
    return Path(root) / "data" / "bulk" / "dart"


def _path(root, endpoint: str, corp: str) -> Path:
    return store_dir(root) / endpoint / f"{corp}.json.gz"


def _check(doc: dict, what: str) -> list[dict]:
    status = str(doc.get("status", ""))
    if status == "000":
        rows = doc.get("list")
        if not isinstance(rows, list):
            raise DartError(f"DART {what}: 응답 형식이 예상과 다릅니다 ({str(doc)[:200]})")
        return rows
    if status == "013":
        return []
    if status == "020":
        raise DailyLimit("DART 하루 호출 한도에 닿았습니다. 다음 실행 때 이어서 받습니다.")
    if status in ("010", "011", "012", "901"):
        raise DartError("OpenDART가 인증키를 거절했습니다. opendart.fss.or.kr 마이페이지에서 키를 확인해 주세요. "
                        f"({status}: {doc.get('message')})")
    raise DartError(f"DART {what}: {status} {doc.get('message')}")


def corp_codes(root: str | Path, key: str, session=None, timeout: float = 60.0, refresh: bool = False) -> dict:
    """{6-digit stock code: 8-digit corp_code} from corpCode.xml (stored; fetched again on `refresh`)."""
    import requests

    p = store_dir(root) / "corpCode.json.gz"
    if p.exists() and not refresh:
        with gzip.open(p, "rb") as g:
            return json.loads(g.read().decode("utf-8"))
    http = session or requests.Session()
    resp = http.get(API + "corpCode.xml", params={"crtfc_key": key}, timeout=timeout)
    if resp.status_code != 200:
        raise DartError(f"DART corpCode: HTTP {resp.status_code}")
    try:
        xml = zipfile.ZipFile(io.BytesIO(resp.content)).read("CORPCODE.xml")
    except zipfile.BadZipFile:  # an error answer: XML or JSON with a status
        try:
            doc = {c.tag: c.text for c in ElementTree.fromstring(resp.content)}
        except ElementTree.ParseError:
            try:
                doc = json.loads(resp.content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                raise DartError(f"DART corpCode: 응답을 읽을 수 없습니다 {resp.content[:200]!r}") from e
        _check(doc if isinstance(doc, dict) else {}, "corpCode")
        raise DartError("DART corpCode: 압축 파일이 아닙니다") from None
    out = {}
    for item in ElementTree.fromstring(xml).iter("list"):
        stock = (item.findtext("stock_code") or "").strip()
        corp = (item.findtext("corp_code") or "").strip()
        if len(stock) == 6 and len(corp) == 8:
            out[stock] = corp
    p.parent.mkdir(parents=True, exist_ok=True)
    _write(p, out)
    return out


def _write(p: Path, doc) -> None:
    tmp = p.with_name(p.name + ".part")
    with gzip.open(tmp, "wb") as g:
        g.write(json.dumps(doc, ensure_ascii=False).encode("utf-8"))
    tmp.replace(p)


def _get(http, endpoint: str, params: dict, timeout: float) -> list[dict]:
    resp = http.get(API + ENDPOINTS[endpoint], params=params, timeout=timeout)
    time.sleep(PAUSE_SECONDS)
    if resp.status_code != 200:
        raise DartError(f"DART {endpoint}: HTTP {resp.status_code}: {resp.content[:200]!r}")
    try:
        doc = json.loads(resp.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise DartError(f"DART {endpoint}: 응답을 읽을 수 없습니다 {resp.content[:200]!r}") from e
    return _check(doc, endpoint)


def _fetch(http, endpoint: str, key: str, corp: str, end: pd.Timestamp, timeout: float) -> list[dict]:
    base = {"crtfc_key": key, "corp_code": corp}
    if endpoint not in DATED:
        return _get(http, endpoint, base, timeout)
    try:
        return _get(http, endpoint, base | {"bgn_de": FIRST_DAY, "end_de": f"{end:%Y%m%d}"}, timeout)
    except DailyLimit:
        raise
    except DartError as e:
        if "100" not in str(e)[:40]:  # 100: a field out of range (a span too long); ask one year at a time
            raise
    rows = []
    for y in range(int(FIRST_DAY[:4]), end.year + 1):
        rows += _get(http, endpoint, base | {"bgn_de": f"{y}0101", "end_de": f"{min(end, pd.Timestamp(f'{y}-12-31')):%Y%m%d}"},
                     timeout)
    return rows


def download(root: str | Path, key: str, codes, today=None, session=None, say=None, timeout: float = 30.0,
             refresh_days: int | None = None) -> dict:
    """Fetch every endpoint for the companies of `codes` (6-digit stock codes) not stored yet, or stored
    more than `refresh_days` days ago. Returns {"fetched", "left", "stopped", "unmapped"}."""
    import requests

    say = say or (lambda s: None)
    http = session or requests.Session()
    today = pd.Timestamp(today or pd.Timestamp.now(tz="Asia/Seoul").date())
    mapping = corp_codes(root, key, session=http)
    corps = sorted({mapping[c] for c in codes if c in mapping})
    unmapped = sorted(c for c in codes if c not in mapping)

    def due(p: Path) -> bool:
        if not p.exists():
            return True
        if refresh_days is None:
            return False
        with gzip.open(p, "rb") as g:
            asked = pd.Timestamp(json.loads(g.read().decode("utf-8"))["asked"])
        return (today - asked).days > refresh_days

    todo = [(e, c) for c in corps for e in ENDPOINTS if due(_path(root, e, c))]
    fetched, stopped = 0, None
    for i, (e, c) in enumerate(todo):
        if i % 50 == 0:
            say(f"DART 공시 받는 중: {i + 1}/{len(todo)}")
        try:
            rows = _fetch(http, e, key, c, today, timeout)
        except DailyLimit as err:
            stopped = str(err)
            break
        p = _path(root, e, c)
        p.parent.mkdir(parents=True, exist_ok=True)
        _write(p, {"asked": f"{today:%Y-%m-%d}", "corp_code": c, "list": rows})
        fetched += 1
    left = sum(1 for e, c in todo if due(_path(root, e, c)))
    return {"fetched": fetched, "left": left, "stopped": stopped, "unmapped": unmapped}


# ----------------------------------------------------------------------------- events
def _num(v) -> float:
    s = re.sub(r"[,\s]", "", str(v if v is not None else ""))
    try:
        return float(s)
    except ValueError:
        return np.nan


def _day(v) -> pd.Timestamp | None:
    d = re.sub(r"\D", "", str(v or ""))[:8]
    try:
        return pd.Timestamp(d) if len(d) == 8 else None
    except ValueError:
        return None


def _read(p: Path) -> list[dict]:
    with gzip.open(p, "rb") as g:
        return json.loads(g.read().decode("utf-8"))["list"]


def records(root: str | Path) -> dict:
    """{file name: sha256} of every stored answer (the dataset's input record)."""
    return {f"{p.parent.name}/{p.name}": file_sha256(p) for p in sorted(store_dir(root).glob("*/*.json.gz"))}


def events(root: str | Path, codes, closes: pd.DataFrame, mktcap: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Events of `codes` (EVENT_COLUMNS) and counts. `closes` / `mktcap`: raw close and market value
    (sessions x codes) to put a value on a report; the last session on or before its receipt date."""
    stored = store_dir(root) / "corpCode.json.gz"
    mapping = corp_codes(root, key="") if stored.exists() else {}
    by_corp = {}
    for code in codes:
        if code in mapping:
            by_corp.setdefault(mapping[code], []).append(code)
    rows, counts = [], {"companies": 0, "elestock_rows": 0, "first_reports": 0, "buybacks": 0, "trusts": 0,
                        "no_date": 0}
    idx = closes.index

    def at(frame: pd.DataFrame, code: str, day: pd.Timestamp) -> float:
        i = idx.searchsorted(day, "right") - 1
        return float(frame[code].iat[i]) if i >= 0 and code in frame else np.nan

    for corp, cs in by_corp.items():
        code = cs[0]
        got = False
        p = _path(root, "elestock", corp)
        if p.exists():
            got = True
            for r in _read(p):
                counts["elestock_rows"] += 1
                day = _day(r.get("rcept_dt")) or _day(str(r.get("rcept_no", ""))[:8])
                change, held = _num(r.get("sp_stock_lmp_irds_cnt")), _num(r.get("sp_stock_lmp_cnt"))
                if day is None:
                    counts["no_date"] += 1
                    continue
                if not np.isfinite(change) or change == 0:
                    continue
                if change > 0 and np.isfinite(held) and change >= held:
                    counts["first_reports"] += 1   # nothing held before: a first report, not a purchase
                    continue
                exec_ = str(r.get("isu_exctv_rgist_at") or "").strip() not in ("", "-")
                major = "주요주주" in str(r.get("isu_main_shrholdr") or "")
                rows.append((day, code, "kr_ins_buy" if change > 0 else "kr_ins_sell",
                             abs(change) * at(closes, code, day), exec_, major or exec_,
                             str(r.get("repror") or "").strip()))
        for e, kind in (("buyback", "kr_buyback"), ("trust", "kr_buyback")):
            p = _path(root, e, corp)
            if not p.exists():
                continue
            got = True
            for r in _read(p):
                day = _day(str(r.get("rcept_no", ""))[:8])
                if day is None:
                    counts["no_date"] += 1
                    continue
                counts["buybacks" if e == "buyback" else "trusts"] += 1
                amount = _num(r.get("aqpln_prc_ostk") if e == "buyback" else r.get("ctr_prc"))
                cap = at(mktcap, code, day)
                rows.append((day, code, kind, amount / cap if np.isfinite(amount) and cap > 0 else np.nan,
                             True, True, f"company:{e}"))
        counts["companies"] += got
    ev = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    ev["known"] = pd.to_datetime(ev["known"])
    ev = ev.sort_values(["known", "symbol", "kind"], kind="stable").reset_index(drop=True)
    counts["mapped"] = len(by_corp)
    return ev, counts


def inputs(root: str | Path) -> str:
    return stable_hash({"parse": PARSE_VERSION, "files": records(root)})
