"""Korean IPOs: the offer price against the first day's open and close (D-042).

A new listing is a common share (code ending in 0, not a SPAC) whose code first appears in the
exchange's daily snapshots (`quantlab.data.krx_open`) on or after `START`; the snapshots from
`WARMUP` on are read, so a stock already trading then is not new. A move between KOSPI and KOSDAQ keeps
its code and is not new.

The offer price comes from OpenDART's 증권신고서(지분증권) 주요정보 (the user's DART key):

    GET /api/estkRs.json?crtfc_key=..&corp_code=..&bgn_de=..&end_de=..   (receipt dates)
    -> {"status": "000", "group": [{"title": "일반사항", "list": [{rcept_no, sbd (청약기일), ...}]},
                                    {"title": "증권의종류", "list": [{rcept_no, stksen, stkcnt, slprc
                                     (모집(매출)가액), slta, slmthn (모집(매출)방법)}]}, ...]}

The layout was taken from a public client of the API (opendart-go's test data); the first download on
the user's PC is the real format check. Per listing, the filings received in the `LOOKBACK_DAYS`
before it are asked for its company (corp_code by stock code, else by an exact name that at most three
companies share). The filing used is the last one received (a correction carries the final price)
whose subscription ended within `SUBSCRIPTION_DAYS` before the listing, leaving out offers to
shareholders or third parties (rights issues of a company already listed elsewhere).

The first real run (2026-10-30) showed estkRs gives the first filing's price, mostly the low end of the
range offered to institutions, not the final one (LG에너지솔루션 257,000 against 300,000), so it is kept
only as `first_offer` and the final price comes from:
- listings from `RULE_CHANGE` (2023-06-26) on: the exchange sets the first day's base price to the
  offer price, so offer = first-day close - change (CMPPREVDD_PRC) in the snapshots;
- otherwise the text of the last filing before listing that carries the final terms, found by
  `list.json` (pblntf_ty C: [발행조건확정]증권신고서(지분증권) first, then 투자설명서, then the last
  correction) and read with `document.xml` (a zip of the filing's XML). Numbers next to the price words
  are scored (words saying "final" count three times) and the best one the exchange's opening range
  allows wins: before `RULE_CHANGE` the first open had to be 90-200% of the offer price, from then on
  60-400%. With no such number the row keeps status `first_only` and stays out of the summary.
For listings from `RULE_CHANGE` on both are known, and how often they agree is reported as a check on
the text reading.

Returns are before costs; `SELL_COST` (commission + tax when selling) is taken off for the net column.
The subscription fee (Kiwoom: 2,000 won per subscription for most customers) is per subscription, not
per share, so it is left to the reader.
"""

from __future__ import annotations

import gzip
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.data import dart_open, krx_open

START = "2015-01-01"
WARMUP = "2014-01-02"
ENDPOINT = "estkRs.json"
LOOKBACK_DAYS = 180
SUBSCRIPTION_DAYS = 60
SELL_COST = 0.0025
MAX_NAME_MATCHES = 3
RULE_CHANGE = "2023-06-26"
BAND_BEFORE, BAND_AFTER = (0.9, 2.0), (0.6, 4.0)
BAND_SLACK = 0.01                          # tick rounding of the opening price
PRICE_WORDS = ("확정공모가", "확정 공모가", "확정발행가", "공모가액", "공모가격", "모집가액", "모집(매출)가액",
               "매출가액", "발행가액", "1주당 공모가")
FINAL_WORDS = ("확정",)
DOCS_PER_LISTING = 2
SAMPLES = ("373220", "323410", "259960")   # LG에너지솔루션, 카카오뱅크, 크래프톤: raw answers in the export
COLUMNS = ["date", "year", "market", "code", "name", "match", "corp_code", "rcept_no", "method", "first_offer",
           "krx_base", "dart_final", "final_rcept_no", "offer_source", "offer", "open", "close", "ret_open",
           "ret_close", "status"]


def store_dir(root: str | Path) -> Path:
    # beside, not inside, the DART store: its files are the KRDART dataset's inputs
    return Path(root) / "data" / "bulk" / "dart-ipo"


def _path(root, code: str) -> Path:
    return store_dir(root) / "estkRs" / f"{code}.json.gz"


def _final_path(root, code: str) -> Path:
    return store_dir(root) / "final" / f"{code}.json.gz"


def _read(p: Path):
    with gzip.open(p, "rb") as g:
        return json.loads(g.read().decode("utf-8"))


# ----------------------------------------------------------------------------- listings
LISTING_COLUMNS = ["date", "market", "code", "name", "open", "close", "change"]


def new_listings(root: str | Path, end, start: str = START, say=None) -> tuple[pd.DataFrame, dict]:
    """First sessions of new common shares from `start` to `end` (LISTING_COLUMNS)
    and counts of what was left out."""
    seen: set[str] = set()
    rows = []
    for y in range(pd.Timestamp(WARMUP).year, pd.Timestamp(end).year + 1):
        snaps, _ = krx_open.load(root, max(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(WARMUP)),
                                 min(pd.Timestamp(f"{y}-12-31"), pd.Timestamp(end)), say=say)
        if not len(snaps):
            continue
        snaps = snaps.sort_values(["date", "market", "code"], kind="stable")
        first = snaps[~snaps["code"].isin(seen)].drop_duplicates("code", keep="first")
        seen.update(first["code"])
        rows.append(first[first["date"] >= pd.Timestamp(start)][LISTING_COLUMNS])
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=LISTING_COLUMNS)
    spac = df["name"].str.contains("스팩", regex=False)
    other = ~spac & (df["code"].str[-1] != "0")
    counts = {"spac": int(spac.sum()), "not_common": int(other.sum())}
    return df[~spac & ~other].reset_index(drop=True), counts


# ----------------------------------------------------------------------------- companies
def _norm(name: str) -> str:
    return re.sub(r"\s|\(주\)|㈜|주식회사", "", str(name)).upper()


def companies(root: str | Path, key: str, session=None, today=None) -> dict:
    """{"stock": {stock code: corp}, "name": {normalised name: [corp, ...]}} from corpCode.xml, fetched
    again once a day (new listings get their stock code there)."""
    today = f"{pd.Timestamp(today or pd.Timestamp.now(tz='Asia/Seoul').date()):%Y-%m-%d}"
    p = store_dir(root) / "corps.json.gz"
    if p.exists():
        doc = _read(p)
        if doc.get("asked") == today:
            return doc
    stock, name = {}, {}
    for corp, n, s in dart_open.corp_list(key, session):
        if s:
            stock[s] = corp
        name.setdefault(_norm(n), []).append(corp)
    doc = {"asked": today, "stock": stock, "name": name}
    p.parent.mkdir(parents=True, exist_ok=True)
    dart_open._write(p, doc)
    return doc


def candidates(corps: dict, code: str, name: str) -> tuple[str, list[str]]:
    if code in corps["stock"]:
        return "code", [corps["stock"][code]]
    found = corps["name"].get(_norm(name), [])
    return ("name", sorted(found)) if 0 < len(found) <= MAX_NAME_MATCHES else ("none", [])


# ----------------------------------------------------------------------------- filings
def _groups(http, params: dict, timeout: float) -> list[dict]:
    resp = http.get(dart_open.API + ENDPOINT, params=params, timeout=timeout)
    time.sleep(dart_open.PAUSE_SECONDS)
    if resp.status_code != 200:
        raise dart_open.DartError(f"DART estkRs: HTTP {resp.status_code}: {resp.content[:200]!r}")
    try:
        doc = json.loads(resp.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise dart_open.DartError(f"DART estkRs: 응답을 읽을 수 없습니다 {resp.content[:200]!r}") from e
    if str(doc.get("status", "")) == "000":
        groups = doc.get("group")
        if not isinstance(groups, list):
            raise dart_open.DartError(f"DART estkRs: 응답 형식이 예상과 다릅니다 ({str(doc)[:200]})")
        return groups
    dart_open._check(doc, "estkRs")  # 013 (nothing filed) returns; other statuses raise
    return []


def download(root: str | Path, key: str, listings: pd.DataFrame, corps: dict, session=None, say=None,
             today=None, timeout: float = 30.0) -> dict:
    """Ask the filings of every listing with a company not stored yet. {"fetched", "left", "stopped"}."""
    import requests

    say = say or (lambda s: None)
    http = session or requests.Session()
    today = pd.Timestamp(today or pd.Timestamp.now(tz="Asia/Seoul").date())
    todo = [r for r in listings.itertuples() if candidates(corps, r.code, r.name)[1] and not _path(root, r.code).exists()]
    fetched, stopped = 0, None
    for i, r in enumerate(todo):
        if i % 50 == 0:
            say(f"공모주 증권신고서 받는 중: {i + 1}/{len(todo)}")
        listed = pd.Timestamp(r.date)
        answers = {}
        try:
            for corp in candidates(corps, r.code, r.name)[1]:
                answers[corp] = _groups(http, {"crtfc_key": key, "corp_code": corp,
                                               "bgn_de": f"{listed - pd.Timedelta(days=LOOKBACK_DAYS):%Y%m%d}",
                                               "end_de": f"{listed:%Y%m%d}"}, timeout)
        except dart_open.DailyLimit as err:
            stopped = str(err)
            break
        p = _path(root, r.code)
        p.parent.mkdir(parents=True, exist_ok=True)
        dart_open._write(p, {"asked": f"{today:%Y-%m-%d}", "listed": f"{listed:%Y-%m-%d}", "answers": answers})
        fetched += 1
    left = sum(1 for r in todo if not _path(root, r.code).exists())
    return {"fetched": fetched, "left": left, "stopped": stopped}


def _group(groups: list[dict], word: str) -> list[dict]:
    for g in groups:
        if word in re.sub(r"\s", "", str(g.get("title", ""))) and isinstance(g.get("list"), list):
            return g["list"]
    return []


def _last_day(text) -> pd.Timestamp | None:
    found = re.findall(r"(\d{4})\D{1,4}(\d{1,2})\D{1,4}(\d{1,2})", str(text or ""))
    if not found:
        return None
    try:
        return pd.Timestamp(int(found[-1][0]), int(found[-1][1]), int(found[-1][2]))
    except ValueError:
        return None


def offers(groups: list[dict]) -> list[dict]:
    """One row per filing: rcept_no, received, subscription end, offer price of the common share, method."""
    general = {str(r.get("rcept_no")): r for r in _group(groups, "일반사항")}
    kinds: dict[str, list[dict]] = {}
    for r in _group(groups, "증권의종류"):
        kinds.setdefault(str(r.get("rcept_no")), []).append(r)
    out = []
    for no, rows in kinds.items():
        common = [r for r in rows if "보통" in str(r.get("stksen", ""))] or rows
        r = common[0]
        out.append({"rcept_no": no, "received": dart_open._day(no[:8]),
                    "sub_end": _last_day((general.get(no) or {}).get("sbd")),
                    "offer": dart_open._num(r.get("slprc")), "method": str(r.get("slmthn") or "").strip()})
    return out


def choose(rows: list[dict], listed: pd.Timestamp) -> dict | None:
    def fits(o: dict) -> bool:
        if "주주" in o["method"] or "제3자" in o["method"]:
            return False
        if not (np.isfinite(o["offer"]) and o["offer"] > 0):
            return False
        if o["sub_end"] is not None:
            return listed - pd.Timedelta(days=SUBSCRIPTION_DAYS) <= o["sub_end"] <= listed
        return o["received"] is not None and listed - pd.Timedelta(days=LOOKBACK_DAYS) <= o["received"] <= listed

    ok = [o for o in rows if fits(o)]
    return max(ok, key=lambda o: o["rcept_no"]) if ok else None


def first_offer(root: str | Path, corps: dict, code: str, name: str, listed: pd.Timestamp) -> tuple[str, dict] | None:
    """(corp, chosen estkRs filing) of a listing, if stored and one fits."""
    cands = candidates(corps, code, name)[1]
    p = _path(root, code)
    if not cands or not p.exists():
        return None
    best = None
    for corp, groups in _read(p)["answers"].items():
        got = choose(offers(groups), listed)
        if got and (best is None or got["rcept_no"] > best[1]["rcept_no"]):
            best = (corp, got)
    return best


# ----------------------------------------------------------------------------- final terms
def _api(http, name: str, params: dict, timeout: float) -> list[dict]:
    resp = http.get(dart_open.API + name, params=params, timeout=timeout)
    time.sleep(dart_open.PAUSE_SECONDS)
    if resp.status_code != 200:
        raise dart_open.DartError(f"DART {name}: HTTP {resp.status_code}: {resp.content[:200]!r}")
    try:
        doc = json.loads(resp.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise dart_open.DartError(f"DART {name}: 응답을 읽을 수 없습니다 {resp.content[:200]!r}") from e
    return dart_open._check(doc, name)


def _text(blob: bytes) -> str:
    for enc in ("utf-8", "cp949"):
        try:
            raw = blob.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raw = blob.decode("utf-8", errors="replace")
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = raw.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", raw).strip()


def snippets(text: str, width: int = 160, cap: int = 300) -> list[str]:
    """The text right after each price word (what the score reads, kept instead of the whole filing)."""
    hits = sorted({m.start() for w in PRICE_WORDS for m in re.finditer(re.escape(w), text)})
    return [text[i:i + width] for i in hits[:cap]]


def _document(http, key: str, rcept_no: str, timeout: float) -> list[str]:
    import io
    import zipfile

    resp = http.get(dart_open.API + "document.xml", params={"crtfc_key": key, "rcept_no": rcept_no}, timeout=timeout)
    time.sleep(dart_open.PAUSE_SECONDS)
    if resp.status_code != 200:
        raise dart_open.DartError(f"DART document: HTTP {resp.status_code}")
    try:
        z = zipfile.ZipFile(io.BytesIO(resp.content))
    except zipfile.BadZipFile:  # an error answer with a status
        from xml.etree import ElementTree

        try:
            doc = {c.tag: c.text for c in ElementTree.fromstring(resp.content)}
        except ElementTree.ParseError:
            try:
                doc = json.loads(resp.content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                raise dart_open.DartError(f"DART document: 응답을 읽을 수 없습니다 {resp.content[:200]!r}") from e
        dart_open._check(doc if isinstance(doc, dict) else {}, "document")
        return []
    out = []
    for n in z.namelist():
        out += snippets(_text(z.read(n)))
    return out


def _rank(row: dict) -> tuple:
    nm = str(row.get("report_nm", ""))
    return ("발행조건확정" in nm, "투자설명서" in nm, str(row.get("rcept_no", "")))


def final_filings(rows: list[dict], listed: pd.Timestamp) -> list[dict]:
    """Equity offering filings received by the listing date, the likeliest to carry final terms first."""
    ok = [r for r in rows if ("지분증권" in str(r.get("report_nm", "")) or "투자설명서" in str(r.get("report_nm", "")))
          and (dart_open._day(str(r.get("rcept_no", ""))[:8]) or listed) <= listed]
    return sorted(ok, key=_rank, reverse=True)


def download_final(root: str | Path, key: str, listings: pd.DataFrame, corps: dict, session=None, say=None,
                   today=None, timeout: float = 60.0) -> dict:
    """The final-terms filings' price passages of every listing with a first filing, not stored yet."""
    import requests

    say = say or (lambda s: None)
    http = session or requests.Session()
    today = pd.Timestamp(today or pd.Timestamp.now(tz="Asia/Seoul").date())
    todo = []
    for r in listings.itertuples():
        listed = pd.Timestamp(r.date)
        got = first_offer(root, corps, r.code, r.name, listed)
        if got and not _final_path(root, r.code).exists():
            todo.append((r.code, got[0], listed, float(r.open)))
    fetched, stopped = 0, None
    for i, (code, corp, listed, opn) in enumerate(todo):
        if i % 50 == 0:
            say(f"공모주 확정 공모가 받는 중: {i + 1}/{len(todo)}")
        try:
            rows = _api(http, "list.json", {"crtfc_key": key, "corp_code": corp, "pblntf_ty": "C", "page_count": 100,
                                            "bgn_de": f"{listed - pd.Timedelta(days=LOOKBACK_DAYS):%Y%m%d}",
                                            "end_de": f"{listed:%Y%m%d}"}, timeout)
            docs = {}
            for f in final_filings(rows, listed)[:DOCS_PER_LISTING]:   # a filing is large: the next only if needed
                no = str(f["rcept_no"])
                docs[no] = {"report_nm": f.get("report_nm"), "snippets": _document(http, key, no, timeout)}
                if any(allowed(v, opn, listed) for v in scores(docs[no]["snippets"])):
                    break
        except dart_open.DailyLimit as err:
            stopped = str(err)
            break
        p = _final_path(root, code)
        p.parent.mkdir(parents=True, exist_ok=True)
        dart_open._write(p, {"asked": f"{today:%Y-%m-%d}", "listed": f"{listed:%Y-%m-%d}", "corp_code": corp,
                             "filings": rows, "docs": docs})
        fetched += 1
    left = sum(1 for code, *_ in todo if not _final_path(root, code).exists())
    return {"fetched": fetched, "left": left, "stopped": stopped}


_NUMBER = re.compile(r"(\d{1,3}(?:,\d{3})+)(?![\d,])|(?<![\d,.])(\d{3,7})\s*원")


def scores(passages: list[str]) -> dict[float, float]:
    """{price: score}: the first two numbers after each price word, three times the weight after a word
    saying the price is final."""
    out: dict[float, float] = {}
    for text in passages:
        w = 3.0 if any(f in text[:12] for f in FINAL_WORDS) else 1.0
        found = [float((a or b).replace(",", "")) for a, b in _NUMBER.findall(text)]
        for k, v in enumerate(v for v in found if 500 <= v <= 5_000_000):
            if k >= 2:
                break
            out[v] = out.get(v, 0.0) + (w if k == 0 else w / 3)
    return out


def band(listed: pd.Timestamp) -> tuple[float, float]:
    return BAND_AFTER if listed >= pd.Timestamp(RULE_CHANGE) else BAND_BEFORE


def allowed(offer: float, opn: float, listed: pd.Timestamp) -> bool:
    lo, hi = band(listed)
    return offer > 0 and lo * offer * (1 - BAND_SLACK) <= opn <= hi * offer * (1 + BAND_SLACK)


def dart_final(root: str | Path, code: str, opn: float, listed: pd.Timestamp) -> tuple[float, str]:
    """(final offer price, filing) read from the stored passages, or (nan, "")."""
    p = _final_path(root, code)
    if not p.exists():
        return np.nan, ""
    for no, d in sorted(_read(p)["docs"].items(), key=lambda kv: _rank({"report_nm": kv[1].get("report_nm"),
                                                                       "rcept_no": kv[0]}), reverse=True):
        ok = {v: sc for v, sc in scores(d["snippets"]).items() if allowed(v, opn, listed)}
        if ok:
            return max(ok, key=lambda v: (ok[v], v)), no
    return np.nan, ""


# ----------------------------------------------------------------------------- results
def table(root: str | Path, listings: pd.DataFrame, corps: dict) -> pd.DataFrame:
    out = []
    for r in listings.itertuples():
        listed = pd.Timestamp(r.date)
        match, cands = candidates(corps, r.code, r.name)
        opn, close = float(r.open), float(r.close)
        base = close - float(r.change) if np.isfinite(float(r.change)) else np.nan
        row = {"date": listed, "year": listed.year, "market": r.market, "code": r.code, "name": r.name,
               "match": match, "corp_code": "", "rcept_no": "", "method": "", "first_offer": np.nan,
               "krx_base": base, "dart_final": np.nan, "final_rcept_no": "", "offer_source": "", "offer": np.nan,
               "open": opn, "close": close, "ret_open": np.nan, "ret_close": np.nan, "status": "no_company"}
        if cands and not _path(root, r.code).exists():
            row["status"] = "not_asked"
        elif cands:
            row["status"] = "no_filing"
            got = first_offer(root, corps, r.code, r.name, listed)
            if got:
                corp, o = got
                row.update(corp_code=corp, rcept_no=o["rcept_no"], method=o["method"], first_offer=o["offer"],
                           status="first_only")
                row["dart_final"], row["final_rcept_no"] = dart_final(root, r.code, opn, listed)
                if listed >= pd.Timestamp(RULE_CHANGE) and allowed(base, opn, listed):
                    row.update(offer=base, offer_source="krx")
                elif np.isfinite(row["dart_final"]):
                    row.update(offer=row["dart_final"], offer_source="dart")
                if row["offer_source"]:
                    row.update(status="ok", ret_open=opn / row["offer"] - 1, ret_close=close / row["offer"] - 1)
        out.append(row)
    return pd.DataFrame(out, columns=COLUMNS)


def agreement(t: pd.DataFrame) -> tuple[int, int]:
    """(agreeing, both known): KRX base price and the text-read DART price, listings from RULE_CHANGE on."""
    both = t[(t["offer_source"] == "krx") & t["dart_final"].notna()]
    return int(((both["dart_final"] - both["krx_base"]).abs() < 0.5).sum()), len(both)


def _stats(g: pd.DataFrame) -> dict:
    o, c = g["ret_open"], g["ret_close"]
    return {"n": len(g), "open_mean": o.mean(), "open_median": o.median(), "open_below": (o < 0).mean(),
            "close_mean": c.mean(), "close_median": c.median(), "close_below": (c < 0).mean(),
            "close_worst": c.min(), "close_best": c.max(), "open_net_mean": (o - SELL_COST).mean(),
            "close_net_mean": (c - SELL_COST).mean(),
            "won_per_share_open": (g["offer"] * o).mean(), "won_per_share_close": (g["offer"] * c).mean()}


def summary(t: pd.DataFrame) -> pd.DataFrame:
    """Per year, per market and overall, over the rows whose offer price was found and checked."""
    ok = t[t["status"] == "ok"]
    rows = [{"group": str(y), **_stats(g)} for y, g in ok.groupby("year")]
    rows += [{"group": m, **_stats(g)} for m, g in ok.groupby("market")]
    if len(ok):
        rows.append({"group": "전체", **_stats(ok)})
    return pd.DataFrame(rows)


def _pct(v: float) -> str:
    return "-" if not np.isfinite(v) else f"{v * 100:+.0f}%"


def report_text(t: pd.DataFrame, s: pd.DataFrame, left_out: dict) -> str:
    counts = t["status"].value_counts().to_dict()
    src = t["offer_source"].value_counts().to_dict()
    same, n = agreement(t)
    lines = [(f"공모주 첫날 조사: 새로 상장한 보통주 {len(t)}곳 (스팩 {left_out['spac']}곳, 우선주 등 "
              f"{left_out['not_common']}곳 제외)"),
             (f"확정 공모가 찾음 {counts.get('ok', 0)}곳 (거래소 기준가 {src.get('krx', 0)}, DART 신고서 {src.get('dart', 0)}) · "
              f"첫 신고서 가격만 있음 {counts.get('first_only', 0)} · 신고서 없음 {counts.get('no_filing', 0)} · "
              f"회사 못 찾음 {counts.get('no_company', 0)}"),
             f"점검: 2023-06-26 이후 상장 {n}곳에서 거래소 기준가와 DART 신고서에서 읽은 가격이 {same}곳 일치",
             "", "연도: 건수 | 첫날 시가에 팔기 평균/중앙값/공모가 아래 | 첫날 종가에 팔기 평균/중앙값/공모가 아래/최악"]
    for r in s.itertuples():
        lines.append(f"{r.group}: {r.n}건 | 시가 {_pct(r.open_mean)} / {_pct(r.open_median)} / {r.open_below * 100:.0f}% | "
                     f"종가 {_pct(r.close_mean)} / {_pct(r.close_median)} / {r.close_below * 100:.0f}% / {_pct(r.close_worst)}")
    lines += ["", f"수익률은 공모가 대비, 매도 비용({SELL_COST * 100:.2f}%)과 청약 수수료(건당) 전입니다."]
    return "\n".join(lines)


def samples(root: str | Path) -> dict:
    """The stored DART answers of a few well-known listings, to check the layouts against."""
    out = {}
    for code in SAMPLES:
        out[code] = {k: _read(p) for k, p in (("estkRs", _path(root, code)), ("final", _final_path(root, code)))
                     if p.exists()}
    return out


def write(folder: Path, t: pd.DataFrame, s: pd.DataFrame, text: str, raw: dict | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        (folder / "dart_samples.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    t.assign(date=t["date"].dt.strftime("%Y-%m-%d")).to_csv(folder / "ipo_firstday.csv", index=False,
                                                            encoding="utf-8-sig")
    s.to_csv(folder / "ipo_summary.csv", index=False, encoding="utf-8-sig")
    (folder / "ipo_summary.txt").write_text(text, encoding="utf-8")
