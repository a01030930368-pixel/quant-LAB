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

What was known before subscribing (D-045): the final-terms filing also reports the institutions' demand
(수요예측 경쟁률, "1,522.85:1") and the price range offered to them (희망공모가액 "9,000원 ~ 11,000원").
For each priced listing the passages after `DEMAND_WORDS` are kept from that filing (the next one only if no
ratio is found, at most `DEMAND_DOCS`), stored in data/bulk/dart-ipo/demand/. The ratio is the one quoted
most often after the word 경쟁률 (the total is in both the summary sentence and the table; the largest wins
a tie); the range is the most frequent "a원 ~ b원" pair after a range word with b/a at most `MAX_RANGE`.
First-day returns are then grouped by fixed ratio buckets (`RATIO_EDGES`) and by where the final price
fell in the range, for the whole period and from `RULE_CHANGE` on. The range's low end should equal
estkRs's first price (`first_offer`), and how often it does is reported as a check on the reading.

The first run (2026-11-03) read the range for 776 of 784 listings but the ratio for only 204: the filings
print it as a table row, "경쟁률 700.09 322.17 .. 1,732.83" (per investor type, then the total), so the
total at the end of that row is taken first (D-046). The same filing's lock-up table gives the share of
the shares bid for with a lock-up (1 - 미확약 total / 합계 total). Both are read again from the stored
passages, with no new DART calls. The rule the run pointed to (final price at or above the range top,
`TOP`) is shown against the rest.
"""

from __future__ import annotations

import gzip
import json
import re
import time
from itertools import pairwise
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
DOCS_PER_LISTING = 3
MISSING = ("013", "014")              # document.xml: no data / 파일이 존재하지 않습니다
SAMPLES = ("373220", "323410", "259960")   # LG에너지솔루션, 카카오뱅크, 크래프톤: raw answers in the export
DEMAND_WORDS = ("경쟁률", "희망공모가", "희망 공모가", "공모희망가", "공모가 밴드", "공모가밴드", "의무보유", "확약")
RANGE_WORDS = ("희망공모가", "희망 공모가", "공모희망가", "공모가 밴드", "공모가밴드")
DEMAND_WIDTH = 400                    # a ratio sits in a table, often far after its header
DEMAND_DOCS = 2
MAX_RANGE = 1.6
RATIO_EDGES = (100, 500, 1000)        # 수요예측 경쟁률 buckets, fixed before looking (D-045)
LOCKUP_EDGES = (0.10, 0.30)          # 기관 의무보유 확약 비율 buckets, fixed before looking (D-046)
TOP = ("범위 상단", "범위 위")          # the rule the 2026-11-03 run pointed to: final price at or above the range top
POSITIONS = ("범위 아래", "범위 하단", "범위 안", "범위 상단", "범위 위")
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


def _demand_path(root, code: str) -> Path:
    return store_dir(root) / "demand" / f"{code}.json.gz"


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


def snippets(text: str, width: int = 160, cap: int = 300, words: tuple = PRICE_WORDS) -> list[str]:
    """The text right after each of `words` (what is read later, kept instead of the whole filing)."""
    hits = sorted({m.start() for w in words for m in re.finditer(re.escape(w), text)})
    return [text[i:i + width] for i in hits[:cap]]


def _document(http, key: str, rcept_no: str, timeout: float, words: tuple = PRICE_WORDS,
              width: int = 160) -> list[str]:
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
        doc = doc if isinstance(doc, dict) else {}
        if str(doc.get("status", "")) in MISSING:
            return []  # the filing's text is not served (seen on the user's PC, 2026-11-01): try the next one
        dart_open._check(doc, "document")
        return []
    out = []
    for n in z.namelist():
        out += snippets(_text(z.read(n)), width=width, words=words)
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


# ----------------------------------------------------------------------------- demand (D-045)
_RATIO = re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?::|대)\s*1(?![\d,.])")
_RANGE = re.compile(r"(\d{1,3}(?:,\d{3})+|\d{3,7})\s*원?\s*[~∼～\-]\s*(\d{1,3}(?:,\d{3})+|\d{3,7})\s*원")


_DECIMAL = re.compile(r"(?:\d{1,3}(?:,\d{3})+|\d+)\.\d+")
_INTEGER = re.compile(r"\d{1,3}(?:,\d{3})+|\d+")


def _run(tokens: list[str], pattern: re.Pattern) -> tuple[list[float], int]:
    """The numbers at the start of `tokens` ("-" for an empty cell counts as 0) and where they stop."""
    out = []
    for i, tok in enumerate(tokens):
        if tok == "-":
            out.append(0.0)
        elif pattern.fullmatch(tok):
            out.append(float(tok.replace(",", "")))
        elif tok[:1].isdigit() and not tok.endswith(")"):
            return out, -1              # a number cut at the end of the passage, or not one this row holds
        else:
            return out, i
    return out, -1                      # ran to the end of the passage: may be cut


def _most(seen: dict) -> float:
    return max(seen, key=lambda v: (seen[v], v)) if seen else np.nan


def demand_ratio(passages: list[str]) -> float:
    """The institutions' demand ratio: the total at the end of the table row "경쟁률 a b .. total"
    (seen on the user's PC, 2026-11-03: per investor type, then all), the one quoted most often; else
    the "x:1" quoted most often after 경쟁률 (the larger on a tie); or nan."""
    table: dict[float, int] = {}
    quoted: dict[float, int] = {}
    for text in passages:
        if not text.startswith("경쟁률"):
            continue
        nums, stop = _run(text[3:].split(), _DECIMAL)
        if stop > 0 and any(v > 0 for v in nums):
            table[nums[-1]] = table.get(nums[-1], 0) + 1
        for m in _RATIO.finditer(text):
            v = float(m.group(1).replace(",", ""))
            if v >= 0.01:
                quoted[v] = quoted.get(v, 0) + 1
    return _most(table) if table else _most(quoted)


def lockup_share(passages: list[str]) -> float:
    """Share of the shares bid for with a lock-up: 1 - 미확약 total / 합계 total from the table
    "... 미확약 n q .. n q 합 계 n q .. n q" (a passage starts at the 확약 inside 미확약), or nan."""
    seen: dict[float, int] = {}
    for text in passages:
        if not text.startswith("확약"):
            continue
        tokens = text[2:].split()
        a, stop = _run(tokens, _INTEGER)
        if stop <= 0 or not a:
            continue
        rest = tokens[stop:]
        if rest[:2] == ["합", "계"]:
            rest = rest[2:]
        elif rest[:1] == ["합계"]:
            rest = rest[1:]
        else:
            continue
        b, stop = _run(rest, _INTEGER)
        if stop > 0 and b and b[-1] > 0 and 0 <= a[-1] <= b[-1]:
            v = round(1 - a[-1] / b[-1], 4)
            seen[v] = seen.get(v, 0) + 1
    return _most(seen)


def price_range(passages: list[str]) -> tuple[float, float]:
    """(low, high) of the price range offered to institutions quoted most often, or (nan, nan)."""
    seen: dict[tuple, int] = {}
    for text in passages:
        if not any(text.startswith(w) for w in RANGE_WORDS):
            continue
        for a, b in _RANGE.findall(text[:120]):
            lo, hi = float(a.replace(",", "")), float(b.replace(",", ""))
            if 500 <= lo < hi <= lo * MAX_RANGE:
                seen[(lo, hi)] = seen.get((lo, hi), 0) + 1
    return max(seen, key=lambda k: (seen[k], k)) if seen else (np.nan, np.nan)


def position(offer: float, lo: float, hi: float) -> str:
    if not (np.isfinite(offer) and np.isfinite(lo) and np.isfinite(hi)):
        return ""
    if offer < lo - 0.5:
        return POSITIONS[0]
    if abs(offer - lo) < 0.5:
        return POSITIONS[1]
    if offer < hi - 0.5:
        return POSITIONS[2]
    return POSITIONS[3] if abs(offer - hi) < 0.5 else POSITIONS[4]


def _demand_order(t_row, filings: list[dict], listed: pd.Timestamp) -> list[dict]:
    ranked = final_filings(filings, listed)
    first = str(t_row.final_rcept_no or "")
    return sorted(ranked, key=lambda f: str(f["rcept_no"]) == first, reverse=True)   # stable: the price's filing first


def download_demand(root: str | Path, key: str, t: pd.DataFrame, session=None, say=None, today=None,
                    timeout: float = 60.0) -> dict:
    """The demand and price-range passages of every priced listing's final-terms filing, not stored yet."""
    import requests

    say = say or (lambda s: None)
    http = session or requests.Session()
    today = pd.Timestamp(today or pd.Timestamp.now(tz="Asia/Seoul").date())
    todo = [r for r in t[t["status"] == "ok"].itertuples()
            if _final_path(root, r.code).exists() and not _demand_path(root, r.code).exists()]
    fetched, stopped = 0, None
    for i, r in enumerate(todo):
        if i % 50 == 0:
            say(f"공모주 수요예측 결과 받는 중: {i + 1}/{len(todo)}")
        listed = pd.Timestamp(r.date)
        docs = {}
        try:
            for f in _demand_order(r, _read(_final_path(root, r.code))["filings"], listed)[:DEMAND_DOCS]:
                no = str(f["rcept_no"])
                docs[no] = {"report_nm": f.get("report_nm"),
                            "snippets": _document(http, key, no, timeout, words=DEMAND_WORDS, width=DEMAND_WIDTH)}
                if np.isfinite(demand_ratio(docs[no]["snippets"])):
                    break
        except dart_open.DailyLimit as err:
            stopped = str(err)
            break
        p = _demand_path(root, r.code)
        p.parent.mkdir(parents=True, exist_ok=True)
        dart_open._write(p, {"asked": f"{today:%Y-%m-%d}", "listed": f"{listed:%Y-%m-%d}", "docs": docs})
        fetched += 1
    left = sum(1 for r in todo if not _demand_path(root, r.code).exists())
    return {"fetched": fetched, "left": left, "stopped": stopped}


def add_demand(root: str | Path, t: pd.DataFrame) -> pd.DataFrame:
    """`t` with the demand ratio, the price range and where the final price fell in it."""
    ratio, lo, hi, pos, src, lock = [], [], [], [], [], []
    for r in t.itertuples():
        p = _demand_path(root, r.code)
        got = (np.nan, np.nan, np.nan, "")
        lk = np.nan
        if p.exists():
            docs = _read(p)["docs"]
            for d in docs.values():
                lk = lockup_share(d["snippets"])
                if np.isfinite(lk):
                    break
            for no, d in docs.items():           # the filing order was kept when storing
                v = demand_ratio(d["snippets"])
                if np.isfinite(v):
                    got = (v, *price_range(d["snippets"]), no)
                    break
            if not got[3]:
                allp = [x for d in docs.values() for x in d["snippets"]]
                got = (np.nan, *price_range(allp), "")
        ratio.append(got[0])
        lo.append(got[1])
        hi.append(got[2])
        src.append(got[3])
        pos.append(position(float(r.offer), got[1], got[2]))
        lock.append(lk)
    return t.assign(demand_ratio=ratio, range_low=lo, range_high=hi, range_pos=pos, demand_rcept_no=src,
                    lockup=lock)


def lockup_bucket(v: float) -> str:
    if not np.isfinite(v):
        return "못 읽음"
    lo, hi = LOCKUP_EDGES
    return f"{lo:.0%} 미만" if v < lo else (f"{lo:.0%}~{hi:.0%}" if v < hi else f"{hi:.0%} 이상")


def ratio_bucket(v: float) -> str:
    if not np.isfinite(v):
        return "못 읽음"
    edges = RATIO_EDGES
    if v < edges[0]:
        return f"{edges[0]}:1 미만"
    for a, b in pairwise(edges):
        if v < b:
            return f"{a}~{b}:1"
    return f"{edges[-1]}:1 이상"


def demand_summary(t: pd.DataFrame) -> pd.DataFrame:
    ok = t[t["status"] == "ok"].copy()
    ok["ratio_bucket"] = ok["demand_ratio"].map(ratio_bucket)
    order = [f"{RATIO_EDGES[0]}:1 미만"] + [f"{a}~{b}:1" for a, b in pairwise(RATIO_EDGES)] + \
            [f"{RATIO_EDGES[-1]}:1 이상", "못 읽음"]
    rows = []
    for period, g in (("전체 기간", ok), (f"{RULE_CHANGE} 이후", ok[ok["date"] >= pd.Timestamp(RULE_CHANGE)])):
        for b in order:
            sub = g[g["ratio_bucket"] == b]
            if len(sub):
                rows.append({"period": period, "kind": "수요예측 경쟁률", "group": b, **_stats(sub)})
        for b in POSITIONS + ("",):
            sub = g[g["range_pos"] == b]
            if len(sub):
                rows.append({"period": period, "kind": "확정 공모가 위치", "group": b or "못 읽음", **_stats(sub)})
        top = g["range_pos"].isin(TOP)
        for name, sub in (("상단 이상만 청약", g[top]), ("나머지", g[~top & (g["range_pos"] != "")])):
            if len(sub):
                rows.append({"period": period, "kind": "청약 규칙", "group": name, **_stats(sub)})
        lb = g["lockup"].map(lockup_bucket)
        lo, hi = LOCKUP_EDGES
        for b in (f"{lo:.0%} 미만", f"{lo:.0%}~{hi:.0%}", f"{hi:.0%} 이상", "못 읽음"):
            sub = g[lb == b]
            if len(sub):
                rows.append({"period": period, "kind": "기관 확약 비율", "group": b, **_stats(sub)})
    return pd.DataFrame(rows)


def demand_text(t: pd.DataFrame, d: pd.DataFrame) -> str:
    ok = t[t["status"] == "ok"]
    got = ok["demand_ratio"].notna().sum()
    both = ok[ok["range_low"].notna() & ok["first_offer"].notna()]
    same = int(((both["range_low"] - both["first_offer"]).abs() < 0.5).sum())
    lines = ["", (f"상장 전에 알 수 있던 것 (공모가 찾은 {len(ok)}곳 중 수요예측 경쟁률 읽음 {got}곳, 희망 범위 읽음 "
                  f"{ok['range_low'].notna().sum()}곳, 기관 확약 비율 읽음 {ok['lockup'].notna().sum()}곳)"),
             f"점검: 희망 범위 아래쪽 값이 첫 신고서 가격과 같은 곳 {same}/{len(both)}"]
    for code in SAMPLES:
        r = ok[ok["code"] == code]
        if len(r):
            r = r.iloc[0]
            ratio = "못 읽음" if pd.isna(r["demand_ratio"]) else f"{r['demand_ratio']:,.2f}:1"
            rng = "못 읽음" if pd.isna(r["range_low"]) else f"{r['range_low']:,.0f}~{r['range_high']:,.0f}원"
            lock = "못 읽음" if pd.isna(r["lockup"]) else f"{r['lockup']:.1%}"
            lines.append(f"확인용: {r['name']} 경쟁률 {ratio}, 희망 범위 {rng}, 확정 {r['offer']:,.0f}원, 확약 {lock}")
    for period in d["period"].unique() if len(d) else []:
        lines += ["", f"[{period}] 묶음: 건수 | 시가에 팔기 평균/중앙값/공모가 아래 | 종가에 팔기 평균/공모가 아래"]
        for r in d[d["period"] == period].itertuples():
            lines.append(f"{r.kind} {r.group}: {r.n}건 | 시가 {_pct(r.open_mean)} / {_pct(r.open_median)} / "
                         f"{r.open_below * 100:.0f}% | 종가 {_pct(r.close_mean)} / {r.close_below * 100:.0f}%")
    return "\n".join(lines)


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
        out[code] = {k: _read(p) for k, p in (("estkRs", _path(root, code)), ("final", _final_path(root, code)),
                                              ("demand", _demand_path(root, code))) if p.exists()}
    return out


def write(folder: Path, t: pd.DataFrame, s: pd.DataFrame, text: str, raw: dict | None = None,
          demand: pd.DataFrame | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        (folder / "dart_samples.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    t.assign(date=t["date"].dt.strftime("%Y-%m-%d")).to_csv(folder / "ipo_firstday.csv", index=False,
                                                            encoding="utf-8-sig")
    s.to_csv(folder / "ipo_summary.csv", index=False, encoding="utf-8-sig")
    if demand is not None:
        demand.to_csv(folder / "ipo_demand.csv", index=False, encoding="utf-8-sig")
    (folder / "ipo_summary.txt").write_text(text, encoding="utf-8")
