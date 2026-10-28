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
shareholders or third parties (rights issues of a company already listed elsewhere). A first-day
open outside `OPEN_BAND` times that price means the filing is not this listing's offer: such a row is
shown but left out of the summary.

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
OPEN_BAND = (0.5, 4.5)
SELL_COST = 0.0025
MAX_NAME_MATCHES = 3
COLUMNS = ["date", "year", "market", "code", "name", "match", "corp_code", "rcept_no", "method", "offer",
           "open", "close", "ret_open", "ret_close", "status"]


def store_dir(root: str | Path) -> Path:
    # beside, not inside, the DART store: its files are the KRDART dataset's inputs
    return Path(root) / "data" / "bulk" / "dart-ipo"


def _path(root, code: str) -> Path:
    return store_dir(root) / "estkRs" / f"{code}.json.gz"


def _read(p: Path):
    with gzip.open(p, "rb") as g:
        return json.loads(g.read().decode("utf-8"))


# ----------------------------------------------------------------------------- listings
def new_listings(root: str | Path, end, start: str = START, say=None) -> tuple[pd.DataFrame, dict]:
    """First sessions of new common shares from `start` to `end` (date, market, code, name, open, close)
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
        rows.append(first[first["date"] >= pd.Timestamp(start)][["date", "market", "code", "name", "open", "close"]])
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "market", "code", "name",
                                                                              "open", "close"])
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


# ----------------------------------------------------------------------------- results
def table(root: str | Path, listings: pd.DataFrame, corps: dict) -> pd.DataFrame:
    out = []
    for r in listings.itertuples():
        match, cands = candidates(corps, r.code, r.name)
        row = {"date": pd.Timestamp(r.date), "year": pd.Timestamp(r.date).year, "market": r.market, "code": r.code,
               "name": r.name, "match": match, "corp_code": "", "rcept_no": "", "method": "", "offer": np.nan,
               "open": float(r.open), "close": float(r.close), "ret_open": np.nan, "ret_close": np.nan,
               "status": "no_company"}
        p = _path(root, r.code)
        if cands and p.exists():
            row["status"] = "no_filing"
            best = None
            for corp, groups in _read(p)["answers"].items():
                got = choose(offers(groups), row["date"])
                if got and (best is None or got["rcept_no"] > best[1]["rcept_no"]):
                    best = (corp, got)
            if best:
                corp, o = best
                row.update(corp_code=corp, rcept_no=o["rcept_no"], method=o["method"], offer=o["offer"],
                           ret_open=row["open"] / o["offer"] - 1, ret_close=row["close"] / o["offer"] - 1)
                ratio = row["open"] / o["offer"]
                row["status"] = "ok" if OPEN_BAND[0] <= ratio <= OPEN_BAND[1] else "price_check"
        elif cands:
            row["status"] = "not_asked"
        out.append(row)
    return pd.DataFrame(out, columns=COLUMNS)


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
    lines = [(f"공모주 첫날 조사: 새로 상장한 보통주 {len(t)}곳 (스팩 {left_out['spac']}곳, 우선주 등 "
              f"{left_out['not_common']}곳 제외)"),
             (f"공모가 찾음 {counts.get('ok', 0)}곳 · 회사 못 찾음 {counts.get('no_company', 0)} · "
              f"신고서 없음 {counts.get('no_filing', 0)} · 가격 확인 필요 {counts.get('price_check', 0)}"),
             "", "연도: 건수 | 첫날 시가에 팔기 평균/중앙값/공모가 아래 | 첫날 종가에 팔기 평균/중앙값/공모가 아래/최악"]
    for r in s.itertuples():
        lines.append(f"{r.group}: {r.n}건 | 시가 {_pct(r.open_mean)} / {_pct(r.open_median)} / {r.open_below * 100:.0f}% | "
                     f"종가 {_pct(r.close_mean)} / {_pct(r.close_median)} / {r.close_below * 100:.0f}% / {_pct(r.close_worst)}")
    lines += ["", f"수익률은 공모가 대비, 매도 비용({SELL_COST * 100:.2f}%)과 청약 수수료(건당) 전입니다."]
    return "\n".join(lines)


def write(folder: Path, t: pd.DataFrame, s: pd.DataFrame, text: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    t.assign(date=t["date"].dt.strftime("%Y-%m-%d")).to_csv(folder / "ipo_firstday.csv", index=False,
                                                            encoding="utf-8-sig")
    s.to_csv(folder / "ipo_summary.csv", index=False, encoding="utf-8-sig")
    (folder / "ipo_summary.txt").write_text(text, encoding="utf-8")
