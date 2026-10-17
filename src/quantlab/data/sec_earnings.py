"""Earnings announcement dates from SEC filings: Form 8-K with item 2.02 "Results of Operations and
Financial Condition" (D-027).

Source: the SEC submissions API, one JSON file per company (free, no key):
    https://data.sec.gov/submissions/CIK##########.json
        filings.recent : columnar arrays (accessionNumber, filingDate, acceptanceDateTime, form, items, ...)
        filings.files  : older pages, each fetched from https://data.sec.gov/submissions/<name>
                         and holding the same columnar arrays at its top level
Companies are the issuers linked to index tickers through their Form 4 filings (sec_insider).

Only the filing date is used, not the acceptance time: a rule reads the reaction over the session of
the filing date (if it is one) and the next session, which covers filings made before the open,
during the session or after the close alike.
"""

from __future__ import annotations

import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quantlab.core.hashing import file_sha256
from quantlab.data.sec_insider import user_agent

API = "https://data.sec.gov/submissions/"
PARSE_VERSION = 1
PAUSE_SECONDS = 0.12   # about 8 requests a second, under the SEC's limit of 10


def store_dir(root: str | Path) -> Path:
    return Path(root) / "data" / "bulk" / "sec-submissions"


def records(root: str | Path) -> dict[str, dict]:
    out = {}
    for p in sorted(store_dir(root).glob("*.meta.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        out[m["cik"]] = m
    return out


def _get(http, url: str, headers: dict, timeout: float):
    resp = http.get(url, headers=headers, timeout=timeout)
    time.sleep(PAUSE_SECONDS)
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise RuntimeError(f"SEC {url}: HTTP {resp.status_code}")
    return resp.content


def _stored_pages(meta: dict) -> dict[str, bytes]:
    """Older pages of a stored company file, by the name its main file gives them."""
    with gzip.open(meta["path"], "rb") as g:
        pages = json.loads(g.read())
    names = [f["name"] for f in pages[0].get("filings", {}).get("files", [])]
    return {n: json.dumps(p).encode() for n, p in zip(names, pages[1:])}


def download(root: str | Path, email: str, ciks: list[str], session=None, say=None, timeout: float = 60.0,
             refresh: bool = False, refresh_before: str | None = None) -> dict:
    """Store each company's submissions JSON (main file and older pages) gzip-compressed, as received.
    Companies already stored are skipped unless `refresh`, or unless fetched before `refresh_before`
    (YYYYMMDD, UTC); such a refresh fetches the main file again and keeps the older pages already
    stored, which the SEC never changes. Returns {"fetched", "missing"}."""
    import requests

    say = say or (lambda s: None)
    http = session or requests
    headers = {"User-Agent": user_agent(email), "Accept-Encoding": "gzip, deflate"}
    d = store_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    have = records(root)
    stale = {c for c, m in have.items() if refresh_before and m["fetched_at"][:8] < refresh_before}
    todo = [c for c in sorted(set(ciks)) if refresh or c not in have or c in stale]
    fetched, missing = [], []
    for i, cik in enumerate(todo):
        if i % 20 == 0:
            say(f"SEC 실적 발표 공시 받는 중: 회사 {i + 1}/{len(todo)}")
        main = _get(http, f"{API}CIK{int(cik):010d}.json", headers, timeout)
        if main is None:
            missing.append(cik)
            continue
        old = _stored_pages(have[cik]) if cik in stale and not refresh else {}
        pages = [main]
        for f in json.loads(main).get("filings", {}).get("files", []):
            page = old.get(f["name"]) or _get(http, API + f["name"], headers, timeout)
            if page is None:
                raise RuntimeError(f"SEC submissions page missing: {f['name']}")
            pages.append(page)
        path = d / f"CIK{int(cik):010d}.json.gz"
        with gzip.open(path, "wb") as g:
            g.write(b"[" + b",".join(pages) + b"]")
        meta = {"cik": cik, "sha256": file_sha256(path), "pages": len(pages), "path": str(path),
                "fetched_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}
        (d / f"CIK{int(cik):010d}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
        fetched.append(cik)
    return {"fetched": fetched, "missing": missing}


def _columns(page: dict) -> dict:
    return page["filings"]["recent"] if "filings" in page else page


def parse_company(path: str | Path, cik: str) -> pd.DataFrame:
    """Original 8-K filings that list item 2.02: cik, filed (date), accession, acceptance (as given)."""
    with gzip.open(path, "rb") as g:
        pages = json.loads(g.read())
    rows = []
    for page in pages:
        cols = _columns(page)
        n = len(cols.get("accessionNumber", []))
        items = cols.get("items") or [""] * n
        acc_time = cols.get("acceptanceDateTime") or [""] * n
        for k in range(n):
            if cols["form"][k] != "8-K":
                continue
            its = {x.strip() for x in str(items[k] or "").split(",")}
            if "2.02" not in its:
                continue
            rows.append({"cik": cik, "filed": cols["filingDate"][k], "accession": cols["accessionNumber"][k],
                         "acceptance": acc_time[k]})
    df = pd.DataFrame(rows, columns=["cik", "filed", "accession", "acceptance"])
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    return df.dropna(subset=["filed"]).drop_duplicates("accession")


def load_parsed(meta: dict) -> pd.DataFrame:
    path = Path(meta["path"])
    if file_sha256(path) != meta["sha256"]:
        raise ValueError(f"SEC 원본 파일이 받은 뒤 바뀌었습니다: {path}")
    cache = path.with_name(f"{path.name}.v{PARSE_VERSION}.{meta['sha256'][:12]}.parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    df = parse_company(path, meta["cik"])
    df.to_parquet(cache, index=False)
    return df


def earnings_events(filings: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Events table rows (kind "earnings"): one per company and filing date, on the ticker whose price
    dates cover it. `links`: issuer_cik, ticker, first_t, last_t (from sec_insider.map_issuers)."""
    if filings.empty:
        return pd.DataFrame(columns=["known", "symbol", "accession", "kind"])
    f = filings.drop_duplicates(["cik", "filed"]).rename(columns={"cik": "issuer_cik"})
    j = f.merge(links[["issuer_cik", "ticker", "first_t", "last_t"]], on="issuer_cik")
    j = j[(j["filed"] >= j["first_t"] - pd.Timedelta(days=5)) & (j["filed"] <= j["last_t"])]
    j = j.drop_duplicates(["issuer_cik", "ticker", "filed"])
    ev = pd.DataFrame({"known": j["filed"].to_numpy(), "symbol": j["ticker"].to_numpy(),
                       "accession": j["accession"].to_numpy(), "kind": "earnings"})
    return ev.sort_values(["known", "symbol"], kind="stable").reset_index(drop=True)
