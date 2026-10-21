"""Financial statement facts from SEC XBRL filings, for value and profitability rules (D-037).

Source: the SEC company facts API, one JSON file per company (free, no key):
    https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
        facts.<taxonomy>.<tag>.units.<unit> : list of {val, end, start?, filed, form, fy, fp, accn, frame?}
Companies are the issuers linked to index tickers through their Form 4 filings (sec_insider).

Kept facts (our name <- tags, first one present for a period wins):
    assets        <- us-gaap Assets                                              (instant)
    equity        <- us-gaap StockholdersEquity, StockholdersEquityIncluding...  (instant)
    gross_profit  <- us-gaap GrossProfit, or revenue minus cost of revenue        (one fiscal year)
    float         <- dei EntityPublicFloat                                       (instant, USD)
A fact counts as known from the day after its `filed` date (a filing can land after the close).
Amended filings are separate rows with their own `filed` date, so a restatement is known only from
then. Public float is the market value of shares held by non-insiders on the last day of the second
fiscal quarter, as the company reports it on its 10-K cover; rules scale it by the stock's price change
since that day, so no share counts or split histories are needed.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quantlab.core.hashing import file_sha256
from quantlab.data.sec_earnings import KEEP_OLD_SECONDS, PAUSE_SECONDS, _get
from quantlab.data.sec_insider import user_agent

API = "https://data.sec.gov/api/xbrl/companyfacts/"
PARSE_VERSION = 1
FACTS = ("assets", "equity", "gross_profit", "float")
INSTANT = {
    "assets": [("us-gaap", "Assets")],
    "equity": [("us-gaap", "StockholdersEquity"),
               ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")],
    "float": [("dei", "EntityPublicFloat")],
}
GROSS = [("us-gaap", "GrossProfit")]
REVENUE = [("us-gaap", "Revenues"), ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
           ("us-gaap", "SalesRevenueNet"), ("us-gaap", "SalesRevenueGoodsNet")]
COST = [("us-gaap", "CostOfRevenue"), ("us-gaap", "CostOfGoodsAndServicesSold"), ("us-gaap", "CostOfGoodsSold")]
YEAR_DAYS = (340, 380)   # a fiscal-year duration
COLUMNS = ["cik", "fact", "value", "start", "end", "filed", "form", "accn"]


def store_dir(root: str | Path) -> Path:
    return Path(root) / "data" / "bulk" / "sec-facts"


def records(root: str | Path) -> dict[str, dict]:
    out = {}
    for p in sorted(store_dir(root).glob("*.meta.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        out[m["cik"]] = m
    return out


def download(root: str | Path, email: str, ciks: list[str], session=None, say=None, timeout: float = 60.0,
             refresh_before: str | None = None) -> dict:
    """Store each company's facts JSON gzip-compressed, as received. Companies already stored are
    skipped unless fetched before `refresh_before` (YYYYMMDD, UTC). A refresh writes a new file and then
    points the meta file at it (see sec_earnings.download). Returns {"fetched", "missing"}; a company
    the SEC has no facts for (404) is remembered and not asked again before `refresh_before`."""
    import requests

    say = say or (lambda s: None)
    http = session or requests
    headers = {"User-Agent": user_agent(email), "Accept-Encoding": "gzip, deflate"}
    d = store_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    have = records(root)
    stale = {c for c, m in have.items() if refresh_before and m["fetched_at"][:8] < refresh_before}
    todo = [c for c in sorted(set(ciks)) if c not in have or c in stale]
    fetched, missing = [], []
    for i, cik in enumerate(todo):
        if i % 20 == 0:
            say(f"SEC 재무제표 자료 받는 중: 회사 {i + 1}/{len(todo)}")
        body = _get(http, f"{API}CIK{int(cik):010d}.json", headers, timeout)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if body is None:
            missing.append(cik)
            body = b"{}"
        path = d / (f"CIK{int(cik):010d}.json.gz" if cik not in have else f"CIK{int(cik):010d}-{stamp}.json.gz")
        with gzip.open(path, "wb") as g:
            g.write(body)
        meta = {"cik": cik, "sha256": file_sha256(path), "path": str(path), "fetched_at": stamp,
                "empty": body == b"{}"}
        tmp = d / f"CIK{int(cik):010d}.meta.json.tmp"
        tmp.write_text(json.dumps(meta), encoding="utf-8")
        os.replace(tmp, d / f"CIK{int(cik):010d}.meta.json")
        fetched.append(cik)
    _prune(d)
    return {"fetched": fetched, "missing": missing}


def _prune(d: Path) -> None:
    current = {Path(m["path"]).name for m in (json.loads(p.read_text(encoding="utf-8")) for p in d.glob("*.meta.json"))}
    now = time.time()
    for p in d.glob("CIK*-*.json.gz*"):
        base = p.name.split(".json.gz")[0] + ".json.gz"
        if base in current or now - p.stat().st_mtime < KEEP_OLD_SECONDS:
            continue
        try:
            p.unlink()
        except OSError:
            pass


def _rows(facts: dict, tags: list[tuple[str, str]], unit: str) -> pd.DataFrame:
    """All reported values of the first-listed tag that has a value for each (start, end) period."""
    out = []
    for rank, (tax, tag) in enumerate(tags):
        for r in facts.get(tax, {}).get(tag, {}).get("units", {}).get(unit, []):
            if "val" in r and "end" in r and "filed" in r:
                out.append({"rank": rank, "value": float(r["val"]), "start": r.get("start"), "end": r["end"],
                            "filed": r["filed"], "form": r.get("form", ""), "accn": r.get("accn", "")})
    df = pd.DataFrame(out, columns=["rank", "value", "start", "end", "filed", "form", "accn"])
    if df.empty:
        return df
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["end", "filed"])
    best = df.groupby(["start", "end"], dropna=False)["rank"].transform("min")
    return df[df["rank"] == best].drop(columns="rank")


def _annual(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    days = (df["end"] - df["start"]).dt.days
    return df[(days >= YEAR_DAYS[0]) & (days <= YEAR_DAYS[1])]


def parse_company(path: str | Path, cik: str) -> pd.DataFrame:
    with gzip.open(path, "rb") as g:
        doc = json.loads(g.read() or b"{}")
    facts = doc.get("facts", {})
    parts = []
    for name, tags in INSTANT.items():
        df = _rows(facts, tags, "USD")
        if not df.empty:
            parts.append(df.assign(fact=name, start=pd.NaT))
    gp = _annual(_rows(facts, GROSS, "USD"))
    rev, cost = _annual(_rows(facts, REVENUE, "USD")), _annual(_rows(facts, COST, "USD"))
    if not rev.empty and not cost.empty:
        # revenue minus cost of revenue, as filed together (same filing, same fiscal year)
        j = rev.merge(cost[["start", "end", "accn", "value"]], on=["start", "end", "accn"], suffixes=("", "_cost"))
        j["value"] = j["value"] - j["value_cost"]
        j = j.drop(columns="value_cost")
        if not gp.empty:
            have = set(zip(gp["start"], gp["end"]))
            j = j[[(s, e) not in have for s, e in zip(j["start"], j["end"])]]
        gp = pd.concat([gp, j], ignore_index=True)
    if not gp.empty:
        parts.append(gp.assign(fact="gross_profit"))
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat(parts, ignore_index=True).assign(cik=cik)
    out = out.drop_duplicates(["fact", "start", "end", "filed", "value"])
    return out[COLUMNS].sort_values(["fact", "filed", "end"], kind="stable").reset_index(drop=True)


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


def fact_events(facts: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Events table rows (kind "fact_<name>"): known the day after filing, on the ticker whose price
    dates cover that day. Columns: known, symbol, kind, value, period_end."""
    cols = ["known", "symbol", "kind", "value", "period_end"]
    if facts.empty:
        return pd.DataFrame(columns=cols)
    f = facts.rename(columns={"cik": "issuer_cik"}).assign(known=facts["filed"] + pd.Timedelta(days=1))
    j = f.merge(links[["issuer_cik", "ticker", "first_t", "last_t"]], on="issuer_cik")
    j = j[(j["known"] >= j["first_t"] - pd.Timedelta(days=5)) & (j["known"] <= j["last_t"])]
    ev = pd.DataFrame({"known": j["known"].to_numpy(), "symbol": j["ticker"].to_numpy(),
                       "kind": ("fact_" + j["fact"]).to_numpy(), "value": j["value"].to_numpy(dtype=float),
                       "period_end": j["end"].to_numpy()})
    return ev.drop_duplicates().sort_values(["known", "symbol", "kind"], kind="stable").reset_index(drop=True)
