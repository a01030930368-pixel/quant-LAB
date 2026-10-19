"""SEC insider transactions (Forms 3/4/5) from the SEC's quarterly "Insider Transactions Data Sets"
(free, no key; D-026).

    https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
    one zip per quarter (2006q1 on), tab-separated UTF-8 tables linked by ACCESSION_NUMBER:
      SUBMISSION      FILING_DATE, DOCUMENT_TYPE (3, 4, 5, 4/A ...), ISSUERCIK, ISSUERTRADINGSYMBOL
      REPORTINGOWNER  RPTOWNERCIK, RPTOWNER_RELATIONSHIP (Director, Officer, TenPercentOwner, Other)
      NONDERIV_TRANS  TRANS_CODE (P = open-market purchase), TRANS_ACQUIRED_DISP_CD (A/D),
                      TRANS_SHARES, TRANS_PRICEPERSHARE
    dates are DD-MON-YYYY.

The SEC asks automated clients to name themselves with a contact e-mail in the User-Agent and to stay
under 10 requests a second; the app asks the user for that e-mail and sends it only to sec.gov.

What the app keeps (per quarter, cached next to the raw zip):
    purchases : one row per original Form 4 (amendments are left out), reporting owner and side with
                open-market purchases (side "buy", P/A rows) or sales (side "sell", S/D rows, D-034):
                known (FILING_DATE), issuer_cik, symbol, owner, value (sum of shares x price over the
                filing's rows of that side), officer, director, accession, side
    filings   : every original Form 4: issuer_cik, symbol, first and last filing date, count
                (used to map issuers to the price data's tickers and to report coverage)

`known` is the filing date. Section 16 filings made before 10 p.m. Eastern carry that day's date, so
a filing is public by the evening of `known` and a rule may act at the next session's open.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.core.hashing import file_sha256

BASES = ("https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/",
         "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/")
FIRST_QUARTER = "2009q3"   # the 30-day window before 2010 needs late-2009 filings
PARSE_VERSION = 2          # 2: open-market sales too (side "sell", D-034)
PAUSE_SECONDS = 0.3        # well under the SEC's 10 requests per second
EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def user_agent(email: str) -> str:
    if not EMAIL.fullmatch(email or ""):
        raise ValueError("SEC 연락용 이메일 형식이 올바르지 않습니다")
    return f"QuantLab personal research {email}"


def sec_dir(root: str | Path) -> Path:
    return Path(root) / "data" / "bulk" / "sec-insider"


def quarter_list(first: str = FIRST_QUARTER, today: date | None = None) -> list[str]:
    """Quarters from `first` up to the last one that has ended."""
    today = today or date.today()
    y, q = int(first[:4]), int(first[-1])
    end_y, end_q = today.year, (today.month - 1) // 3 + 1
    end_y, end_q = (end_y, end_q - 1) if end_q > 1 else (end_y - 1, 4)
    out = []
    while (y, q) <= (end_y, end_q):
        out.append(f"{y}q{q}")
        y, q = (y, q + 1) if q < 4 else (y + 1, 1)
    return out


def records(root: str | Path) -> dict[str, dict]:
    out = {}
    for p in sorted(sec_dir(root).glob("*.meta.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        out[m["quarter"]] = m
    return out


def download(root: str | Path, email: str, quarters: list[str] | None = None, session=None, say=None,
             timeout: float = 120.0) -> dict:
    """Fetch the quarters not yet stored. Quarters the SEC has not published yet (404 on every base
    URL, only at the end of the list) are reported, not raised. Returns {"fetched", "missing"}."""
    import requests

    say = say or (lambda s: None)
    http = session or requests
    headers = {"User-Agent": user_agent(email), "Accept-Encoding": "gzip, deflate"}
    have = records(root)
    d = sec_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    fetched, missing = [], []
    todo = [q for q in (quarters or quarter_list()) if q not in have]
    for i, q in enumerate(todo):
        say(f"SEC 내부자 거래 자료 받는 중: {q} ({i + 1}/{len(todo)})")
        got = None
        for base in BASES:
            url = f"{base}{q}_form345.zip"
            resp = http.get(url, headers=headers, timeout=timeout, stream=True)
            time.sleep(PAUSE_SECONDS)
            if resp.status_code == 404:
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"SEC {q}: HTTP {resp.status_code} ({url})")
            tmp = d / f"{q}_form345.zip.part"
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    f.write(chunk)
            got = (url, tmp)
            break
        if got is None:
            missing.append(q)
            continue
        url, tmp = got
        if not zipfile.is_zipfile(tmp):
            tmp.unlink()
            raise RuntimeError(f"SEC {q}: 받은 파일이 zip이 아닙니다 (차단 페이지일 수 있음)")
        path = d / f"{q}_form345.zip"
        tmp.replace(path)
        meta = {"quarter": q, "url": url, "sha256": file_sha256(path), "bytes": path.stat().st_size,
                "fetched_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"), "path": str(path)}
        (d / f"{q}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        fetched.append(q)
    gaps = [q for q in missing if q != (quarters or quarter_list())[-1]]
    if gaps:
        raise RuntimeError(f"SEC 자료가 없는 분기가 있습니다: {gaps}")
    return {"fetched": fetched, "missing": missing}


# --- parsing --------------------------------------------------------------------------------
def _table(z: zipfile.ZipFile, stem: str, cols: list[str]) -> pd.DataFrame:
    names = [n for n in z.namelist() if Path(n).stem.upper() == stem and n.lower().endswith((".tsv", ".txt"))]
    if not names:
        raise ValueError(f"{stem} table missing from {z.filename}")
    text = z.read(names[0]).decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False,
                     on_bad_lines="skip")
    df.columns = [c.strip().upper() for c in df.columns]
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{stem} lacks columns {sorted(missing)}")
    return df[cols]


def parse_date(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s.str.strip().str.title(), format="%d-%b-%Y", errors="coerce")
    rest = out.isna() & s.str.strip().ne("")
    if rest.any():  # other spellings, if a quarter uses them
        out[rest] = pd.to_datetime(s[rest], format="mixed", errors="coerce")
    return out


def norm_symbol(s: pd.Series) -> pd.Series:
    return s.str.upper().str.strip().str.replace(r"[./ ]+", "-", regex=True).str.strip("-")


def parse_quarter(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(purchases, filings, counts) for one quarterly zip."""
    with zipfile.ZipFile(path) as z:
        sub = _table(z, "SUBMISSION", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK",
                                       "ISSUERTRADINGSYMBOL"])
        own = _table(z, "REPORTINGOWNER", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
        tr = _table(z, "NONDERIV_TRANS", ["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_ACQUIRED_DISP_CD",
                                          "TRANS_SHARES", "TRANS_PRICEPERSHARE"])
    counts = {"submissions": int(len(sub))}
    sub = sub[sub["DOCUMENT_TYPE"].str.strip() == "4"].copy()
    sub["known"] = parse_date(sub["FILING_DATE"])
    counts["bad_dates"] = int(sub["known"].isna().sum())
    sub = sub[sub["known"].notna()]
    sub["issuer_cik"] = sub["ISSUERCIK"].str.strip().str.lstrip("0")
    sub["symbol"] = norm_symbol(sub["ISSUERTRADINGSYMBOL"])
    counts["form4"] = int(len(sub))
    filings = (sub.groupby(["issuer_cik", "symbol"]).agg(first=("known", "min"), last=("known", "max"),
                                                          n=("known", "size")).reset_index())
    code = tr["TRANS_CODE"].str.strip().str.upper()
    ad = tr["TRANS_ACQUIRED_DISP_CD"].str.strip().str.upper()
    tr = tr.assign(value=pd.to_numeric(tr["TRANS_SHARES"], errors="coerce")
                   * pd.to_numeric(tr["TRANS_PRICEPERSHARE"], errors="coerce"))
    rel = own["RPTOWNER_RELATIONSHIP"].str.upper()
    own = own.assign(owner=own["RPTOWNERCIK"].str.strip().str.lstrip("0"),
                     officer=rel.str.contains("OFFICER"), director=rel.str.contains("DIRECTOR"))
    parts = []
    for side, mask in (("buy", (code == "P") & (ad == "A")), ("sell", (code == "S") & (ad == "D"))):
        rows = tr[mask]
        name = "purchase" if side == "buy" else "sale"
        counts[f"{name}_rows"] = int(len(rows))
        counts[f"{name}_rows_without_value"] = int(rows["value"].isna().sum())
        val = rows.groupby("ACCESSION_NUMBER")["value"].sum(min_count=1).rename("value").reset_index()
        part = (sub.merge(val, on="ACCESSION_NUMBER")
                .merge(own[["ACCESSION_NUMBER", "owner", "officer", "director"]], on="ACCESSION_NUMBER"))
        part = part.rename(columns={"ACCESSION_NUMBER": "accession"})[
            ["known", "issuer_cik", "symbol", "owner", "value", "officer", "director", "accession"]]
        part = part.drop_duplicates(["accession", "owner"]).assign(side=side)
        counts[f"{name}_filings"] = int(part["accession"].nunique())
        parts.append(part)
    pur = pd.concat(parts, ignore_index=True).sort_values(["known", "issuer_cik", "owner", "side"], kind="stable")
    return pur.reset_index(drop=True), filings, counts


def load_parsed(meta: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Parsed tables for one stored quarter, cached beside the zip (keyed by its hash)."""
    path = Path(meta["path"])
    if file_sha256(path) != meta["sha256"]:
        raise ValueError(f"SEC 원본 파일이 받은 뒤 바뀌었습니다: {path}")
    stem = path.with_name(f"{meta['quarter']}.v{PARSE_VERSION}.{meta['sha256'][:12]}")
    pp, fp, cp = (stem.with_suffix(stem.suffix + ext) for ext in (".purchases.parquet", ".filings.parquet",
                                                                    ".counts.json"))
    if pp.exists() and fp.exists() and cp.exists():
        return pd.read_parquet(pp), pd.read_parquet(fp), json.loads(cp.read_text(encoding="utf-8"))
    pur, fil, counts = parse_quarter(path)
    pur.to_parquet(pp)
    fil.to_parquet(fp)
    cp.write_text(json.dumps(counts), encoding="utf-8")
    return pur, fil, counts


# --- issuer -> price ticker ------------------------------------------------------------------
DEAD = re.compile(r"^(.+)-\d{6}$")


def ticker_base(t: str) -> str:
    """AAL-199702 -> AAL (delisted names in the price data carry -YYYYMM)."""
    m = DEAD.match(t)
    return m.group(1) if m else t


def map_issuers(purchases: pd.DataFrame, filings: pd.DataFrame, ranges: pd.DataFrame,
                slack_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the price data's ticker to each purchase row.

    `ranges`: ticker, first, last (dates with prices). An issuer (CIK) is linked to a ticker when one of
    the symbols it filed under equals the ticker's base and the filing dates overlap the ticker's price
    dates. Through the CIK, filings made under an older symbol (FB) reach the renamed ticker (META).
    Each purchase then goes to the linked ticker whose price dates cover its filing date.
    Returns (purchases with `ticker`, unmapped rows dropped; the issuer -> ticker links)."""
    r = ranges.assign(base=ranges["ticker"].map(ticker_base))
    f = filings.groupby(["issuer_cik", "symbol"]).agg(first=("first", "min"), last=("last", "max")).reset_index()
    cand = f.merge(r, left_on="symbol", right_on="base", suffixes=("_f", "_t"))
    slack = pd.Timedelta(days=slack_days)
    cand = cand[(cand["first_f"] <= cand["last_t"] + slack) & (cand["last_f"] >= cand["first_t"] - slack)]
    links = cand[["issuer_cik", "ticker", "first_t", "last_t", "symbol"]].drop_duplicates(["issuer_cik", "ticker"])
    j = purchases.reset_index(drop=True).reset_index().merge(
        links[["issuer_cik", "ticker", "first_t", "last_t", "symbol"]].rename(columns={"symbol": "link_symbol"}),
        on="issuer_cik")
    j = j[(j["known"] >= j["first_t"] - pd.Timedelta(days=5)) & (j["known"] <= j["last_t"] + pd.Timedelta(days=5))]
    j["own_symbol"] = j["link_symbol"] == j["symbol"]
    j = j.sort_values(["index", "own_symbol", "ticker"], ascending=[True, False, True]).drop_duplicates("index")
    out = j.drop(columns=["index", "first_t", "last_t", "link_symbol", "own_symbol"])
    return out.sort_values(["known", "ticker", "owner"], kind="stable").reset_index(drop=True), links


def insider_events(mapped: pd.DataFrame) -> pd.DataFrame:
    """Events table for MarketData (see quantlab.core.market_data)."""
    ev = mapped.rename(columns={"symbol": "filed_symbol"}).rename(columns={"ticker": "symbol"})[
        ["known", "symbol", "owner", "value", "officer", "director", "filed_symbol", "accession"]].copy()
    ev["value"] = ev["value"].astype(float).fillna(0.0)
    side = mapped["side"] if "side" in mapped else pd.Series("buy", index=mapped.index)
    ev["kind"] = np.where(side.to_numpy() == "sell", "insider_sell", "insider_buy")
    return ev.sort_values(["known", "symbol", "owner"], kind="stable").reset_index(drop=True)
