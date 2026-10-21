"""Synthetic S&P 500 bundle and SEC insider zips in the shapes the real sources use (D-026)."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.core.hashing import file_sha256
from quantlab.data import sec_insider, sp500

SUB_COLS = ["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERNAME",
            "ISSUERTRADINGSYMBOL"]
OWN_COLS = ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"]
TR_COLS = ["ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "SECURITY_TITLE", "TRANS_DATE", "TRANS_CODE",
           "TRANS_SHARES", "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD", "DIRECT_INDIRECT_OWNERSHIP"]


def sec_date(d) -> str:
    return pd.Timestamp(d).strftime("%d-%b-%Y").upper()


def write_quarter_zip(path: Path, filings: list[dict]) -> None:
    """filings: {acc, date, doc, cik, symbol, owners: [(cik, relationship)], trans: [(code, ad, shares, price)]}"""
    sub, own, tr = [], [], []
    sk = 1
    for f in filings:
        sub.append([f["acc"], sec_date(f["date"]), sec_date(f["date"]), f.get("doc", "4"), f["cik"], "X CORP",
                    f["symbol"]])
        for cik, rel in f["owners"]:
            own.append([f["acc"], cik, "SOMEONE", rel, ""])
        for code, ad, shares, price in f.get("trans", []):
            tr.append([f["acc"], str(sk), "Common Stock", sec_date(f["date"]), code, str(shares), str(price), ad, "D"])
            sk += 1
    with zipfile.ZipFile(path, "w") as z:
        for name, cols, rows in (("SUBMISSION.tsv", SUB_COLS, sub), ("REPORTINGOWNER.tsv", OWN_COLS, own),
                                 ("NONDERIV_TRANS.tsv", TR_COLS, tr)):
            z.writestr(name, "\n".join("\t".join(r) for r in [cols] + rows) + "\n")


def quarter_of(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.year}q{(d.month - 1) // 3 + 1}"


def store_sec(root: Path, filings: list[dict], first="2009q3", last="2021q2") -> None:
    d = sec_insider.sec_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    qs = sec_insider.quarter_list(first, pd.Timestamp(last[:4] + "-12-31").date())
    qs = qs[: qs.index(last) + 1]
    for q in qs:
        path = d / f"{q}_form345.zip"
        write_quarter_zip(path, [f for f in filings if quarter_of(f["date"]) == q])
        meta = {"quarter": q, "url": "fixture", "sha256": file_sha256(path), "bytes": path.stat().st_size,
                "fetched_at": "20261016T000000000000Z", "path": str(path)}
        (d / f"{q}.meta.json").write_text(json.dumps(meta), encoding="utf-8")


def store_bundle(root: Path, prices: pd.DataFrame, members: pd.DataFrame) -> dict:
    d = sp500.bundle_dir(root) / "20261016T000000000000Z"
    d.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(d / "prices.parquet", index=False)
    members.to_parquet(d / "members.parquet", index=False)
    meta = {"source": sp500.SOURCE, "url": "fixture", "fetched_at": "20261016T000000000000Z", "dir": str(d),
            "files": {n: {"sha256": file_sha256(d / n), "bytes": (d / n).stat().st_size} for n in sp500.FILES}}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def bundle_frames(sessions: pd.DatetimeIndex, spans: dict[str, tuple], seed=3, crsp=()) -> pd.DataFrame:
    """Long price table like prices.parquet. spans: ticker -> (first, last) price dates.
    Tickers in `crsp` have no open (like the CRSP-derived rows)."""
    rng = np.random.default_rng(seed)
    rows = []
    for t, (a, b) in spans.items():
        idx = sessions[(sessions >= pd.Timestamp(a)) & (sessions <= pd.Timestamp(b))]
        close = 40 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(idx))))
        opn = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.003, len(idx)))
        adj = close * np.linspace(0.8, 1.0, len(idx))  # dividends folded into the adjusted close
        rows.append(pd.DataFrame({"date": idx, "ticker": t, "open": np.nan if t in crsp else opn,
                                  "high": np.nan if t in crsp else np.fmax(opn, close) * 1.003,
                                  "low": np.nan if t in crsp else np.fmin(opn, close) * 0.997, "close": close,
                                  "volume": 1e6, "adj_close": adj, "dividends": 0.0, "stock_splits": 0.0,
                                  "source": "crsp" if t in crsp else "yahoo"}))
    return pd.concat(rows, ignore_index=True)


def store_submissions(root: Path, companies: dict[str, list[tuple]], page_size: int = 3) -> None:
    """companies: cik -> [(filingDate, form, items)]. Older filings go to extra pages, as for big filers."""
    import gzip

    from quantlab.data import sec_earnings

    d = sec_earnings.store_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    for cik, filings in companies.items():
        filings = sorted(filings, key=lambda f: f[0], reverse=True)

        def cols(rows, start):
            return {"accessionNumber": [f"{cik}-{start + i}" for i in range(len(rows))],
                    "filingDate": [r[0] for r in rows], "form": [r[1] for r in rows], "items": [r[2] for r in rows],
                    "acceptanceDateTime": [f"{r[0]}T16:05:00.000Z" for r in rows]}

        recent, older = filings[:page_size], filings[page_size:]
        pages_older = [older[i:i + page_size] for i in range(0, len(older), page_size)]
        main = {"cik": cik, "filings": {"recent": cols(recent, 0),
                                        "files": [{"name": f"CIK{int(cik):010d}-submissions-{k:03d}.json"}
                                                  for k in range(len(pages_older))]}}
        pages = [main] + [cols(p, 1000 * (k + 1)) for k, p in enumerate(pages_older)]
        path = d / f"CIK{int(cik):010d}.json.gz"
        with gzip.open(path, "wb") as g:
            g.write(json.dumps(pages).encode())
        meta = {"cik": cik, "sha256": file_sha256(path), "pages": len(pages), "path": str(path),
                "fetched_at": "20261016T000000000000Z"}
        (d / f"CIK{int(cik):010d}.meta.json").write_text(json.dumps(meta), encoding="utf-8")


def facts_doc(cik: str, years, assets, equity, gross=None, revenue_cost=None, float_=None, filed_day="02-20",
              amend: dict | None = None) -> dict:
    """A companyfacts JSON in the SEC's layout: one 10-K per fiscal year (calendar years), filed on
    `filed_day` of the next year, plus a 10-Q with a quarter's gross profit (not annual: ignored).
    Values are callables of the year. `amend`: {year: (filed date, assets)} restates assets later."""
    def fact(rows):
        return {"units": {"USD": rows}}

    a, e, g, r, c, fl = [], [], [], [], [], []
    for y in years:
        filed = f"{y + 1}-{filed_day}"
        base = {"filed": filed, "form": "10-K", "fy": y, "fp": "FY", "accn": f"{cik}-{y}"}
        a.append(base | {"end": f"{y}-12-31", "val": assets(y)})
        e.append(base | {"end": f"{y}-12-31", "val": equity(y)})
        if gross:
            g.append(base | {"start": f"{y}-01-01", "end": f"{y}-12-31", "val": gross(y)})
            g.append(base | {"start": f"{y}-10-01", "end": f"{y}-12-31", "val": gross(y) / 4, "form": "10-K"})
        if revenue_cost:
            rv, cs = revenue_cost(y)
            r.append(base | {"start": f"{y}-01-01", "end": f"{y}-12-31", "val": rv})
            c.append(base | {"start": f"{y}-01-01", "end": f"{y}-12-31", "val": cs})
        if float_:
            fl.append(base | {"end": f"{y}-06-30", "val": float_(y)})
    for y, (filed, val) in (amend or {}).items():
        a.append({"filed": filed, "form": "10-K/A", "fy": y, "fp": "FY", "accn": f"{cik}-{y}-A",
                  "end": f"{y}-12-31", "val": val})
    gaap = {"Assets": fact(a), "StockholdersEquity": fact(e)}
    if g:
        gaap["GrossProfit"] = fact(g)
    if r:
        gaap["Revenues"], gaap["CostOfRevenue"] = fact(r), fact(c)
    return {"cik": int(cik), "facts": {"us-gaap": gaap, "dei": {"EntityPublicFloat": fact(fl)} if fl else {}}}


def store_facts(root: Path, docs: dict[str, dict]) -> None:
    import gzip

    from quantlab.data import sec_facts

    d = sec_facts.store_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    for cik, doc in docs.items():
        path = d / f"CIK{int(cik):010d}.json.gz"
        with gzip.open(path, "wb") as g:
            g.write(json.dumps(doc).encode())
        meta = {"cik": cik, "sha256": file_sha256(path), "path": str(path), "fetched_at": "20261022T000000000000Z",
                "empty": False}
        (d / f"CIK{int(cik):010d}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
