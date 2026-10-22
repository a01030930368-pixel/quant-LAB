"""Korean stocks from the Korea Exchange's KRX Open API: one snapshot of every listed stock per session
and market, since 2010 (D-038).

Source: https://openapi.krx.co.kr (free; the user's own key, applied for per service). One call per
session and market returns every stock listed that day, so names delisted later are in the history
(no survivorship bias):

    GET https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd?basDd=YYYYMMDD    (KOSPI)
    GET https://data-dbg.krx.co.kr/svc/apis/sto/ksq_bydd_trd?basDd=YYYYMMDD    (KOSDAQ)
    header AUTH_KEY: <key>
    -> {"OutBlock_1": [{BAS_DD, ISU_CD, ISU_NM, MKT_NM, SECT_TP_NM, TDD_CLSPRC, CMPPREVDD_PRC, FLUC_RT,
                        TDD_OPNPRC, TDD_HGPRC, TDD_LWPRC, ACC_TRDVOL, ACC_TRDVAL, MKTCAP, LIST_SHRS}, ...]}

The layout was taken from public clients of the API (krx-cli, korea-stock-mcp); the first download on
the user's PC is the real format check. The daily limit is about 10,000 calls; the whole history of
both markets is about 8,300, and a download that hits the limit stops and goes on in a later run.
Each response is stored gzip-compressed as received, one file per session and market, and never
changes afterwards. Prices are the traded ones (not adjusted); `quantlab.data.kr_stocks` turns them
into adjusted prices with the exchange's own base price (close minus CMPPREVDD_PRC).
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.core.hashing import file_sha256, stable_hash

API = "https://data-dbg.krx.co.kr/svc/apis/sto/"
MARKETS = {"KOSPI": "stk_bydd_trd", "KOSDAQ": "ksq_bydd_trd"}
FIRST_DAY = "2010-01-04"
PAUSE_SECONDS = 0.05
RECENT_DAYS = 10          # an empty answer this close to today may only mean "not published yet"
PARSE_VERSION = 2          # 2: typed columns (empty days made every column an object)
NUMERIC = {"TDD_OPNPRC": "open", "TDD_HGPRC": "high", "TDD_LWPRC": "low", "TDD_CLSPRC": "close",
           "CMPPREVDD_PRC": "change", "ACC_TRDVOL": "volume", "ACC_TRDVAL": "value", "MKTCAP": "mktcap",
           "LIST_SHRS": "shares"}
COLUMNS = ["date", "market", "code", "name", "section", "open", "high", "low", "close", "change", "volume",
           "value", "mktcap", "shares"]


def _empty() -> pd.DataFrame:
    """No rows, but the columns' types: an untyped empty frame turns every column of a concat into objects."""
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in COLUMNS})
    df["date"] = pd.Series(dtype="datetime64[ns]")
    for c in ("market", "code", "name", "section"):
        df[c] = pd.Series(dtype="object")
    return df


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    for c in NUMERIC.values():
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


class KrxError(RuntimeError):
    pass


class DailyLimit(KrxError):
    pass


def store_dir(root: str | Path) -> Path:
    return Path(root) / "data" / "bulk" / "krx-open"


def _path(root, market: str, day: pd.Timestamp) -> Path:
    return store_dir(root) / market / f"{day.year}" / f"{day:%Y%m%d}.json.gz"


def stored(root: str | Path, market: str) -> dict[pd.Timestamp, Path]:
    out = {}
    for p in sorted((store_dir(root) / market).glob("*/*.json.gz")):
        out[pd.Timestamp(p.name[:8])] = p
    return out


def sessions(start: str, end) -> pd.DatetimeIndex:
    from quantlab.data.quality import exchange_sessions

    return exchange_sessions(pd.Timestamp(start), pd.Timestamp(end), "XKRX")


def _rows(body: bytes) -> list[dict]:
    try:
        doc = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise KrxError(f"KRX 응답을 읽을 수 없습니다: {body[:200]!r}") from e
    rows = doc.get("OutBlock_1") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        msg = (doc.get("respMsg") or doc.get("message")) if isinstance(doc, dict) else None
        msg = msg or str(doc)[:200]
        raise KrxError(f"KRX 응답 형식이 예상과 다릅니다: {msg}")
    return rows


def _get(http, market: str, day: pd.Timestamp, key: str, timeout: float) -> list[dict]:
    resp = http.get(API + MARKETS[market], params={"basDd": f"{day:%Y%m%d}"},
                    headers={"AUTH_KEY": key, "Accept": "application/json"}, timeout=timeout)
    time.sleep(PAUSE_SECONDS)
    text = resp.content[:400].decode("utf-8", errors="replace")
    if resp.status_code == 429 or "limit" in text.lower() and resp.status_code != 200:
        raise DailyLimit("KRX 하루 호출 한도에 닿았습니다. 내일 다시 실행하면 이어서 받습니다.")
    if resp.status_code in (401, 403):
        raise KrxError("KRX Open API가 키를 거절했습니다. 키가 맞는지, 유가증권·코스닥 일별매매정보 서비스가 "
                       f"승인됐는지 openapi.krx.co.kr 마이페이지에서 확인해 주세요. ({resp.status_code}: {text[:120]})")
    if resp.status_code != 200:
        raise KrxError(f"KRX {market} {day.date()}: HTTP {resp.status_code}: {text[:200]}")
    return _rows(resp.content)


def download(root: str | Path, key: str, end, start: str = FIRST_DAY, session=None, say=None,
             timeout: float = 30.0, today=None) -> dict:
    """Fetch every XKRX session from `start` to `end` not stored yet, for both markets. Returns
    {"fetched", "left", "stopped"}: `stopped` names why the run ended early (the daily limit)."""
    import requests

    say = say or (lambda s: None)
    http = session or requests.Session()
    today = pd.Timestamp(today or pd.Timestamp.now(tz="Asia/Seoul").date())
    days = sessions(start, end)
    todo = [(m, d) for m in MARKETS for d in days if not _path(root, m, d).exists()]
    fetched, stopped = 0, None
    for i, (m, d) in enumerate(todo):
        if i % 50 == 0:
            say(f"한국거래소 자료 받는 중: {i + 1}/{len(todo)} ({m} {d:%Y-%m-%d})")
        try:
            rows = _get(http, m, d, key, timeout)
        except DailyLimit as e:
            stopped = str(e)
            break
        if not rows and (today - d).days <= RECENT_DAYS:
            continue  # not published yet; asked again next time
        p = _path(root, m, d)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        with gzip.open(tmp, "wb") as g:
            g.write(json.dumps({"OutBlock_1": rows}, ensure_ascii=False).encode("utf-8"))
        tmp.replace(p)
        fetched += 1
    left = sum(1 for m, d in todo if not _path(root, m, d).exists())
    return {"fetched": fetched, "left": left, "stopped": stopped}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip()
                         .replace({"-": None, "": None}), errors="coerce")


def parse_day(path: str | Path, market: str) -> pd.DataFrame:
    with gzip.open(path, "rb") as g:
        rows = _rows(g.read())
    day = pd.Timestamp(Path(path).name[:8])
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    if "ISU_CD" not in df or "TDD_CLSPRC" not in df:
        raise KrxError(f"KRX {market} {day.date()}: 종목코드·종가 열이 없습니다 (형식이 바뀐 것 같습니다)")
    out = pd.DataFrame({"date": day, "market": market}, index=df.index)
    code = df["ISU_CD"].astype(str).str.strip()
    # a 12-character ISIN (KR7005930003) carries the 6-character short code at [3:9]
    out["code"] = np.where(code.str.len() == 12, code.str[3:9], code)
    out["name"] = df.get("ISU_NM", pd.Series("", index=df.index)).astype(str)
    out["section"] = df.get("SECT_TP_NM", pd.Series("", index=df.index)).fillna("").astype(str)
    for src, dst in NUMERIC.items():
        out[dst] = _num(df[src]) if src in df else np.nan
    return out[COLUMNS]


def load(root: str | Path, start=None, end=None, say=None) -> tuple[pd.DataFrame, dict]:
    """Every stored snapshot as one long table (COLUMNS), and the input record (sha256 per file).
    Parsed years are cached beside the files, keyed by their contents."""
    say = say or (lambda s: None)
    parts, inputs = [], {}
    for m in MARKETS:
        files = stored(root, m)
        years = sorted({d.year for d in files if (start is None or d >= pd.Timestamp(start))
                        and (end is None or d <= pd.Timestamp(end))})
        for y in years:
            fs = {d: p for d, p in files.items() if d.year == y}
            shas = {f"{d:%Y%m%d}": file_sha256(p) for d, p in sorted(fs.items())}
            inputs[f"{m}/{y}"] = stable_hash(shas)
            cache = store_dir(root) / m / f"{y}.v{PARSE_VERSION}.{inputs[f'{m}/{y}'][:16]}.parquet"
            if cache.exists():
                parts.append(pd.read_parquet(cache))
                continue
            say(f"한국거래소 자료 정리 중: {m} {y}")
            frames = [f for f in (parse_day(p, m) for _, p in sorted(fs.items())) if len(f)]
            year = _typed(pd.concat(frames, ignore_index=True)) if frames else _empty()
            year.to_parquet(cache, index=False)
            parts.append(year)
    parts = [p for p in parts if len(p)]
    df = _typed(pd.concat(parts, ignore_index=True)) if parts else _empty()
    if len(df):
        if start is not None:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True), inputs


def missing_sessions(root: str | Path, start, end) -> dict[str, list[str]]:
    """XKRX sessions in [start, end] without a stored snapshot, per market."""
    days = sessions(start, end)
    return {m: [str(d.date()) for d in days if not _path(root, m, d).exists()] for m in MARKETS}
