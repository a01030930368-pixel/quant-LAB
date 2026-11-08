"""Korean 5% holding reports (대량보유 상황보고) from OpenDART, as events (D-047).

Anyone who comes to hold 5% or more of a listed company's shares, and then every change of 1%p or more,
must report it. Institutions (pension funds, asset managers, banks, insurers) holding for investment may
file the short form (약식), due by the 10th of the month after the change, so the news can be up to
about six weeks old when it is filed; owners, companies and activists file the general form (일반)
within five days. Per company (the user's DART key, about 20,000 calls a day):

    GET /api/majorstock.json?crtfc_key=..&corp_code=..
    -> {"status": "000", "list": [{rcept_no, rcept_dt, corp_code, corp_name, report_tp (보고구분),
        repror (대표보고자), stkqy, stkqy_irds, stkrt (보유비율 %), stkrt_irds (증감 %p), ctr_stkqy,
        ctr_stkrt, report_resn (보고사유)}, ...]}

The layout is OpenDART's published one (지분공시 종합정보 > 대량보유 상황보고); the first download on
the user's PC is the real format check, and the counts of each report form are kept in the manifest.
Answers are stored apart from the KRDART files (data/bulk/dart-block/), so those datasets keep their
versions.

Events, known on the receipt date:
- `kr_block_buy`: a short-form (institutional) report whose holding went up (stkrt_irds > 0); value =
  the increase in %p, owner = the reporter.
- `kr_block_new`: the same report when it is the first at 5% or more (stkrt - stkrt_irds < 5).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.core.hashing import file_sha256, stable_hash
from quantlab.data import dart_open

ENDPOINT = "majorstock.json"
PARSE_VERSION = 1
SHORT_FORM = "약식"
THRESHOLD = 5.0
KINDS = ("kr_block_buy", "kr_block_new")


def store_dir(root: str | Path) -> Path:
    return Path(root) / "data" / "bulk" / "dart-block"


def _path(root, corp: str) -> Path:
    return store_dir(root) / "majorstock" / f"{corp}.json.gz"


def _get(http, key: str, corp: str, timeout: float) -> list[dict]:
    import time

    resp = http.get(dart_open.API + ENDPOINT, params={"crtfc_key": key, "corp_code": corp}, timeout=timeout)
    time.sleep(dart_open.PAUSE_SECONDS)
    if resp.status_code != 200:
        raise dart_open.DartError(f"DART majorstock: HTTP {resp.status_code}: {resp.content[:200]!r}")
    try:
        doc = json.loads(resp.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise dart_open.DartError(f"DART majorstock: 응답을 읽을 수 없습니다 {resp.content[:200]!r}") from e
    return dart_open._check(doc, "majorstock")


def download(root: str | Path, key: str, codes, today=None, session=None, say=None, timeout: float = 30.0) -> dict:
    """The 5% reports of the companies of `codes` (6-digit stock codes) not stored yet.
    Returns {"fetched", "left", "stopped", "unmapped"}."""
    import requests

    say = say or (lambda s: None)
    http = session or requests.Session()
    today = pd.Timestamp(today or pd.Timestamp.now(tz="Asia/Seoul").date())
    mapping = dart_open.corp_codes(root, key, session=http)
    corps = sorted({mapping[c] for c in codes if c in mapping})
    todo = [c for c in corps if not _path(root, c).exists()]
    fetched, stopped = 0, None
    for i, c in enumerate(todo):
        if i % 50 == 0:
            say(f"DART 5% 대량보유 공시 받는 중: {i + 1}/{len(todo)}")
        try:
            rows = _get(http, key, c, timeout)
        except dart_open.DailyLimit as err:
            stopped = str(err)
            break
        p = _path(root, c)
        p.parent.mkdir(parents=True, exist_ok=True)
        dart_open._write(p, {"asked": f"{today:%Y-%m-%d}", "corp_code": c, "list": rows})
        fetched += 1
    left = sum(1 for c in todo if not _path(root, c).exists())
    return {"fetched": fetched, "left": left, "stopped": stopped,
            "unmapped": sorted(c for c in codes if c not in mapping)}


def _read(p: Path) -> list[dict]:
    with gzip.open(p, "rb") as g:
        return json.loads(g.read().decode("utf-8"))["list"]


def events(root: str | Path, codes) -> tuple[pd.DataFrame, dict]:
    """Events of `codes` (dart_open.EVENT_COLUMNS) and counts of what was read."""
    stored = dart_open.store_dir(root) / "corpCode.json.gz"
    mapping = dart_open.corp_codes(root, key="") if stored.exists() else {}
    by_corp: dict[str, str] = {}
    for code in codes:
        if code in mapping:
            by_corp.setdefault(mapping[code], code)
    rows = []
    counts = {"companies": 0, "reports": 0, "short_form": 0, "general_form": 0, "other_form": 0, "up": 0,
              "new": 0, "no_date": 0}
    for corp, code in by_corp.items():
        p = _path(root, corp)
        if not p.exists():
            continue
        counts["companies"] += 1
        for r in _read(p):
            counts["reports"] += 1
            form = str(r.get("report_tp") or "")
            if SHORT_FORM in form:
                counts["short_form"] += 1
            elif "일반" in form:
                counts["general_form"] += 1
            else:
                counts["other_form"] += 1
            day = dart_open._day(r.get("rcept_dt")) or dart_open._day(str(r.get("rcept_no", ""))[:8])
            if day is None:
                counts["no_date"] += 1
                continue
            if SHORT_FORM not in form:
                continue
            change, held = dart_open._num(r.get("stkrt_irds")), dart_open._num(r.get("stkrt"))
            if not np.isfinite(change) or change <= 0:
                continue
            owner = str(r.get("repror") or "").strip()
            counts["up"] += 1
            rows.append((day, code, KINDS[0], change, False, False, owner))
            if np.isfinite(held) and held >= THRESHOLD and held - change < THRESHOLD:
                counts["new"] += 1
                rows.append((day, code, KINDS[1], change, False, False, owner))
    ev = pd.DataFrame(rows, columns=dart_open.EVENT_COLUMNS)
    ev["known"] = pd.to_datetime(ev["known"])
    ev = ev.sort_values(["known", "symbol", "kind"], kind="stable").reset_index(drop=True)
    counts["mapped"] = len(by_corp)
    return ev, counts


def inputs(root: str | Path) -> str:
    files = {p.name: file_sha256(p) for p in sorted((store_dir(root) / "majorstock").glob("*.json.gz"))}
    return stable_hash({"parse": PARSE_VERSION, "files": files})
