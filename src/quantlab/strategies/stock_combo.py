"""Building blocks for the automatic search on index stocks (rules block `stock_combo`, D-028).

A `stock_combo` spec trades the stocks of a point-in-time index with one rule: buy a member stock at
the next open on the session its chosen conditions all become true (they were not all true the
session before), hold `hold` sessions unless a close-based stop or take fires first.

    universe: {market: US_STOCK, index: SP500, symbols: [SPY, BIL]}
    rules:
      kind: declarative
      stock_combo:
        catalog: 1
        park: SPY            # equity not in a position is held here (refilled in batches, as PEAD-SP500)
        park_band: 2
    parameters:
      entry: {default: earn5, search: [earn5, earn5+ins1, ...]}   # 1 to 3 conditions from different groups
      hold:  {default: 20, search: [5, 10, 20, 30, 60]}
      stop:  {default: 0, search: [0, 0.1, 0.2]}
      take:  {default: 0, search: [0, 0.1, 0.2]}
      slots: {default: 10, search: [10, 20]}

Every condition is a whole-dataset table built once by `build_panels` from data up to each row only
(rolling windows, earlier events); a strategy reads it one row at a time through `MarketView.panel`.
The catalog is part of the app code: a new meaning needs a new catalog number.

When more stocks signal than there are free slots, a fixed pseudo-random order picks them (seeded,
the same on every run), so no condition decides the order of another.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from quantlab.strategies.base import MarketView, PartialWeights, Strategy

CATALOGS = (1, 2, 3, 4, 5)  # 2 adds 5 slots (D-029); 3 conditions and exits (D-033); 4 index changes, insider sales,
                           # the weeks before earnings (D-034); 5 Korean stocks (D-038); older ones keep their meaning
BENCHMARK = "SPY"
KR_BENCHMARK = "069500"     # KODEX 200: market filters and trade comparison of catalog 5
KR_LIMIT_CHANGE = pd.Timestamp("2015-06-15")  # KRX daily price limit ±15% before, ±30% from this day
PARAMS = ("entry", "hold", "stop", "take", "slots")
KEYS = ("catalog", "park", "park_band")
ORDER_SEED = 20261016
EARN_DAYS = 5          # an earnings reaction counts for this many sessions, its own included
INSIDER_DAYS = 30      # calendar days of insider filings, as INS-CLUSTER
INSIDER_MIN_VALUE = 10_000
INSIDER_BIG_VALUE = 250_000  # catalog 3: one officer or director buying at least this much
SELL_BIG_VALUE = 100_000     # catalog 4: a sale counts toward "3 insiders selling" from this size
NOSELL_DAYS = 90             # catalog 4: no officer or director sale (>= INSIDER_MIN_VALUE) in this many days
INDEX_CHANGE_SESSIONS = 20   # catalog 4: joined or left the index within this many sessions
PRE_EARN_DAYS = (80, 88)     # catalog 4: calendar days since the last earnings filing (next one ~91 days on)

# key -> (Korean label, group, first catalog); groups in this order, items in a group from mild to strict.
# A later catalog only adds items, so a catalog's list is the items with `since <= catalog`, in order.
_CONDITIONS: dict[str, tuple[str, str, int]] = {
    "earn3": ("실적 발표 반응이 SPY보다 +3% 이상", "earn", 1),
    "earn5": ("실적 발표 반응이 SPY보다 +5% 이상", "earn", 1),
    "earn8": ("실적 발표 반응이 SPY보다 +8% 이상", "earn", 1),
    "earnneg5": ("실적 발표 반응이 SPY보다 -5% 이하", "earn", 3),
    "earnneg8": ("실적 발표 반응이 SPY보다 -8% 이하", "earn", 3),
    "ins1": ("임원·이사 1명 이상 매수 (30일)", "ins", 1),
    "ins2": ("임원·이사 2명 이상 매수 (30일)", "ins", 1),
    "insbig": ("임원·이사 한 명이 25만 달러 이상 매수 (30일)", "ins", 3),
    "drop5": ("5일 수익률 -5% 이하", "drop", 1),
    "drop10": ("5일 수익률 -10% 이하", "drop", 1),
    "rsi14_30": ("RSI(14) < 30", "rsi", 1),
    "rsi2_10": ("RSI(2) < 10", "rsi", 1),
    "gap3": ("시가가 전날 종가보다 +3% 이상", "gap", 1),
    "gap5": ("시가가 전날 종가보다 +5% 이상", "gap", 1),
    "gapdn3": ("시가가 전날 종가보다 -3% 이하", "gap", 3),
    "gapdn5": ("시가가 전날 종가보다 -5% 이하", "gap", 3),
    "up200": ("200일선 위", "trend", 1),
    "down200": ("200일선 아래", "trend", 1),
    "spy_up200": ("SPY가 200일선 위", "spy", 1),
    "spy_down200": ("SPY가 200일선 아래", "spy", 3),
    "spy_drop5": ("SPY 5일 수익률 -5% 이하", "spy", 3),
    "high252": ("52주 신고가", "high", 1),
    "low252": ("52주 신저가", "high", 3),
    "drop1_4": ("하루 수익률 -4% 이하", "drop1", 3),
    "drop1_7": ("하루 수익률 -7% 이하", "drop1", 3),
    "drop20_15": ("20일 수익률 -15% 이하", "drop20", 3),
    "dn3": ("3일 연속 하락", "streak", 3),
    "dn5": ("5일 연속 하락", "streak", 3),
    "squeeze": ("최근 20일 변동성이 120일 변동성의 60% 미만", "vol", 3),
    "idxadd": ("최근 20거래일 안에 S&P 500에 새로 편입", "index", 4),
    "idxdel": ("최근 20거래일 안에 S&P 500에서 빠짐", "index", 4),
    "inssell3": ("임원·이사 3명 이상이 10만 달러 이상 매도 (30일)", "sell", 4),
    "nosell": ("임원·이사 매도 없음 (90일)", "sell", 4),
    "preearn": ("지난 실적 발표 후 80~88일 (다음 발표 1~2주 전)", "pre", 4),
    "mkt_up200": ("KODEX 200이 200일선 위", "mkt", 5),
    "mkt_down200": ("KODEX 200이 200일선 아래", "mkt", 5),
    "mkt_drop5": ("KODEX 200 5일 수익률 -5% 이하", "mkt", 5),
    "uplimit": ("상한가 마감 (가격제한폭까지 상승)", "limit", 5),
    "dnlimit": ("하한가 마감 (가격제한폭까지 하락)", "limit", 5),
}
# catalog 5 (Korean stocks, no filings yet): the price items of catalog 3 without the SPY ones, and its own
KR_FROM_CATALOG3 = ("drop", "rsi", "gap", "trend", "high", "drop1", "drop20", "streak", "vol")
CONDITIONS: dict[str, tuple[str, str]] = {k: v[:2] for k, v in _CONDITIONS.items()}
GROUPS = ("earn", "ins", "drop", "rsi", "gap", "trend", "spy", "high", "drop1", "drop20", "streak", "vol",
          "index", "sell", "pre", "mkt", "limit")
# adjacent thresholds of the same measure (neighbours); rsi and trend items are different measures
LADDERS = (("earn3", "earn5", "earn8"), ("earnneg5", "earnneg8"), ("ins1", "ins2"), ("drop5", "drop10"),
           ("gap3", "gap5"), ("gapdn3", "gapdn5"), ("drop1_4", "drop1_7"), ("dn3", "dn5"))
_HOLDS = {1: (5, 10, 20, 30, 60), 2: (5, 10, 20, 30, 60), 3: (3, 5, 10, 20, 30, 60), 4: (3, 5, 10, 20, 30, 60),
          5: (3, 5, 10, 20, 30, 60)}
_STOPS = {1: (0, 0.1, 0.2), 2: (0, 0.1, 0.2), 3: (0, 0.1, 0.2), 4: (0, 0.1, 0.2), 5: (0, 0.1, 0.2)}
_TAKES = {1: (0, 0.1, 0.2), 2: (0, 0.1, 0.2), 3: (0, 0.1, 0.2), 4: (0, 0.1, 0.2), 5: (0, 0.1, 0.2)}
HOLDS, STOPS, TAKES = _HOLDS[1], _STOPS[1], _TAKES[1]
SLOTS = {1: (10, 20), 2: (5, 10, 20), 3: (5, 10, 20), 4: (5, 10, 20), 5: (5, 10, 20)}


def _in_catalog(key: str, catalog: int) -> bool:
    _, group, since = _CONDITIONS[key]
    if catalog == 5:
        return since == 5 or (since <= 3 and group in KR_FROM_CATALOG3)
    return since <= catalog and since < 5


def conditions(catalog: int = 1) -> list[str]:
    return [k for k in _CONDITIONS if _in_catalog(k, catalog)]


def benchmark(catalog: int) -> str:
    return KR_BENCHMARK if catalog == 5 else BENCHMARK


def entries(catalog: int = 1) -> list[str]:
    """Every entry key: 1 to 3 conditions from different groups, joined with '+' in catalog order."""
    keys = conditions(catalog)
    out = []
    for n in (1, 2, 3):
        for combo in itertools.combinations(keys, n):
            if len({CONDITIONS[k][1] for k in combo}) == n:
                out.append("+".join(combo))
    return out


def full_space(catalog: int = 1) -> dict[str, list]:
    return {"entry": entries(catalog), "hold": list(_HOLDS[catalog]), "stop": list(_STOPS[catalog]),
            "take": list(_TAKES[catalog]), "slots": list(SLOTS[catalog])}


def already_tried(params: dict, catalog: int) -> bool:
    """A combination an earlier search ran (its trials are counted as prior trials), so the newer search
    skips it instead of running it twice: in catalog 3, one made only of catalog-1 items and exits
    (AUTO-SP500); in catalog 4, one made only of catalog-3 items and exits (AUTO-SP500 and AUTO-SP500X)."""
    if catalog < 3 or catalog == 5:  # catalog 5 is a new market
        return False
    prev = 1 if catalog == 3 else 3
    return (all(_CONDITIONS[c][2] <= prev for c in str(params["entry"]).split("+"))
            and params["hold"] in _HOLDS[prev] and params["stop"] in _STOPS[prev] and params["take"] in _TAKES[prev]
            and params["slots"] in SLOTS[prev])


def catalog_of(rules: dict) -> int:
    return int(rules.get("catalog", 1))


def validate_stock_combo(cfg: Any, symbols: list[str]) -> list[str]:
    if not isinstance(cfg, dict):
        return ["rules.stock_combo must be a mapping"]
    errs = []
    extra = set(cfg) - set(KEYS)
    if extra:
        errs.append(f"rules.stock_combo: unknown keys {sorted(extra)}")
    if cfg.get("catalog") not in CATALOGS:
        errs.append(f"rules.stock_combo.catalog must be one of {CATALOGS}")
    if cfg.get("park") is not None and cfg["park"] not in symbols:
        errs.append("rules.stock_combo.park must be one of universe.symbols")
    v = cfg.get("park_band", 1)
    if not isinstance(v, int) or isinstance(v, bool) or v < 1:
        errs.append("rules.stock_combo.park_band must be a whole number >= 1")
    bench = benchmark(cfg.get("catalog", 1)) if cfg.get("catalog") in CATALOGS else BENCHMARK
    if bench not in symbols:
        errs.append(f"stock_combo rules read {bench} (earnings reaction, market trend): add it to universe.symbols")
    return errs


def validate_params(parameters: dict, catalog: int = 1) -> list[str]:
    if catalog not in CATALOGS:
        return []
    errs = []
    space = full_space(catalog)
    allowed = {k: set(v) for k, v in space.items()}
    for name in PARAMS:
        p = parameters.get(name)
        if not isinstance(p, dict) or "default" not in p:
            errs.append(f"parameters.{name} is required for stock_combo rules")
            continue
        bad = [v for v in [p["default"]] + list(p.get("search") or []) if v not in allowed[name]]
        if bad:
            errs.append(f"parameters.{name}: {bad[:3]} not in stock catalog {catalog}")
    extra = set(parameters) - set(PARAMS)
    if extra:
        errs.append(f"stock_combo rules take only {PARAMS}; unknown {sorted(extra)}")
    return errs


def describe(params: dict) -> str:
    parts = [CONDITIONS[k][0] for k in params["entry"].split("+")]
    parts.append(f"{params['hold']}일 보유")
    if params.get("stop"):
        parts.append(f"손절 {params['stop'] * 100:g}%")
    if params.get("take"):
        parts.append(f"익절 {params['take'] * 100:g}%")
    parts.append(f"{params['slots']}종목")
    return " · ".join(parts)


def neighbours(params: dict, catalog: int = 1) -> list[dict]:
    """One step away: the adjacent threshold of one condition, or the adjacent holding period."""
    out = []
    conds = params["entry"].split("+")
    order = list(CONDITIONS)
    for i, c in enumerate(conds):
        for ladder in LADDERS:
            if c in ladder:
                j = ladder.index(c)
                for k in (j - 1, j + 1):
                    if 0 <= k < len(ladder):
                        new = sorted(conds[:i] + [ladder[k]] + conds[i + 1:], key=order.index)
                        out.append(dict(params, entry="+".join(new)))
    holds = _HOLDS.get(catalog, HOLDS)
    h = holds.index(params["hold"]) if params["hold"] in holds else None
    if h is not None:
        for k in (h - 1, h + 1):
            if 0 <= k < len(holds):
                out.append(dict(params, hold=holds[k]))
    return out


# ----------------------------------------------------------------------------- tables
def _adjusted(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split-adjusted open and close (ratios between rows are what a holder saw)."""
    k = np.nan_to_num(values[:, 5], nan=1.0)
    later = np.concatenate([np.cumprod(k[::-1])[::-1][1:], [1.0]])
    return values[:, 0] / later, values[:, 3] / later


def _wilder_rsi(c: np.ndarray, n: int) -> np.ndarray:
    """RSI on every column of `c` (rows: sessions); NaN rows of a column are skipped, as in
    `quantlab.strategies.indicators`."""
    from quantlab.strategies.indicators import _wilder

    out = np.full(c.shape, np.nan)
    for j in range(c.shape[1]):
        rows = np.flatnonzero(np.isfinite(c[:, j]))
        if len(rows) > n:
            out[rows, j] = _wilder(c[rows, j], n)
    return out


def build_panels(data) -> dict:
    """Condition tables (sessions x index stocks), membership, close and open, and the pick order.
    Row t uses prices up to t and events known by t only."""
    cal = data.calendar
    syms = [s for s in data.members.columns if s in data.frames] if data.members is not None else []
    T, N = len(cal), len(syms)
    O = np.full((T, N), np.nan)
    C = np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        O[:, j], C[:, j] = _adjusted(data.values(s))
    _, spy = _adjusted(data.values(BENCHMARK if BENCHMARK in data.frames else KR_BENCHMARK))
    col = {s: j for j, s in enumerate(syms)}
    cdf = pd.DataFrame(C)

    def shifted(a: np.ndarray, n: int) -> np.ndarray:
        out = np.full_like(a, np.nan)
        out[n:] = a[:-n]
        return out

    with np.errstate(invalid="ignore", divide="ignore"):
        ret5 = C / shifted(C, 5) - 1
        gap = O / shifted(C, 1) - 1
        sma200 = cdf.rolling(200, min_periods=200).mean().to_numpy()
        high252 = cdf.rolling(252, min_periods=252).max().to_numpy()
        spy_s = pd.Series(spy)
        spy_up = (spy_s / spy_s.rolling(200, min_periods=200).mean() - 1).to_numpy() > 0
    p: dict[str, Any] = {}
    p["drop5"], p["drop10"] = ret5 <= -0.05, ret5 <= -0.10
    p["gap3"], p["gap5"] = gap >= 0.03, gap >= 0.05
    with np.errstate(invalid="ignore", divide="ignore"):
        ret1 = C / shifted(C, 1) - 1
        ret20 = C / shifted(C, 20) - 1
        low252 = cdf.rolling(252, min_periods=252).min().to_numpy()
        r1 = pd.DataFrame(ret1)
        vol20 = r1.rolling(20, min_periods=20).std().to_numpy()
        vol120 = r1.rolling(120, min_periods=120).std().to_numpy()
        spy5 = spy_s / spy_s.shift(5) - 1
    p["gapdn3"], p["gapdn5"] = gap <= -0.03, gap <= -0.05
    p["drop1_4"], p["drop1_7"] = ret1 <= -0.04, ret1 <= -0.07
    p["drop20_15"] = ret20 <= -0.15
    p["low252"] = np.isfinite(low252) & (C <= low252)
    down = np.nan_to_num(ret1, nan=0.0) < 0
    run = np.zeros((T, N), dtype=np.int32)
    for t in range(1, T):
        run[t] = np.where(down[t], run[t - 1] + 1, 0)
    p["dn3"], p["dn5"] = run >= 3, run >= 5
    p["squeeze"] = np.isfinite(vol20) & np.isfinite(vol120) & (vol20 < 0.6 * vol120)
    spy_down = (spy_s / spy_s.rolling(200, min_periods=200).mean() - 1).to_numpy() < 0
    p["spy_down200"] = np.repeat(spy_down[:, None], N, axis=1)
    p["spy_drop5"] = np.repeat((spy5.to_numpy() <= -0.05)[:, None], N, axis=1)
    p["up200"], p["down200"] = C > sma200, C < sma200
    p["high252"] = np.isfinite(high252) & (C >= high252)
    p["spy_up200"] = np.repeat(spy_up[:, None], N, axis=1)
    # catalog 5: the KODEX 200 filters (the same tables as the SPY ones of a Korean dataset) and price limits
    p["mkt_up200"], p["mkt_down200"], p["mkt_drop5"] = p["spy_up200"], p["spy_down200"], p["spy_drop5"]
    limit = np.where(cal < KR_LIMIT_CHANGE, 0.15, 0.30)[:, None] - 0.01
    p["uplimit"], p["dnlimit"] = ret1 >= limit, ret1 <= -limit
    rsi14 = _wilder_rsi(C, 14)
    rsi2 = _wilder_rsi(C, 2)
    p["rsi14_30"], p["rsi2_10"] = rsi14 < 30, rsi2 < 10

    ev = data.events if data.events is not None else pd.DataFrame(columns=["known", "symbol", "kind"])
    known_all = pd.to_datetime(ev["known"]).to_numpy(dtype="datetime64[ns]") if len(ev) else np.array([], "datetime64[ns]")
    cal64 = cal.to_numpy(dtype="datetime64[ns]")

    # earnings: abnormal return over SPY from the last close before the filing date to the first close after it
    abn = np.full((T, N), -np.inf)
    neg = np.full((T, N), np.inf)  # the most negative reaction in the window (catalog 3)
    if len(ev) and "kind" in ev:
        e_mask = (ev["kind"] == "earnings").to_numpy()
        for k in np.flatnonzero(e_mask):
            j = col.get(ev["symbol"].iat[k])
            if j is None:
                continue
            e = int(np.searchsorted(cal64, known_all[k], "right"))
            b = int(np.searchsorted(cal64, known_all[k], "left")) - 1
            if b < 0 or e >= T:
                continue
            r = C[e, j] / C[b, j] - spy[e] / spy[b]
            if np.isfinite(r):
                seg = abn[e:e + EARN_DAYS, j]
                np.maximum(seg, r, out=seg)
                seg = neg[e:e + EARN_DAYS, j]
                np.minimum(seg, r, out=seg)
    p["earn3"], p["earn5"], p["earn8"] = abn >= 0.03, abn >= 0.05, abn >= 0.08
    p["earnneg5"], p["earnneg8"] = neg <= -0.05, neg <= -0.08

    # insiders: distinct officers/directors with a purchase filing of at least $10k in the last 30 days
    n_ins = _insider_counts(ev, known_all, cal64, col, T, N, INSIDER_MIN_VALUE)
    p["ins1"], p["ins2"] = n_ins >= 1, n_ins >= 2
    p["insbig"] = _insider_counts(ev, known_all, cal64, col, T, N, INSIDER_BIG_VALUE) >= 1
    # catalog 4: sales (filings made public by the session), the earnings calendar, index changes
    p["inssell3"] = _insider_counts(ev, known_all, cal64, col, T, N, SELL_BIG_VALUE, kind="insider_sell") >= 3
    p["nosell"] = _insider_counts(ev, known_all, cal64, col, T, N, INSIDER_MIN_VALUE, kind="insider_sell",
                                  days=NOSELL_DAYS) == 0
    last_earn = np.full((T, N), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    if len(ev) and "kind" in ev:
        for k in np.flatnonzero((ev["kind"] == "earnings").to_numpy()):
            j = col.get(ev["symbol"].iat[k])
            if j is None:
                continue
            lo = int(np.searchsorted(cal64, known_all[k], "left"))  # public by the evening of its filing date
            if lo < T:
                seg = last_earn[lo:, j]
                np.copyto(seg, known_all[k], where=np.isnat(seg) | (seg < known_all[k]))
    age = (cal64[:, None] - last_earn) / np.timedelta64(1, "D")
    with np.errstate(invalid="ignore"):
        p["preearn"] = (age >= PRE_EARN_DAYS[0]) & (age <= PRE_EARN_DAYS[1])
    raw = data.members[syms].to_numpy(dtype=float) if N else np.zeros((T, 0))
    window = pd.DataFrame(raw).rolling(INDEX_CHANGE_SESSIONS + 1, min_periods=INDEX_CHANGE_SESSIONS + 1)
    lo_m, hi_m = window.min().to_numpy(), window.max().to_numpy()
    p["idxadd"] = (raw == 1) & (lo_m == 0)
    p["idxdel"] = (raw == 0) & (hi_m == 1) & np.isfinite(C)

    for k in list(p):
        p[k] = np.ascontiguousarray(p[k], dtype=bool)
    members = data.members[syms].to_numpy(dtype=bool) if N else np.zeros((T, 0), bool)
    p["member"] = members & np.isfinite(C)
    p["close"], p["open"] = C, O
    p["order"] = np.random.default_rng(ORDER_SEED).random((T, N))
    for v in p.values():
        v.setflags(write=False)
    p["symbols"] = tuple(syms)
    return p


def _insider_counts(ev, known_all, cal64, col, T: int, N: int, min_value: float, kind: str = "insider_buy",
                    days: int = INSIDER_DAYS) -> np.ndarray:
    """Per session and stock: distinct officers/directors with a filing of `kind` (purchase or sale)
    of at least `min_value` dollars made public in the last `days` calendar days."""
    count = np.zeros((T + 1, N), dtype=np.int32)
    if len(ev) and "kind" in ev and "owner" in ev:
        role = np.zeros(len(ev), dtype=bool)
        for r in ("officer", "director"):
            if r in ev:
                role |= ev[r].fillna(False).astype(bool).to_numpy()
        val = pd.to_numeric(ev["value"], errors="coerce").fillna(0).to_numpy() if "value" in ev else np.zeros(len(ev))
        i_mask = (ev["kind"] == kind).to_numpy() & role & (val >= min_value)
        spans: dict[tuple, list[tuple[int, int]]] = {}
        for k in np.flatnonzero(i_mask):
            j = col.get(ev["symbol"].iat[k])
            if j is None:
                continue
            lo = int(np.searchsorted(cal64, known_all[k], "left"))
            hi = int(np.searchsorted(cal64, known_all[k] + np.timedelta64(days - 1, "D"), "right"))
            if lo < hi:
                spans.setdefault((j, ev["owner"].iat[k]), []).append((lo, hi))
        for (j, _), iv in spans.items():
            iv.sort()
            a, b = iv[0]
            for lo, hi in iv[1:] + [(T + 1, T + 1)]:
                if lo > b:
                    count[a, j] += 1
                    count[b, j] -= 1
                    a, b = lo, hi
                else:
                    b = max(b, hi)
    return np.cumsum(count, axis=0)[:T]


# ----------------------------------------------------------------------------- strategy
class StockComboStrategy(Strategy):
    def __init__(self, *, spec_id: str, spec_version_str: str, rules: dict, **params: Any) -> None:
        super().__init__(**params)
        self.strategy_id, self.version = spec_id, spec_version_str
        self.cfg = dict(rules["stock_combo"])
        for name in PARAMS:
            if name not in params:
                raise ValueError(f"stock_combo needs parameter {name}")
        self.conds = str(params["entry"]).split("+")
        unknown = [c for c in self.conds if c not in CONDITIONS]
        if unknown:
            raise ValueError(f"unknown stock_combo conditions {unknown}")
        self.hold, self.slots = int(params["hold"]), int(params["slots"])
        self.stop, self.take = float(params["stop"] or 0), float(params["take"] or 0)
        self.park_symbol = self.cfg.get("park")  # idle equity, not a trade of the rule
        self.excess_benchmark = benchmark(catalog_of(self.cfg))  # each trade is also compared with the market
                                                                # (SPY; KODEX 200 in catalog 5) for the same days
        self.held: dict[int, list] = {}         # column -> [sessions since signal, entry open or None]
        self.park_w: float | None = None

    def describe(self) -> dict:
        return {"strategy_id": self.strategy_id, "version": self.version, "params": self.params,
                "rules": {"stock_combo": self.cfg}}

    def _all(self, view: MarketView, back: int) -> np.ndarray | None:
        rows = [view.panel(build_panels, c, back) for c in self.conds]
        if rows[0] is None:
            return None
        out = rows[0].copy()
        for r in rows[1:]:
            out &= r
        return out

    def on_bar(self, view: MarketView):
        syms = view.panel_columns(build_panels)
        close = view.panel(build_panels, "close")
        opn = view.panel(build_panels, "open")
        w = PartialWeights()
        for j in list(self.held):
            h = self.held[j]
            h[0] += 1
            if h[1] is None and h[0] >= 1:
                h[1] = opn[j] if np.isfinite(opn[j]) and opn[j] > 0 else None
            out = h[0] >= self.hold
            if not out and h[1] and np.isfinite(close[j]):
                r = close[j] / h[1] - 1
                out = (self.stop and r <= -self.stop) or (self.take and r >= self.take)
            if out:
                w[syms[j]] = 0.0
                del self.held[j]
        free = self.slots - len(self.held)
        if free > 0:
            now = self._all(view, 0)
            before = self._all(view, 1)
            sig = now & view.panel(build_panels, "member")
            if "idxdel" in self.conds:  # a stock that just left the index is no longer a member
                sig = now
            if before is not None:
                sig &= ~before
            idx = np.flatnonzero(sig)
            idx = idx[[j not in self.held for j in idx]] if len(idx) else idx
            if len(idx):
                order = view.panel(build_panels, "order")
                for j in idx[np.argsort(-order[idx], kind="stable")][:free]:
                    w[syms[j]] = 1.0 / self.slots
                    self.held[int(j)] = [0, None]
        park = self.park_symbol
        if park:
            room = 1.0 - len(self.held) / self.slots
            band = int(self.cfg.get("park_band", 1)) / self.slots
            if self.park_w is None or self.park_w > room + 1e-9 or room - self.park_w >= band - 1e-9:
                self.park_w = room
                w[park] = room
        return w or None
