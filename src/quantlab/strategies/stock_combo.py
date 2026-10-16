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

CATALOGS = (1,)
BENCHMARK = "SPY"
PARAMS = ("entry", "hold", "stop", "take", "slots")
KEYS = ("catalog", "park", "park_band")
ORDER_SEED = 20261016
EARN_DAYS = 5          # an earnings reaction counts for this many sessions, its own included
INSIDER_DAYS = 30      # calendar days of insider filings, as INS-CLUSTER
INSIDER_MIN_VALUE = 10_000

# key -> (Korean label, group); groups in this order, items in a group from mild to strict
CONDITIONS: dict[str, tuple[str, str]] = {
    "earn3": ("실적 발표 반응이 SPY보다 +3% 이상", "earn"),
    "earn5": ("실적 발표 반응이 SPY보다 +5% 이상", "earn"),
    "earn8": ("실적 발표 반응이 SPY보다 +8% 이상", "earn"),
    "ins1": ("임원·이사 1명 이상 매수 (30일)", "ins"),
    "ins2": ("임원·이사 2명 이상 매수 (30일)", "ins"),
    "drop5": ("5일 수익률 -5% 이하", "drop"),
    "drop10": ("5일 수익률 -10% 이하", "drop"),
    "rsi14_30": ("RSI(14) < 30", "rsi"),
    "rsi2_10": ("RSI(2) < 10", "rsi"),
    "gap3": ("시가가 전날 종가보다 +3% 이상", "gap"),
    "gap5": ("시가가 전날 종가보다 +5% 이상", "gap"),
    "up200": ("200일선 위", "trend"),
    "down200": ("200일선 아래", "trend"),
    "spy_up200": ("SPY가 200일선 위", "spy"),
    "high252": ("52주 신고가", "high"),
}
GROUPS = ("earn", "ins", "drop", "rsi", "gap", "trend", "spy", "high")
# adjacent thresholds of the same measure (neighbours); rsi and trend items are different measures
LADDERS = (("earn3", "earn5", "earn8"), ("ins1", "ins2"), ("drop5", "drop10"), ("gap3", "gap5"))
HOLDS = (5, 10, 20, 30, 60)
STOPS = (0, 0.1, 0.2)
TAKES = (0, 0.1, 0.2)
SLOTS = (10, 20)


def entries(catalog: int = 1) -> list[str]:
    """Every entry key: 1 to 3 conditions from different groups, joined with '+' in catalog order."""
    keys = list(CONDITIONS)
    out = []
    for n in (1, 2, 3):
        for combo in itertools.combinations(keys, n):
            if len({CONDITIONS[k][1] for k in combo}) == n:
                out.append("+".join(combo))
    return out


def full_space(catalog: int = 1) -> dict[str, list]:
    return {"entry": entries(catalog), "hold": list(HOLDS), "stop": list(STOPS), "take": list(TAKES),
            "slots": list(SLOTS)}


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
    if BENCHMARK not in symbols:
        errs.append(f"stock_combo rules read {BENCHMARK} (earnings reaction, market trend): add it to universe.symbols")
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
    h = HOLDS.index(params["hold"]) if params["hold"] in HOLDS else None
    if h is not None:
        for k in (h - 1, h + 1):
            if 0 <= k < len(HOLDS):
                out.append(dict(params, hold=HOLDS[k]))
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
    _, spy = _adjusted(data.values(BENCHMARK))
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
    p["up200"], p["down200"] = C > sma200, C < sma200
    p["high252"] = np.isfinite(high252) & (C >= high252)
    p["spy_up200"] = np.repeat(spy_up[:, None], N, axis=1)
    rsi14 = _wilder_rsi(C, 14)
    rsi2 = _wilder_rsi(C, 2)
    p["rsi14_30"], p["rsi2_10"] = rsi14 < 30, rsi2 < 10

    ev = data.events if data.events is not None else pd.DataFrame(columns=["known", "symbol", "kind"])
    known_all = pd.to_datetime(ev["known"]).to_numpy(dtype="datetime64[ns]") if len(ev) else np.array([], "datetime64[ns]")
    cal64 = cal.to_numpy(dtype="datetime64[ns]")

    # earnings: abnormal return over SPY from the last close before the filing date to the first close after it
    abn = np.full((T, N), -np.inf)
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
    p["earn3"], p["earn5"], p["earn8"] = abn >= 0.03, abn >= 0.05, abn >= 0.08

    # insiders: distinct officers/directors with a purchase filing of at least $10k in the last 30 days
    count = np.zeros((T + 1, N), dtype=np.int32)
    if len(ev) and "kind" in ev and "owner" in ev:
        role = np.zeros(len(ev), dtype=bool)
        for r in ("officer", "director"):
            if r in ev:
                role |= ev[r].fillna(False).astype(bool).to_numpy()
        val = pd.to_numeric(ev["value"], errors="coerce").fillna(0).to_numpy() if "value" in ev else np.zeros(len(ev))
        i_mask = (ev["kind"] == "insider_buy").to_numpy() & role & (val >= INSIDER_MIN_VALUE)
        spans: dict[tuple, list[tuple[int, int]]] = {}
        for k in np.flatnonzero(i_mask):
            j = col.get(ev["symbol"].iat[k])
            if j is None:
                continue
            lo = int(np.searchsorted(cal64, known_all[k], "left"))
            hi = int(np.searchsorted(cal64, known_all[k] + np.timedelta64(INSIDER_DAYS - 1, "D"), "right"))
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
    n_ins = np.cumsum(count, axis=0)[:T]
    p["ins1"], p["ins2"] = n_ins >= 1, n_ins >= 2

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
        self.excess_benchmark = BENCHMARK       # each trade is also compared with SPY held for the same days
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
