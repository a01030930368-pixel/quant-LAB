"""Building blocks for the automatic strategy search (rules block `combo`, catalog 1).

A `combo` spec trades one leg per listed ETF with the same rule on each. Its parameters choose one
item from each list below; the search runs combinations of them on TRAIN+VALIDATION data only.

    rules:
      kind: declarative
      combo:
        catalog: 1
        trade: [EWJ, EWG, ...]
        slots: 5                  # each open leg holds 1/5 of equity (D-018 slots sizing)
        park: SPY                 # optional: equity not in an open leg is held in SPY
    parameters:
      entry:    {default: z20_2.0, search: [...]}   # keys of ENTRIES
      filter:   {default: none, search: [...]}      # keys of FILTERS
      exit:     {default: none, search: [...]}      # keys of EXITS
      max_hold: {default: 10, search: [5, 10, 20, 30]}
      stop:     {default: 0, search: [0, 0.10]}     # close-based stop loss, 0 = none
      take:     {default: 0, search: [0, 0.10]}     # close-based take profit, 0 = none

The catalog is part of the app code: changing what a key means requires a new catalog number, so a
recorded trial always means the same rule. Every indicator is read through the whole-series path
(`quantlab.strategies.indicators`), the same in the search and in the final evaluation.

Entry items in one `group` differ only by threshold, ordered from mild to strict; the search uses
the adjacent items as the "neighbours" of a result (does a slightly different threshold also work?).
"""

from __future__ import annotations

from typing import Any

CATALOGS = (1, 2)
CATALOG = CATALOGS[-1]  # newest; catalog 1 is kept as it was so its recorded trials keep their meaning


def _c(indicator: str, op: str, value: float, **kw) -> dict:
    return {"indicator": indicator, "op": op, "value": value, **kw}


# key -> (Korean label, conditions, group, first catalog). Declaration order is the list order; a
# later catalog only adds items, so a catalog's lists are the items with `since <= catalog`, in order.
_ENTRIES: dict[str, tuple[str, list[dict], str, int]] = {}


def _entry(key: str, label: str, conds: list[dict], group: str | None = None, since: int = 1) -> None:
    _ENTRIES[key] = (label, conds, group or key, since)


for z, c in ((1.5, 1), (2.0, 1), (2.5, 1), (3.0, 2)):
    _entry(f"z20_{z}", f"20일 볼린저 z < -{z}", [_c("bollinger_z", "<", -z, days=20)], "z20", c)
for v, c in ((2, 2), (5, 1), (10, 1), (15, 1)):
    _entry(f"rsi2_{v}", f"RSI(2) < {v}", [_c("rsi", "<", v, days=2)], "rsi2", c)
for n, c in ((3, 1), (4, 1), (5, 1), (6, 2)):
    _entry(f"down{n}", f"{n}일 연속 하락", [_c("down_streak", ">=", n)], "down", c)
for p, c in ((3, 1), (5, 1), (8, 1), (12, 2), (15, 2)):
    _entry(f"ret5_{p}", f"5일 수익률 < -{p}%", [_c("return", "<", -p / 100, days=5)], "ret5", c)
for p, c in ((5, 1), (8, 1), (12, 1), (16, 2), (20, 2)):
    _entry(f"ret10_{p}", f"10일 수익률 < -{p}%", [_c("return", "<", -p / 100, days=10)], "ret10", c)
for d, c in ((20, 1), (60, 2), (120, 2)):
    _entry(f"low{d}", f"{d}일 최저가", [_c("close_vs_min", "<=", 0, days=d)], "low", c)
for v, c in ((0.05, 2), (0.1, 1), (0.2, 1)):
    _entry(f"ibs_{v}", f"IBS < {v}", [_c("ibs", "<", v)], "ibs", c)
for d, levels in {5: ((2, 1), (4, 1), (6, 2)), 20: ((4, 1), (7, 1), (10, 2), (15, 2)),
                  60: ((7, 1), (12, 1), (16, 2), (20, 2)), 120: ((10, 1), (15, 1), (20, 2), (25, 2))}.items():
    for p, c in levels:
        _entry(f"below{d}_{p}", f"종가가 {d}일선보다 {p}% 아래", [_c("close_vs_sma", "<", -p / 100, days=d)],
               f"below{d}", c)
for f, s in ((5, 20), (20, 60), (60, 120)):
    _entry(f"golden{f}_{s}", f"{f}일선이 {s}일선 상향 돌파", [_c("sma_cross", ">", 0, fast=f, slow=s)])
for d in (20, 60, 120):
    _entry(f"breakup{d}", f"종가가 {d}일선 상향 돌파", [_c("close_cross_sma", ">", 0, days=d)])
_entry("macd_up", "MACD 골든크로스", [_c("macd_cross", ">", 0)])
_entry("macd_up_neg", "MACD 골든크로스 (0선 아래)", [_c("macd_cross", ">", 0), _c("macd", "<", 0)])
for v in (30, 25, 20):
    _entry(f"rsi14_{v}", f"RSI(14) < {v}", [_c("rsi", "<", v, days=14)], "rsi14", 2)
for p in (10, 15, 20):
    _entry(f"ret20_{p}", f"20일 수익률 < -{p}%", [_c("return", "<", -p / 100, days=20)], "ret20", 2)

_FILTERS: dict[str, tuple[str, list[dict], int]] = {
    "none": ("추세 조건 없음", [], 1),
    "up200": ("200일선 위", [_c("close_vs_sma", ">", 0, days=200)], 1),
    "up120": ("120일선 위", [_c("close_vs_sma", ">", 0, days=120)], 1),
    "aligned": ("정배열 (5>20>60>120일선)", [_c("sma_ratio", ">", 0, fast=5, slow=20),
                                         _c("sma_ratio", ">", 0, fast=20, slow=60),
                                         _c("sma_ratio", ">", 0, fast=60, slow=120)], 1),
    "down200": ("200일선 아래", [_c("close_vs_sma", "<", 0, days=200)], 1),
    "macd_pos": ("MACD 0선 위", [_c("macd", ">", 0)], 1),
    "up60": ("60일선 위", [_c("close_vs_sma", ">", 0, days=60)], 2),
    "macd_neg": ("MACD 0선 아래", [_c("macd", "<", 0)], 2),
    "spy_up200": ("SPY가 200일선 위", [_c("close_vs_sma", ">", 0, days=200, of="SPY")], 2),
    "spy_down200": ("SPY가 200일선 아래", [_c("close_vs_sma", "<", 0, days=200, of="SPY")], 2),
}

_EXITS: dict[str, tuple[str, list[dict], int]] = {
    "none": ("보유 기간·손절·익절만", [], 1),
    "above5": ("5일선 회복", [_c("close_vs_sma", ">", 0, days=5)], 1),
    "above10": ("10일선 회복", [_c("close_vs_sma", ">", 0, days=10)], 1),
    "above20": ("20일선 회복", [_c("close_vs_sma", ">", 0, days=20)], 1),
    "rsi2_70": ("RSI(2) > 70", [_c("rsi", ">", 70, days=2)], 1),
    "macd_dn": ("MACD 데드크로스", [_c("macd_cross", "<", 0)], 1),
    "below20": ("20일선 이탈", [_c("close_vs_sma", "<", 0, days=20)], 1),
    "rsi2_50": ("RSI(2) > 50", [_c("rsi", ">", 50, days=2)], 2),
    "rsi2_90": ("RSI(2) > 90", [_c("rsi", ">", 90, days=2)], 2),
    "high5": ("5일 최고가", [_c("close_vs_max", ">=", 0, days=5)], 2),
}

_MAX_HOLD = {1: [5, 10, 20, 30], 2: [3, 5, 10, 20, 30]}
_STOPS = {1: [0, 0.10], 2: [0, 0.10, 0.20]}
_TAKES = {1: [0, 0.10], 2: [0, 0.10, 0.20, 0.30]}
PARAMS = ("entry", "filter", "exit", "max_hold", "stop", "take")


def entries(catalog: int = 1) -> dict[str, tuple[str, list[dict], str]]:
    return {k: v[:3] for k, v in _ENTRIES.items() if v[3] <= catalog}


def filters(catalog: int = 1) -> dict[str, tuple[str, list[dict]]]:
    return {k: v[:2] for k, v in _FILTERS.items() if v[2] <= catalog}


def exits(catalog: int = 1) -> dict[str, tuple[str, list[dict]]]:
    return {k: v[:2] for k, v in _EXITS.items() if v[2] <= catalog}


# catalog 1, unchanged names for existing callers
ENTRIES, FILTERS, EXITS = entries(1), filters(1), exits(1)
MAX_HOLD, STOPS, TAKES = _MAX_HOLD[1], _STOPS[1], _TAKES[1]


def full_space(catalog: int = 1) -> dict[str, list]:
    return {"entry": list(entries(catalog)), "filter": list(filters(catalog)), "exit": list(exits(catalog)),
            "max_hold": list(_MAX_HOLD[catalog]), "stop": list(_STOPS[catalog]), "take": list(_TAKES[catalog])}


def catalog_of(combo: dict) -> int:
    return int(combo.get("catalog", 1))


def validate_combo(combo: Any, symbols: list[str]) -> list[str]:
    if not isinstance(combo, dict):
        return ["rules.combo must be a mapping"]
    errs = []
    if combo.get("catalog") not in CATALOGS:
        errs.append(f"rules.combo.catalog must be one of {CATALOGS} (this app's building-block lists)")
    trade = combo.get("trade")
    if not isinstance(trade, list) or not trade or not set(trade) <= set(symbols):
        errs.append("rules.combo.trade must be a non-empty list of universe.symbols")
    if not isinstance(combo.get("slots"), int) or combo["slots"] < 1:
        errs.append("rules.combo.slots must be a whole number >= 1")
    if combo.get("park") is not None and (combo["park"] not in symbols or combo["park"] in (trade or [])):
        errs.append("rules.combo.park must be a universe symbol that is not traded")
    return errs


def validate_params(parameters: dict, catalog: int = 1, symbols: list[str] | None = None) -> list[str]:
    errs = []
    if catalog not in CATALOGS:
        return []  # reported by validate_combo
    space = full_space(catalog)
    for name in PARAMS:
        p = parameters.get(name)
        if not isinstance(p, dict) or "default" not in p:
            errs.append(f"parameters.{name} is required for combo rules")
            continue
        for v in [p["default"]] + list(p.get("search") or []):
            if v not in space[name]:
                errs.append(f"parameters.{name}: '{v}' is not in building-block catalog {catalog}")
    if symbols is not None and isinstance(parameters.get("filter"), dict):
        # market filters read another symbol (e.g. SPY): only the filters this spec lists need it
        listed = [parameters["filter"].get("default")] + list(parameters["filter"].get("search") or [])
        catalog_filters = filters(catalog)
        used = {c.get("of") for k in listed if k in catalog_filters for c in catalog_filters[k][1]} - {None}
        if not used <= set(symbols):
            errs.append(f"parameters.filter reads {sorted(used)}: add them to universe.symbols")
    extra = set(parameters) - set(PARAMS)
    if extra:
        errs.append(f"combo rules take only {PARAMS}; unknown {sorted(extra)}")
    return errs


def build_rules(combo: dict, params: dict) -> dict:
    """entry_exit rules for one combination."""
    cat = catalog_of(combo)
    space = full_space(cat)
    for name in PARAMS:
        if params.get(name) not in space[name]:
            raise ValueError(f"combo parameter {name}={params.get(name)!r} is not in catalog {cat}")
    entry = _ENTRIES[params["entry"]][1] + _FILTERS[params["filter"]][1]
    ex: dict[str, Any] = {"any": list(_EXITS[params["exit"]][1]), "max_hold_days": int(params["max_hold"])}
    if params["stop"]:
        ex["stop_loss"] = float(params["stop"])
    if params["take"]:
        ex["take_profit"] = float(params["take"])
    legs = [{"name": f"{params['entry']}-{sym}", "trade": sym, "entry": entry, "exit": ex} for sym in combo["trade"]]
    sizing: dict[str, Any] = {"slots": int(combo["slots"])}
    if combo.get("park"):
        sizing["park"] = combo["park"]  # idle equity held in this symbol
    return {"entry_exit": {"sizing": sizing, "legs": legs}}


def describe(params: dict) -> str:
    """One Korean line for tables: 진입 · 추세 · 청산 · 보유 · 손절/익절."""
    parts = [_ENTRIES[params["entry"]][0], _FILTERS[params["filter"]][0], _EXITS[params["exit"]][0],
             f"최대 {params['max_hold']}일"]
    if params.get("stop"):
        parts.append(f"손절 {params['stop'] * 100:g}%")
    if params.get("take"):
        parts.append(f"익절 {params['take'] * 100:g}%")
    return " · ".join(parts)


def neighbours(params: dict, catalog: int = 1) -> list[dict]:
    """Combinations one step away within the catalog: the adjacent threshold in the same entry group,
    and the adjacent maximum holding period. Singleton entries only vary the holding period."""
    out = []
    ent = entries(catalog)
    group = ent[params["entry"]][2]
    same = [k for k, v in ent.items() if v[2] == group]
    i = same.index(params["entry"])
    for j in (i - 1, i + 1):
        if 0 <= j < len(same):
            out.append(dict(params, entry=same[j]))
    holds = _MAX_HOLD[catalog]
    h = holds.index(params["max_hold"]) if params["max_hold"] in holds else None
    if h is not None:
        for j in (h - 1, h + 1):
            if 0 <= j < len(holds):
                out.append(dict(params, max_hold=holds[j]))
    return out
