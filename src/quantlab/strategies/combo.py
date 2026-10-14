"""Building blocks for the automatic strategy search (rules block `combo`, catalog 1).

A `combo` spec trades one leg per listed ETF with the same rule on each. Its parameters choose one
item from each list below; the search runs combinations of them on TRAIN+VALIDATION data only.

    rules:
      kind: declarative
      combo:
        catalog: 1
        trade: [EWJ, EWG, ...]
        slots: 5                  # each open leg holds 1/5 of equity (D-018 slots sizing)
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

CATALOG = 1


def _c(indicator: str, op: str, value: float, **kw) -> dict:
    return {"indicator": indicator, "op": op, "value": value, **kw}


# key -> (Korean label, conditions, group)
ENTRIES: dict[str, tuple[str, list[dict], str]] = {}


def _entry(key: str, label: str, conds: list[dict], group: str | None = None) -> None:
    ENTRIES[key] = (label, conds, group or key)


for z in (1.5, 2.0, 2.5):
    _entry(f"z20_{z}", f"20일 볼린저 z < -{z}", [_c("bollinger_z", "<", -z, days=20)], "z20")
for v in (5, 10, 15):
    _entry(f"rsi2_{v}", f"RSI(2) < {v}", [_c("rsi", "<", v, days=2)], "rsi2")
for n in (3, 4, 5):
    _entry(f"down{n}", f"{n}일 연속 하락", [_c("down_streak", ">=", n)], "down")
for p in (3, 5, 8):
    _entry(f"ret5_{p}", f"5일 수익률 < -{p}%", [_c("return", "<", -p / 100, days=5)], "ret5")
for p in (5, 8, 12):
    _entry(f"ret10_{p}", f"10일 수익률 < -{p}%", [_c("return", "<", -p / 100, days=10)], "ret10")
_entry("low20", "20일 최저가", [_c("close_vs_min", "<=", 0, days=20)])
for v in (0.1, 0.2):
    _entry(f"ibs_{v}", f"IBS < {v}", [_c("ibs", "<", v)], "ibs")
for d, (a, b) in {5: (2, 4), 20: (4, 7), 60: (7, 12), 120: (10, 15)}.items():
    for p in (a, b):
        _entry(f"below{d}_{p}", f"종가가 {d}일선보다 {p}% 아래", [_c("close_vs_sma", "<", -p / 100, days=d)], f"below{d}")
for f, s in ((5, 20), (20, 60), (60, 120)):
    _entry(f"golden{f}_{s}", f"{f}일선이 {s}일선 상향 돌파", [_c("sma_cross", ">", 0, fast=f, slow=s)])
for d in (20, 60, 120):
    _entry(f"breakup{d}", f"종가가 {d}일선 상향 돌파", [_c("close_cross_sma", ">", 0, days=d)])
_entry("macd_up", "MACD 골든크로스", [_c("macd_cross", ">", 0)])
_entry("macd_up_neg", "MACD 골든크로스 (0선 아래)", [_c("macd_cross", ">", 0), _c("macd", "<", 0)])

FILTERS: dict[str, tuple[str, list[dict]]] = {
    "none": ("추세 조건 없음", []),
    "up200": ("200일선 위", [_c("close_vs_sma", ">", 0, days=200)]),
    "up120": ("120일선 위", [_c("close_vs_sma", ">", 0, days=120)]),
    "aligned": ("정배열 (5>20>60>120일선)", [_c("sma_ratio", ">", 0, fast=5, slow=20),
                                         _c("sma_ratio", ">", 0, fast=20, slow=60),
                                         _c("sma_ratio", ">", 0, fast=60, slow=120)]),
    "down200": ("200일선 아래", [_c("close_vs_sma", "<", 0, days=200)]),
    "macd_pos": ("MACD 0선 위", [_c("macd", ">", 0)]),
}

EXITS: dict[str, tuple[str, list[dict]]] = {
    "none": ("보유 기간·손절·익절만", []),
    "above5": ("5일선 회복", [_c("close_vs_sma", ">", 0, days=5)]),
    "above10": ("10일선 회복", [_c("close_vs_sma", ">", 0, days=10)]),
    "above20": ("20일선 회복", [_c("close_vs_sma", ">", 0, days=20)]),
    "rsi2_70": ("RSI(2) > 70", [_c("rsi", ">", 70, days=2)]),
    "macd_dn": ("MACD 데드크로스", [_c("macd_cross", "<", 0)]),
    "below20": ("20일선 이탈", [_c("close_vs_sma", "<", 0, days=20)]),
}

MAX_HOLD = [5, 10, 20, 30]
STOPS = [0, 0.10]
TAKES = [0, 0.10]
PARAMS = ("entry", "filter", "exit", "max_hold", "stop", "take")


def full_space() -> dict[str, list]:
    return {"entry": list(ENTRIES), "filter": list(FILTERS), "exit": list(EXITS),
            "max_hold": list(MAX_HOLD), "stop": list(STOPS), "take": list(TAKES)}


def validate_combo(combo: Any, symbols: list[str]) -> list[str]:
    if not isinstance(combo, dict):
        return ["rules.combo must be a mapping"]
    errs = []
    if combo.get("catalog") != CATALOG:
        errs.append(f"rules.combo.catalog must be {CATALOG} (this app's building-block list)")
    trade = combo.get("trade")
    if not isinstance(trade, list) or not trade or not set(trade) <= set(symbols):
        errs.append("rules.combo.trade must be a non-empty list of universe.symbols")
    if not isinstance(combo.get("slots"), int) or combo["slots"] < 1:
        errs.append("rules.combo.slots must be a whole number >= 1")
    return errs


def validate_params(parameters: dict) -> list[str]:
    errs = []
    space = full_space()
    for name in PARAMS:
        p = parameters.get(name)
        if not isinstance(p, dict) or "default" not in p:
            errs.append(f"parameters.{name} is required for combo rules")
            continue
        for v in [p["default"]] + list(p.get("search") or []):
            if v not in space[name]:
                errs.append(f"parameters.{name}: '{v}' is not in building-block catalog {CATALOG}")
    extra = set(parameters) - set(PARAMS)
    if extra:
        errs.append(f"combo rules take only {PARAMS}; unknown {sorted(extra)}")
    return errs


def build_rules(combo: dict, params: dict) -> dict:
    """entry_exit rules for one combination."""
    for name in PARAMS:
        if params.get(name) not in full_space()[name]:
            raise ValueError(f"combo parameter {name}={params.get(name)!r} is not in catalog {CATALOG}")
    entry = ENTRIES[params["entry"]][1] + FILTERS[params["filter"]][1]
    ex: dict[str, Any] = {"any": list(EXITS[params["exit"]][1]), "max_hold_days": int(params["max_hold"])}
    if params["stop"]:
        ex["stop_loss"] = float(params["stop"])
    if params["take"]:
        ex["take_profit"] = float(params["take"])
    legs = [{"name": f"{params['entry']}-{sym}", "trade": sym, "entry": entry, "exit": ex} for sym in combo["trade"]]
    return {"entry_exit": {"sizing": {"slots": int(combo["slots"])}, "legs": legs}}


def describe(params: dict) -> str:
    """One Korean line for tables: 진입 · 추세 · 청산 · 보유 · 손절/익절."""
    parts = [ENTRIES[params["entry"]][0], FILTERS[params["filter"]][0], EXITS[params["exit"]][0],
             f"최대 {params['max_hold']}일"]
    if params.get("stop"):
        parts.append(f"손절 {params['stop'] * 100:g}%")
    if params.get("take"):
        parts.append(f"익절 {params['take'] * 100:g}%")
    return " · ".join(parts)


def neighbours(params: dict) -> list[dict]:
    """Combinations one step away: the adjacent threshold in the same entry group, and the
    adjacent maximum holding period. Singleton entries only vary the holding period."""
    out = []
    group = ENTRIES[params["entry"]][2]
    same = [k for k, v in ENTRIES.items() if v[2] == group]
    i = same.index(params["entry"])
    for j in (i - 1, i + 1):
        if 0 <= j < len(same):
            out.append(dict(params, entry=same[j]))
    h = MAX_HOLD.index(params["max_hold"]) if params["max_hold"] in MAX_HOLD else None
    if h is not None:
        for j in (h - 1, h + 1):
            if 0 <= j < len(MAX_HOLD):
                out.append(dict(params, max_hold=MAX_HOLD[j]))
    return out
