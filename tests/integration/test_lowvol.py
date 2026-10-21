"""D-036: rules held for months — low-volatility S&P 500 stocks (LOWVOL-SP500) and insider clusters
held six months (INS-HOLD6M)."""

import numpy as np
import pandas as pd

from quantlab.backtesting.engine import run_backtest
from quantlab.core.market_data import MarketData
from quantlab.services.lab import builtin_path
from quantlab.strategies import lowvol as lv
from quantlab.strategies.insider import InsiderClusterStrategy
from quantlab.strategies.spec import load_spec
from quantlab.validation.engine import leak_test
from synthetic import frame

STOCKS = ["A", "B", "C", "D", "E"]
RULES = {"lowvol": {"months": [3, 6, 9, 12], "park": "SPY"}}


def market(n=400, members_off=None, seed=3):
    """Five members whose daily returns swing by 0.5% (A) up to 2.5% (E), and SPY."""
    idx = pd.bdate_range("2019-01-01", periods=n)
    rng = np.random.default_rng(seed)
    frames = {}
    for k, s in enumerate(["SPY"] + STOCKS):
        sd = 0.01 if s == "SPY" else 0.005 * k
        c = 100 * np.exp(np.cumsum(rng.normal(0.0003, sd, n)))
        f = frame(c, c)
        f.index = idx
        frames[s] = f
    members = pd.DataFrame(True, index=idx, columns=STOCKS)
    for s, (a, b) in (members_off or {}).items():
        members.loc[idx[a]:idx[b], s] = False
    return MarketData(idx, frames, members, None)


def strat(**params):
    params = {"lookback": 126, "top_n": 2} | params
    return lv.LowVolStrategy(spec_id="T", spec_version_str="1", rules=RULES, **params)


def holdings(data, **params):
    """Symbols held after each fill date (a fill changes holdings at the open)."""
    t = run_backtest(strat(**params), data).trades
    out, qty = {}, {}
    for d, g in t.groupby("date"):
        for s, q in zip(g["symbol"], g["qty"]):
            qty[s] = qty.get(s, 0) + q
        out[str(d.date())] = sorted(s for s, q in qty.items() if q > 1e-9 and s != "SPY")
    return out, t


def test_specs_and_policies_load():
    x = load_spec(builtin_path("specs", "LOWVOL-SP500.qls.yaml"))
    assert x.default_params == {"lookback": 252, "top_n": 50} and x.spec["rules"]["lowvol"]["months"] == [3, 6, 9, 12]
    y = load_spec(builtin_path("specs", "INS-HOLD6M.qls.yaml"))
    assert y.spec["rules"]["insider"]["hold_days"] == 126 and y.spec["strategy"]["parent"] == "INS-CLUSTER@1.0.0"


def test_parks_in_spy_then_holds_the_calmest_and_swaps_only_at_quarter_ends():
    data = market()
    h, t = holdings(data)
    first = t[t["symbol"] != "SPY"]["date"].min()
    # SPY bought on the first open; the first pick waits for 126 returns (127 closes)
    assert t.iloc[0]["symbol"] == "SPY" and first == data.calendar[127]
    spy = t[t["symbol"] == "SPY"]
    assert spy["qty"].iloc[-1] < 0 and spy["date"].iloc[-1] == first  # all of SPY sold at the first pick
    assert h[str(first.date())] == ["A", "B"]
    later = t[(t["symbol"] != "SPY") & (t["date"] > first)]["date"].dt.date.unique()
    # every later trade fills on the first session of a quarter (signal at the last close of Mar/Jun/Sep/Dec)
    assert len(later) and all(d.month in (1, 4, 7, 10) and d.day <= 4 for d in later)


def test_a_member_that_leaves_is_replaced_at_the_next_pick():
    data = market(members_off={"A": (200, 399)})
    h, _ = holdings(data)
    last = h[max(h)]
    assert "A" not in last and last == ["B", "C"]


def test_equal_weights_of_top_n():
    data = market()
    res = run_backtest(strat(top_n=3), data)
    eq = res.equity
    assert np.isfinite(eq.iloc[-1]) and eq.iloc[-1] > 0
    buys = res.trades[(res.trades["symbol"] != "SPY") & (res.trades["qty"] > 0)]
    first = buys[buys["date"] == buys["date"].min()]
    value = first["notional"].to_numpy()
    assert len(first) == 3 and value.max() / value.min() < 1.02


def test_volatility_uses_only_the_past():
    data = market(n=360)
    for name in ("vol126", "vol252"):
        full = lv.build_panels(data)[name]
        cut = MarketData(data.calendar[:300], {s: f.iloc[:300] for s, f in data.frames.items()},
                         data.members.iloc[:300], None)
        part = lv.build_panels(cut)[name]
        np.testing.assert_array_equal(np.isnan(full[:300]), np.isnan(part))
        np.testing.assert_allclose(full[:300][np.isfinite(part)], part[np.isfinite(part)])
    assert leak_test(lambda **p: strat(**p), {"lookback": 126, "top_n": 2}, market(), 4, 7) == []


def test_insider_park_band_refills_spy_in_batches():
    n = 200
    idx = pd.bdate_range("2020-01-01", periods=n)
    frames = {}
    for s in ["SPY"] + STOCKS:
        f = frame(np.full(n, 100.0), np.full(n, 100.0))
        f.index = idx
        frames[s] = f
    members = pd.DataFrame(True, index=idx, columns=STOCKS)
    ev = []
    for k, s in enumerate(STOCKS):  # a two-insider cluster in each stock, one week apart
        d = str(idx[10 + 5 * k].date())
        ev += [(d, s, "insider_buy", f"{s}{o}", 50_000, True, False) for o in "ab"]
    ev = pd.DataFrame(ev, columns=["known", "symbol", "kind", "owner", "value", "officer", "director"])
    ev["known"] = pd.to_datetime(ev["known"])
    data = MarketData(idx, frames, members, ev)

    def spy_trades(band):
        cfg = {"roles": ["officer", "director"], "min_value": 10000, "min_insiders": 2, "window_days": 30,
               "hold_days": 126, "slots": 20, "park": "SPY"} | ({"park_band": band} if band else {})
        t = run_backtest(InsiderClusterStrategy(spec_id="T", spec_version_str="1", rules={"insider": cfg}), data).trades
        return t[t["symbol"] == "SPY"], t[(t["symbol"] != "SPY") & (t["qty"] > 0)]

    one, buys = spy_trades(None)
    two, buys2 = spy_trades(2)
    assert len(buys) == len(buys2) == 5
    # without a band SPY trades at the first fill, at each of the 5 entries and at each of the 5 exits;
    # with a band of 2 slots it still shrinks at each entry (never over-invested) but is refilled only
    # once two positions have closed
    assert len(one) == 1 + 5 + 5 and len(two) == 1 + 5 + 2
