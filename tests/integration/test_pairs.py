"""D-041: Korean sector funds paired with the 1x inverse KOSPI 200 fund, long only, prices from Naver."""

import numpy as np
import pandas as pd
import pytest

from quantlab.data import providers
from quantlab.data.quality import exchange_sessions
from quantlab.services.lab import Lab, builtin_path
from quantlab.strategies.pairs import validate_pairs
from quantlab.strategies.spec import load_spec

SPEC = builtin_path("specs", "PAIR-KRSECT.qls.yaml")
SECTORS = ["091160", "091170", "091180", "102960", "102970", "117460", "117680", "117700", "140700", "140710"]
BASE, HEDGE = "069500", "114800"
LATE = {"140700": "2011-04-26", "140710": "2011-04-26"}  # the insurance and transport funds list in 2011
END = "2021-03-31"  # under Naver's 3,000-session chart cap from 2009-06


def test_rule_is_fixed_before_any_run():
    loaded = load_spec(SPEC)
    cfg = loaded.spec["rules"]["pairs"]
    assert cfg["trade"] == SECTORS and (cfg["base"], cfg["hedge"]) == (BASE, HEDGE)
    assert 2 * cfg["slots"] * cfg["leg_weight"] == pytest.approx(1.0)
    assert loaded.default_params == {"lookback": 60, "entry_z": 2.0, "max_hold": 20}
    assert loaded.spec["strategy"]["derived_after_holdout_view"] is False


def test_validation_rejects_bad_blocks():
    syms = ["A", "B", "M", "H"]
    ok = {"trade": ["A", "B"], "base": "M", "hedge": "H", "leg_weight": 0.25, "slots": 2}
    assert validate_pairs(ok, syms) == []
    assert validate_pairs(dict(ok, leg_weight=0.3), syms)   # 2 pairs x 2 legs x 30% > 100%
    assert validate_pairs(dict(ok, hedge="M"), syms)
    assert validate_pairs(dict(ok, base="A"), syms)
    assert validate_pairs(dict(ok, trade=["C"]), syms)
    assert validate_pairs(dict(ok, slots=0), syms)


class FakeResponse:
    status_code = 200
    encoding = "euc-kr"

    def __init__(self, text: str) -> None:
        self.content = text.encode("euc-kr")
        self.text = text


def prices(seed: int) -> dict[str, pd.DataFrame]:
    """A common market factor; each sector = market x its own mean-reverting gap; inverse = -1x daily."""
    sessions = exchange_sessions("2009-06-01", END, "XKRX")
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0002, 0.011, len(sessions))
    out = {BASE: np.cumsum(mkt)}
    for i, s in enumerate(SECTORS):
        gap = np.zeros(len(sessions))
        shock = rng.normal(0, 0.012, len(sessions))
        for t in range(1, len(sessions)):
            gap[t] = 0.9 * gap[t - 1] + shock[t]
        out[s] = out[BASE] * (0.8 + 0.05 * i) + gap
    frames = {}
    for s, logp in out.items():
        close = np.round(10000 * np.exp(logp))
        frames[s] = pd.DataFrame({"close": close}, index=sessions)
    inv = np.round(10000 * np.cumprod(1 - mkt))
    frames[HEDGE] = pd.DataFrame({"close": inv}, index=sessions)
    frames[HEDGE] = frames[HEDGE][frames[HEDGE].index >= "2009-09-16"]
    for s, d in LATE.items():
        frames[s] = frames[s][frames[s].index >= d]
    return frames


def naver_text(symbol: str, df: pd.DataFrame) -> str:
    c = df["close"].to_numpy()
    o = np.r_[c[0], c[:-1]]
    items = "".join(f'<item data="{d:%Y%m%d}|{a:.0f}|{max(a, b) + 5:.0f}|{min(a, b) - 5:.0f}|{b:.0f}|1000" />\n'
                    for d, a, b in zip(df.index, o, c))
    return f'<protocol><chartdata symbol="{symbol}" count="{len(df)}">\n{items}</chartdata></protocol>'


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    texts = {s: naver_text(s, df) for s, df in prices(5).items()}
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        assert url == providers.NAVER_URL
        calls.append(params["symbol"])
        return FakeResponse(texts[params["symbol"]])

    import requests

    mp = pytest.MonkeyPatch()
    mp.setattr(requests, "get", fake_get)
    mp.setattr(providers, "get_api_key", lambda provider="tiingo": None)
    lab = Lab(tmp_path_factory.mktemp("pairs") / "QL")
    lab.naver_calls = calls
    yield lab
    lab.close()
    mp.undo()


def test_backtest_opens_pairs_with_the_inverse_fund_on_dev_only(lab):
    bt = lab.quick_backtest(SPEC)
    assert sorted(set(lab.naver_calls)) == sorted(SECTORS + [BASE, HEDGE])
    e = lab.registry.experiment(bt.experiment_id)
    assert e["kind"] == "exploratory" and e["metrics"]["window"][1] == "2019-12-30"  # KRX is shut on 12-31
    tr = bt.result.trades
    tr = tr.assign(date=pd.to_datetime(tr["date"]))
    buys = tr[tr["symbol"].isin(SECTORS) & (tr["qty"] > 0)]
    sells = tr[tr["symbol"].isin(SECTORS) & (tr["qty"] < 0)]
    assert len(buys) >= 40 and len(sells) >= 40
    hedge_buys = tr[(tr["symbol"] == HEDGE) & (tr["qty"] > 0)]
    # every sector buy comes with an inverse buy, unless another pair closed that day (the hedge stays)
    assert set(buys["date"]) <= set(hedge_buys["date"]) | set(sells["date"])
    assert len(set(buys["date"]) & set(hedge_buys["date"])) >= 0.5 * buys["date"].nunique()
    assert not len(buys[buys["symbol"].isin(LATE) & (buys["date"] < "2011-04-26")])
    assert not len(tr[tr["symbol"] == BASE])               # KODEX 200 is the yardstick, never traded
    # never more than two sectors at once, and each pair closes within 20 sessions of its signal
    held: dict[str, pd.Timestamp] = {}
    sessions = exchange_sessions("2009-06-01", END, "XKRX")
    for _, r in tr[tr["symbol"].isin(SECTORS)].sort_values(["date", "qty"]).iterrows():
        if r["qty"] > 0:
            held[r["symbol"]] = r["date"]
            assert len(held) <= 2
        else:
            opened = held.pop(r["symbol"])
            n = int(sessions.get_loc(r["date"]) - sessions.get_loc(opened))
            assert 1 <= n <= 20
    assert lab.registry.holdout_ledger() == []


def test_the_inverse_leg_keeps_the_market_out(lab):
    # synthetic gaps revert by construction, so the pairs earn while hardly moving with the market
    bt = lab.quick_backtest(SPEC)
    r = bt.result.returns()
    shares = bt.result.shares
    held = shares[SECTORS].abs().sum(axis=1) > 0
    base = prices(5)[BASE]["close"].reindex(shares.index).pct_change().fillna(0.0)
    assert held.mean() > 0.2
    assert abs(np.corrcoef(r[held], base[held])[0, 1]) < 0.5  # one leg alone would be about 0.9
    assert bt.result.equity.iloc[-1] > bt.result.equity.iloc[0]


def test_final_evaluation_passes_the_integrity_checks(lab):
    out = lab.quick_final(SPEC)
    gates = {v["gate"]: v["verdict"] for v in lab.registry.verdicts(out.experiment_id)}
    assert gates["GATE1"] == "PASS"
