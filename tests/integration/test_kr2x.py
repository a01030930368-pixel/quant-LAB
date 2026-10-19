"""KRX group (D-035): prices from Naver without an API key on the Korea Exchange schedule, the 2x
inverse leg, KODEX 200 as the yardstick and idle cash at 0% (no risk-free fund)."""

import numpy as np
import pytest

from quantlab.data import providers
from quantlab.data.quality import exchange_sessions
from quantlab.services import autosearch as A
from quantlab.services.lab import Lab
from quantlab.strategies import combo

LISTED = {"122630": "2010-02-22", "233740": "2015-12-17", "252670": "2016-09-22", "069500": "2002-10-14"}


class FakeResponse:
    status_code = 200
    encoding = "euc-kr"

    def __init__(self, text: str) -> None:
        self.content = text.encode("euc-kr")
        self.text = text


def naver_text(symbol: str, seed: int) -> str:
    sessions = exchange_sessions(LISTED[symbol], "2023-12-28", "XKRX")
    rng = np.random.default_rng(seed)
    vol = 0.012 if symbol == "069500" else 0.024
    close = np.round(10000 * np.exp(np.cumsum(rng.normal(0.0002, vol, len(sessions)))))
    opn = np.round(np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, vol / 4, len(sessions))))
    hi, lo = np.maximum(opn, close) + 5, np.minimum(opn, close) - 5
    items = "".join(f'<item data="{d:%Y%m%d}|{o:.0f}|{h:.0f}|{l:.0f}|{c:.0f}|1000" />\n'
                    for d, o, h, l, c in zip(sessions, opn, hi, lo, close))
    return f'<protocol><chartdata symbol="{symbol}" count="{len(sessions)}">\n{items}</chartdata></protocol>'


@pytest.fixture()
def kr_lab(tmp_path, monkeypatch):
    texts = {s: naver_text(s, i) for i, s in enumerate(LISTED)}
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        assert url == providers.NAVER_URL
        calls.append(params["symbol"])
        return FakeResponse(texts[params["symbol"]])

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(providers, "get_api_key", lambda provider="tiingo": None)
    lab = Lab(tmp_path / "QL")
    lab.naver_calls = calls
    yield lab
    lab.close()


def test_kr2x_group_runs_without_a_key_on_krx_sessions(kr_lab):
    auto = kr_lab.auto_search()
    out = auto.run("AUTO-KR2X", workers=1, limit=6)
    assert sorted(kr_lab.naver_calls) == sorted(LISTED)  # every symbol came from Naver, no Tiingo key needed
    assert out["done"] >= 6 and out["failed"] == 0 and out["pending"] == 0
    assert kr_lab.registry.holdout_ledger() == []
    assert out["beat_symbol"] == "069500" and out["beat_metric"] == "sharpe" and out["beat_value"] is not None
    assert next(f["label"] for f in out["funnel"] if f["code"] == "spy") == "KODEX 200 보유보다 낮은 샤프 (위험 대비 수익)"
    _, loaded = auto.loaded("AUTO-KR2X")
    _, policy = kr_lab.policy_for(loaded)
    assert out["family_trials"] == out["done"]  # a new family: no prior trials
    assert auto._grid_size(loaded, auto._keyed_order(loaded, policy)) == 57 * 8 * 10 * 5 * 3 * 4
    meta = kr_lab.data.raw_records("252670")[-1]
    assert meta["source"]["provider"] == "naver" and meta["first"] == "2016-09-22"
    assert not [c for c in meta["qc"]["checks"] if c["status"] == "fatal"]  # matches the XKRX schedule


def test_inverse_leg_trades_the_other_direction(kr_lab):
    """The same oversold rule buys the inverse fund after the index rallies and the 2x fund after it falls."""
    auto = kr_lab.auto_search()
    _, loaded = auto.loaded("AUTO-KR2X")
    _, policy = kr_lab.policy_for(loaded)
    version = kr_lab.ensure_spec_dataset(loaded, policy)
    data, _ = kr_lab._strategy_symbols_data(loaded, version)
    from quantlab.validation.engine import _rf, dev_data

    dev = dev_data(data, policy)
    assert (_rf(dev, policy) == 0).all()
    params = dict(loaded.default_params, entry="rsi2_10", exit="above5", max_hold=10)
    summary = kr_lab.backtest(auto.loaded("AUTO-KR2X")[0], version, params)
    traded = set(summary.result.trades["symbol"]) if "symbol" in summary.result.trades else set()
    assert {"122630", "252670"} <= traded and "069500" not in traded


def test_spy_market_filters_need_spy_only_when_listed():
    space = combo.full_space(2)
    params = {k: {"default": v[0], "search": v} for k, v in space.items()}
    assert any("SPY" in e for e in combo.validate_params(params, 2, ["122630", "069500"]))
    params["filter"]["search"] = [f for f in space["filter"] if not f.startswith("spy_")]
    assert combo.validate_params(params, 2, ["122630", "069500"]) == []
    assert A.GROUPS[-1] == "AUTO-KR2X"


def test_unadjusted_split_is_refused(monkeypatch):
    text = ('<chartdata symbol="122630">\n<item data="20260105|100000|101000|99000|100000|1" />\n'
            '<item data="20260106|20000|20500|19800|20100|1" />\n</chartdata>')
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(text))
    with pytest.raises(providers.ProviderError, match="2026-01-06"):
        providers.fetch_naver("122630", "2010-01-01")
