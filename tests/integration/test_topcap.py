"""D-044: the ten largest Korean stocks by market value at 10% each, rebalanced monthly (TOPCAP-KR), on
synthetic exchange snapshots."""

import numpy as np
import pandas as pd
import pytest

from quantlab.data import kr_dataset, kr_stocks, krx_open, providers
from quantlab.services.lab import Lab, builtin_path
from quantlab.strategies.spec import load_spec
from quantlab.strategies.topcap import validate_topcap
from kr_fixtures import FakeKrx, snapshots
from test_kr2x import FakeResponse, naver_text

END = "2021-12-30"
SPEC = builtin_path("specs", "TOPCAP-KR.qls.yaml")


def test_rule_is_fixed_before_any_run():
    loaded = load_spec(SPEC)
    assert loaded.default_params == {"top_n": 10}
    assert loaded.spec["rules"]["topcap"] == {"months": list(range(1, 13)), "park": "069500"}
    assert loaded.spec["universe"]["index"] == "KRTOP300"
    assert validate_topcap({"months": [13], "park": "069500"}, ["069500"])
    assert validate_topcap({"months": [3], "park": "X", "extra": 1}, ["069500"])


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    snaps = snapshots(end=END)[0]
    text = naver_text("069500", 0)
    mp = pytest.MonkeyPatch()
    import requests

    mp.setattr(requests, "get", lambda url, params=None, headers=None, timeout=None: FakeResponse(text))
    mp.setattr(providers, "get_api_key", lambda provider="tiingo": "good-key" if provider == "krx" else None)
    mp.setattr(kr_dataset, "latest_day", lambda today=None: pd.Timestamp(END))
    mp.setattr(krx_open, "PAUSE_SECONDS", 0.0)
    http = FakeKrx(snaps)
    real = krx_open.download
    mp.setattr(krx_open, "download", lambda *a, **k: real(*a, **(k | {"session": http, "today": "2022-06-01"})))
    lab = Lab(tmp_path_factory.mktemp("topcap") / "QL")
    lab.snaps = snaps
    yield lab
    lab.close()
    mp.undo()


def test_holds_the_ten_largest_members_at_equal_weight(lab):
    bt = lab.quick_backtest(SPEC)
    e = lab.registry.experiment(bt.experiment_id)
    assert e["kind"] == "exploratory" and lab.registry.holdout_ledger() == []
    version = e["dataset_version"]
    data, manifest = lab.load_dataset(version)
    assert manifest["qc_fatal"] == [] and manifest["inputs"]["caps"] == 1
    caps = data.events[data.events["kind"] == "kr_mktcap"]
    assert len(caps) and caps["value"].gt(0).all()
    # the market values are the exchange's, on each month's last session
    raw = pd.concat([pd.DataFrame(v) for v in lab.snaps.values() if v], ignore_index=True)
    one = caps.iloc[len(caps) // 2]
    want = raw[(raw["ISU_CD"] == one["symbol"]) & (raw["BAS_DD"] == f"{one['known']:%Y%m%d}")]["MKTCAP"]
    assert float(want.iloc[0]) == one["value"]
    sh = bt.result.shares
    stocks = [c for c in sh.columns if c != "069500"]
    tr = bt.result.trades.assign(date=lambda d: pd.to_datetime(d["date"]))
    assert tr.loc[tr["symbol"] == "069500", "qty"].iloc[0] > 0           # parked before the first pick
    # at every month end: next session holds exactly the ten largest members by that close's value
    wide = caps.pivot(index="known", columns="symbol", values="value")
    member = data.members
    checked = 0
    for day in wide.index[3:-1]:
        i = sh.index.get_indexer([day])[0]
        if i < 0 or i + 2 >= len(sh):
            continue
        ok = [s for s in wide.columns if member.at[day, s] and np.isfinite(wide.at[day, s])]
        top = sorted(ok, key=lambda s: (-wide.at[day, s], s))[:10]
        traded = {s for s in stocks if np.isfinite(data.frames[s]["close"].iloc[i + 1])}   # a delisted name stays
        held = {s for s in stocks if sh.iloc[i + 1][s] > 0} & traded
        stuck = {s for s in stocks if sh.iloc[i + 1][s] > 0} - traded
        if not set(top) <= traded or stuck:
            continue      # a delisted name cannot be sold, so its money is not there to buy the new pick
        assert held == set(top)
        px = {s: data.frames[s]["open"].iloc[i + 1] for s in top}
        values = np.array([sh.iloc[i + 1][s] * px[s] for s in top])
        assert np.nanmax(values) / np.nanmin(values) < 1.15             # brought back to ~10% each (a halt has no open)
        checked += 1
    assert checked > 50
    sells = tr[(tr["qty"] < 0) & (tr["symbol"] != "069500")]
    assert len(sells) and (sells["commission"] > sells["notional"].abs() * 21e-4).all()   # tax paid


def test_the_stock_datasets_of_other_rules_keep_their_version(lab):
    loaded = load_spec(builtin_path("specs", "AUTO-KRSTK.qls.yaml"))
    _, policy = lab.policy_for(loaded)
    version = lab.ensure_spec_dataset(loaded, policy)
    manifest = lab.load_dataset(version)[1]
    assert "caps" not in manifest["inputs"]
    data = lab.load_dataset(version)[0]
    assert not (data.events["kind"] == "kr_mktcap").any() if len(data.events) else True
    assert kr_stocks.TOP_N == 300
