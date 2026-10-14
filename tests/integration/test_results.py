"""Results: report items, packet contents and holdout-sharing lineage (M12)."""

import json
import zipfile

import pytest
import yaml

from quantlab.analysis.report import SECTIONS
from quantlab.services.lab import Lab, builtin_path
from quantlab.services.selftest import write_synthetic_csvs


@pytest.fixture(scope="module")
def done(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("res")
    lab = Lab(tmp / "QL")
    for sym, p in write_synthetic_csvs(tmp).items():
        lab.import_csv(sym, p)
    lab.version, _ = lab.build_dataset(["SPY", "BIL"])
    lab.imp = lab.import_strategy(builtin_path("specs", "TT-SPY-SMA.qls.yaml"))
    lab.bt = lab.backtest(lab.imp.strategy_hash, lab.version)
    lab.confirm_candidate(lab.imp.strategy_hash)
    lab.final = lab.final_evaluation(lab.imp.strategy_hash, lab.version)
    yield lab, tmp
    lab.close()


def test_report_has_all_14_items_and_charts(done):
    lab, _ = done
    for eid in (lab.bt.experiment_id, lab.final.experiment_id):
        p = lab.write_report(eid)
        text = p.read_text(encoding="utf-8")
        for title in SECTIONS:
            assert f"## {title}" in text, title
        assert (p.parent / "report.html").exists() and (p.parent / "equity.png").exists()


def test_packet_contents_and_no_key(done, monkeypatch):
    lab, _ = done
    monkeypatch.setenv("TIINGO_API_KEY", "sk-test-not-a-real-key-123")
    pk = lab.export_packet(lab.final.experiment_id)
    with zipfile.ZipFile(pk) as z:
        names = set(z.namelist())
        blob = b"".join(z.read(n) for n in names)
    assert {"report.md", "metrics.json", "policy.yaml", "strategy.qls.yaml", "daily_equity.csv", "trades.csv",
            "app.json"} <= names
    assert b"sk-test-not-a-real-key-123" not in blob
    assert not any(n.endswith(".parquet") or n.startswith(("data/", "raw/")) for n in names)


def test_derived_strategy_after_shared_holdout_cannot_pass(done, tmp_path):
    lab, _ = done
    lab.export_packet(lab.final.experiment_id)
    assert lab.registry.holdout_shared("TT-SPY-SMA")
    spec = yaml.safe_load(builtin_path("specs", "TT-SPY-SMA.qls.yaml").read_text(encoding="utf-8"))
    spec["strategy"].update(id="TT-SPY-SMA", version="2.0.0", parent="TT-SPY-SMA@1.0.0", derived_after_holdout_view=False)
    spec["rules"]["signal"]["lookback_months"] = 9
    p = tmp_path / "d.qls.yaml"
    p.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    imp = lab.import_strategy(p)
    from quantlab.validation.engine import lineage_independence
    probs = lineage_independence(lab.registry, "TT-SPY-SMA", "2.0.0")
    assert any("exported" in x for x in probs)


def test_compare_table(done):
    lab, _ = done
    df = lab.compare([lab.bt.experiment_id, lab.final.experiment_id])
    assert list(df["kind"]) == ["exploratory", "formal"]
