"""Service-layer data access rules (M10)."""

import pytest

from quantlab.services.lab import Lab, builtin_path
from quantlab.services.selftest import write_synthetic_csvs


@pytest.fixture
def lab(tmp_path):
    lab = Lab(tmp_path / "QL")
    for sym, p in write_synthetic_csvs(tmp_path).items():
        lab.import_csv(sym, p)
    lab.version, _ = lab.build_dataset(["SPY", "BIL"])
    lab.imp = lab.import_strategy(builtin_path("specs", "TT-SPY-SMA.qls.yaml"))
    yield lab
    lab.close()


def test_exploratory_backtest_never_sees_holdout(lab):
    out = lab.backtest(lab.imp.strategy_hash, lab.version)
    assert out.result.equity.index.max() <= __import__("pandas").Timestamp("2018-12-31")
    assert lab.registry.holdout_ledger() == []


def test_final_evaluation_requires_confirmation_and_uses_confirmed_params(lab):
    with pytest.raises(PermissionError):
        lab.final_evaluation(lab.imp.strategy_hash, lab.version)
    lab.confirm_candidate(lab.imp.strategy_hash, {"lookback_months": 12})
    assert lab.registry.current_state("TT-SPY-SMA") == "VALIDATION"
    out = lab.final_evaluation(lab.imp.strategy_hash, lab.version)
    e = lab.registry.experiment(out.experiment_id)
    assert e["config"]["strategy"]["params"]["lookback_months"] == 12
    assert e["strategy_hash"] == lab.imp.strategy_hash and e["strategy_origin"] == "claude"
    assert len(lab.registry.holdout_ledger()) == 1


def test_cli_selftest_exit_code():
    from quantlab.cli import main
    assert main(["selftest"]) == 0


def test_example_specs_copied_to_work_folder(tmp_path):
    from quantlab.services.lab import Lab

    lab = Lab(tmp_path / "QL")
    try:
        assert (tmp_path / "QL" / "examples" / "TT-SPY-SMA.qls.yaml").is_file()
    finally:
        lab.close()


def test_builtin_specs_import_and_resolve_their_policies(tmp_path):
    from quantlab.services.lab import Lab, builtin_path

    lab = Lab(tmp_path / "QL")
    try:
        for spec in sorted(builtin_path("specs").glob("*.qls.yaml")):
            res = lab.import_strategy(spec)
            assert res.state == "RESEARCH", (spec.name, res.problems)
            _, policy = lab.policy_for(res.loaded)
            assert policy["strategy_family"] == res.strategy_id
            un = res.loaded.spec["universe"]
        assert set(un["symbols"]) | ({un["index"]} if "index" in un else set()) == set(policy["universe"])
    finally:
        lab.close()
