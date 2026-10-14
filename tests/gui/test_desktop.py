"""Desktop screens (M13): the main flow through the GUI, headless (QT_QPA_PLATFORM=offscreen)."""

import os
import subprocess
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from quantlab.gui.main_window import MainWindow  # noqa: E402
from quantlab.services.lab import Lab, builtin_path  # noqa: E402
from quantlab.services.selftest import write_synthetic_csvs  # noqa: E402


@pytest.fixture
def win(qtbot, tmp_path):
    w = MainWindow(Lab(tmp_path / "QuantLab"), sync=True)
    qtbot.addWidget(w)
    yield w
    w.lab.close()


def test_six_screens(win):
    assert win.version_label.text() == "개발 버전"  # a frozen build shows its release tag here
    assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == ["데이터", "전략", "실험", "검증", "결과", "기록"]


def test_csv_to_results_flow(win, tmp_path):
    csvs = write_synthetic_csvs(tmp_path)
    for sym, p in csvs.items():
        win.data.csv_symbol.setText(sym)
        win.data.pick_csv(str(p))
    assert win.data.raw.rowCount() == 2
    win.data.ds_symbols.setText(" ".join(csvs))
    win.data.build()
    assert win.data.ds.rowCount() == 1

    win.strategies.import_spec(str(builtin_path("specs", "TT-SPY-SMA.qls.yaml")))
    assert win.strategies.table.rowCount() == 1
    assert win.strategies.rows[0]["state"] == "RESEARCH"

    win.experiments.refresh()
    win.experiments.run_one()
    assert win.runner.last_error is None, win.runner.last_error
    assert "샤프 차이" in win.experiments.summary.text()
    assert win.results.charts.count() == 3  # equity, drawdown, monthly
    eid = win.results.exp.currentData()
    assert eid and win.lab.registry.experiment(eid)["kind"] == "exploratory"

    # final evaluation: confirm first, then the dialog must be accepted
    win.validation.final()  # no confirmation yet -> nothing happens
    assert not any(e["kind"] == "formal" for e in win.lab.registry.history())
    win.validation.confirm()
    win.validation.ask = lambda text: False
    win.validation.final()
    assert not any(e["kind"] == "formal" for e in win.lab.registry.history())
    asked = []
    win.validation.ask = lambda text: asked.append(text) or True
    win.validation.final()
    assert "한 번만" in asked[0]
    assert "최종 판정" in win.validation.info.text()
    assert win.validation.ledger.rowCount() == 1

    win.results.report()
    assert win.results.last_path.exists()
    win.results.packet()
    assert win.results.last_path.suffix == ".zip"
    assert win.history.exps.rowCount() >= 2


def test_engine_does_not_import_qt():
    code = ("import sys, quantlab.services.lab, quantlab.services.batch, quantlab.analysis.report, quantlab.cli; "
            "sys.exit(1 if any(m.startswith('PySide6') for m in sys.modules) else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_app_selftest_flag():
    from quantlab.gui.app import main

    assert main(["QuantLab", "--selftest"]) == 0


def test_dataset_symbols_prefilled_and_builtin_button(win, tmp_path):
    for sym, p in write_synthetic_csvs(tmp_path).items():
        win.data.csv_symbol.setText(sym)
        win.data.pick_csv(str(p))
    assert set(win.data.ds_symbols.text().split()) == {"SPY", "BIL"}
    [b for b in win.strategies.actions if "기준 전략" in b.text()][0].click()
    assert win.strategies.rows[0]["strategy_id"] == "TT-SPY-SMA"


def test_home_one_click_backtest_and_final(win, tmp_path, monkeypatch):
    """Home: one button runs import + data + backtest and lands on the result page."""
    monkeypatch.setattr(win.lab, "has_api_key", lambda: False)
    assert win.stack.currentWidget() is win.home
    ids = [p.name for p in win.lab.strategy_files()]
    assert "TT-SPY-SMA.qls.yaml" in ids and len(win.home.cards) == len(ids)

    # no price data and no key: a clear Korean error, nothing recorded
    tt = next(p for p in win.lab.strategy_files() if p.name == "TT-SPY-SMA.qls.yaml")
    win.home.run(tt)
    assert "API 키" in (win.runner.last_error or "")
    assert not win.home.progress.isVisible()
    win.runner.last_error = None

    for sym, p in write_synthetic_csvs(tmp_path).items():
        win.lab.import_csv(sym, p)
    win.home.run(tt)
    assert win.runner.last_error is None, win.runner.last_error
    assert win.stack.currentWidget() is win.result
    eid = win.result.pick.currentData()
    assert win.lab.registry.experiment(eid)["kind"] == "exploratory"

    asked = []
    win.ask = lambda text: asked.append(text) or True
    win.home.final(tt, "TT-SPY-SMA")
    assert win.runner.last_error is None, win.runner.last_error
    assert "최종 평가" in asked[0]
    formal = [e for e in win.lab.registry.history() if e["kind"] == "formal"]
    assert len(formal) == 1 and win.result.pick.currentData() == formal[0]["experiment_id"]
