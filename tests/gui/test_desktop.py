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
