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


def test_auto_search_page_runs_and_lists_results(win, tmp_path):
    import numpy as np

    from quantlab.data.quality import exchange_sessions
    from synthetic import frame

    sessions = exchange_sessions("2007-06-01", "2018-12-31")
    rng = np.random.default_rng(1)
    for sym in ["TLT", "IEF", "LQD", "GLD", "SLV", "DBC", "VNQ", "SPY", "BIL"]:
        close = 50 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(sessions))))
        df = frame(close, close)
        df.index = sessions
        p = tmp_path / f"{sym}.csv"
        df.rename_axis("date").reset_index().to_csv(p, index=False, date_format="%Y-%m-%d")
        win.lab.import_csv(sym, p)
    assert list(win.pages)[:2] == ["홈", "자동 탐색"]
    win.go("자동 탐색")
    page = win.auto
    page.choose("AUTO-REAL")
    assert page.kpi_values[0].text().startswith("0 / ")
    page.workers.setValue(1)
    page.limit = 5
    # "끝나면 계속": a finished run starts the next one by itself; stop the chain after two runs
    from quantlab.services.autosearch import AutoSearch

    runs = []
    real = AutoSearch.run

    def counted(self, *a, **k):
        runs.append(1)
        if len(runs) == 2:
            page.keep_going.setChecked(False)
        return real(self, *a, **k)

    AutoSearch.run = counted
    try:
        page.start()
    finally:
        AutoSearch.run = real
    assert len(runs) == 2
    assert not page.running
    assert page.summary["done"] >= 10 and page.summary["pending"] == 0
    assert page.funnel.rowCount() == 10
    assert win.lab.registry.holdout_ledger() == []
    # AUTO specs never appear as strategy cards on the home page
    assert not any("AUTO-" in str(p) for p in win.lab.strategy_files())


def test_auto_page_unlocks_after_a_failed_run(win, monkeypatch):
    """build-26: a run that failed (and a summary that could not be read) left the page looking busy,
    with the other groups' tabs disabled."""
    from quantlab.services.autosearch import AutoSearch

    win.go("자동 탐색")
    page = win.auto
    page.choose("AUTO-LEV")

    def broken(self, *a, **k):
        raise RuntimeError("spec refused")

    monkeypatch.setattr(AutoSearch, "run", broken)
    monkeypatch.setattr(AutoSearch, "summary", broken)
    page.start()
    assert not page.running
    assert page.start_btn.isEnabled() and not page.stop_btn.isEnabled()
    assert all(b.isEnabled() for b in page.seg_buttons.values())
    page.choose("AUTO-LEVP")
    assert page.group == "AUTO-LEVP"


def test_auto_page_lists_near_misses_and_saves_them(win, tmp_path, monkeypatch):
    import csv

    from quantlab.gui import autopage

    win.go("자동 탐색")
    page = win.auto
    near = [{"params": {"entry": "rsi2_5", "k": i}, "rule": f"규칙 {i}", "fail": "spy", "status": "SPY 보유보다 낮은 샤프",
             "n": 50, "gross": 0.02, "net": 0.01, "cagr": 0.05, "sharpe": 1.0 - i / 1000, "mdd": -0.2, "t": 2.0}
            for i in range(autopage.NEAR_SHOWN + 20)]
    s = {"group": page.group, "done": 10, "grid": 10, "family_trials": 10, "t_required": 4.0, "beat_metric": "sharpe",
         "beat_value": 1.2, "pending": 0, "funnel": [], "candidates": [], "near": near, "holdout_reads": 0}
    page._show(s)
    assert page.cands.rowCount() == 0
    page.show_near(True)
    assert page.cands.rowCount() == autopage.NEAR_SHOWN
    assert page.cands.item(0, 0).text() == "규칙 0" and page.cands.item(0, 8).text() == "—"
    assert not page.final_btn.isEnabled() and page.view_btn.isEnabled() and page.csv_btn.isEnabled()
    assert page._chosen()["rule"] == "규칙 0"
    page._show(s)  # a refresh keeps the near-miss view
    assert page.cands.rowCount() == autopage.NEAR_SHOWN
    out = tmp_path / "near.csv"
    monkeypatch.setattr(autopage.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    page.save_near_csv()
    with open(out, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "규칙" and len(rows) == len(near) + 1 and rows[1][0] == "규칙 0"
    page.show_near(False)
    assert page.cands.rowCount() == 0 and not page.final_btn.isEnabled()


def test_auto_page_final_only_for_the_preregistered_near_miss(win, monkeypatch):
    win.go("자동 탐색")
    page = win.auto
    base = {"fail": "tvalue", "n": 50, "gross": 0.02, "net": 0.01, "cagr": 0.15, "sharpe": 0.8, "mdd": -0.2, "t": 2.4}
    near = [dict(base, params={"k": 0}, rule="고정", status="사전 고정 (최종 평가 가능)", preregistered=True),
            dict(base, params={"k": 1}, rule="다른 것", status="시험 개수 대비 통계 강도 부족", preregistered=False)]
    s = {"group": page.group, "done": 2, "grid": 2, "family_trials": 2, "t_required": 4.0, "beat_metric": "cagr",
         "beat_value": 0.1, "pending": 0, "funnel": [], "candidates": [], "near": near, "holdout_reads": 0,
         "holdout_start": "2020-01-01"}
    page._show(s)
    page.show_near(True)
    page.cands.selectRow(0)
    assert page.final_btn.isEnabled()
    page.cands.selectRow(1)
    assert not page.final_btn.isEnabled()
    asked = []
    monkeypatch.setattr(win, "ask", lambda text: asked.append(text) or False)
    page.final_candidate()
    assert asked == []
    page.cands.selectRow(0)
    page.final_candidate()
    assert len(asked) == 1 and "미리 고정" in asked[0] and "2020년 이후" in asked[0]


def test_home_asks_for_the_sec_email_before_an_index_strategy(win, monkeypatch):
    path = next(p for p in win.lab.strategy_files() if p.name == "INS-CLUSTER.qls.yaml")
    started = []
    monkeypatch.setattr(win.runner, "start", lambda *a, **k: started.append(a[0]) or True)
    win.home.ask_email = lambda: None          # cancelled: nothing starts
    win.home.run(path)
    assert started == [] and win.lab.sec_email() is None
    notes = []
    monkeypatch.setattr(win, "notify", notes.append)
    win.home.ask_email = lambda: "not-an-email"
    win.home.run(path)
    assert started == [] and notes
    win.home.ask_email = lambda: "me@example.com"
    win.home.run(path)
    assert started == ["백테스트"] and win.lab.sec_email() == "me@example.com"
    win.home.ask_email = lambda: pytest.fail("asked again")
    win.home.run(path)
    assert started == ["백테스트", "백테스트"]


def test_paper_page_shows_the_next_orders_and_the_slots(win, monkeypatch):
    """D-030: the 모의 투자 page draws one refresh: orders at the next open, five slots, the score."""
    from PySide6.QtWidgets import QLabel

    from quantlab.services import paper as P

    win.go("모의 투자")
    assert win.paper.current()["id"] == "GAP-EARN-SP500"
    assert "갱신" in win.paper.todo.findChildren(QLabel)[0].text()
    snap = {"strategy_id": "GAP-EARN-SP500", "version": "1.0.0", "start": "2026-10-17", "asof": "2026-12-18",
            "refreshed_at": "20261218T224000Z", "started": True, "params": {"slots": 5, "hold": 30},
            "hold": 30, "ret": 0.034, "spy_ret": 0.019, "mdd": -0.021, "spy_mdd": -0.016,
            "orders": [{"side": "buy", "symbol": "AAAA", "weight": 0.2}, {"side": "sell", "symbol": "BBBB", "weight": 0.0},
                       {"side": "park", "symbol": "SPY", "weight": 0.2}],
            "holdings": [{"symbol": "BBBB", "park": False, "ret": 0.062, "spy_ret": 0.022, "sessions": 30,
                          "entry_date": "2026-11-05", "weight": 0.21},
                         {"symbol": "SPY", "park": True, "weight": 0.2}],
            "closed": [{"symbol": "FFFF", "entry": "2026-10-21", "exit": "2026-12-03", "gross": 0.05, "net": 0.044,
                        "bench": 0.019, "excess": 0.031}],
            "ledger": {"changed": [{"kind": "different"}]}}
    monkeypatch.setattr(P.Paper, "refresh", lambda self, path, say=None, today=None: snap)
    win.lab.set_sec_email("someone@example.com")
    win.paper.refresh_now()
    assert win.runner.last_error is None, win.runner.last_error
    texts = [w.text() for w in win.paper.todo.findChildren(QLabel)]
    assert any("12-21 월요일 시가" in t for t in texts)
    assert "매수" in texts and "AAAA" in texts and "BBBB" in texts
    assert win.paper.slots.count() == 5 and win.paper.kpis.count() == 4
    assert win.paper.warn.isVisibleTo(win.paper)
    assert win.paper.table.rowCount() == 1 and win.paper.table.item(0, 0).text() == "FFFF"
    win.paper._show("days")
    assert win.paper.table.rowCount() == 0          # nothing saved in this work folder


def test_paper_page_switches_the_automatic_refresh(win, monkeypatch):
    """D-031: the checkbox registers the Windows task (only in the installed app)."""
    from quantlab.services import scheduler

    win.go("모의 투자")
    assert not win.paper.auto.isEnabled()          # a source checkout cannot schedule QuantLab.exe
    done = []
    monkeypatch.setattr(scheduler, "supported", lambda: True)
    monkeypatch.setattr(scheduler, "install", lambda exe=None: done.append("install"))
    monkeypatch.setattr(scheduler, "remove", lambda: done.append("remove"))
    monkeypatch.setattr(win.lab, "has_api_key", lambda: True)
    win.lab.set_sec_email("someone@example.com")
    win.paper.refresh()
    assert win.paper.auto.isEnabled()
    win.paper.auto.setChecked(True)
    assert done == ["install"] and win.lab.paper().auto_ids() == ["GAP-EARN-SP500"]
    win.paper.auto.setChecked(False)
    assert done == ["install", "remove"] and win.lab.paper().auto_ids() == []


def test_factory_switch_registers_both_tasks_and_news_shows_once(win, monkeypatch):
    """D-033: the factory card on 자동 탐색; the app shows the factory's news when it arrives."""
    from quantlab.services import scheduler

    win.go("자동 탐색")
    page = win.auto
    assert not page.factory_on.isEnabled() and "꺼져" in page.factory_line.text()
    done = []
    monkeypatch.setattr(scheduler, "supported", lambda: True)
    monkeypatch.setattr(scheduler, "install", lambda exe=None: done.append("paper"))
    monkeypatch.setattr(scheduler, "install_factory", lambda exe=None, run_now=True: done.append("factory"))
    monkeypatch.setattr(scheduler, "remove_factory", lambda: done.append("-factory"))
    monkeypatch.setattr(scheduler, "remove", lambda: done.append("-paper"))
    page.refresh()
    page.factory_on.setChecked(True)
    assert done == ["paper", "factory"] and win.lab.factory().enabled()
    assert "켜져 있어요" in page.factory_line.text()
    page.factory_on.setChecked(False)
    assert done[2:] == ["-factory", "-paper"] and not win.lab.factory().enabled()
    page.choose("AUTO-SP500X")
    assert page.seg_buttons["AUTO-SP500X"].isChecked()
    f = win.lab.factory()
    f._news({"kind": "candidate", "id": "W01", "rule": "규칙", "t": 5.5, "cagr": 0.2, "mdd": -0.2, "start": "2026-10-20"})
    win.factory_news()
    assert "모든 기준을 통과한 전략" in win.statusBar().currentMessage()
    assert f.unseen() == []
