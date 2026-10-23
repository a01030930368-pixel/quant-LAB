"""QUANT LAB desktop window: six screens over the shared `Lab` services (03 v2.0.0 §6).

The screens hold no research logic: every action calls a `Lab` method, so the GUI and the
CLI produce identical records. Long actions run one at a time on a worker thread; while a
job runs, the action buttons are disabled and the registry is not read from the UI thread.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QComboBox, QDateEdit, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter, QStackedWidget, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from quantlab.analysis import charts
from quantlab.services.lab import Lab, builtin_path

PCT = {"cagr", "total_return", "ann_vol", "max_drawdown", "excess_cagr", "trade.mean_gross", "trade.median_gross",
       "trade.mean_gross_ex_top10"}
LABELS = {"cagr": "CAGR", "total_return": "총수익", "ann_vol": "연변동성", "sharpe": "샤프", "sortino": "소르티노",
          "max_drawdown": "최대 낙폭", "calmar": "칼마", "n_trades": "체결 수", "turnover": "연 회전율",
          "psr_vs_0": "PSR(샤프>0)", "sharpe_diff": "샤프 차이", "trade.n": "거래 수",
          "trade.mean_gross": "거래당 평균(비용 전)", "trade.median_gross": "거래당 중앙값(비용 전)",
          "trade.mean_gross_ex_top10": "상위 10건 뺀 평균", "trade.t_day_clustered": "t값(같은 날 묶음)"}


# ----------------------------------------------------------------------------- jobs
class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)
    text = Signal(str)


class _Job(QRunnable):
    def __init__(self, fn: Callable, signals: _Signals) -> None:
        super().__init__()
        self.fn, self.signals = fn, signals

    def run(self) -> None:
        try:
            def prog(a, b):
                self.signals.progress.emit(a, b)

            prog.say = self.signals.text.emit  # step text for the progress card
            out = self.fn(prog)
        except Exception as e:  # shown to the user, never swallowed
            self.signals.failed.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc(limit=4)}")
        else:
            self.signals.done.emit(out)


class Runner(QObject):
    """One background job at a time. `sync=True` runs inline (tests)."""

    busy_changed = Signal(bool)
    message = Signal(str)

    def __init__(self, sync: bool = False) -> None:
        super().__init__()
        self.sync = sync
        self.busy = False
        self.pool = QThreadPool.globalInstance()
        self._keep: list[_Signals] = []
        self.last_error: str | None = None

    def start(self, label: str, fn: Callable[[Callable[[int, int], None]], Any], on_done: Callable[[Any], None],
              on_progress: Callable[[int, int], None] | None = None,
              on_text: Callable[[str], None] | None = None) -> bool:
        if self.busy:
            self.message.emit("다른 작업이 진행 중입니다. 끝난 뒤 다시 실행하세요.")
            return False
        self._set_busy(True)
        self.message.emit(f"{label} 실행 중…")
        sig = _Signals()
        self._keep.append(sig)

        def finish(out):
            self._set_busy(False)
            self._keep.remove(sig)
            self.message.emit(f"{label} 완료")
            on_done(out)

        def fail(text):
            self._set_busy(False)
            self._keep.remove(sig)
            self.message.emit(f"{label} 실패")
            self.last_error = text
            QMessageBox.warning(None, "QUANT LAB", f"{label} 실패\n\n{text}") if not self.sync else None

        sig.done.connect(finish)
        sig.failed.connect(fail)
        if on_progress:
            sig.progress.connect(on_progress)
        sig.text.connect(self.message.emit)
        if on_text:
            sig.text.connect(on_text)
        job = _Job(fn, sig)
        if self.sync:
            job.run()
        else:
            self.pool.start(job)
        return True

    def _set_busy(self, b: bool) -> None:
        self.busy = b
        self.busy_changed.emit(b)


# ----------------------------------------------------------------------------- helpers
def _fmt(k: str, v: Any) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, float):
        return f"{v * 100:.2f}%" if k in PCT else f"{v:.3f}"
    return str(v)


def _fill(table: QTableWidget, rows: list[dict], cols: list[tuple[str, str]]) -> None:
    table.clear()
    table.setColumnCount(len(cols))
    table.setHorizontalHeaderLabels([c[1] for c in cols])
    table.setRowCount(len(rows))
    for i, r in enumerate(rows):
        for j, (k, _) in enumerate(cols):
            it = QTableWidgetItem(_fmt(k, r.get(k)))
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            table.setItem(i, j, it)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)


def _table() -> QTableWidget:
    t = QTableWidget()
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setAlternatingRowColors(True)
    t.verticalHeader().setVisible(False)
    return t


def _note(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setObjectName("muted")
    return lab


class Screen(QWidget):
    def __init__(self, win: "MainWindow") -> None:
        super().__init__()
        self.win = win
        self.actions: list[QPushButton] = []

    @property
    def lab(self) -> Lab:
        return self.win.lab

    def button(self, text: str, slot) -> QPushButton:
        b = QPushButton(text)
        b.clicked.connect(slot)
        self.actions.append(b)
        return b

    def refresh(self) -> None:
        pass


# ----------------------------------------------------------------------------- 1. data
class DataScreen(Screen):
    def __init__(self, win):
        super().__init__(win)
        lay = QVBoxLayout(self)
        key = QGroupBox("Tiingo API 키")
        kl = QHBoxLayout(key)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("키를 붙여넣고 저장 (Windows 자격 증명 관리자에 보관, 파일·기록에 남지 않음)")
        kl.addWidget(self.key)
        kl.addWidget(self.button("저장", self.save_key))
        lay.addWidget(key)

        krx = QGroupBox("KRX Open API 인증키 (한국 주식, openapi.krx.co.kr)")
        xl = QHBoxLayout(krx)
        self.krx_key = QLineEdit()
        self.krx_key.setEchoMode(QLineEdit.Password)
        self.krx_key.setPlaceholderText("키를 붙여넣고 저장 (Windows 자격 증명 관리자에 보관)")
        xl.addWidget(self.krx_key)
        xl.addWidget(self.button("저장", self.save_krx_key))
        lay.addWidget(krx)

        dart = QGroupBox("OpenDART 인증키 (한국 공시, opendart.fss.or.kr)")
        dl = QHBoxLayout(dart)
        self.dart_key = QLineEdit()
        self.dart_key.setEchoMode(QLineEdit.Password)
        self.dart_key.setPlaceholderText("키를 붙여넣고 저장 (Windows 자격 증명 관리자에 보관)")
        dl.addWidget(self.dart_key)
        dl.addWidget(self.button("저장", self.save_dart_key))
        lay.addWidget(dart)

        fetch = QGroupBox("가격 내려받기 (일봉, 원가격 + 배당·분할)")
        fl = QHBoxLayout(fetch)
        self.symbols = QLineEdit("SPY BIL")
        self.start = QDateEdit()
        self.start.setDisplayFormat("yyyy-MM-dd")
        self.start.setDate(pd.Timestamp("2007-06-01").date())
        fl.addWidget(QLabel("종목"))
        fl.addWidget(self.symbols, 2)
        fl.addWidget(QLabel("시작일"))
        fl.addWidget(self.start)
        fl.addWidget(self.button("내려받기", self.fetch))
        lay.addWidget(fetch)

        csv = QGroupBox("CSV 가져오기 (열: date, open, high, low, close, dividend, split)")
        cl = QHBoxLayout(csv)
        self.csv_symbol = QLineEdit()
        self.csv_symbol.setPlaceholderText("종목 코드")
        cl.addWidget(self.csv_symbol)
        cl.addWidget(self.button("파일 선택…", self.pick_csv))
        lay.addWidget(csv)

        idx = QGroupBox("S&P 500 종목 · SEC 내부자 거래 (지수 전략용)")
        il = QHBoxLayout(idx)
        self.sec_email = QLineEdit()
        self.sec_email.setPlaceholderText("SEC 연락용 이메일 (SEC 규칙; 이 PC와 sec.gov에만 전달)")
        il.addWidget(self.sec_email, 2)
        il.addWidget(self.button("저장", self.save_sec_email))
        il.addWidget(self.button("S&P 500 가격 새로 받기", self.refresh_sp500))
        self.index_info = QLabel("")
        il.addWidget(self.index_info, 2)
        lay.addWidget(idx)

        lay.addWidget(QLabel("받은 자료 (최신 기록, 품질 검사 결과)"))
        self.raw = _table()
        lay.addWidget(self.raw, 2)
        row = QHBoxLayout()
        self.ds_symbols = QLineEdit()
        self.ds_symbols.setPlaceholderText("데이터셋에 넣을 종목 (예: SPY BIL)")
        row.addWidget(self.ds_symbols)
        row.addWidget(self.button("데이터셋 만들기", self.build))
        lay.addLayout(row)
        lay.addWidget(QLabel("데이터셋"))
        self.ds = _table()
        lay.addWidget(self.ds, 1)

    def save_key(self):
        k = self.key.text().strip()
        if not k:
            return
        self.lab.set_api_key(k)
        self.key.clear()
        self.win.status("API 키를 저장했습니다.")

    def save_krx_key(self):
        k = self.krx_key.text().strip()
        if k:
            self.lab.set_api_key(k, "krx")
            self.krx_key.clear()
            self.win.status("KRX 인증키를 저장했습니다.")

    def save_dart_key(self):
        k = self.dart_key.text().strip()
        if k:
            self.lab.set_api_key(k, "dart")
            self.dart_key.clear()
            self.win.status("OpenDART 인증키를 저장했습니다.")

    def save_sec_email(self):
        try:
            self.lab.set_sec_email(self.sec_email.text())
        except ValueError as e:
            self.win.notify(str(e))
            return
        self.win.status("SEC 연락용 이메일을 저장했습니다.")
        self.refresh()

    def refresh_sp500(self):
        from quantlab.data import sp500

        self.win.runner.start("S&P 500 가격 받기", lambda p: sp500.download_bundle(self.lab.ws.root, say=p.say),
                              lambda _: self.win.refresh_all())

    def fetch(self):
        syms = self.symbols.text().upper().split()
        start = self.start.date().toString("yyyy-MM-dd")
        self.win.runner.start("내려받기", lambda p: self.lab.fetch(syms, start), lambda _: self.win.refresh_all())

    def pick_csv(self, path: str | None = None):
        sym = self.csv_symbol.text().strip().upper()
        if not sym:
            self.win.notify("먼저 종목 코드를 입력하세요.")
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "CSV 선택", "", "CSV (*.csv)")
        if path:
            self.win.runner.start("CSV 가져오기", lambda p: self.lab.import_csv(sym, path),
                                  lambda _: self.win.refresh_all())

    def build(self):
        syms = self.ds_symbols.text().upper().split()
        if syms:
            self.win.runner.start("데이터셋 만들기", lambda p: self.lab.build_dataset(syms),
                                  lambda _: self.win.refresh_all())

    def refresh(self):
        from quantlab.data import sec_insider, sp500

        if self.lab.sec_email() and not self.sec_email.text():
            self.sec_email.setText(self.lab.sec_email())
        b = sp500.bundles(self.lab.ws.root)
        q = sorted(sec_insider.records(self.lab.ws.root))
        self.index_info.setText(f"S&P 500 가격: {b[-1]['fetched_at'][:8] if b else '없음'} · "
                                f"SEC 분기: {f'{q[0]}~{q[-1]} ({len(q)}개)' if q else '없음'}")
        latest: dict[str, dict] = {}
        for m in self.lab.data.raw_records():
            latest[m["symbol"]] = m
        rows = []
        for s, m in sorted(latest.items()):
            checks = m["qc"]["checks"]
            rows.append({"symbol": s, "provider": m["source"].get("provider"), "first": m["first"], "last": m["last"],
                         "rows": m["rows"], "fatal": sum(c["status"] == "fatal" for c in checks),
                         "warn": sum(c["status"] == "warn" for c in checks), "sha": m["raw_sha256"][:12]})
        # default: every symbol received, until the user types their own list
        auto = " ".join(r["symbol"] for r in rows)
        if self.ds_symbols.text().strip() in ("", getattr(self, "_auto", "")):
            self.ds_symbols.setText(auto)
        self._auto = auto
        _fill(self.raw, rows, [("symbol", "종목"), ("provider", "출처"), ("first", "시작"), ("last", "끝"),
                               ("rows", "행"), ("fatal", "치명"), ("warn", "경고"), ("sha", "원본 해시")])
        ds = [{"version": d["dataset_version"][:16], "symbols": " ".join(d["symbols"]), "fatal": len(d["qc_fatal"]),
               "warn": len(d["qc_warnings"])} for d in self.lab.data.datasets()]
        _fill(self.ds, ds, [("version", "버전"), ("symbols", "종목"), ("fatal", "치명"), ("warn", "경고")])


# ----------------------------------------------------------------------------- 2. strategies
class StrategyScreen(Screen):
    def __init__(self, win):
        super().__init__(win)
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(self.button("명세서 가져오기 (.qls.yaml)…", self.import_spec))
        top.addWidget(self.button("기준 전략 가져오기 (앱 동작 확인용)",
                                  lambda: self.import_spec(str(builtin_path("specs", "TT-SPY-SMA.qls.yaml")))))
        top.addStretch()
        lay.addLayout(top)
        lay.addWidget(_note("가져올 때 형식 검사, 합성 데이터 시험, 미래 데이터 누출 검사를 거칩니다. "
                            "통과하면 RESEARCH, 아니면 IDEA로 남습니다. 코드형 전략은 믿을 수 있는 출처의 파일만 가져오세요."))
        split = QSplitter(Qt.Vertical)
        self.table = _table()
        self.table.itemSelectionChanged.connect(self.show_spec)
        split.addWidget(self.table)
        self.spec = QPlainTextEdit()
        self.spec.setReadOnly(True)
        split.addWidget(self.spec)
        lay.addWidget(split)
        self.rows: list[dict] = []

    def import_spec(self, path: str | None = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "전략 명세서", str(self.lab.ws.path("examples")), "QUANT LAB 명세서 (*.qls.yaml *.yaml)")
        if not path:
            return

        def done(res):
            self.win.refresh_all()
            msg = f"{res.strategy_id}@{res.version}: {res.state}"
            if res.problems:
                msg += "\n\n" + "\n".join(res.problems)
            self.win.status(msg.splitlines()[0])
            if not self.win.runner.sync:
                QMessageBox.information(self, "가져오기 결과", msg)

        self.win.runner.start("전략 가져오기", lambda p: self.lab.import_strategy(path), done)

    def refresh(self):
        self.rows = []
        for s in self.lab.registry.strategies():
            self.rows.append({**s, "state": self.lab.registry.current_state(s["strategy_id"]),
                              "hash": s["strategy_hash"][:12],
                              "derived": "예" if s["derived_after_holdout_view"] else "아니오"})
        _fill(self.table, self.rows, [("strategy_id", "전략"), ("version", "버전"), ("state", "상태"),
                                      ("origin", "출처"), ("parent", "부모"), ("derived", "보류 결과 본 뒤 파생"),
                                      ("hash", "해시")])

    def show_spec(self):
        r = self.table.currentRow()
        if 0 <= r < len(self.rows):
            body = self.lab.registry.strategy(self.rows[r]["strategy_hash"])["body"]
            text = json.dumps(body["spec"], indent=1, ensure_ascii=False)
            if body.get("code"):
                text += "\n\n# ---- code ----\n" + body["code"]
            self.spec.setPlainText(text)


# ----------------------------------------------------------------------------- shared selectors
class Pickers(QWidget):
    """Strategy + dataset selectors used by the experiment and validation screens."""

    def __init__(self, screen: Screen) -> None:
        super().__init__()
        self.screen = screen
        f = QFormLayout(self)
        f.setContentsMargins(0, 0, 0, 0)
        self.strategy = QComboBox()
        self.dataset = QComboBox()
        f.addRow("전략", self.strategy)
        f.addRow("데이터셋", self.dataset)

    def refresh(self):
        lab = self.screen.lab
        cur_s, cur_d = self.strategy.currentData(), self.dataset.currentData()
        self.strategy.clear()
        for s in lab.registry.strategies():
            state = lab.registry.current_state(s["strategy_id"])
            self.strategy.addItem(f"{s['strategy_id']}@{s['version']} ({state})", s["strategy_hash"])
        self.dataset.clear()
        for d in lab.data.datasets():
            self.dataset.addItem(f"{d['dataset_version'][:16]} · {' '.join(d['symbols'])}", d["dataset_version"])
        for box, cur in ((self.strategy, cur_s), (self.dataset, cur_d)):
            i = box.findData(cur)
            if i >= 0:
                box.setCurrentIndex(i)

    def chosen(self) -> tuple[str | None, str | None]:
        return self.strategy.currentData(), self.dataset.currentData()


# ----------------------------------------------------------------------------- 3. experiments
class ExperimentScreen(Screen):
    def __init__(self, win):
        super().__init__(win)
        lay = QVBoxLayout(self)
        lay.addWidget(_note("탐색 실험은 TRAIN+VALIDATION 구간만 사용합니다. 최종 보류 구간(HOLDOUT)은 이 화면에서 절대 열리지 않습니다."))
        self.pick = Pickers(self)
        lay.addWidget(self.pick)
        one = QGroupBox("단일 백테스트")
        ol = QHBoxLayout(one)
        self.params = QLineEdit()
        self.params.setPlaceholderText('파라미터 (비우면 기본값). 예: {"lookback_months": 8}')
        ol.addWidget(self.params, 2)
        ol.addWidget(self.button("실행", self.run_one))
        lay.addWidget(one)
        many = QGroupBox("파라미터 탐색 (명세서의 search 목록 격자)")
        ml = QHBoxLayout(many)
        self.workers = QSpinBox()
        self.workers.setRange(0, 64)
        self.workers.setSpecialValueText("자동")
        ml.addWidget(QLabel("병렬 작업 수"))
        ml.addWidget(self.workers)
        ml.addWidget(self.button("탐색 시작", self.run_search))
        self.cancel = QPushButton("취소")
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self._cancel)
        ml.addWidget(self.cancel)
        self.bar = QProgressBar()
        ml.addWidget(self.bar, 2)
        lay.addWidget(many)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.trials = _table()
        lay.addWidget(self.trials, 2)
        self.canvas_box = QVBoxLayout()
        lay.addLayout(self.canvas_box, 2)
        self._stop = False

    def refresh(self):
        self.pick.refresh()

    def _choose(self):
        s, d = self.pick.chosen()
        if not s or not d:
            self.win.notify("전략과 데이터셋을 먼저 준비하세요.")
        return s, d

    def run_one(self):
        s, d = self._choose()
        if not s:
            return
        try:
            params = json.loads(self.params.text()) if self.params.text().strip() else None
        except json.JSONDecodeError as e:
            self.win.notify(f"파라미터 형식 오류: {e}")
            return

        def done(bt):
            m = bt.metrics
            self.summary.setText(
                f"실험 {bt.experiment_id} · 구간 {m['window'][0]}~{m['window'][1]} · 샤프 차이 {m['sharpe_diff']:.3f} · "
                f"CAGR {m['strategy']['cagr'] * 100:.2f}% (벤치마크 {m['benchmark']['cagr'] * 100:.2f}%) · "
                f"MDD {m['strategy']['max_drawdown'] * 100:.1f}%")
            self.win.results.select(bt.experiment_id)
            self.win.refresh_all()

        self.win.runner.start("백테스트", lambda p: self.lab.backtest(s, d, params), done)

    def run_search(self):
        s, d = self._choose()
        if not s:
            return
        self._stop = False
        self.cancel.setEnabled(True)
        workers = self.workers.value()

        def done(batch):
            self.cancel.setEnabled(False)
            best = batch.get("best") or {}
            dsr = batch.get("best_dsr")
            self.summary.setText(
                f"탐색 {batch['batch_id']}: 완료 {batch['done']}/{batch['total']}, 실패 {batch['failed']}, 취소 {batch['cancelled']}. "
                f"최고 시행 {best.get('params')} 샤프 차이 {_fmt('x', best.get('sharpe_diff'))}, "
                f"DSR {_fmt('x', dsr)} (이 계열 누적 시행 {batch['family_trials']}회). 최고 시행은 자동으로 채택되지 않습니다.")
            _fill(self.trials, batch["rows"], [("trial", "#"), ("params", "파라미터"), ("status", "상태"),
                                               ("sharpe_diff", "샤프 차이"), ("sharpe", "샤프"), ("cagr", "CAGR"),
                                               ("max_drawdown", "MDD")])
            self._show_sensitivity(batch["rows"])
            self.win.refresh_all()

        def progress(a, b):
            self.bar.setMaximum(b)
            self.bar.setValue(a)

        self.win.runner.start(
            "파라미터 탐색",
            lambda p: self.lab.search(s, d, workers=workers, progress=p, should_stop=lambda: self._stop),
            done, progress)

    def _cancel(self):
        self._stop = True
        self.win.status("취소 요청: 진행 중인 시행이 끝나면 멈춥니다. 같은 탐색은 CLI `search --resume`으로 이어갈 수 있습니다.")

    def _show_sensitivity(self, rows):
        while self.canvas_box.count():
            w = self.canvas_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        with charts.dark():
            self.canvas_box.addWidget(FigureCanvasQTAgg(charts.parameter_sensitivity(rows)))


# ----------------------------------------------------------------------------- 4. validation
class ValidationScreen(Screen):
    def __init__(self, win):
        super().__init__(win)
        lay = QVBoxLayout(self)
        lay.addWidget(_note(
            "정식 검증 순서: ① 후보 확정(파라미터와 판정 설정을 고정) → ② 최종 평가. 최종 평가는 보류 구간을 한 번 읽고 "
            "판정(PASS/HOLD/FAIL/INVALID)을 기록합니다. 같은 계열의 두 번째 열람은 PASS가 될 수 없습니다."))
        self.pick = Pickers(self)
        lay.addWidget(self.pick)
        row = QHBoxLayout()
        self.params = QLineEdit()
        self.params.setPlaceholderText("확정할 파라미터 JSON (비우면 기본값)")
        row.addWidget(self.params, 2)
        row.addWidget(self.button("① 후보 확정", self.confirm))
        row.addWidget(self.button("② 최종 평가…", self.final))
        lay.addLayout(row)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.gates = _table()
        lay.addWidget(self.gates, 1)
        lay.addWidget(QLabel("보류 구간 열람 기록"))
        self.ledger = _table()
        lay.addWidget(self.ledger, 1)
        self.ask = self._ask_dialog

    def refresh(self):
        self.pick.refresh()
        rows = self.lab.registry.holdout_ledger()
        _fill(self.ledger, rows, [(k, k) for k in (rows[0].keys() if rows else ["strategy_family", "accessed_at"])])

    def confirm(self):
        s, _ = self.pick.chosen()
        if not s:
            return
        try:
            params = json.loads(self.params.text()) if self.params.text().strip() else None
        except json.JSONDecodeError as e:
            self.win.notify(f"파라미터 형식 오류: {e}")
            return
        p = self.lab.confirm_candidate(s, params, note="desktop app")
        self.info.setText(f"후보 확정: {p}. 이 파라미터로만 최종 평가가 실행됩니다.")
        self.win.refresh_all()

    def _ask_dialog(self, text: str) -> bool:
        return QMessageBox.question(self, "최종 평가 확인", text) == QMessageBox.Yes

    def final(self):
        s, d = self.pick.chosen()
        if not s or not d:
            return
        conf = self.lab.registry.confirmation(s)
        if conf is None:
            self.win.notify("먼저 후보를 확정하세요.")
            return
        sid = self.lab.registry.strategy(s)["strategy_id"]
        prior = len([r for r in self.lab.registry.holdout_ledger() if r.get("strategy_family") == sid])
        text = (f"{sid} 최종 평가를 실행합니다.\n\n확정 파라미터: {conf['params']}\n"
                f"이 계열의 이전 보류 구간 열람: {prior}회\n\n"
                "보류 구간은 한 번만 읽는 것이 원칙입니다. 결과를 본 뒤 규칙을 고쳐 다시 평가하면 그 평가는 독립 검증이 아닙니다. "
                "진행할까요?")
        if not self.ask(text):
            return

        def done(out):
            self.info.setText(f"최종 판정: {out.overall} (실험 {out.experiment_id})")
            _fill(self.gates, [{"gate": g.gate, "verdict": g.verdict, "reasons": "; ".join(g.reasons)} for g in out.gates],
                  [("gate", "단계"), ("verdict", "판정"), ("reasons", "근거")])
            self.win.results.select(out.experiment_id)
            self.win.refresh_all()

        self.win.runner.start("최종 평가", lambda p: self.lab.final_evaluation(s, d), done)


# ----------------------------------------------------------------------------- 5. results
class ResultsScreen(Screen):
    def __init__(self, win):
        super().__init__(win)
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.exp = QComboBox()
        self.exp.currentIndexChanged.connect(self.show)
        row.addWidget(QLabel("실험"))
        row.addWidget(self.exp, 2)
        row.addWidget(self.button("보고서 만들기", self.report))
        row.addWidget(self.button("결과 패킷 내보내기", self.packet))
        row.addWidget(self.button("여러 실험 비교…", self.compare))
        lay.addLayout(row)
        split = QSplitter(Qt.Horizontal)
        self.metrics = _table()
        split.addWidget(self.metrics)
        self.charts = QTabWidget()
        split.addWidget(self.charts)
        split.setSizes([320, 800])
        lay.addWidget(split, 3)
        self.cmp = _table()
        lay.addWidget(self.cmp, 1)
        self.last_path: Path | None = None

    def refresh(self):
        cur = self.exp.currentData()
        self.exp.blockSignals(True)
        self.exp.clear()
        for e in reversed(self.lab.registry.history()):
            label = f"{e['experiment_id']} · {e['strategy_id']}@{e['strategy_version']} · {e['kind']}"
            if e.get("overall"):
                label += f" · {e['overall']}"
            self.exp.addItem(label, e["experiment_id"])
        i = self.exp.findData(cur)
        self.exp.setCurrentIndex(i if i >= 0 else 0)
        self.exp.blockSignals(False)
        self.show()

    def select(self, eid: str):
        self.refresh()
        i = self.exp.findData(eid)
        if i >= 0:
            self.exp.setCurrentIndex(i)
            self.show()

    def show(self):
        eid = self.exp.currentData()
        self.charts.clear()
        if not eid:
            return
        e = self.lab.registry.experiment(eid)
        m = e["metrics"]
        src = m.get("holdout") or m.get("dev") or m
        rows = []
        for k in ("cagr", "total_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "n_trades",
                  "turnover", "psr_vs_0"):
            rows.append({"metric": k, "strategy": src.get("strategy", {}).get(k), "benchmark": src.get("benchmark", {}).get(k)})
        rows.append({"metric": "sharpe_diff", "strategy": src.get("sharpe_diff")})
        ts = src.get("trades") or m.get("trades")
        if ts and ts.get("n"):
            for k in ("n", "mean_gross", "median_gross", "mean_gross_ex_top10", "t_day_clustered"):
                rows.append({"metric": f"trade.{k}", "strategy": ts.get(k)})
        _fill(self.metrics, [{**r, "label": LABELS.get(r["metric"], r["metric"]), "strategy": _fmt(r["metric"], r["strategy"]),
                              "benchmark": _fmt(r["metric"], r.get("benchmark"))} for r in rows],
              [("label", "지표"), ("strategy", "전략"), ("benchmark", "벤치마크")])
        try:
            eq, _ = self.lab.artifacts(eid)
        except FileNotFoundError:
            self.charts.addTab(QLabel("이 실험은 저장된 곡선이 없습니다(Gate 1에서 중단 등)."), "곡선")
            return
        policy = self.lab.registry.get_policy(e["validation_policy_version"]) if e.get("validation_policy_version") else {}
        hs = policy.get("splits", {}).get("holdout", [None])[0] if e["kind"] == "formal" else None
        with charts.dark():
            self.charts.addTab(FigureCanvasQTAgg(charts.equity(eq, hs)), "누적 수익")
            self.charts.addTab(FigureCanvasQTAgg(charts.drawdown(eq)), "낙폭")
            self.charts.addTab(FigureCanvasQTAgg(charts.monthly_returns(eq["strategy"])), "월별 수익")

    def report(self):
        eid = self.exp.currentData()
        if not eid:
            self.win.notify("먼저 위 목록에서 실험을 고르세요.")
            return

        def done(path):
            self.last_path = path
            self.win.status(f"보고서: {path}")
            if not self.win.runner.sync:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.with_suffix(".html"))))

        self.win.runner.start("보고서 만들기", lambda p: self.lab.write_report(eid), done)

    def packet(self):
        eid = self.exp.currentData()
        if not eid:
            self.win.notify("먼저 위 목록에서 실험을 고르세요.")
            return

        def done(path):
            self.last_path = path
            self.win.notify(f"결과 패킷을 만들었습니다 (API 키·원본 가격 자료 제외).\n\n{path}\n\n"
                            "이 zip을 Claude 프로젝트에 올리면 함께 검토할 수 있습니다.")
            if not self.win.runner.sync:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))  # show it in Explorer

        self.win.runner.start("결과 패킷 내보내기", lambda p: self.lab.export_packet(eid), done)

    def compare(self, ids: list[str] | None = None):
        if ids is None:
            text, ok = QInputDialog.getText(self, "실험 비교", "비교할 실험 번호 (공백으로 구분)")
            if not ok:
                return
            ids = text.split()
        if ids:
            df = self.lab.compare(ids)
            _fill(self.cmp, df.to_dict("records"), [(c, c) for c in df.columns])


# ----------------------------------------------------------------------------- 6. history
class HistoryScreen(Screen):
    def __init__(self, win):
        super().__init__(win)
        lay = QVBoxLayout(self)
        lay.addWidget(_note("모든 기록은 추가만 되고 수정·삭제되지 않습니다(작업 폴더의 registry.sqlite)."))
        lay.addWidget(QLabel("실험"))
        self.exps = _table()
        lay.addWidget(self.exps, 2)
        lay.addWidget(QLabel("전략 상태 변경"))
        self.states = _table()
        lay.addWidget(self.states, 1)

    def refresh(self):
        _fill(self.exps, list(reversed(self.lab.registry.history())),
              [("experiment_id", "실험"), ("created_at", "시각"), ("kind", "종류"), ("strategy_id", "전략"),
               ("strategy_version", "버전"), ("overall", "판정"), ("failure_reason", "실패 사유")])
        rows = list(reversed(self.lab.registry.state_history()))
        cols = [(k, k) for k in (rows[0].keys() if rows else ["strategy_id", "to_state", "reason"])]
        _fill(self.states, rows, cols)


# ----------------------------------------------------------------------------- window
class MainWindow(QMainWindow):
    """Top tab bar + pages. 홈/결과 are the one-click flow; 고급 keeps the six step-by-step screens."""

    def __init__(self, lab: Lab, sync: bool = False) -> None:
        from quantlab.gui.autopage import AutoPage
        from quantlab.gui.home import HomePage, ResultPage
        from quantlab.gui.paperpage import PaperPage

        super().__init__()
        self.lab = lab
        self.runner = Runner(sync)
        self.setWindowTitle(f"QUANT LAB — {lab.ws.root}")
        self.resize(1240, 860)
        self.tabs = QTabWidget()
        self.data = DataScreen(self)
        self.strategies = StrategyScreen(self)
        self.experiments = ExperimentScreen(self)
        self.validation = ValidationScreen(self)
        self.results = ResultsScreen(self)
        self.history = HistoryScreen(self)
        self.screens = [self.data, self.strategies, self.experiments, self.validation, self.results, self.history]
        for s, name in zip(self.screens, ("데이터", "전략", "실험", "검증", "결과", "기록")):
            self.tabs.addTab(s, name)
        self.home = HomePage(self)
        self.result = ResultPage(self)
        self.auto = AutoPage(self)
        self.paper = PaperPage(self)
        advanced = QWidget()
        advanced.setObjectName("page")
        al = QVBoxLayout(advanced)
        al.setContentsMargins(14, 12, 14, 12)
        hint = QLabel("단계별로 직접 조작하는 화면입니다. 평소에는 홈에서 버튼 하나로 실행하면 됩니다.")
        hint.setObjectName("muted")
        al.addWidget(hint)
        al.addWidget(self.tabs)

        self.stack = QStackedWidget()
        self.pages = {"홈": self.home, "자동 탐색": self.auto, "모의 투자": self.paper, "결과": self.result, "고급": advanced}
        top = QWidget()
        top.setObjectName("topbar")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(14, 0, 16, 0)
        tl.setSpacing(0)
        logo = QLabel("QUANT▸LAB")
        logo.setObjectName("logo")
        tl.addWidget(logo)
        self.nav: dict[str, QPushButton] = {}
        group = QButtonGroup(self)
        for name, page in self.pages.items():
            self.stack.addWidget(page)
            b = QPushButton(name)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, n=name: self.go(n))
            group.addButton(b)
            tl.addWidget(b)
            self.nav[name] = b
        tl.addStretch(1)
        from quantlab.core.app_info import build_label

        folder = QLabel()
        folder.setObjectName("topinfo")
        folder.setToolTip(f"작업 폴더\n{lab.ws.root}")
        folder.setText(folder.fontMetrics().elidedText(f"작업 폴더 {lab.ws.root}", Qt.ElideMiddle, 320))
        tl.addWidget(folder)
        tl.addSpacing(18)
        dot = QLabel("●")
        dot.setObjectName("dot")
        tl.addWidget(dot)
        tl.addSpacing(6)
        self.version_label = QLabel(build_label())
        self.version_label.setObjectName("topinfo")
        tl.addWidget(self.version_label)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        bl.addWidget(top)
        bl.addWidget(self.stack, 1)
        self.setCentralWidget(body)
        self.runner.message.connect(self.status)
        self.runner.busy_changed.connect(self._busy)
        self.refresh_all()
        self.go("홈")
        # the research factory runs in its own process: show its news when the app opens, then every 10 minutes
        self._news_timer = QTimer(self)
        self._news_timer.timeout.connect(self.factory_news)
        self._news_timer.start(10 * 60 * 1000)
        QTimer.singleShot(1500, self.factory_news)

    def factory_news(self) -> None:
        from quantlab.services.factory import report_text

        try:
            news = self.lab.factory().unseen()
        except Exception:  # noqa: BLE001 - a damaged news file must not stop the app
            return
        if not news:
            return
        self.lab.factory().mark_seen()
        for n in news:
            self.notify(report_text(n))
        self.auto.refresh()

    def go(self, name: str) -> None:
        self.stack.setCurrentWidget(self.pages[name])
        self.nav[name].setChecked(True)
        if name == "자동 탐색":
            self.auto.refresh()
        elif name == "모의 투자":
            self.paper.refresh()

    def show_result(self, experiment_id: str) -> None:
        self.result.select(experiment_id)
        self.go("결과")

    def ask(self, text: str) -> bool:
        """A yes/no question; headless tests answer yes."""
        if self.runner.sync:
            return True
        return QMessageBox.question(self, "QUANT LAB", text) == QMessageBox.Yes

    def status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def notify(self, text: str) -> None:
        """Information for the user: a dialog in the app, the status bar in headless tests."""
        self.status(text)
        if not self.runner.sync:
            QMessageBox.information(self, "QUANT LAB", text)

    def _busy(self, busy: bool) -> None:
        for s in self.screens:
            for b in s.actions:
                b.setEnabled(not busy)
        self.auto._buttons()
        self.paper._buttons()
        if not busy:
            self.home._idle()
            self.home.refresh()  # restores each card's own enabled state
        else:
            for b in self.home.run_buttons:
                b.setEnabled(False)

    def refresh_all(self) -> None:
        for s in self.screens:
            s.refresh()
        self.home.refresh()
        self.result.refresh()

    def closeEvent(self, ev):  # noqa: N802 (Qt name)
        if self.auto.running:
            if self.ask("자동 탐색이 진행 중입니다. 멈추고 닫을까요? (남은 조합은 다음에 '이어서 하기'로 시험합니다)"):
                self.auto.stop()
                self.notify("지금 시험 중인 조합이 끝나면 멈춥니다. 잠시 뒤 다시 닫아 주세요.")
            ev.ignore()
            return
        if self.runner.busy:
            self.notify("작업이 끝난 뒤 닫아 주세요.")
            ev.ignore()
            return
        self.lab.close()
        ev.accept()
