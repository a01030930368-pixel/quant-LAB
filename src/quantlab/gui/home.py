"""Simple screens: Home (pick a strategy, one button) and Result (big numbers, one chart).

The detailed screens remain under "고급" for step-by-step control; these pages call the same
`Lab` services (quick_backtest / quick_final), so records are identical either way.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QPalette
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QComboBox, QFileDialog, QFrame, QHBoxLayout,
                               QHeaderView, QInputDialog, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QStyledItemDelegate, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from quantlab.analysis import charts
from quantlab.gui import theme


# ----------------------------------------------------------------------------- building blocks
def _empty(lay) -> None:
    """Remove everything now: hidden and detached at once, so old cards never show behind new ones."""
    while lay.count():
        item = lay.takeAt(0)
        w = item.widget()
        if w is not None:
            w.hide()
            w.setParent(None)
            w.deleteLater()
        elif item.layout() is not None:
            _empty(item.layout())


def label(text: str, role: str | None = None, wrap: bool = False) -> QLabel:
    lab = QLabel(text)
    if role:
        lab.setObjectName(role)
    lab.setWordWrap(wrap)
    lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lab


def button(text: str, role: str | None = None, slot=None) -> QPushButton:
    b = QPushButton(text)
    if role:
        b.setObjectName(role)
    b.setCursor(Qt.PointingHandCursor)
    if slot:
        b.clicked.connect(slot)
    return b


class Card(QFrame):
    def __init__(self, padding: int = 22, spacing: int = 10):
        super().__init__()
        self.setObjectName("card")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(padding, padding, padding, padding)
        self.lay.setSpacing(spacing)


def chip(text: str, fg: str | None = None, bg: str | None = None) -> QLabel:
    c = label(text, "chip")
    if fg:
        c.setStyleSheet(f"color: {fg}; background: {bg or theme.BG};")
    c.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
    return c


def pct(x, signed=False) -> str:
    if x is None or x != x:
        return "—"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


class Page(QScrollArea):
    """Scrollable gray page with a centered content column."""

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        inner.setObjectName("page")
        outer = QHBoxLayout(inner)
        outer.setContentsMargins(14, 14, 14, 14)
        col = QWidget()
        col.setObjectName("page")
        col.setMaximumWidth(1480)
        self.body = QVBoxLayout(col)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(12)
        outer.addWidget(col)
        self.setWidget(inner)

    def clear(self, keep: int = 0):
        while self.body.count() > keep:
            item = self.body.takeAt(keep)
            if item.widget():
                item.widget().setParent(None)


# ----------------------------------------------------------------------------- home
STATE_KO = {"RESEARCH": "연구 중", "VALIDATION": "검증 중", "IDEA": "검사 미통과", "REJECTED": "기각",
            "HOLD": "보류", "PAPER_TRADING": "모의운용", "CANDIDATE": "후보", "LIVE": "실거래", "SUSPENDED": "중단"}


def pane_header(text: str, right: QWidget | str | None = None) -> QHBoxLayout:
    """Small uppercase-style caption at the top of a pane, with an optional item on the right."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(label(text, "ph"))
    row.addStretch()
    if isinstance(right, str):
        row.addWidget(label(right, "ph"))
    elif right is not None:
        row.addWidget(right)
    return row


def separator() -> QFrame:
    f = QFrame()
    f.setObjectName("sep")
    f.setFrameShape(QFrame.VLine)
    return f


def hline() -> QFrame:
    f = QFrame()
    f.setObjectName("hsep")
    f.setFrameShape(QFrame.HLine)
    return f


def kpi_strip(items: list[tuple[str, str, str | None]]) -> tuple[Card, list[QLabel]]:
    """One pane split into equal cells: caption, big fixed-width number, optional note."""
    c = Card(0, 0)
    row = QHBoxLayout()
    row.setSpacing(0)
    row.setContentsMargins(0, 0, 0, 0)
    values = []
    for i, (name, value, note) in enumerate(items):
        if i:
            row.addWidget(separator())
        cell = QVBoxLayout()
        cell.setContentsMargins(14, 10, 14, 10)
        cell.setSpacing(2)
        cell.addWidget(label(name, "muted"))
        v = label(value, "big")
        values.append(v)
        cell.addWidget(v)
        if note is not None:
            cell.addWidget(label(note, "muted"))
        row.addLayout(cell, 1)
    c.lay.addLayout(row)
    return c, values


def cell(text: str, color: str | None = None, mono: bool = True, align=Qt.AlignRight) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    it.setTextAlignment(align | Qt.AlignVCenter)
    if mono:
        it.setFont(theme.mono_font(13))
    if color:
        it.setForeground(QColor(color))
    return it


class KeepColors(QStyledItemDelegate):
    """Selected rows keep each cell's own text color (rise red, fall blue) instead of one highlight color."""

    def initStyleOption(self, opt, index):  # noqa: N802 (Qt name)
        super().initStyleOption(opt, index)
        fg = index.data(Qt.ForegroundRole)
        if fg is not None:
            opt.palette.setBrush(QPalette.HighlightedText, fg)


class HomePage(Page):
    """Strategy table on the left, the selected strategy's curve and the work log on the right."""

    COLS = ("전략", "이름", "상태", "연수익", "벤치", "MDD", "샤프", "판정")

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.run_buttons: list[QPushButton] = []
        self.cards: dict[str, int] = {}  # strategy id -> table row
        self.paths: list[Path] = []
        self.selected: str | None = None
        self._seen = -1  # history length at the last refresh
        self.kpis, self.kpi_values = kpi_strip([("등록 전략", "0", None), ("탐색 실험", "0", None),
                                                ("최종 평가", "0", None), ("PASS 판정", "0", None)])
        self.body.addWidget(self.kpis)
        # progress pane (hidden until a job runs)
        self.progress = Card(12, 6)
        self.step = label("", "h3")
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setTextVisible(False)
        self.progress.lay.addWidget(self.step)
        self.progress.lay.addWidget(self.bar)
        self.progress.hide()
        self.body.addWidget(self.progress)
        self.key_card = self._key_card()
        self.body.addWidget(self.key_card)
        self.krx_card = self._krx_card()
        self.body.addWidget(self.krx_card)
        self.dart_card = self._dart_card()
        self.body.addWidget(self.dart_card)

        row = QHBoxLayout()
        row.setSpacing(12)
        # left: strategy table + actions
        left = Card(0, 0)
        head = QWidget()
        hl = QVBoxLayout(head)
        hl.setContentsMargins(12, 8, 12, 8)
        self.count = label("", "ph")
        hl.addLayout(pane_header("전략 목록", self.count))
        left.lay.addWidget(head)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setObjectName("plain")
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(36)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setItemDelegate(KeepColors(self.table))
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        for i in range(len(self.COLS)):
            self.table.horizontalHeaderItem(i).setTextAlignment((Qt.AlignLeft if i < 2 else Qt.AlignRight) | Qt.AlignVCenter)
        self.table.setMinimumHeight(220)
        self.table.itemSelectionChanged.connect(self._on_select)
        left.lay.addWidget(self.table, 1)
        acts = QHBoxLayout()
        acts.setContentsMargins(12, 10, 12, 12)
        acts.setSpacing(8)
        self.run_btn = button("▶  백테스트 실행", "primary", lambda: self._selected_path() and self.run(self._selected_path()))
        self.final_btn = button("최종 평가", "secondary",
                                lambda: self._selected_path() and self.final(self._selected_path(), self.selected))
        self.run_buttons = [self.run_btn, self.final_btn]
        acts.addWidget(self.run_btn)
        acts.addWidget(self.final_btn)
        acts.addWidget(button("명세서 추가", "secondary", self.add_spec))
        acts.addWidget(button("공모주 첫날 조사", "secondary", self.ipo))
        acts.addStretch()
        acts.addWidget(label("탐색 실험은 최종 평가 구간을 열지 않아요", "muted"))
        left.lay.addLayout(acts)
        row.addWidget(left, 1)

        # right: curve of the selected strategy + work log
        right = Card(0, 0)
        right.setFixedWidth(340)
        ch = QWidget()
        cl = QVBoxLayout(ch)
        cl.setContentsMargins(12, 8, 12, 4)
        self.chart_title = label("누적 수익", "ph")
        cl.addWidget(self.chart_title)
        self.chart_host = QVBoxLayout()
        cl.addLayout(self.chart_host)
        right.lay.addWidget(ch)
        right.lay.addWidget(hline())
        lh = QWidget()
        ll = QVBoxLayout(lh)
        ll.setContentsMargins(12, 8, 12, 0)
        ll.addWidget(label("작업 기록", "ph"))
        right.lay.addWidget(lh)
        self.log = QPlainTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        right.lay.addWidget(self.log, 1)
        row.addWidget(right)
        self.body.addLayout(row, 1)
        self.win.runner.message.connect(self.log_line)

    @property
    def lab(self):
        return self.win.lab

    def log_line(self, text: str) -> None:
        self.log.appendPlainText(f"{datetime.now():%H:%M}  {text}")

    def _key_card(self) -> Card:
        c = Card(12, 8)
        c.lay.addLayout(pane_header("가격 자료 연결"))
        c.lay.addWidget(label("가격 자료를 받으려면 Tiingo API 키가 필요해요. 키는 Windows 자격 증명 관리자에만 저장되고 "
                              "파일·기록·결과 패킷에는 남지 않아요.", "body2", wrap=True))
        row = QHBoxLayout()
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("API 키 붙여넣기")
        row.addWidget(self.key, 1)
        row.addWidget(button("저장", "primary", self.save_key))
        c.lay.addLayout(row)
        return c

    def _krx_card(self) -> Card:
        c = Card(12, 8)
        c.lay.addLayout(pane_header("한국 주식 자료 연결"))
        c.lay.addWidget(label("한국 주식 자동 탐색(공장 3차)은 한국거래소 KRX Open API 인증키가 있어야 해요. "
                              "openapi.krx.co.kr에서 받은 키를 넣어 주세요. 키는 Windows 자격 증명 관리자에만 저장돼요.",
                              "body2", wrap=True))
        row = QHBoxLayout()
        self.krx_key = QLineEdit()
        self.krx_key.setEchoMode(QLineEdit.Password)
        self.krx_key.setPlaceholderText("KRX 인증키 붙여넣기")
        row.addWidget(self.krx_key, 1)
        row.addWidget(button("저장", "primary", self.save_krx_key))
        c.lay.addLayout(row)
        return c

    def save_krx_key(self):
        k = self.krx_key.text().strip()
        if k:
            self.lab.set_api_key(k, "krx")
            self.krx_key.clear()
            self.refresh()

    def _dart_card(self) -> Card:
        c = Card(12, 8)
        c.lay.addLayout(pane_header("한국 공시 자료 연결"))
        c.lay.addWidget(label("한국 공시 자동 탐색(공장 4차)은 OpenDART 인증키가 있어야 해요. "
                              "opendart.fss.or.kr에서 받은 키를 넣어 주세요. 키는 Windows 자격 증명 관리자에만 저장돼요.",
                              "body2", wrap=True))
        row = QHBoxLayout()
        self.dart_key = QLineEdit()
        self.dart_key.setEchoMode(QLineEdit.Password)
        self.dart_key.setPlaceholderText("OpenDART 인증키 붙여넣기")
        row.addWidget(self.dart_key, 1)
        row.addWidget(button("저장", "primary", self.save_dart_key))
        c.lay.addLayout(row)
        return c

    def save_dart_key(self):
        k = self.dart_key.text().strip()
        if k:
            self.lab.set_api_key(k, "dart")
            self.dart_key.clear()
            self.refresh()

    def save_key(self):
        k = self.key.text().strip()
        if k:
            self.lab.set_api_key(k)
            self.key.clear()
            self.refresh()

    def add_spec(self, path: str | None = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "전략 명세서", str(self.lab.ws.path("examples")),
                                                  "QUANT LAB 명세서 (*.qls.yaml *.yaml)")
        if path:
            self.win.runner.start("전략 가져오기", lambda p: self.lab.import_strategy(path),
                                  lambda res: (self.win.refresh_all(),
                                               self.win.notify(f"{res.strategy_id}: {res.state}"
                                                               + ("\n" + "\n".join(res.problems) if res.problems else ""))))

    # strategy table -------------------------------------------------------------
    def refresh(self):
        self.key_card.setVisible(not self.lab.has_api_key())
        self.krx_card.setVisible(not self.lab.has_krx_key())
        self.dart_card.setVisible(not self.lab.has_dart_key())
        history = self.lab.registry.history()
        formal = [e for e in history if e["kind"] == "formal"]
        self.paths = list(self.lab.strategy_files())
        for v, n in zip(self.kpi_values, (len(self.paths), len(history) - len(formal), len(formal),
                                          sum(1 for e in formal if e.get("overall") == "PASS"))):
            v.setText(str(n))
        self.cards = {}
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.paths))
        for i, path in enumerate(self.paths):
            self._fill_row(i, path)
        self.table.blockSignals(False)
        self.count.setText(f"{len(self.paths)}개")
        if self.selected not in self.cards or len(history) != self._seen:
            # after a new experiment (or at start) show the strategy run most recently, else the first row
            recent = [e["strategy_id"] for e in history if e["strategy_id"] in self.cards]
            self.selected = recent[-1] if recent else None
        self._seen = len(history)
        row = self.cards.get(self.selected, 0 if self.paths else -1)
        if row >= 0:
            self.table.selectRow(row)
        self._on_select()

    def _last(self, sid: str) -> tuple[dict | None, dict | None]:
        last_dev = last_formal = None
        for e in self.lab.registry.history():
            if e["strategy_id"] != sid:
                continue
            if e["kind"] == "formal":
                last_formal = e
            else:
                last_dev = e
        return last_dev, last_formal

    def _fill_row(self, i: int, path: Path) -> None:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        st = spec["strategy"]
        sid = st["id"]
        self.cards[sid] = i
        first = cell(sid, theme.TEXT, align=Qt.AlignLeft)
        first.setData(Qt.UserRole, sid)
        first.setToolTip(f"v{st['version']} · " + " · ".join(spec["universe"]["symbols"]))
        self.table.setItem(i, 0, first)
        self.table.setItem(i, 1, cell(st["name"], theme.TEXT2, mono=False, align=Qt.AlignLeft))
        state = self.lab.registry.current_state(sid)
        self.table.setItem(i, 2, cell(STATE_KO.get(state, "새 전략"), theme.TEXT2, mono=False))
        dev, formal = self._last(sid)
        m = self.lab.registry.experiment(dev["experiment_id"])["metrics"] if dev else {}
        s, b = m.get("strategy", {}), m.get("benchmark", {})
        for j, val in ((3, s.get("cagr")), (4, b.get("cagr")), (5, s.get("max_drawdown"))):
            self.table.setItem(i, j, cell(pct(val, True), theme.signed_color(val) if val is not None else theme.MUTED))
        sh = s.get("sharpe")
        self.table.setItem(i, 6, cell("—" if sh is None or sh != sh else f"{sh:.2f}"))
        verdict = formal["overall"] if formal and formal.get("overall") else None
        fg = theme.VERDICT.get(verdict or "INVALID", theme.VERDICT["INVALID"])[0] if verdict else theme.MUTED
        self.table.setItem(i, 7, cell(verdict or "—", fg))

    def _selected_path(self) -> Path | None:
        row = self.cards.get(self.selected) if self.selected else None
        return self.paths[row] if row is not None and row < len(self.paths) else None

    def _on_select(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.selected = self.table.item(rows[0].row(), 0).data(Qt.UserRole)
        elif not self.paths:
            self.selected = None
        dev, formal = self._last(self.selected) if self.selected else (None, None)
        self.run_btn.setEnabled(self.selected is not None)
        self.final_btn.setEnabled(formal is None and dev is not None)
        self.final_btn.setToolTip("탐색 실험을 먼저 해 보세요." if dev is None else
                                  "이 전략은 이미 최종 평가를 했습니다." if formal else "보류 구간을 한 번 열어 판정합니다.")
        self._show_chart(dev)

    def _show_chart(self, dev: dict | None) -> None:
        _empty(self.chart_host)
        self.chart_title.setText(f"{self.selected} 누적 수익 · 주황 전략 · 회색 벤치마크" if self.selected else "누적 수익")
        eq = None
        if dev:
            try:
                eq, _ = self.lab.artifacts(dev["experiment_id"])
            except FileNotFoundError:
                eq = None
        if eq is None:
            msg = label("백테스트를 실행하면 여기에 누적 수익 곡선이 나와요.", "muted", wrap=True)
            msg.setMinimumHeight(160)
            msg.setAlignment(Qt.AlignCenter)
            self.chart_host.addWidget(msg)
            return
        with charts.dark():
            fig = charts.equity(eq, None, title="", names=charts.KO)
        for ax in fig.axes:
            if ax.get_legend():
                ax.get_legend().remove()  # too small for a legend; the caption names the lines
        canvas = FigureCanvasQTAgg(fig)
        canvas.setFixedHeight(200)
        self.chart_host.addWidget(canvas)

    # actions ------------------------------------------------------------------
    def _busy(self, text: str):
        self.step.setText(text)
        self.progress.show()

    def _idle(self):
        self.progress.hide()

    SEC_EMAIL_TEXT = ("이 전략은 S&P 500 종목 가격과 SEC(미국 증권거래위원회)의 내부자 거래 공시를 씁니다.\n"
                      "SEC는 자료를 자동으로 받는 프로그램에 연락용 이메일을 밝히라고 요구해요.\n"
                      "입력한 이메일은 이 PC에만 저장되고 SEC 서버에만 전달돼요.\n\n연락용 이메일:")

    def ask_email(self) -> str | None:
        text, ok = QInputDialog.getText(self, "SEC 연락용 이메일", self.SEC_EMAIL_TEXT)
        return text.strip() if ok and text.strip() else None

    def _ensure_sec_email(self, path: Path) -> bool:
        """Index strategies need the SEC contact e-mail before their data can be downloaded."""
        if not self.lab.needs_sec_email(path):
            return True
        email = self.ask_email()
        if not email:
            return False
        try:
            self.lab.set_sec_email(email)
        except ValueError as e:
            self.win.notify(str(e))
            return False
        return True

    def run(self, path: Path):
        if not self._ensure_sec_email(path):
            return

        def done(bt):
            self._idle()
            self.win.refresh_all()
            self.win.show_result(bt.experiment_id)

        self._busy("준비 중")
        if not self.win.runner.start("백테스트", lambda p: self.lab.quick_backtest(path, say=p.say), done,
                                     on_text=self.step.setText):
            self._idle()
        elif self.win.runner.last_error and self.win.runner.sync:
            self._idle()

    def ipo(self):
        """D-042: every Korean IPO from 2015, offer price against the first day's open and close."""
        def done(res):
            folder, text = res
            self._idle()
            self.log_line(f"공모주 조사 결과: {folder}")
            self.win.notify(f"{text}\n\n결과 파일: {folder}\n이 폴더의 파일을 Claude 프로젝트에 올리면 함께 볼 수 있어요.")
            if not self.win.runner.sync:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

        self._busy("공모주 조사 준비 중")
        if not self.win.runner.start("공모주 첫날 조사", lambda p: self.lab.ipo_study(say=p.say), done,
                                     on_text=self.step.setText):
            self._idle()
        elif self.win.runner.last_error and self.win.runner.sync:
            self._idle()

    def final(self, path: Path, sid: str):
        if not self._ensure_sec_email(path):
            return
        reads = self.lab.holdout_reads(sid)
        text = (f"{sid}의 최종 평가를 시작할까요?\n\n"
                "• 지금의 기본 설정이 후보로 확정됩니다.\n"
                "• 숨겨 둔 최종 평가 구간을 한 번 열어 PASS·HOLD·FAIL을 판정합니다.\n"
                "• 결과를 본 뒤 전략을 고쳐 다시 평가하면 그 평가는 독립 검증이 아닙니다.\n\n"
                f"이 전략의 이전 열람: {reads}회")
        if not self.win.ask(text):
            return

        def done(out):
            self._idle()
            self.win.refresh_all()
            self.win.show_result(out.experiment_id)

        self._busy("최종 평가 준비 중")
        self.win.runner.start("최종 평가", lambda p: self.lab.quick_final(path, say=p.say), done,
                              on_text=self.step.setText)


# ----------------------------------------------------------------------------- result
class ResultPage(Page):
    def __init__(self, win):
        super().__init__()
        self.win = win
        head = QHBoxLayout()
        self.title = label("결과", "h1")
        head.addWidget(self.title)
        head.addStretch()
        self.pick = QComboBox()
        self.pick.setMinimumWidth(320)
        self.pick.currentIndexChanged.connect(self.show_current)
        head.addWidget(self.pick)
        self.body.addLayout(head)
        self.content = QVBoxLayout()
        self.content.setSpacing(12)
        self.body.addLayout(self.content)
        self.body.addStretch()
        self.eid: str | None = None
        self.last_path: Path | None = None

    @property
    def lab(self):
        return self.win.lab

    def refresh(self):
        cur = self.pick.currentData()
        self.pick.blockSignals(True)
        self.pick.clear()
        for e in reversed(self.lab.registry.history()):
            kind = "최종 평가" if e["kind"] == "formal" else "탐색"
            self.pick.addItem(f"{e['strategy_id']} · {kind} · {e['created_at'][:16].replace('T', ' ')}", e["experiment_id"])
        i = self.pick.findData(cur)
        self.pick.setCurrentIndex(i if i >= 0 else 0)
        self.pick.blockSignals(False)
        self.show_current()

    def select(self, eid: str):
        self.refresh()
        i = self.pick.findData(eid)
        if i >= 0 and i != self.pick.currentIndex():
            self.pick.setCurrentIndex(i)  # re-renders through currentIndexChanged

    def _clear(self):
        _empty(self.content)

    def show_current(self):
        self._clear()
        eid = self.pick.currentData()
        self.eid = eid
        if not eid:
            self.content.addWidget(label("아직 실험이 없어요. 홈에서 전략을 실행해 보세요.", "body2"))
            return
        e = self.lab.registry.experiment(eid)
        m = e["metrics"]
        formal = e["kind"] == "formal"
        src = m.get("holdout") or m.get("dev") or m
        s, b = src.get("strategy", {}), src.get("benchmark", {})

        # summary pane: title row, then a strip of fixed-width numbers
        top = Card(12, 8)
        r = QHBoxLayout()
        r.addWidget(label(f"{e['strategy_id']} · v{e['strategy_version']}", "h2"))
        r.addStretch()
        verdicts = {v["gate"]: v for v in self.lab.registry.verdicts(eid)}
        if formal and "OVERALL" in verdicts:
            v = verdicts["OVERALL"]["verdict"]
            fg, bg = theme.VERDICT.get(v, theme.VERDICT["INVALID"])
            r.addWidget(chip(f"판정 {v}", fg, bg))
        r.addWidget(chip("최종 평가" if formal else "탐색 실험"))
        top.lay.addLayout(r)
        period = f"{s.get('start', '')} ~ {s.get('end', '')}"
        pol = self.lab.registry.get_policy(e["validation_policy_version"]) if e.get("validation_policy_version") else {}
        c = pol.get("costs", {})
        cost = f" · 비용 반영(편도 {(c.get('commission_bps', 0) + c.get('slippage_bps', 0)) / 100:.2f}%)" if c else ""
        top.lay.addWidget(label(("최종 평가 구간 " if formal and m.get("holdout") else "탐색 구간(학습+검증) ") + period
                                + cost, "muted"))
        diff = src.get("sharpe_diff")
        if diff is not None:
            better = diff > 0
            top.lay.addWidget(label(("위험 대비 성과가 벤치마크보다 좋아요" if better else "위험 대비 성과가 벤치마크보다 낮아요")
                                    + f" (샤프 차이 {diff:+.2f})", "body2"))
        self.content.addWidget(top)

        ts = src.get("trades") or m.get("trades") or {}
        items, signs = [], []
        for name, val, sub, signed in (
                ("연 수익률", s.get("cagr"), f"벤치마크 {pct(b.get('cagr'), True)}", True),
                ("샤프 지수", s.get("sharpe"), f"벤치마크 {b.get('sharpe', 0):.2f}" if b else "", False),
                ("최대 낙폭", s.get("max_drawdown"), f"벤치마크 {pct(b.get('max_drawdown'), True)}", True),
                ("거래 수", ts.get("n", s.get("n_trades")),
                 f"매수·매도 체결 {s.get('n_trades')}회" if ts else "체결 기준", False)):
            text = pct(val, True) if signed else ("—" if val is None else (f"{val:.2f}" if isinstance(val, float) else str(val)))
            items.append((name, text, sub))
            signs.append(val if signed else None)
        strip, values = kpi_strip(items)
        for v, val in zip(values, signs):
            if val is not None:
                v.setStyleSheet(f"color: {theme.signed_color(val)};")
        self.content.addWidget(strip)

        # verdict reasons
        if formal and verdicts:
            vc = Card(12, 6)
            vc.lay.addLayout(pane_header("판정 근거"))
            names = {"GATE1": "1단계 · 데이터·누출·계산 검사", "GATE2": "2단계 · 탐색 구간 가치",
                     "GATE3": "3단계 · 최종 평가 구간", "GATE4_PARTIAL": "4단계 · 비용 2배·주변 설정"}
            for g in ("GATE1", "GATE2", "GATE3", "GATE4_PARTIAL"):
                if g in verdicts:
                    v = verdicts[g]
                    fg, bg = theme.VERDICT.get(v["verdict"], theme.VERDICT["INVALID"])
                    row = QHBoxLayout()
                    row.addWidget(chip(v["verdict"], fg, bg))
                    row.addWidget(label(names[g], "h3"))
                    row.addStretch()
                    vc.lay.addLayout(row)
                    vc.lay.addWidget(label("; ".join(v["reasons"]), "muted", wrap=True))
            self.content.addWidget(vc)

        # chart card with segmented control
        try:
            eq, _ = self.lab.artifacts(eid)
        except FileNotFoundError:
            eq = None
        if eq is not None:
            cc = Card(12, 8)
            seg = QHBoxLayout()
            group = QButtonGroup(cc)
            host = QVBoxLayout()
            policy = self.lab.registry.get_policy(e["validation_policy_version"]) if e.get("validation_policy_version") else {}
            hs = policy.get("splits", {}).get("holdout", [None])[0] if formal else None
            figs = {"누적 수익": lambda: charts.equity(eq, hs, title="", names=charts.KO), "낙폭": lambda: charts.drawdown(eq, title="", names=charts.KO),
                    "월별 수익": lambda: charts.monthly_returns(eq["strategy"], title="")}

            def show(name):
                _empty(host)
                with charts.dark():
                    canvas = FigureCanvasQTAgg(figs[name]())
                canvas.setMinimumHeight(330)
                host.addWidget(canvas)

            for i, name in enumerate(figs):
                sb = button(name, "seg")
                sb.setCheckable(True)
                sb.setChecked(i == 0)
                sb.clicked.connect(lambda _=False, n=name: show(n))
                group.addButton(sb)
                seg.addWidget(sb)
            seg.addStretch()
            cc.lay.addLayout(seg)
            cc.lay.addLayout(host)
            show("누적 수익")
            self.content.addWidget(cc)

        # trade-level checks
        ts = src.get("trades") or m.get("trades")
        if ts and ts.get("n"):
            tc = Card(12, 6)
            tc.lay.addLayout(pane_header("거래 단위 점검"))
            tc.lay.addWidget(label(
                f"{ts['n']}건 · 거래당 평균 {pct(ts['mean_gross'], True)} (비용 전), 중앙값 {pct(ts['median_gross'], True)}, "
                f"상위 10건 뺀 평균 {pct(ts.get('mean_gross_ex_top10'), True)}, 비용 후 승률 {pct(ts['win_rate_net'])}, "
                f"평균 보유 {ts['mean_sessions']:.0f}거래일", "body2", wrap=True))
            self.content.addWidget(tc)

        # actions
        acts = QHBoxLayout()
        acts.addWidget(button("보고서 열기", "secondary", self.report))
        acts.addWidget(button("결과 패킷 만들기", "secondary", self.packet))
        acts.addStretch()
        self.content.addLayout(acts)

    def report(self):
        eid = self.eid
        if eid:
            self.win.runner.start("보고서 만들기", lambda p: self.lab.write_report(eid), self._opened_report)

    def _opened_report(self, path):
        self.last_path = path
        if not self.win.runner.sync:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.with_suffix(".html"))))

    def packet(self):
        eid = self.eid
        if not eid:
            return

        def done(path):
            self.last_path = path
            self.win.notify(f"결과 패킷을 만들었어요 (API 키·원본 가격 자료 제외).\n\n{path}\n\n"
                            "이 zip을 Claude 프로젝트에 올리면 함께 검토할 수 있어요.")
            if not self.win.runner.sync:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))

        self.win.runner.start("결과 패킷 만들기", lambda p: self.lab.export_packet(eid), done)


__all__ = ["HomePage", "ResultPage", "QMessageBox"]
