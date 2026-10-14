"""Simple screens: Home (pick a strategy, one button) and Result (big numbers, one chart).

The detailed screens remain under "고급" for step-by-step control; these pages call the same
`Lab` services (quick_backtest / quick_final), so records are identical either way.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
                               QVBoxLayout, QWidget)

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
        outer.setContentsMargins(32, 28, 32, 32)
        col = QWidget()
        col.setObjectName("page")
        col.setMaximumWidth(980)
        self.body = QVBoxLayout(col)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(16)
        outer.addWidget(col)
        self.setWidget(inner)

    def clear(self, keep: int = 0):
        while self.body.count() > keep:
            item = self.body.takeAt(keep)
            if item.widget():
                item.widget().setParent(None)


# ----------------------------------------------------------------------------- home
class HomePage(Page):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.run_buttons: list[QPushButton] = []
        self.cards: dict[str, QFrame] = {}
        self.body.addWidget(label("전략을 고르고 실행하세요", "h1"))
        self.body.addWidget(label("버튼 하나로 가격 자료 준비부터 백테스트까지 진행합니다. "
                                  "탐색 실험은 최종 평가 구간을 열지 않습니다.", "body2", wrap=True))
        # progress card (hidden until a job runs)
        self.progress = Card(18, 8)
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
        self.list_head = QHBoxLayout()
        self.list_head.addWidget(label("전략", "h2"))
        self.list_head.addStretch()
        self.list_head.addWidget(button("명세서 파일 추가", "link", self.add_spec))
        self.body.addLayout(self.list_head)
        self.grid_host = QWidget()
        self.grid_host.setObjectName("page")
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(16)
        self.body.addWidget(self.grid_host)
        self.body.addStretch()

    @property
    def lab(self):
        return self.win.lab

    def _key_card(self) -> Card:
        c = Card()
        c.lay.addWidget(label("가격 자료를 받으려면 Tiingo API 키가 필요해요", "h3"))
        c.lay.addWidget(label("키는 Windows 자격 증명 관리자에만 저장되고 파일·기록·결과 패킷에는 남지 않아요.", "muted", wrap=True))
        row = QHBoxLayout()
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("API 키 붙여넣기")
        row.addWidget(self.key, 1)
        row.addWidget(button("저장", "primary", self.save_key))
        c.lay.addLayout(row)
        return c

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

    # strategy cards -----------------------------------------------------------
    def refresh(self):
        self.key_card.setVisible(not self.lab.has_api_key())
        _empty(self.grid)
        self.run_buttons = []
        self.cards = {}
        for i, path in enumerate(self.lab.strategy_files()):
            card = self._strategy_card(path)
            self.grid.addWidget(card, i // 2, i % 2)

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

    def _strategy_card(self, path: Path) -> Card:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        st = spec["strategy"]
        sid = st["id"]
        c = Card()
        self.cards[sid] = c
        top = QHBoxLayout()
        state = self.lab.registry.current_state(sid)
        top.addWidget(chip({"RESEARCH": "연구 중", "VALIDATION": "검증 중", "IDEA": "검사 미통과",
                            "REJECTED": "기각"}.get(state, "새 전략")))
        top.addWidget(chip(spec["hypothesis"]["class"] == "theory" and "이론 기반" or "데이터 탐색"))
        top.addStretch()
        top.addWidget(label(f"{sid} · v{st['version']}", "muted"))
        c.lay.addLayout(top)
        c.lay.addWidget(label(st["name"], "h3", wrap=True))
        syms = spec["universe"]["symbols"]
        c.lay.addWidget(label(" · ".join(syms[:8]) + (f" 외 {len(syms) - 8}" if len(syms) > 8 else ""), "muted", wrap=True))
        dev, formal = self._last(sid)
        if dev:
            m = self.lab.registry.experiment(dev["experiment_id"])["metrics"]
            row = QHBoxLayout()
            for name, val, signed in (("연 수익률", m["strategy"]["cagr"], True), ("벤치마크", m["benchmark"]["cagr"], True),
                                      ("최대 낙폭", m["strategy"]["max_drawdown"], True)):
                box = QVBoxLayout()
                box.addWidget(label(name, "muted"))
                v = label(pct(val, signed), "h3")
                v.setStyleSheet(f"color: {theme.signed_color(val)};")
                box.addWidget(v)
                row.addLayout(box)
            row.addStretch()
            c.lay.addLayout(row)
            c.lay.addWidget(label("최근 탐색 실험 (최종 평가 구간 제외)", "muted"))
        if formal:
            fg, bg = theme.VERDICT.get(formal["overall"] or "INVALID", theme.VERDICT["INVALID"])
            r = QHBoxLayout()
            r.addWidget(label("최종 평가", "muted"))
            r.addWidget(chip(formal["overall"] or "—", fg, bg))
            r.addStretch()
            c.lay.addLayout(r)
        c.lay.addStretch()
        btns = QHBoxLayout()
        run = button("백테스트 실행", "primary", lambda: self.run(path))
        self.run_buttons.append(run)
        btns.addWidget(run, 1)
        fin = button("최종 평가", "secondary", lambda: self.final(path, sid))
        fin.setEnabled(formal is None and dev is not None)
        fin.setToolTip("탐색 실험을 먼저 해 보세요." if dev is None else
                       "이 전략은 이미 최종 평가를 했습니다." if formal else "보류 구간을 한 번 열어 판정합니다.")
        self.run_buttons.append(fin)
        btns.addWidget(fin)
        c.lay.addLayout(btns)
        return c

    # actions ------------------------------------------------------------------
    def _busy(self, text: str):
        self.step.setText(text)
        self.progress.show()

    def _idle(self):
        self.progress.hide()

    def run(self, path: Path):
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

    def final(self, path: Path, sid: str):
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
        self.content.setSpacing(16)
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

        # summary card
        top = Card()
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
        ts = src.get("trades") or m.get("trades") or {}
        tiles = QHBoxLayout()
        tiles.setSpacing(28)
        for name, val, sub, signed in (
                ("연 수익률", s.get("cagr"), f"벤치마크 {pct(b.get('cagr'), True)}", True),
                ("샤프 지수", s.get("sharpe"), f"벤치마크 {b.get('sharpe', 0):.2f}" if b else "", False),
                ("최대 낙폭", s.get("max_drawdown"), f"벤치마크 {pct(b.get('max_drawdown'), True)}", True),
                ("거래 수", ts.get("n", s.get("n_trades")),
                 f"매수·매도 체결 {s.get('n_trades')}회" if ts else "체결 기준", False)):
            box = QVBoxLayout()
            box.setSpacing(2)
            box.addWidget(label(name, "muted"))
            text = pct(val, True) if signed else ("—" if val is None else (f"{val:.2f}" if isinstance(val, float) else str(val)))
            big = label(text, "big")
            if signed:
                big.setStyleSheet(f"color: {theme.signed_color(val)};")
            box.addWidget(big)
            box.addWidget(label(sub, "muted"))
            tiles.addLayout(box)
        tiles.addStretch()
        top.lay.addLayout(tiles)
        diff = src.get("sharpe_diff")
        if diff is not None:
            better = diff > 0
            top.lay.addWidget(label(("위험 대비 성과가 벤치마크보다 좋아요" if better else "위험 대비 성과가 벤치마크보다 낮아요")
                                    + f" (샤프 차이 {diff:+.2f})", "body2"))
        self.content.addWidget(top)

        # verdict reasons
        if formal and verdicts:
            vc = Card(20, 6)
            vc.lay.addWidget(label("판정 근거", "h3"))
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
            cc = Card(18, 8)
            seg = QHBoxLayout()
            group = QButtonGroup(cc)
            host = QVBoxLayout()
            policy = self.lab.registry.get_policy(e["validation_policy_version"]) if e.get("validation_policy_version") else {}
            hs = policy.get("splits", {}).get("holdout", [None])[0] if formal else None
            figs = {"누적 수익": lambda: charts.equity(eq, hs, title="", names=charts.KO), "낙폭": lambda: charts.drawdown(eq, title="", names=charts.KO),
                    "월별 수익": lambda: charts.monthly_returns(eq["strategy"], title="")}

            def show(name):
                _empty(host)
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
            tc = Card(20, 6)
            tc.lay.addWidget(label("거래 단위 점검", "h3"))
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
