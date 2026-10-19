"""자동 탐색 page: run the automatic strategy search for one ETF group and review its candidates.

The search runs on its own thread with its own registry connection, so the other pages stay usable
while it works. It only ever sees TRAIN+VALIDATION data; the final evaluation of a candidate is a
separate, confirmed action by the user (`quantlab.services.autosearch`).
"""

from __future__ import annotations

import csv
import json
import os

from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QFileDialog, QHBoxLayout, QHeaderView, QProgressBar, QPushButton,
                               QSpinBox, QTableWidget, QVBoxLayout)

from quantlab.data.providers import display_name
from quantlab.gui import theme
from quantlab.gui.home import Card, KeepColors, Page, button, cell, hline, kpi_strip, label, pane_header, pct
from quantlab.services.autosearch import GROUPS
from quantlab.services.factory import GROUPS as FACTORY_GROUPS

NEAR_SHOWN = 500  # near misses listed on the page; the CSV has all of them


def write_near_csv(rows: list[dict], path) -> None:
    """All near misses of a group, best first, for a spreadsheet (UTF-8 with BOM so Excel reads Korean)."""
    cols = [("규칙", "rule"), ("탈락 이유", "status"), ("샤프", "sharpe"), ("연수익", "cagr"), ("비용 2배 연수익", "cagr_2x"),
            ("MDD", "mdd"), ("거래", "n"), ("비용 전/거래", "gross"), ("비용 후/거래", "net"), ("t값", "t"),
            ("TRAIN 수익", "ret_train"), ("VALIDATION 수익", "ret_validation")]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([h for h, _ in cols] + ["설정"])
        for r in rows:
            w.writerow([r.get(k) for _, k in cols] + [json.dumps(r["params"], ensure_ascii=False, sort_keys=True)])

def _keep_awake(on: bool) -> None:
    """Windows: stop the PC from going to sleep while the search runs (the screen may still turn off)."""
    if os.name != "nt":
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))


GROUP_NAME = {"AUTO-DEVA": "선진국 A", "AUTO-DEVB": "선진국 B", "AUTO-EM": "신흥국", "AUTO-REAL": "채권·금·원자재·리츠",
             "AUTO-LEV": "3배 ETF", "AUTO-LEVP": "3배 ETF + 쉬는 돈 SPY", "AUTO-SP500": "S&P 500 종목",
             "AUTO-SP500X": "S&P 500 종목 · 공장 1차", "AUTO-SP500Y": "S&P 500 종목 · 공장 2차",
             "AUTO-KR2X": "한국 레버리지·인버스"}
GROUP_NOTE = {
    "AUTO-DEVA": "일본·독일·영국·캐나다·호주·프랑스·스위스·홍콩·싱가포르 ETF 9종",
    "AUTO-DEVB": "스웨덴·네덜란드·스페인·이탈리아·벨기에·오스트리아 ETF 6종",
    "AUTO-EM": "브라질·멕시코·대만·한국·남아공·말레이시아·중국 ETF 7종",
    "AUTO-REAL": "미국 장기·중기 국채, 회사채, 금, 은, 원자재, 리츠 ETF 7종",
    "AUTO-LEV": "신흥국·중국·한국·유럽 3배 레버리지 ETF 4종 (EDC·YINN·KORU·EURL), 2010년부터. SPY와 샤프로 비교",
    "AUTO-LEVP": "같은 3배 ETF 4종, 거래하지 않는 돈은 SPY에 넣어 둠. 합계 연수익으로 SPY와 비교. "
                 "3배 ETF 묶음과 보류 구간을 함께 써서 최종 평가는 둘 중 하나만 가능",
    "AUTO-SP500": "그날 S&P 500에 속한 종목. 실적 반응·내부자 매수·급락·RSI·갭 상승·200일선·SPY 추세·52주 신고가 중 "
                  "1~3개를 겹친 조건. 거래마다 같은 기간 SPY보다 얼마나 더 벌었는지로 거르고, 쉬는 돈은 SPY",
    "AUTO-SP500X": "자동 연구 공장의 묶음. S&P 500 종목 재료 28개(실적 실망·하락 갭·52주 신저가·연속 하락·변동성 수축 등 추가) "
                   "중 1~3개. 앞 묶음이 이미 시험한 조합은 건너뛰고, 통과하면 앞으로의 자료로 모의 투자해요 "
                   "(2020년 이후 보류 구간은 이미 썼어요)",
    "AUTO-SP500Y": "자동 연구 공장 2차. 1차 재료에 새 정보 5개(S&P 500 편입·편출, 임원 3명 이상 매도, 90일간 매도 없음, "
                   "다음 실적 발표 1~2주 전)를 더해, 새 재료가 하나 이상 들어간 조합만 시험해요",
    "AUTO-KR2X": "KODEX 레버리지·코스닥150레버리지·200선물인버스2X (2010년부터, 인버스는 2016-09부터). "
                 "가격은 네이버 증권(키 필요 없음), 비용은 편도 0.065%. KODEX 200 보유와 샤프로 비교",
}


def factory_text(p: dict) -> str:
    """One line under the factory switch: progress only (the user gets a report when a round is done)."""
    rnd = f"{p['round']}차 " if p.get("round", 1) > 1 else ""
    if p["finished"]:
        text = f"목록을 모두 시험했어요 ({p['done']:,}개). 새 재료가 들어올 때까지 쉬어요."
    elif not p["enabled"]:
        text = f"{rnd}꺼져 있어요." + (f" 지금까지 {p['done']:,}개 시험." if p["done"] else "")
    else:
        state = ("돌고 있어요" if p["running"] else
                 "충전기를 뽑아서 쉬는 중이에요 (꽂으면 1분쯤 뒤 이어서 해요)" if p.get("battery") else
                 "켜져 있어요 (PC 로그인 뒤 몇 분 안에, 또는 한 시간 안에 시작)")
        text = rnd + state
        if p["grid"]:
            text += f" · {p['grid']:,}개 중 {p['done']:,}개 시험 ({p['done'] / p['grid'] * 100:.1f}%)"
    if p["done"]:
        text += f" · 통과 {p['candidates']}개 · 모의 투자 중 {p['watch']}개"
    if p.get("error") and not p["running"]:
        text += f"\n마지막 오류: {p['error']}"
    return text


class AutoPage(Page):
    COLS = ("규칙", "거래", "비용 전/거래", "비용 후/거래", "연수익", "샤프", "MDD", "t값", "이웃", "상태")

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.group = GROUPS[0]
        self.running = False
        self._stop = False
        self._keep = []
        self.summary: dict | None = None
        self.limit: int | None = None  # tests run a few combinations instead of a full run
        self._exhausted: set[str] = set()  # groups whose whole grid has been tried
        self._reading = self._read_again = False  # a summary read is running / another was asked for
        self.near_mode = False  # the right-hand table lists near misses instead of candidates

        head = QHBoxLayout()
        head.addWidget(label("자동 탐색", "h1"))
        head.addStretch()
        self.body.addLayout(head)
        self.body.addWidget(label(
            "앱이 재료(진입 신호·추세 조건·청산 규칙·보유 기간·손절·익절)를 조합해 TRAIN+VALIDATION 구간에서만 시험하고, "
            "걸러내는 기준을 모두 통과한 조합만 후보로 남깁니다. 시험한 개수만큼 통과 기준(t값)이 높아집니다. "
            "최종 평가(맨 뒤 보류 구간)는 앱이 스스로 열지 않습니다.", "muted", wrap=True))

        fac = Card(16, 8)
        frow = QHBoxLayout()
        frow.addWidget(label("자동 연구 공장", "ph"))
        frow.addSpacing(12)
        self.factory_on = QCheckBox("켜기: PC가 켜져 있고 충전기가 꽂혀 있으면 앱을 열지 않아도 계속 시험 (다른 작업에 CPU를 양보)")
        self.factory_on.toggled.connect(self._factory_toggled)
        frow.addWidget(self.factory_on)
        frow.addStretch()
        fac.lay.addLayout(frow)
        self.factory_line = label("", "muted", wrap=True)
        fac.lay.addWidget(self.factory_line)
        self.body.addWidget(fac)

        seg = QHBoxLayout()
        seg.setSpacing(0)
        self.seg = QButtonGroup(self)
        self.seg_buttons: dict[str, QPushButton] = {}
        for g in GROUPS:
            b = button(GROUP_NAME[g], "seg")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, gg=g: self.choose(gg))
            self.seg.addButton(b)
            seg.addWidget(b)
            self.seg_buttons[g] = b
        seg.addStretch()
        self.body.addLayout(seg)
        self.group_note = label("", "muted")
        self.body.addWidget(self.group_note)

        self.kpis, self.kpi_values = kpi_strip([("시험한 조합", "—", None), ("이 묶음 누적 시험", "—", None),
                                                ("필요한 t값", "—", None), ("비교 기준 보유", "—", None),
                                                ("후보", "—", None)])
        self.body.addWidget(self.kpis)

        ctl = Card(16, 8)
        row = QHBoxLayout()
        row.addWidget(label("동시에 쓸 CPU", "muted"))
        self.workers = QSpinBox()
        self.workers.setRange(1, max(1, os.cpu_count() or 1))
        self.workers.setValue(max(1, (os.cpu_count() or 2) - 1))
        row.addWidget(self.workers)
        row.addSpacing(12)
        self.start_btn = button("탐색 시작", "primary", self.start)
        row.addWidget(self.start_btn)
        self.stop_btn = button("멈추기", "secondary", self.stop)
        self.stop_btn.setEnabled(False)
        row.addWidget(self.stop_btn)
        self.keep_going = QCheckBox("끝나면 계속")
        self.keep_going.setChecked(True)
        self.keep_going.setToolTip("3,000개가 끝나면 다음 3,000개를 바로 시작하고, 묶음을 다 시험하면 다음 묶음으로 넘어갑니다.")
        row.addWidget(self.keep_going)
        row.addSpacing(12)
        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        row.addWidget(self.bar, 1)
        ctl.lay.addLayout(row)
        self.step = label("", "muted", wrap=True)
        ctl.lay.addWidget(self.step)
        self.body.addWidget(ctl)

        body = QHBoxLayout()
        body.setSpacing(12)
        fun = Card(14, 6)
        fun.lay.addLayout(pane_header("탈락 이유 (먼저 걸린 기준)"))
        self.funnel = self._table(("기준", "탈락"))
        fun.lay.addWidget(self.funnel)
        body.addWidget(fun, 1)
        cand = Card(14, 6)
        self.cand_head = label("후보", "ph")
        top = QHBoxLayout()
        self.list_seg = QButtonGroup(self)
        self.list_buttons = {}
        for near, text in ((False, "후보"), (True, "아깝게 떨어진 전략")):
            b = button(text, "seg")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, n=near: self.show_near(n))
            self.list_seg.addButton(b)
            top.addWidget(b)
            self.list_buttons[near] = b
        self.list_buttons[False].setChecked(True)
        top.addSpacing(12)
        top.addWidget(self.cand_head)
        top.addStretch()
        self.view_btn = button("백테스트 보기", "secondary", self.view_candidate)
        self.csv_btn = button("CSV로 저장…", "secondary", self.save_near_csv)
        self.csv_btn.setVisible(False)
        self.final_btn = button("최종 평가…", "primary", self.final_candidate)
        top.addWidget(self.view_btn)
        top.addWidget(self.csv_btn)
        top.addWidget(self.final_btn)
        cand.lay.addLayout(top)
        cand.lay.addWidget(hline())
        self.cands = self._table(self.COLS)
        self.cands.setItemDelegate(KeepColors(self.cands))
        self.cands.itemSelectionChanged.connect(self._buttons)
        cand.lay.addWidget(self.cands)
        self.cand_note = label("", "muted", wrap=True)
        cand.lay.addWidget(self.cand_note)
        body.addWidget(cand, 3)
        self.body.addLayout(body)
        self.body.addStretch()
        self.seg_buttons[self.group].setChecked(True)

    @property
    def lab(self):
        return self.win.lab

    def _table(self, cols) -> QTableWidget:
        t = QTableWidget(0, len(cols))
        t.setObjectName("plain")
        t.setHorizontalHeaderLabels(list(cols))
        t.verticalHeader().setVisible(False)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setSelectionMode(QAbstractItemView.SingleSelection)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setMinimumHeight(300)
        return t

    # --- state ---------------------------------------------------------------------
    def choose(self, group: str) -> None:
        if self.running and group != self.group:
            self.seg_buttons[self.group].setChecked(True)
            self.win.notify("탐색이 끝나거나 멈춘 뒤 다른 묶음으로 바꿀 수 있습니다.")
            return
        self.group = group
        self.seg_buttons[group].setChecked(True)
        self.refresh()

    def factory_refresh(self) -> None:
        from quantlab.services import scheduler

        f = self.lab.factory()
        p = f.progress()
        self.factory_on.blockSignals(True)
        self.factory_on.setChecked(p["enabled"])
        self.factory_on.blockSignals(False)
        self.factory_on.setEnabled(scheduler.supported())
        self.factory_on.setToolTip("" if scheduler.supported() else "설치된 Windows 앱에서만 쓸 수 있어요")
        self.factory_line.setText(factory_text(p))

    def _factory_toggled(self, on: bool) -> None:
        from quantlab.services import scheduler

        f = self.lab.factory()
        if on and self.running and self.group in FACTORY_GROUPS:
            self.win.notify("이 화면에서 공장 묶음을 탐색하는 중이에요. 멈춘 뒤에 공장을 켜 주세요.")
            self.factory_refresh()
            return
        f.set_enabled(on)
        try:
            if on:
                scheduler.install()  # the factory's rules are paper traded every day
                scheduler.install_factory()
            else:
                scheduler.remove_factory()
                if not self.lab.paper().auto_ids():
                    scheduler.remove()
        except RuntimeError as e:
            f.set_enabled(not on)
            self.win.notify(str(e))
        self.factory_refresh()
        self.win.status("자동 연구 공장을 켰어요" if on else "자동 연구 공장을 껐어요 (지금 시험 중인 조합까지 하고 멈춰요)")

    def refresh(self) -> None:
        """Read the group's summary off the window thread (the first read of a large group takes a
        while; later reads only add the trials written since) and show it when it arrives."""
        self.group_note.setText(GROUP_NOTE.get(self.group, ""))
        try:
            self.factory_refresh()
        except Exception as e:  # noqa: BLE001 - the search page must open even if the factory files are damaged
            self.factory_line.setText(f"공장 상태를 읽지 못했어요: {e}")
        if self.win.runner.sync:
            try:
                self._show(self.lab.auto_search().summary(self.group))
            except Exception as e:  # shown, never swallowed
                self.step.setText(f"요약을 읽지 못했습니다: {e}")
                self._buttons()
            return
        if self._reading:
            self._read_again = True
            return
        from quantlab.gui.main_window import _Job, _Signals

        self._reading, self._read_again = True, False
        group, root = self.group, self.lab.ws.root
        sig = _Signals()
        self._keep.append(sig)

        def job(prog):
            from quantlab.services.lab import Lab

            lab = Lab(root)  # its own registry connection; the window keeps using its own
            try:
                return lab.auto_search().summary(group)
            finally:
                lab.close()

        def arrived(s):
            self._read_done(sig)
            if s["group"] == self.group:
                self._show(s)

        def failed(text):
            self._read_done(sig)
            self.step.setText(f"요약을 읽지 못했습니다: {text.splitlines()[0]}")
            self._buttons()

        sig.done.connect(arrived)
        sig.failed.connect(failed)
        if not self.summary or self.summary.get("group") != group:
            self.step.setText("탐색 기록을 읽는 중")
        QThreadPool.globalInstance().start(_Job(job, sig))

    def _read_done(self, sig) -> None:
        self._reading = False
        if sig in self._keep:
            self._keep.remove(sig)
        if self._read_again:
            self.refresh()

    def _show(self, s: dict) -> None:
        if self.step.text() == "탐색 기록을 읽는 중":
            self.step.setText("")
        self.summary = s
        v = self.kpi_values
        v[0].setText(f"{s['done']:,} / {s['grid']:,}")
        v[1].setText(f"{s['family_trials']:,}")
        v[2].setText(f"{s['t_required']:.2f}")
        b = s.get("beat_value")
        v[3].setText("—" if b is None else display_name(s.get("beat_symbol", "SPY")) + " "
                     + (f"샤프 {b:.2f}" if s.get("beat_metric") == "sharpe" else f"연 {pct(b)}"))
        v[4].setText(f"{sum(1 for c in s['candidates'] if c['status'] == '후보')}")
        if not self.running:
            self.start_btn.setText(f"이어서 하기 ({s['pending']:,}개 남음)" if s["pending"] else "탐색 시작 (3,000개)")
        self.funnel.setRowCount(len(s["funnel"]))
        for i, f in enumerate(s["funnel"]):
            self.funnel.setItem(i, 0, cell(f["label"], mono=False, align=Qt.AlignLeft))
            self.funnel.setItem(i, 1, cell(f"{f['count']:,}"))
        self.funnel.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._fill_list()

    def _rows(self) -> list[dict]:
        s = self.summary or {}
        return (s.get("near") or [])[:NEAR_SHOWN] if self.near_mode else s.get("candidates") or []

    def show_near(self, near: bool) -> None:
        self.near_mode = near
        self.list_buttons[near].setChecked(True)
        self._fill_list()

    def _fill_list(self) -> None:
        s = self.summary or {}
        rows = self._rows()
        self.cands.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [cell(r["rule"], mono=False, align=Qt.AlignLeft), cell(f"{r['n']}"), cell(pct(r.get("gross"), True)),
                    cell(pct(r.get("net"), True)), cell(pct(r.get("cagr"), True)),
                    cell(f"{r['sharpe']:.2f}" if r.get("sharpe") is not None else "—"), cell(pct(r.get("mdd"))),
                    cell(f"{r['t']:.2f}" if r.get("t") is not None else "—"), cell(r.get("neighbours", "—")),
                    cell(r["status"], theme.OK if r["status"] == "후보" or r.get("preregistered") else theme.ACCENT, mono=False)]
            for j, it in enumerate(vals):
                self.cands.setItem(i, j, it)
        self.cands.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.csv_btn.setVisible(self.near_mode)
        self.final_btn.setVisible(not self.near_mode or any(r.get("preregistered") for r in rows))
        if self.near_mode:
            total = len(s.get("near") or [])
            by = "샤프" if s.get("beat_metric") == "sharpe" else "연수익"
            self.cand_head.setText(f"{total:,}개" + (f" (위 {NEAR_SHOWN}개 표시)" if total > NEAR_SHOWN else ""))
            self.cand_note.setText(
                "아직 없습니다." if not total else
                f"앞의 네 기준(거래 수·거래당 수익·두 구간 이익·비용 2배)은 통과했지만 {display_name(s.get('beat_symbol', 'SPY'))} 비교나 그 뒤 기준에서 떨어진 조합입니다. "
                f"{by}이 높은 순서입니다. 아이디어를 얻는 용도로만 보세요. 수천 개 중에 골라내면 운 좋은 조합을 집기 쉬워서, "
                "여기서는 최종 평가를 할 수 없습니다. 맨 위 '사전 고정' 조합만 예외입니다 (미리 정해 둔 하나).")
        else:
            reads = s.get("holdout_reads", 0)
            self.cand_head.setText(f"{len(rows)}개")
            self.cand_note.setText(
                "아직 후보가 없습니다. 탐색을 더 돌리거나 다른 묶음을 시험해 보세요." if not rows else
                f"이 묶음의 보류 구간은 {'아직 열지 않았습니다' if not reads else f'이미 {reads}번 열었습니다 (다음 최종 평가는 PASS가 될 수 없음)'}. "
                "최종 평가는 묶음마다 후보 하나만 하는 것이 원칙입니다.")
        self._buttons()

    def save_near_csv(self) -> None:
        rows = (self.summary or {}).get("near") or []
        if not rows:
            self.win.notify("저장할 조합이 없습니다.")
            return
        default = str(self.lab.ws.root / f"{self.group}_아깝게_떨어진_전략.csv")
        path, _ = QFileDialog.getSaveFileName(self, "CSV로 저장", default, "CSV (*.csv)")
        if path:
            write_near_csv(rows, path)
            self.win.notify(f"{len(rows):,}개를 저장했습니다: {path}")

    def _buttons(self) -> None:
        has = self.cands.rowCount() > 0
        self.view_btn.setEnabled(has and not self.win.runner.busy)
        self.final_btn.setEnabled(has and not self.win.runner.busy
                                  and (not self.near_mode or bool((self._selected() or {}).get("preregistered"))))
        self.csv_btn.setEnabled(bool((self.summary or {}).get("near")))
        self.start_btn.setEnabled(not self.running)
        self.stop_btn.setEnabled(self.running)
        for b in self.seg_buttons.values():
            b.setEnabled(not self.running or b.isChecked())

    # --- running -------------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        from quantlab.gui.main_window import _Job, _Signals

        if self.group in FACTORY_GROUPS and self.lab.factory().running():
            self.win.notify("자동 연구 공장이 이 묶음을 시험하고 있어요. 공장을 끄면 여기서 직접 돌릴 수 있어요.")
            return
        self.running, self._stop = True, False
        _keep_awake(True)
        self._buttons()
        group, workers, root, limit = self.group, self.workers.value(), self.lab.ws.root, self.limit
        sig = _Signals()
        self._keep.append(sig)

        def job(prog):
            from quantlab.services.lab import Lab

            lab = Lab(root)  # its own registry connection; the window keeps using its own
            try:
                return lab.auto_search().run(group, workers=workers, progress=prog, should_stop=lambda: self._stop,
                                             say=prog.say, limit=limit)
            finally:
                lab.close()

        def progress(a, b):
            self.bar.setMaximum(b)
            self.bar.setValue(a)
            self.step.setText(f"{a:,} / {b:,} 조합 시험함")
            if a % 200 == 0:
                self.refresh()

        def finish(out):
            self._done(sig)
            note = out.get("note") or ("멈췄습니다. '이어서 하기'로 남은 조합을 마저 시험합니다." if out.get("stopped")
                                       else "이번 탐색을 마쳤습니다.")
            self.step.setText(note)
            self.win.status(f"자동 탐색 {group}: {note}")
            if self.keep_going.isChecked() and not out.get("stopped"):
                self._continue(group, exhausted=bool(out.get("note")))

        def fail(text):
            self._done(sig)
            self.step.setText("탐색이 중단되었습니다.")
            self.win.notify(f"자동 탐색 실패\n\n{text}")

        sig.done.connect(finish)
        sig.failed.connect(fail)
        sig.progress.connect(progress)
        sig.text.connect(self.step.setText)
        j = _Job(job, sig)
        if self.win.runner.sync:
            j.run()
        else:
            QThreadPool.globalInstance().start(j)

    def _continue(self, group: str, exhausted: bool) -> None:
        """Start the next run: the same group, or the next group once this one is fully tried."""
        if exhausted:
            self._exhausted.add(group)
            rest = [g for g in GROUPS if g not in self._exhausted]
            if not rest:
                self.step.setText("모든 묶음의 조합을 시험했습니다.")
                return
            nxt = [g for g in GROUPS[GROUPS.index(self.group) + 1:] + GROUPS if g not in self._exhausted]
            group = nxt[0]
        self.choose(group)
        if self.win.runner.sync:
            self.start()
        else:
            QTimer.singleShot(2000, self.start)

    def _done(self, sig) -> None:
        self.running = False
        _keep_awake(False)
        if sig in self._keep:
            self._keep.remove(sig)
        self._buttons()  # at once: the summary read that follows may fail or take a while
        self.refresh()

    def stop(self) -> None:
        self._stop = True
        self.step.setText("멈추는 중: 지금 시험 중인 조합이 끝나면 멈춥니다.")

    # --- candidates ----------------------------------------------------------------
    def _chosen(self) -> dict | None:
        r = self.cands.currentRow()
        rows = self._rows()
        if not 0 <= r < len(rows):
            r = 0 if rows else -1
        if r < 0:
            self.win.notify("후보를 먼저 고르세요.")
            return None
        return rows[r]

    def view_candidate(self) -> None:
        c = self._chosen()
        if not c:
            return
        group = self.group

        def done(bt):
            self.win.refresh_all()
            self.win.show_result(bt.experiment_id)

        self.win.runner.start("후보 백테스트", lambda p: self.lab.auto_search().open_candidate(group, c["params"]), done)

    def _selected(self) -> dict | None:
        rows = self._rows()
        r = self.cands.currentRow()
        return rows[r] if 0 <= r < len(rows) else (rows[0] if rows else None)

    def final_candidate(self) -> None:
        c = self._chosen()
        if c and self.near_mode and not c.get("preregistered"):
            self.win.notify("이 목록에서는 '사전 고정' 조합만 최종 평가할 수 있습니다.")
            return
        if not c:
            return
        s = self.summary or {}
        reads = s.get("holdout_reads", 0)
        what = "미리 고정해 둔 조합(정식 후보 아님)" if c.get("preregistered") else "후보"
        text = (f"{self.group} {what}을 최종 평가합니다.\n\n규칙: {c['rule']}\n"
                f"이 묶음 누적 시험: {s.get('family_trials', 0):,}개 (판정의 보정에 반영)\n"
                f"이 묶음의 이전 보류 구간 열람: {reads}회\n\n"
                f"{str(s.get('holdout_start', ''))[:4]}년 이후 보류 구간을 한 번 읽습니다. 이 묶음에서는 후보 하나만 최종 평가하는 것이 원칙이고, "
                "두 번째부터는 PASS가 될 수 없습니다. 진행할까요?")
        if not self.win.ask(text):
            return
        group = self.group

        def done(out):
            self.win.refresh_all()
            self.win.show_result(out.experiment_id)

        self.win.runner.start("최종 평가", lambda p: self.lab.auto_search().final_evaluation(group, c["params"]), done)
