"""자동 탐색 page: run the automatic strategy search for one ETF group and review its candidates.

The search runs on its own thread with its own registry connection, so the other pages stay usable
while it works. It only ever sees TRAIN+VALIDATION data; the final evaluation of a candidate is a
separate, confirmed action by the user (`quantlab.services.autosearch`).
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QHBoxLayout, QHeaderView, QProgressBar, QPushButton,
                               QSpinBox, QTableWidget, QVBoxLayout)

from quantlab.gui import theme
from quantlab.gui.home import Card, KeepColors, Page, button, cell, hline, kpi_strip, label, pane_header, pct
from quantlab.services.autosearch import GROUPS

def _keep_awake(on: bool) -> None:
    """Windows: stop the PC from going to sleep while the search runs (the screen may still turn off)."""
    if os.name != "nt":
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))


GROUP_NAME = {"AUTO-DEVA": "선진국 A", "AUTO-DEVB": "선진국 B", "AUTO-EM": "신흥국", "AUTO-REAL": "채권·금·원자재·리츠"}
GROUP_NOTE = {
    "AUTO-DEVA": "일본·독일·영국·캐나다·호주·프랑스·스위스·홍콩·싱가포르 ETF 9종",
    "AUTO-DEVB": "스웨덴·네덜란드·스페인·이탈리아·벨기에·오스트리아 ETF 6종",
    "AUTO-EM": "브라질·멕시코·대만·한국·남아공·말레이시아·중국 ETF 7종",
    "AUTO-REAL": "미국 장기·중기 국채, 회사채, 금, 은, 원자재, 리츠 ETF 7종",
}


class AutoPage(Page):
    COLS = ("규칙", "거래", "비용 전/거래", "비용 후/거래", "연수익", "MDD", "t값", "이웃", "상태")

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

        head = QHBoxLayout()
        head.addWidget(label("자동 탐색", "h1"))
        head.addStretch()
        self.body.addLayout(head)
        self.body.addWidget(label(
            "앱이 재료(진입 신호·추세 조건·청산 규칙·보유 기간·손절·익절)를 조합해 TRAIN+VALIDATION 구간에서만 시험하고, "
            "걸러내는 기준을 모두 통과한 조합만 후보로 남깁니다. 시험한 개수만큼 통과 기준(t값)이 높아집니다. "
            "최종 평가(2017년 이후 보류 구간)는 앱이 스스로 열지 않습니다.", "muted", wrap=True))

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
                                                ("필요한 t값", "—", None), ("SPY 보유 연수익", "—", None),
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
        top.addWidget(self.cand_head)
        top.addStretch()
        self.view_btn = button("백테스트 보기", "secondary", self.view_candidate)
        self.final_btn = button("최종 평가…", "primary", self.final_candidate)
        top.addWidget(self.view_btn)
        top.addWidget(self.final_btn)
        cand.lay.addLayout(top)
        cand.lay.addWidget(hline())
        self.cands = self._table(self.COLS)
        self.cands.setItemDelegate(KeepColors(self.cands))
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

    def refresh(self) -> None:
        self.group_note.setText(GROUP_NOTE.get(self.group, ""))
        try:
            s = self.lab.auto_search().summary(self.group)
        except Exception as e:  # shown, never swallowed
            self.step.setText(f"요약을 읽지 못했습니다: {e}")
            return
        self.summary = s
        v = self.kpi_values
        v[0].setText(f"{s['done']:,} / {s['grid']:,}")
        v[1].setText(f"{s['family_trials']:,}")
        v[2].setText(f"{s['t_required']:.2f}")
        v[3].setText(pct(s["beat_cagr"]))
        v[4].setText(f"{sum(1 for c in s['candidates'] if c['status'] == '후보')}")
        if not self.running:
            self.start_btn.setText(f"이어서 하기 ({s['pending']:,}개 남음)" if s["pending"] else "탐색 시작 (3,000개)")
        self.funnel.setRowCount(len(s["funnel"]))
        for i, f in enumerate(s["funnel"]):
            self.funnel.setItem(i, 0, cell(f["label"], mono=False, align=Qt.AlignLeft))
            self.funnel.setItem(i, 1, cell(f"{f['count']:,}"))
        self.funnel.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        rows = s["candidates"]
        self.cands.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [cell(r["rule"], mono=False, align=Qt.AlignLeft), cell(f"{r['n']}"), cell(pct(r.get("gross"), True)),
                    cell(pct(r.get("net"), True)), cell(pct(r.get("cagr"), True)), cell(pct(r.get("mdd"))),
                    cell(f"{r['t']:.2f}" if r.get("t") is not None else "—"), cell(r["neighbours"]),
                    cell(r["status"], theme.OK if r["status"] == "후보" else theme.ACCENT, mono=False)]
            for j, it in enumerate(vals):
                self.cands.setItem(i, j, it)
        self.cands.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.cand_head.setText(f"후보 {len(rows)}")
        reads = s["holdout_reads"]
        self.cand_note.setText(
            "아직 후보가 없습니다. 탐색을 더 돌리거나 다른 묶음을 시험해 보세요." if not rows else
            f"이 묶음의 보류 구간은 {'아직 열지 않았습니다' if not reads else f'이미 {reads}번 열었습니다 (다음 최종 평가는 PASS가 될 수 없음)'}. "
            "최종 평가는 묶음마다 후보 하나만 하는 것이 원칙입니다.")
        self._buttons()

    def _buttons(self) -> None:
        has = self.cands.rowCount() > 0
        self.view_btn.setEnabled(has and not self.win.runner.busy)
        self.final_btn.setEnabled(has and not self.win.runner.busy)
        self.start_btn.setEnabled(not self.running)
        self.stop_btn.setEnabled(self.running)
        for b in self.seg_buttons.values():
            b.setEnabled(not self.running or b.isChecked())

    # --- running -------------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        from quantlab.gui.main_window import _Job, _Signals

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
                self.step.setText("네 묶음의 조합을 모두 시험했습니다.")
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
        self.refresh()

    def stop(self) -> None:
        self._stop = True
        self.step.setText("멈추는 중: 지금 시험 중인 조합이 끝나면 멈춥니다.")

    # --- candidates ----------------------------------------------------------------
    def _chosen(self) -> dict | None:
        r = self.cands.currentRow()
        rows = (self.summary or {}).get("candidates") or []
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

    def final_candidate(self) -> None:
        c = self._chosen()
        if not c:
            return
        s = self.summary or {}
        reads = s.get("holdout_reads", 0)
        text = (f"{self.group} 후보를 최종 평가합니다.\n\n규칙: {c['rule']}\n"
                f"이 묶음 누적 시험: {s.get('family_trials', 0):,}개 (판정의 보정에 반영)\n"
                f"이 묶음의 이전 보류 구간 열람: {reads}회\n\n"
                "2017년 이후 보류 구간을 한 번 읽습니다. 이 묶음에서는 후보 하나만 최종 평가하는 것이 원칙이고, "
                "두 번째부터는 PASS가 될 수 없습니다. 진행할까요?")
        if not self.win.ask(text):
            return
        group = self.group

        def done(out):
            self.win.refresh_all()
            self.win.show_result(out.experiment_id)

        self.win.runner.start("최종 평가", lambda p: self.lab.auto_search().final_evaluation(group, c["params"]), done)
