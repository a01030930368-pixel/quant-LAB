"""모의 투자 page (design option B, "today's orders first"): what the rule buys and sells at the next
open, the five slots, the score against SPY since the start, and the record below.

One button fetches today's data and recomputes (`quantlab.services.paper`); no order reaches a broker.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QGridLayout, QHBoxLayout, QHeaderView, QProgressBar,
                               QTableWidget)

from quantlab.gui import theme
from quantlab.gui.home import Card, KeepColors, Page, _empty, button, cell, chip, label, pane_header, pct

KST = timezone(timedelta(hours=9))
WEEKDAY = "월화수목금토일"


def _color(x) -> str | None:
    if x is None or x != x or abs(x) < 1e-12:
        return None
    return theme.RISE if x > 0 else theme.FALL


def next_session(asof: str) -> pd.Timestamp:
    from quantlab.data.quality import exchange_sessions

    a = pd.Timestamp(asof)
    s = exchange_sessions(str((a + pd.Timedelta(days=1)).date()), str((a + pd.Timedelta(days=10)).date()))
    return s[0] if len(s) else a + pd.offsets.BDay(1)


def _day(ts) -> str:
    t = pd.Timestamp(ts)
    return f"{t.month:02d}-{t.day:02d} {WEEKDAY[t.weekday()]}요일"


def order_text(o: dict, snap: dict) -> str:
    if o["side"] == "buy":
        return f"계좌의 {o['weight'] * 100:.0f}%만큼 삽니다. 매수 조건이 오늘 처음 모두 맞았어요."
    if o["side"] == "sell":
        return f"{snap.get('hold') or ''}거래일을 채워서 전부 팝니다." if snap.get("hold") else "전부 팝니다."
    return f"쉬는 돈을 SPY에 넣어 SPY 비중을 {o['weight'] * 100:.0f}%로 맞춥니다."


class PaperPage(Page):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.snap: dict | None = None
        self.view = "closed"
        head = QHBoxLayout()
        self.title = label("모의 투자", "h1")
        head.addWidget(self.title)
        self.idchip = chip("")
        head.addWidget(self.idchip)
        self.pick = QComboBox()
        self.pick.setMinimumWidth(260)
        self.pick.currentIndexChanged.connect(self._picked)
        head.addWidget(self.pick)
        head.addStretch()
        self.when = label("", "muted")
        head.addWidget(self.when)
        self.go_btn = button("오늘 자료로 갱신", "primary", self.refresh_now)
        head.addWidget(self.go_btn)
        self.body.addLayout(head)

        self.progress = Card(12, 6)
        self.step = label("", "h3")
        bar = QProgressBar()
        bar.setRange(0, 0)
        bar.setTextVisible(False)
        self.progress.lay.addWidget(self.step)
        self.progress.lay.addWidget(bar)
        self.progress.hide()
        self.body.addWidget(self.progress)

        self.todo = Card(18, 10)
        self.todo.setStyleSheet(f"QFrame#card {{ border: 1px solid {theme.ACCENT}; }}")
        self.body.addWidget(self.todo)
        self.slots_title = label("", "ph")
        self.body.addWidget(self.slots_title)
        self.slots = QGridLayout()
        self.slots.setSpacing(10)
        self.body.addLayout(self.slots)
        self.kpis = QGridLayout()
        self.kpis.setSpacing(10)
        self.body.addLayout(self.kpis)
        self.warn = label("", "body2", wrap=True)
        self.warn.setStyleSheet(f"color: {theme.ACCENT};")
        self.warn.hide()
        self.body.addWidget(self.warn)

        links = QHBoxLayout()
        self.closed_btn = button("마친 거래 보기", "seg", lambda: self._show("closed"))
        self.days_btn = button("날짜별 장부 보기", "seg", lambda: self._show("days"))
        for b in (self.closed_btn, self.days_btn):
            b.setCheckable(True)
            links.addWidget(b)
        links.addStretch()
        self.body.addLayout(links)
        self.table = QTableWidget(0, 0)
        self.table.setObjectName("plain")
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setItemDelegate(KeepColors(self.table))
        self.table.setMinimumHeight(260)
        self.body.addWidget(self.table, 1)
        self.rows: list[dict] = []

    @property
    def lab(self):
        return self.win.lab

    # --- data ----------------------------------------------------------------------
    def refresh(self) -> None:
        """Re-read the list of rules and the last saved day (no download)."""
        try:
            self.rows = self.lab.paper().candidates()
        except Exception:  # noqa: BLE001 - the page must open even when a spec is broken
            self.rows = []
        cur = self.pick.currentData()
        self.pick.blockSignals(True)
        self.pick.clear()
        for r in self.rows:
            self.pick.addItem(f"{r['id']} · {r['name']}", r["path"])
        i = max(0, self.pick.findData(cur)) if cur else 0
        self.pick.setCurrentIndex(i)
        self.pick.blockSignals(False)
        self._picked()

    def current(self) -> dict | None:
        i = self.pick.currentIndex()
        return self.rows[i] if 0 <= i < len(self.rows) else None

    def _picked(self) -> None:
        r = self.current()
        self.snap = self.lab.paper().latest(r["id"], r["version"]) if r else None
        self.render()

    def refresh_now(self) -> None:
        r = self.current()
        if not r:
            return
        home = self.win.home
        if not home._ensure_sec_email(r["path"]):
            return

        def done(snap):
            self.progress.hide()
            self.snap = snap
            self.render()

        self.step.setText("준비 중")
        self.progress.show()
        if not self.win.runner.start("모의 투자 갱신", lambda p: self.lab.paper().refresh(r["path"], say=p.say), done,
                                     on_text=self.step.setText) or self.win.runner.last_error and self.win.runner.sync:
            self.progress.hide()

    # --- drawing -------------------------------------------------------------------
    def render(self) -> None:
        r, s = self.current(), self.snap
        self.idchip.setText(r["id"] if r else "")
        _empty(self.todo.lay)
        _empty(self.slots)
        _empty(self.kpis)
        self.warn.hide()
        if not r:
            self.todo.lay.addWidget(label("모의 투자할 수 있는 전략이 없어요 (정책에 전진 보류 시작일이 있는 전략만 돼요).", "body2"))
            self._fill_table([], [])
            return
        if not s:
            self.when.setText("아직 갱신 전")
            self.todo.lay.addWidget(label(f"{r['start']} 이후 새로 쌓이는 자료로 매일 확인해요. "
                                          "'오늘 자료로 갱신'을 누르면 시작해요.", "body2", wrap=True))
            self._fill_table([], [])
            return
        t = (datetime.strptime(s["refreshed_at"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).astimezone(KST)
             if s.get("refreshed_at") else None)
        self.when.setText(f"기준일 {s['asof']} · 갱신 {t:%m-%d %H:%M} 한국 시간" if t is not None else "")
        if not s.get("started"):
            self.todo.lay.addWidget(label(s.get("note", ""), "body2", wrap=True))
            self._fill_table([], [])
            return
        nxt = next_session(s["asof"])
        self.todo.lay.addLayout(pane_header(f"다음 미국 장 시작 때 ({_day(nxt)} 시가)"))
        orders = sorted(s["orders"], key=lambda o: {"sell": 0, "buy": 1, "park": 2}[o["side"]])
        if not [o for o in orders if o["side"] != "park"]:
            self.todo.lay.addWidget(label("사고팔 종목이 없어요. 들고 있는 그대로 둡니다.", "h3"))
        for o in orders:
            row = QHBoxLayout()
            row.setSpacing(14)
            name, color = {"buy": ("매수", theme.RISE), "sell": ("매도", theme.FALL), "park": ("SPY", theme.TEXT2)}[o["side"]]
            tag = label(name, "h3")
            tag.setFixedWidth(52)
            tag.setAlignment(Qt.AlignCenter)
            tag.setStyleSheet(f"background: {color}; color: {theme.BG}; border-radius: 3px; padding: 3px 0;")
            row.addWidget(tag)
            sym = label(o["symbol"], "num")
            sym.setFixedWidth(90)
            row.addWidget(sym)
            row.addWidget(label(order_text(o, s), "body2", wrap=True), 1)
            self.todo.lay.addLayout(row)
        self.todo.lay.addWidget(label("실제 주문은 나가지 않아요. 앱이 다음 날 시가에 수수료 0.25%와 슬리피지를 붙여 "
                                      "샀다고 장부에 적어요.", "muted", wrap=True))

        n = int(s.get("params", {}).get("slots") or 0)
        held = [h for h in s["holdings"] if not h["park"]]
        park = next((h for h in s["holdings"] if h["park"]), None)
        for i in range(max(n, 5)):
            self.slots.setColumnStretch(i, 1)
        for i in range(4):
            self.kpis.setColumnStretch(i, 1)
        if n:
            self.slots_title.setText(f"{n}칸 (한 칸 = 계좌의 {100 / n:.0f}%)")
            for i in range(n):
                self.slots.addWidget(self._slot(held[i] if i < len(held) else None, park, s), 0, i)
        else:
            self.slots_title.setText("보유 종목")
            for i, h in enumerate(held):
                self.slots.addWidget(self._slot(h, park, s), 0, i)
        closed = s.get("closed", [])
        ex = [c["excess"] for c in closed if c.get("excess") is not None]
        kpis = [("전략", pct(s["ret"], True), _color(s["ret"]), f"{s['start']} 시작 뒤"),
                ("SPY", pct(s["spy_ret"], True), _color(s["spy_ret"]), "같은 기간"),
                ("최대 낙폭", pct(s["mdd"]), theme.FALL if s["mdd"] < 0 else None, f"SPY {pct(s['spy_mdd'])}"),
                ("마친 거래", f"{len(closed)}건", None,
                 f"평균 SPY 대비 {pct(sum(ex) / len(ex), True)}" if ex else "아직 없음")]
        for i, (name, value, color, note) in enumerate(kpis):
            c = Card(12, 2)
            c.lay.addWidget(label(name, "ph"))
            v = label(value, "big")
            if color:
                v.setStyleSheet(f"color: {color};")
            c.lay.addWidget(v)
            c.lay.addWidget(label(note, "muted"))
            self.kpis.addWidget(c, 0, i)
        ch = s.get("ledger", {}).get("changed") or []
        if ch:
            self.warn.setText(f"이전에 적어 둔 거래 {len(ch)}건이 오늘 다시 계산한 결과와 달라요 (가격 자료가 고쳐졌을 수 있어요). "
                              "장부에는 처음 적은 내용이 그대로 남아 있어요.")
            self.warn.show()
        self._show(self.view)

    def _slot(self, h: dict | None, park: dict | None, s: dict) -> Card:
        c = Card(14, 8)
        c.setMinimumHeight(140)
        top = QHBoxLayout()
        if h:
            top.addWidget(label(h["symbol"], "num"))
            top.addStretch()
            r = label(pct(h.get("ret"), True), "num")
            if _color(h.get("ret")):
                r.setStyleSheet(f"color: {_color(h.get('ret'))};")
            top.addWidget(r)
            c.lay.addLayout(top)
            vs = h.get("ret", 0) - h["spy_ret"] if h.get("spy_ret") is not None and h.get("ret") is not None else None
            c.lay.addWidget(label(f"SPY보다 {pct(vs, True)}" if vs is not None else "", "body2"))
            c.lay.addStretch()
            hold = s.get("hold")
            bar = QProgressBar()
            bar.setTextVisible(False)
            bar.setRange(0, hold or 1)
            bar.setValue(min(h.get("sessions", 0), hold or 1))
            c.lay.addWidget(bar)
            c.lay.addWidget(label(f"{h.get('sessions', 0)} / {hold}거래일 · {h.get('entry_date', '')} 매수" if hold
                                  else f"{h.get('entry_date', '')} 매수", "muted"))
        else:
            top.addWidget(label("빈칸", "num"))
            c.lay.addLayout(top)
            c.lay.addWidget(label("SPY에 넣어 둠" if park else "현금", "body2"))
            c.lay.addStretch()
        return c

    def _show(self, view: str) -> None:
        self.view = view
        self.closed_btn.setChecked(view == "closed")
        self.days_btn.setChecked(view == "days")
        s = self.snap
        if not s or not s.get("started"):
            self._fill_table([], [])
            return
        if view == "closed":
            rows = []
            for c in reversed(s.get("closed", [])):
                rows.append([cell(c["symbol"], align=Qt.AlignLeft), cell(f"{c['entry']} → {c['exit']}", theme.TEXT2),
                             cell(pct(c.get("gross"), True), _color(c.get("gross"))),
                             cell(pct(c.get("net"), True), _color(c.get("net"))),
                             cell(pct(c.get("bench"), True), _color(c.get("bench"))),
                             cell(pct(c.get("excess"), True), _color(c.get("excess")))])
            self._fill_table(["종목", "기간", "비용 전", "비용 후", "SPY", "SPY 대비"], rows)
        else:
            r = self.current()
            days = list(reversed(self.lab.paper().days(r["id"], r["version"])))
            rows = []
            for d in days:
                if not d.get("started"):
                    continue
                side = {"buy": "매수", "sell": "매도"}
                acts = ", ".join(f"{side[o['side']]} {o['symbol']}" for o in d["orders"] if o["side"] in side) or "없음"
                rows.append([cell(d["asof"], align=Qt.AlignLeft), cell(pct(d["ret"], True), _color(d["ret"])),
                             cell(pct(d["spy_ret"], True), _color(d["spy_ret"])),
                             cell(str(len([h for h in d["holdings"] if not h["park"]]))),
                             cell(acts, theme.TEXT2, mono=False, align=Qt.AlignLeft)])
            self._fill_table(["기준일", "전략", "SPY", "보유", "다음 날 주문"], rows)

    def _fill_table(self, cols: list[str], rows: list[list]) -> None:
        self.table.clear()
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, it in enumerate(row):
                self.table.setItem(i, j, it)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setStretchLastSection(False)

    def _buttons(self) -> None:
        self.go_btn.setEnabled(not self.win.runner.busy)
