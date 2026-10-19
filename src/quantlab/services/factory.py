"""자동 연구 공장 (D-033): the automatic search of AUTO-SP500X, run without the window.

Windows Task Scheduler starts `QuantLab.exe --factory` when the user signs in, when the charger is
plugged in, and once an hour in case it was closed, never on battery; the process stops itself within
half a minute of the charger being unplugged (finished trials are kept, the unfinished ones run again
on the next start). It lowers its own priority, so the user's other programs come first,
and runs the search 3,000 combinations at a time until the whole grid is tried, the user switches
the factory off, or the PC shuts down. It never needs Claude and never reads the holdout (the
search sees TRAIN+VALIDATION only; the 2020+ holdout of these stocks was spent by GAP-EARN-SP500).

Kept in <work folder>/factory/:

    log.jsonl      one line per run of 3,000: when, how many, candidates so far
    watch.json     rules sent to paper trading on forward data: every candidate, and the few best
                   near misses that failed only the t-value (labelled as such; selected on noise
                   among hundreds of thousands, so forward data is their only test)
    news.json      what the app shows when it opens: a new candidate, or the final report once the
                   grid is done (the user asked for a report at the end, not a weekly one)
    running.lock   held while a factory process runs; refreshed every minute
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

GROUP = "AUTO-SP500X"
KST = timezone(timedelta(hours=9))
NEAR_IN_WATCH = 5        # t-only near misses kept in paper trading at once
LOCK_STALE_SECONDS = 15 * 60
WATCH_REV = 2           # 2: the near misses in paper trading are the strongest so far, not the first found
CHECK_SECONDS = 30       # how often a running factory re-reads the on/off switch


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _kst(ts: datetime | None = None) -> str:
    return (ts or _now()).astimezone(KST).strftime("%Y-%m-%d %H:%M")


def low_priority() -> None:
    """Below every program the user starts; the search's worker processes inherit it."""
    if os.name == "nt":
        import ctypes

        idle_priority_class = 0x40
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), idle_priority_class)
    else:
        try:
            os.nice(10)
        except OSError:
            pass


def on_battery() -> bool:
    """Windows laptop unplugged. The user asked for the factory to run on the charger only; a desktop
    or an unknown state counts as plugged in."""
    if os.name != "nt":
        return False
    import ctypes

    class PowerStatus(ctypes.Structure):
        _fields_ = [("ac_line", ctypes.c_ubyte), ("battery_flag", ctypes.c_ubyte), ("battery_percent", ctypes.c_ubyte),
                    ("system_status", ctypes.c_ubyte), ("lifetime", ctypes.c_ulong), ("full_lifetime", ctypes.c_ulong)]

    st = PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
        return False
    return st.ac_line == 0


class FactoryBusy(RuntimeError):
    pass


class Factory:
    def __init__(self, lab) -> None:
        self.lab = lab

    # --- files -----------------------------------------------------------------------
    def path(self, name: str) -> Path:
        p = self.lab.ws.path("factory", name)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _read(self, name: str, default):
        p = self.path(name)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    def _write(self, name: str, value) -> None:
        p = self.path(name)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, p)

    # --- switch ----------------------------------------------------------------------
    def enabled(self) -> bool:
        return bool(self.lab._settings().get("factory_on"))

    def set_enabled(self, on: bool) -> None:
        self.lab.set_setting("factory_on", bool(on))

    def running(self) -> bool:
        p = self.path("running.lock")
        return p.exists() and time.time() - p.stat().st_mtime < LOCK_STALE_SECONDS

    @contextmanager
    def _lock(self):
        p = self.path("running.lock")
        if p.exists() and time.time() - p.stat().st_mtime >= LOCK_STALE_SECONDS:
            p.unlink(missing_ok=True)  # left behind by a process that was killed
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise FactoryBusy("자동 연구 공장이 이미 돌고 있어요") from None
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        try:
            yield
        finally:
            p.unlink(missing_ok=True)

    # --- records -----------------------------------------------------------------------
    def log(self) -> list[dict]:
        p = self.path("log.jsonl")
        if not p.exists():
            return []
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

    def _log(self, row: dict) -> None:
        with open(self.path("log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def record_error(self, e: BaseException) -> None:
        with open(self.path("errors.log"), "a", encoding="utf-8") as f:
            f.write(f"{_kst()} {type(e).__name__}: {e}".replace("\n", " ") + "\n")

    def last_error(self) -> str | None:
        p = self.path("errors.log")
        lines = p.read_text(encoding="utf-8").strip().splitlines() if p.exists() else []
        return lines[-1] if lines else None

    def watch(self, all_: bool = False) -> list[dict]:
        """Rules in paper trading (`all_`: also those replaced by stronger near misses)."""
        w = self._read("watch.json", [])
        return w if all_ else [x for x in w if not x.get("dropped")]

    def news(self) -> list[dict]:
        return self._read("news.json", [])

    def unseen(self) -> list[dict]:
        return [n for n in self.news() if not n.get("seen")]

    def mark_seen(self) -> None:
        self._write("news.json", [dict(n, seen=True) for n in self.news()])

    def _news(self, item: dict) -> None:
        self._write("news.json", self.news() + [{"at": _kst(), "seen": False, **item}])

    def finished(self) -> bool:
        """The whole grid of the current spec has been tried (checked without loading any data)."""
        done = self._read("finished.json", {})
        if not done:
            return False
        strategy_hash, _ = self.lab.auto_search().loaded(GROUP)
        return done.get("strategy_hash") == strategy_hash

    # --- running ----------------------------------------------------------------------
    def run(self, should_stop=None, max_runs: int | None = None, limit: int | None = None, workers: int = 0,
            today=None) -> str:
        """Search until the grid is done, the switch is off, or `should_stop`. Returns why it ended."""
        if self.finished():
            if self._read("finished.json", {}).get("watch_rev", 1) < WATCH_REV:
                with self._lock():  # a grid finished by an older version: pick the final top near misses once
                    self._update_watch(self.lab.auto_search().summary(GROUP), today)
                    self._write("finished.json", self._read("finished.json", {}) | {"watch_rev": WATCH_REV})
            return "finished"
        with self._lock():
            low_priority()
            auto = self.lab.auto_search()
            last = [0.0, True]

            def stop() -> bool:
                if should_stop and should_stop():
                    return True
                if time.time() - last[0] >= CHECK_SECONDS:
                    last[0] = time.time()
                    self.path("running.lock").touch()
                    last[1] = self.enabled() and not on_battery()
                return not last[1]

            runs = 0
            while True:
                if stop():
                    return "stopped"
                before = sum(1 for v in auto.trials(GROUP).values() if v["status"] == "done")
                start = _now()
                out = auto.run(GROUP, workers=workers, should_stop=stop, limit=limit)
                row = {"start": _kst(start), "end": _kst(), "minutes": round((_now() - start).total_seconds() / 60, 1),
                       "tried": out["done"] - before, "done": out["done"], "grid": out["grid"],
                       "candidates": len(out["candidates"]), "t_required": out["t_required"],
                       "best_t": max((r.get("t") or 0 for r in out["near"] if r["fail"] == "tvalue"), default=None)}
                self._log(row)
                self._update_watch(out, today)
                runs += 1
                if out.get("note"):  # every combination tried
                    self._finish(out)
                    return "finished"
                if out.get("stopped"):
                    return "stopped"
                if max_runs and runs >= max_runs:
                    return "max_runs"

    def _start_date(self, today=None) -> str:
        """Paper trading of a rule picked now starts on the next calendar day: its first signal is
        computed after that day's close, so nothing it is judged on existed when it was picked."""
        d = (today or _now().astimezone(KST).date())
        return str(d + timedelta(days=1))

    def _update_watch(self, out: dict, today=None) -> None:
        from quantlab.services.autosearch import key

        watch = self.watch(all_=True)
        have = {key(w["params"]) for w in watch}
        added = []

        def add(r: dict, kind: str) -> None:
            n = 1 + max((int(w["id"][1:]) for w in watch), default=0)
            w = {"id": f"W{n:02d}", "kind": kind, "params": r["params"], "rule": r["rule"],
                 "t": r.get("t"), "cagr": r.get("cagr"), "sharpe": r.get("sharpe"), "mdd": r.get("mdd"),
                 "n": r.get("n"), "added": _kst(), "start": self._start_date(today)}
            watch.append(w)
            have.add(key(r["params"]))
            added.append(w)

        for r in out["candidates"]:
            if r.get("status") == "후보" and key(r["params"]) not in have:
                add(r, "candidate")
        # the NEAR_IN_WATCH strongest rules that failed only the t-value so far; a weaker one is replaced
        # (it stays in the file, marked dropped, and leaves paper trading)
        top = sorted((r for r in out["near"] if r["fail"] == "tvalue"), key=lambda r: -(r.get("t") or 0))[:NEAR_IN_WATCH]
        keep = {key(r["params"]) for r in top}
        dropped = False
        for w in watch:
            if w["kind"] == "near" and not w.get("dropped") and key(w["params"]) not in keep:
                w["dropped"] = _kst()
                dropped = True
        active = {key(w["params"]) for w in watch if not w.get("dropped")}
        for r in top:
            if key(r["params"]) not in active:
                if key(r["params"]) in have:  # dropped earlier, strong again: back in, from the next day
                    w = next(w for w in watch if key(w["params"]) == key(r["params"]))
                    w.pop("dropped")
                    w.update(t=r.get("t"), start=self._start_date(today), added=_kst())
                    dropped = True
                else:
                    add(r, "near")
        if added or dropped:
            self._write("watch.json", watch)
        for w in added:
            if w["kind"] == "candidate":
                self._news({"kind": "candidate", "id": w["id"], "rule": w["rule"], "t": w["t"], "cagr": w["cagr"],
                            "mdd": w["mdd"], "start": w["start"]})

    def _finish(self, out: dict) -> None:
        strategy_hash, _ = self.lab.auto_search().loaded(GROUP)
        log = self.log()
        report = {"kind": "finished", "grid": out["grid"], "done": out["done"], "family_trials": out["family_trials"],
                  "t_required": out["t_required"], "candidates": len(out["candidates"]),
                  "funnel": [(f["label"], f["count"]) for f in out["funnel"]],
                  "best_near": [{"rule": r["rule"], "t": r.get("t"), "cagr": r.get("cagr"), "mdd": r.get("mdd")}
                                for r in sorted((r for r in out["near"] if r["fail"] == "tvalue"),
                                                key=lambda r: -(r.get("t") or 0))[:5]],
                  "watch": len(self.watch()), "began": log[0]["start"] if log else None,
                  "hours": round(sum(x["minutes"] for x in log) / 60, 1)}
        self._write("finished.json", {"strategy_hash": strategy_hash, "at": _kst(), "watch_rev": WATCH_REV})
        self._news(report)

    # --- the page ----------------------------------------------------------------------
    def progress(self) -> dict:
        log = self.log()
        last = log[-1] if log else None
        return {"enabled": self.enabled(), "running": self.running(), "finished": self.finished(),
                "done": last["done"] if last else 0, "grid": last["grid"] if last else None,
                "candidates": last["candidates"] if last else 0, "last": last,
                "watch": len(self.watch()), "error": self.last_error(), "battery": on_battery()}


def report_text(n: dict) -> str:
    """The message box the app shows for one news item."""
    pct = lambda x: "—" if x is None else f"{x * 100:.1f}%"  # noqa: E731
    if n["kind"] == "candidate":
        return (f"자동 연구 공장에서 모든 기준을 통과한 전략이 나왔어요 ({n['at']}).\n\n규칙: {n['rule']}\n"
                f"t값 {n['t']:.2f} · 연수익 {pct(n['cagr'])} · MDD {pct(n['mdd'])} (2010~2019년)\n\n"
                f"{n['start']}부터 모의 투자 화면에서 공장 {n['id']}로 매일 따라가요. "
                "2020년 이후 보류 구간은 이미 썼기 때문에, 진짜 판정은 앞으로의 모의 투자 결과예요.")
    lines = [f"자동 연구 공장이 목록을 모두 시험했어요 ({n['at']}).", "",
             f"시험한 조합 {n['done']:,}개 (공장 가동 {n['hours']}시간, {n['began']} 시작)",
             f"통과한 전략 {n['candidates']}개 · 필요한 t값 {n['t_required']:.2f}", "", "탈락 이유:"]
    lines += [f"  {lbl} {cnt:,}" for lbl, cnt in n["funnel"] if cnt]
    if n["best_near"]:
        lines += ["", "t값만 모자랐던 상위 규칙:"]
        lines += [f"  {r['rule']} · t {r['t']:.2f} · 연수익 {pct(r['cagr'])}" for r in n["best_near"]]
    lines += ["", f"모의 투자로 따라가는 규칙 {n['watch']}개.",
              "공장은 새 재료가 들어올 때까지 쉬어요. 이 보고서를 Claude에게 보여 주면 다음 재료를 설계해요."]
    return "\n".join(lines)
