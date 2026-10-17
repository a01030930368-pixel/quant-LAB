"""Windows Task Scheduler entries: `QuantLab.exe --paper-refresh` (D-031) and `--factory` (D-033).

Twice a day, 08:30 and 21:00 on the PC's clock (Korea: after the US close, and before the next US
open in case the stock bundle was late). A run missed while the PC was off starts when it is next on
(StartWhenAvailable). It runs only while the user is signed in, as the user, so the Tiingo key in the
Windows credential store and the work folder chosen in the app are the same ones.

The research factory starts when the user signs in and is tried again every hour (a second copy is
not started while one runs), with no time limit and a low priority; it ends by itself when the
switch is off or the whole grid is tried.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

TASK = "QUANT LAB 모의 투자 자동 갱신"
TIMES = ("08:30", "21:00")
FACTORY_TASK = "QUANT LAB 자동 연구 공장"


def supported() -> bool:
    """Only the installed Windows app can be scheduled (a source checkout has no QuantLab.exe)."""
    return os.name == "nt" and bool(getattr(sys, "frozen", False))


def task_xml(exe: str, times=TIMES, start: date | None = None) -> str:
    start = start or date.today()
    triggers = "".join(
        f"<CalendarTrigger><StartBoundary>{start.isoformat()}T{t}:00</StartBoundary><Enabled>true</Enabled>"
        "<ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay></CalendarTrigger>" for t in times)
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>QUANT LAB: 모의 투자 전략을 오늘 자료로 갱신합니다 (실제 주문 없음)</Description></RegistrationInfo>
  <Triggers>{triggers}</Triggers>
  <Principals><Principal id="Author"><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <ExecutionTimeLimit>PT2H</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author"><Exec><Command>{escape(exe)}</Command><Arguments>--paper-refresh</Arguments>
    <WorkingDirectory>{escape(str(Path(exe).parent))}</WorkingDirectory></Exec></Actions>
</Task>
"""


def factory_xml(exe: str, start: date | None = None) -> str:
    start = start or date.today()
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>QUANT LAB: 자동 연구 공장 (S&amp;P 500 종목 규칙을 2010~2019년 자료로 계속 시험, 실제 주문 없음)</Description></RegistrationInfo>
  <Triggers>
    <LogonTrigger><Enabled>true</Enabled><Delay>PT5M</Delay></LogonTrigger>
    <CalendarTrigger><StartBoundary>{start.isoformat()}T00:00:00</StartBoundary><Enabled>true</Enabled>
      <Repetition><Interval>PT1H</Interval><Duration>P1D</Duration></Repetition>
      <ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay></CalendarTrigger>
  </Triggers>
  <Principals><Principal id="Author"><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>8</Priority>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author"><Exec><Command>{escape(exe)}</Command><Arguments>--factory</Arguments>
    <WorkingDirectory>{escape(str(Path(exe).parent))}</WorkingDirectory></Exec></Actions>
</Task>
"""


def _schtasks(*args: str) -> subprocess.CompletedProcess:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(["schtasks", *args], capture_output=True, text=True, creationflags=flags)


def install(exe: str | None = None) -> None:
    exe = exe or sys.executable
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "task.xml"
        p.write_text(task_xml(exe), encoding="utf-16")
        r = _schtasks("/Create", "/TN", TASK, "/XML", str(p), "/F")
    if r.returncode != 0:
        raise RuntimeError(f"자동 갱신 예약을 만들지 못했습니다: {(r.stderr or r.stdout).strip()}")


def remove() -> None:
    r = _schtasks("/Delete", "/TN", TASK, "/F")
    if r.returncode != 0 and installed():
        raise RuntimeError(f"자동 갱신 예약을 지우지 못했습니다: {(r.stderr or r.stdout).strip()}")


def installed() -> bool:
    return _schtasks("/Query", "/TN", TASK).returncode == 0


def install_factory(exe: str | None = None, run_now: bool = True) -> None:
    exe = exe or sys.executable
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "task.xml"
        p.write_text(factory_xml(exe), encoding="utf-16")
        r = _schtasks("/Create", "/TN", FACTORY_TASK, "/XML", str(p), "/F")
    if r.returncode != 0:
        raise RuntimeError(f"자동 연구 공장 예약을 만들지 못했습니다: {(r.stderr or r.stdout).strip()}")
    if run_now:
        _schtasks("/Run", "/TN", FACTORY_TASK)


def remove_factory() -> None:
    r = _schtasks("/Delete", "/TN", FACTORY_TASK, "/F")
    if r.returncode != 0 and factory_installed():
        raise RuntimeError(f"자동 연구 공장 예약을 지우지 못했습니다: {(r.stderr or r.stdout).strip()}")


def factory_installed() -> bool:
    return _schtasks("/Query", "/TN", FACTORY_TASK).returncode == 0
