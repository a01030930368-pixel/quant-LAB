"""Windows Task Scheduler entry that runs `QuantLab.exe --paper-refresh` (D-031).

Twice a day, 08:30 and 21:00 on the PC's clock (Korea: after the US close, and before the next US
open in case the stock bundle was late). A run missed while the PC was off starts when it is next on
(StartWhenAvailable). It runs only while the user is signed in, as the user, so the Tiingo key in the
Windows credential store and the work folder chosen in the app are the same ones.
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
