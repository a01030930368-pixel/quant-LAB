"""Desktop entry point (`quantlab-app`, and the packaged QuantLab.exe).

    QuantLab.exe              open the app (asks for a work folder on first run)
    QuantLab.exe --selftest   run the synthetic end-to-end flow without a window; exit code 0 = pass
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path


def _selftest() -> int:
    from quantlab.services.selftest import run_selftest

    try:
        res = run_selftest(log=lambda s: print(s, flush=True))
    except Exception as e:  # noqa: BLE001 - CI reads the exit code
        print(f"SELFTEST FAILED: {type(e).__name__}: {e}", flush=True)
        return 1
    print("SELFTEST OK" if res.get("ok") else "SELFTEST FAILED", flush=True)
    return 0 if res.get("ok") else 1


def choose_workspace(settings, parent=None) -> Path | None:
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from quantlab.core.workspace import default_root

    saved = settings.value("workspace")
    if saved and Path(saved).exists():
        return Path(saved)
    QMessageBox.information(parent, "QUANT LAB",
                            "연구 자료와 기록을 둘 작업 폴더를 고르세요. 백업은 이 폴더를 복사하면 됩니다.\n"
                            f"기본값: {default_root()}")
    default_root().mkdir(parents=True, exist_ok=True)
    path = QFileDialog.getExistingDirectory(parent, "작업 폴더 선택", str(default_root()))
    if not path:
        return None
    settings.setValue("workspace", path)
    return Path(path)


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()  # parameter search uses spawned worker processes
    argv = list(sys.argv if argv is None else argv)
    if "--selftest" in argv:
        return _selftest()
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from quantlab.gui.main_window import MainWindow
    from quantlab.services.lab import Lab

    app = QApplication(argv)
    app.setApplicationName("QUANT LAB")
    settings = QSettings("QuantLab", "QuantLab")
    ws = choose_workspace(settings)
    if ws is None:
        return 0
    win = MainWindow(Lab(ws))
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
