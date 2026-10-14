"""Desktop entry point (`quantlab-app`, and the packaged QuantLab.exe).

    QuantLab.exe              open the app (asks for a work folder on first run)
    QuantLab.exe --selftest   run the synthetic end-to-end flow without a window; exit code 0 = pass
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path


def _selftest() -> int:
    """A windowed .exe has no console: the log goes to QUANTLAB_SELFTEST_LOG or the temp folder."""
    import os
    import tempfile

    from quantlab.services.selftest import run_selftest

    log_path = Path(os.environ.get("QUANTLAB_SELFTEST_LOG", Path(tempfile.gettempdir()) / "quantlab-selftest.log"))
    with open(log_path, "w", encoding="utf-8") as fh:
        def log(line: str) -> None:
            fh.write(line + "\n")
            fh.flush()
            if sys.stdout is not None:
                print(line, flush=True)

        try:
            res = run_selftest(log=log)
        except Exception as e:  # noqa: BLE001 - CI reads the exit code
            log(f"SELFTEST FAILED: {type(e).__name__}: {e}")
            return 1
        log("SELFTEST OK" if res.get("ok") else "SELFTEST FAILED")
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


def _show_uncaught_errors() -> None:
    """A windowed app has no console, so an error inside a button handler would vanish silently."""
    import traceback

    from PySide6.QtWidgets import QMessageBox

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb, limit=6))
        QMessageBox.warning(None, "QUANT LAB 오류", f"{exc_type.__name__}: {exc}\n\n{text}")

    sys.excepthook = hook


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
    from quantlab.gui import theme

    theme.apply(app)
    _show_uncaught_errors()
    settings = QSettings("QuantLab", "QuantLab")
    ws = choose_workspace(settings)
    if ws is None:
        return 0
    win = MainWindow(Lab(ws))
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
