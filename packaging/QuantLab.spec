# PyInstaller spec for the QUANT LAB desktop app (folder build).
# Build:  uv run pyinstaller packaging/QuantLab.spec --noconfirm
# Check:  dist/QuantLab/QuantLab(.exe) --selftest   (exit code 0 = pass)
# The quantlab .py sources ship as data so app_code_hash in a frozen build equals the
# hash of the same sources in a checkout (03 v2.0.0 §7.2).
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

datas = collect_data_files("quantlab", include_py_files=True) + copy_metadata("quantlab")
binaries = []
hiddenimports = collect_submodules("quantlab") + collect_submodules("keyring.backends")
for pkg in ("exchange_calendars",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += ["matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_agg"]

a = Analysis(
    [os.path.join(SPECPATH, "quantlab_app.py")],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PyQt5", "PyQt6", "IPython",
              "pytest", "hypothesis"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="QuantLab",
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="QuantLab", upx=False)
