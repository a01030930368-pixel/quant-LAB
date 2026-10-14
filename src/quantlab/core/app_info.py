"""Identity of the running application build (03 v2.0.0 §7.2: app_version + app_code_hash).

`app_code_hash` hashes every .py file of the installed `quantlab` package, so a
recorded experiment can be tied to the exact engine code that produced it, in a
source checkout and in a frozen PyInstaller build alike.
"""

from __future__ import annotations

import functools
import hashlib
from importlib import metadata
from pathlib import Path

import quantlab

PACKAGE_ROOT = Path(quantlab.__file__).resolve().parent


def app_version() -> str:
    try:
        return metadata.version("quantlab")
    except metadata.PackageNotFoundError:
        return getattr(quantlab, "__version__", "0+unknown")


def build_label() -> str:
    """The build tag written next to QuantLab.exe by CI ('build-12'); source checkouts say so."""
    import sys

    if not getattr(sys, "frozen", False):
        return "개발 버전"
    info = Path(sys.executable).parent / "build-info.txt"
    try:
        return info.read_text(encoding="utf-8-sig").splitlines()[0].strip()
    except (OSError, IndexError):
        return "버전 정보 없음"


@functools.lru_cache(maxsize=1)
def app_code_hash() -> str:
    h = hashlib.sha256()
    files = sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)
    if not files:  # frozen build without sources: fall back to compiled modules
        files = sorted(PACKAGE_ROOT.rglob("*.pyc"))
    for p in files:
        h.update(p.relative_to(PACKAGE_ROOT).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest()
