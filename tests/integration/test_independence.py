"""03 v2.0.0 §1.2 R1–R3 and §8.1 independence tests."""

import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
AI_CLIENTS = re.compile(r"\b(anthropic|openai|google[-_.]generativeai|google[-_.]genai|cohere|mistralai|"
                        r"langchain|llama[-_]index|ollama)\b", re.I)


def test_no_ai_client_in_dependencies_or_source():
    assert not AI_CLIENTS.search((ROOT / "pyproject.toml").read_text())
    lock = (ROOT / "uv.lock").read_text()
    names = re.findall(r'^name = "([^"]+)"', lock, re.M)
    assert not [n for n in names if AI_CLIENTS.search(n)], names
    for p in (ROOT / "src").rglob("*.py"):
        assert not re.search(r"^\s*(import|from)\s+(anthropic|openai)", p.read_text(), re.M), p


def test_full_flow_runs_with_network_blocked(monkeypatch, tmp_path):
    def refuse(*a, **k):
        raise OSError("network disabled for this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    from quantlab.services.selftest import run_selftest

    res = run_selftest(tmp_path, log=lambda *_: None)
    assert res["ok"] and res["verdict"] in {"PASS", "HOLD", "FAIL", "INVALID"}


def test_fetch_contacts_only_the_configured_provider(tmp_path):
    from quantlab.data import providers

    seen = []

    class FakeResp:
        status_code = 200
        content = b"[]"
        text = "[]"

        def json(self):
            return []

    class FakeSession:
        def get(self, url, **kw):
            seen.append(url)
            assert "Token k" in kw["headers"]["Authorization"]
            return FakeResp()

    providers.fetch_tiingo("SPY", "2010-01-01", api_key="k", session=FakeSession())
    assert seen and all(u.startswith("https://api.tiingo.com/") for u in seen)


def test_engine_modules_do_not_import_gui():
    code = ("import sys, quantlab.services.lab, quantlab.services.selftest, quantlab.cli; "
            "bad=[m for m in sys.modules if m.split('.')[0] in ('PySide6','shiboken6','matplotlib.backends.backend_qtagg')]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
