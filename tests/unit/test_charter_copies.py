"""docs/charter holds copies of the governing documents; the project's charter/
folder is the original. Fail loudly if a copy drifts from its recorded hash."""

from pathlib import Path

from quantlab.core.hashing import file_sha256

DOCS = Path(__file__).resolve().parents[2] / "docs" / "charter"


def test_charter_copies_match_recorded_hashes():
    lines = (DOCS / "SHA256SUMS").read_text().split("\n")
    entries = [l.split() for l in lines if l.strip()]
    assert len(entries) >= 4
    for digest, name in entries:
        assert file_sha256(DOCS / name) == digest, f"{name} differs from its recorded hash"
