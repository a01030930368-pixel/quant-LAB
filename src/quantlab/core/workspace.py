"""User work folder (03 v2.0.0 §7.1). All research data and records live here,
never in the source repository. Backing up = copying this folder."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SUBDIRS = ("data/raw", "data/processed", "strategies", "policies", "artifacts", "reports", "exports")


def default_root() -> Path:
    # Windows: Documents\QuantLab ; elsewhere ~/QuantLab
    docs = Path.home() / "Documents"
    return (docs if os.name == "nt" or docs.exists() else Path.home()) / "QuantLab"


@dataclass(frozen=True)
class Workspace:
    root: Path

    @classmethod
    def open(cls, root: str | Path | None = None) -> "Workspace":
        ws = cls(Path(root) if root else default_root())
        for sub in SUBDIRS:
            (ws.root / sub).mkdir(parents=True, exist_ok=True)
        return ws

    @property
    def registry_path(self) -> Path:
        return self.root / "registry.sqlite"

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)
