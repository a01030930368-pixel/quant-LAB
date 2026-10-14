"""Raw data preservation and dataset versions (03 v2.0.0 §7.3).

Raw downloads are stored exactly as received (bytes + sha256) next to their
normalized Parquet form. A dataset is a set of raw files, one per symbol; its
version is the hash of those raw hashes and the processing version, so the same
raw inputs always give the same dataset_version.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quantlab.core.hashing import file_sha256, stable_hash
from quantlab.core.market_data import MarketData
from quantlab.data.quality import QCReport, run_qc

PROCESSING_VERSION = 1


@dataclass
class RawRecord:
    symbol: str
    raw_sha256: str
    raw_path: Path
    frame_path: Path
    qc: QCReport
    source: dict


class DataStore:
    def __init__(self, root: str | Path, registry=None) -> None:
        self.root = Path(root)
        self.raw_dir = self.root / "data" / "raw"
        self.proc_dir = self.root / "data" / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.proc_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry

    # --- raw -------------------------------------------------------------------
    def _latest_frame(self, symbol: str) -> pd.DataFrame | None:
        metas = self.raw_records(symbol)
        return pd.read_parquet(metas[-1]["frame_path"]) if metas else None

    def ingest(self, symbol: str, frame: pd.DataFrame, raw: bytes | Path, source: dict,
               sessions: pd.DatetimeIndex | None = None) -> RawRecord:
        symbol = symbol.upper()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        d = self.raw_dir / symbol
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{ts}_{source.get('provider', 'unknown')}"
        if isinstance(raw, Path):
            raw_path = d / f"{stem}{raw.suffix or '.bin'}"
            shutil.copy2(raw, raw_path)
        else:
            raw_path = d / f"{stem}.json"
            raw_path.write_bytes(raw)
        sha = file_sha256(raw_path)
        qc = run_qc(symbol, frame, sessions=sessions, previous=self._latest_frame(symbol))
        frame_path = d / f"{stem}.parquet"
        frame.to_parquet(frame_path)
        meta = {"symbol": symbol, "raw_sha256": sha, "raw_path": str(raw_path), "frame_path": str(frame_path),
                "source": source, "fetched_at": ts, "rows": int(len(frame)),
                "first": str(frame.index.min().date()) if len(frame) else None,
                "last": str(frame.index.max().date()) if len(frame) else None, "qc": qc.to_dict()}
        (d / f"{stem}.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        return RawRecord(symbol, sha, raw_path, frame_path, qc, source)

    def raw_records(self, symbol: str | None = None) -> list[dict]:
        pattern = f"{symbol.upper()}/*.meta.json" if symbol else "*/*.meta.json"
        metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.raw_dir.glob(pattern))]
        return sorted(metas, key=lambda m: m["fetched_at"])

    # --- datasets -----------------------------------------------------------------
    def build_dataset(self, symbols: list[str]) -> tuple[str, dict]:
        """Dataset from the latest raw record of each symbol."""
        chosen = {}
        for s in symbols:
            recs = self.raw_records(s)
            if not recs:
                raise FileNotFoundError(f"no data for {s}; fetch or import it first")
            chosen[s.upper()] = recs[-1]
        version = stable_hash({"processing": PROCESSING_VERSION,
                               "raw": {s: m["raw_sha256"] for s, m in sorted(chosen.items())}})
        out = self.proc_dir / version[:16]
        out.mkdir(parents=True, exist_ok=True)
        fatal, warnings = [], []
        for s, m in chosen.items():
            shutil.copy2(m["frame_path"], out / f"{s}.parquet")
            for c in m["qc"]["checks"]:
                msg = f"{s}: {c['name']}: {c['detail']}"
                if c["status"] == "fatal":
                    fatal.append(msg)
                elif c["status"] == "warn":
                    warnings.append(msg)
        manifest = {"dataset_version": version, "processing_version": PROCESSING_VERSION,
                    "symbols": {s: {k: m[k] for k in ("raw_sha256", "source", "fetched_at", "first", "last", "rows")}
                                for s, m in chosen.items()},
                    "qc_fatal": fatal, "qc_warnings": warnings}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
        if self.registry is not None:
            self.registry.register_dataset(version, manifest)
        return version, manifest

    def datasets(self) -> list[dict]:
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.proc_dir.glob("*/manifest.json"))]

    def load_dataset(self, version: str) -> tuple[MarketData, dict]:
        d = self.proc_dir / version[:16]
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if manifest["dataset_version"] != version and not manifest["dataset_version"].startswith(version):
            raise KeyError(version)
        frames = {}
        for s, meta in manifest["symbols"].items():
            if file_sha256(meta_path := Path(self._raw_path_for(s, meta))) != meta["raw_sha256"]:
                raise ValueError(f"raw file for {s} changed on disk: {meta_path}")
            f = pd.read_parquet(d / f"{s}.parquet")
            f = f[~f.index.duplicated()]
            frames[s] = f[["open", "high", "low", "close", "dividend", "split"]]
        return MarketData.from_frames(frames), manifest

    def _raw_path_for(self, symbol: str, meta: dict) -> str:
        for m in self.raw_records(symbol):
            if m["raw_sha256"] == meta["raw_sha256"]:
                return m["raw_path"]
        raise FileNotFoundError(f"raw file for {symbol} ({meta['raw_sha256'][:12]}) is missing")
