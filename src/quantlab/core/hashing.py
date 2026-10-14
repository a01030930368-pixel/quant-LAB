"""Stable hashing for configs, datasets and results.

Every identifier that the registry stores (dataset_version, configuration_version,
validation_policy_version, result hash) comes from these functions, so the same
input must always produce the same hash across runs and machines.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f):
            return "NaN"
        if math.isinf(f):
            return "Inf" if f > 0 else "-Inf"
        # repr round-trips exactly; avoids platform-dependent formatting
        return repr(f)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(_normalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_hash(df: pd.DataFrame) -> str:
    """Hash of a DataFrame's values, index and columns (order-sensitive)."""
    h = hashlib.sha256()
    h.update(canonical_json([str(c) for c in df.columns]).encode())
    h.update(canonical_json([str(i) for i in df.index]).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()
