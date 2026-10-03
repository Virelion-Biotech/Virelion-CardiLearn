"""Content fingerprints and reproducibility metadata."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any


def dataframe_fingerprint(df) -> str:
    """Return a stable hash of dataframe values, columns, and dtypes."""
    payload = df.copy()
    payload = payload.reindex(sorted(payload.columns), axis=1)
    raw = payload.to_json(orient="split", date_format="iso", default_handler=str)
    dtype = json.dumps({c: str(payload[c].dtype) for c in payload.columns}, sort_keys=True)
    return hashlib.sha256((raw + dtype).encode("utf-8")).hexdigest()


def runtime_metadata() -> dict[str, Any]:
    try:
        cardilearn_version = version("virelion-cardilearn")
    except PackageNotFoundError:
        cardilearn_version = "source"
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cardilearn_version": cardilearn_version,
    }
