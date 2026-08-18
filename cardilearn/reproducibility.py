"""Content-addressed fingerprints for datasets and experiment inputs."""
from __future__ import annotations
import hashlib
import json
from typing import Any
import pandas as pd

def dataframe_fingerprint(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    schema = json.dumps([(str(c), str(df[c].dtype)) for c in df.columns], sort_keys=True).encode()
    return hashlib.sha256(schema + payload).hexdigest()

def config_fingerprint(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
