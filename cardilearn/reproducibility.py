"""Reproducibility primitives for CardiLearn experiments.

The manifest records configuration, data/split fingerprints, seeds and code identity.
These utilities never fit preprocessing or models and cannot silently contaminate held-out evaluation.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_mapping(value: Mapping[str, Any]) -> str:
    """Return a stable SHA256 fingerprint for a JSON-compatible mapping."""
    return sha256_text(canonical_json(dict(value)))


def fingerprint_ids(ids: list[str] | tuple[str, ...]) -> str:
    """Fingerprint an ordered biological-unit ID manifest."""
    return sha256_text(canonical_json(list(ids)))


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Fingerprint dataframe values, index, column names and dtypes."""
    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    schema = canonical_json([(str(c), str(df[c].dtype)) for c in df.columns]).encode("utf-8")
    return hashlib.sha256(schema + payload).hexdigest()


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Backward-compatible alias for the canonical configuration fingerprint."""
    return fingerprint_mapping(config)


@dataclass(frozen=True)
class ReproducibilityManifest:
    """Complete provenance record required to reproduce an experiment."""

    schema_version: str
    project_version: str
    config_name: str
    config_fingerprint: str
    data_fingerprint: str
    split_fingerprint: str
    seeds: tuple[int, ...]
    primary_seed: int
    git_commit: str = ""
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    python_version: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)
    notes: str = ""

    def __post_init__(self) -> None:
        for name, value in (("config_fingerprint", self.config_fingerprint), ("data_fingerprint", self.data_fingerprint), ("split_fingerprint", self.split_fingerprint)):
            if value and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower())):
                raise ValueError(f"{name} must be a SHA256 hex digest")
        if not self.seeds:
            raise ValueError("at least one experiment seed is required")
        if self.primary_seed not in self.seeds:
            raise ValueError("primary_seed must be included in seeds")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["seeds"] = list(self.seeds)
        return payload

    def fingerprint(self) -> str:
        """Fingerprint scientific provenance, excluding machine/time metadata."""
        stable = {
            "schema_version": self.schema_version,
            "project_version": self.project_version,
            "config_name": self.config_name,
            "config_fingerprint": self.config_fingerprint,
            "data_fingerprint": self.data_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "seeds": list(self.seeds),
            "primary_seed": self.primary_seed,
            "git_commit": self.git_commit,
        }
        return fingerprint_mapping(stable)


def make_manifest(*, config: Mapping[str, Any], config_name: str, data_fingerprint: str = "", split_fingerprint: str = "", seeds: list[int] | tuple[int, ...] = (42,), primary_seed: int | None = None, git_commit: str = "", project_version: str = "0.3.0", notes: str = "") -> ReproducibilityManifest:
    """Construct a manifest from already-computed provenance inputs."""
    ordered_seeds = tuple(int(seed) for seed in seeds)
    if not ordered_seeds:
        raise ValueError("seeds must contain at least one seed")
    return ReproducibilityManifest(schema_version="1.0", project_version=project_version, config_name=config_name, config_fingerprint=fingerprint_mapping(config), data_fingerprint=data_fingerprint, split_fingerprint=split_fingerprint, seeds=ordered_seeds, primary_seed=ordered_seeds[0] if primary_seed is None else int(primary_seed), git_commit=git_commit, notes=notes)


def save_manifest(manifest: ReproducibilityManifest, path: str | Path) -> None:
    """Write a human-readable manifest JSON file."""
    Path(path).write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> ReproducibilityManifest:
    """Load and validate a manifest written by :func:`save_manifest`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["seeds"] = tuple(int(seed) for seed in payload["seeds"])
    return ReproducibilityManifest(**payload)
