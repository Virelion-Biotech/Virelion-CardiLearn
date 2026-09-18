"""Immutable filesystem registry for CardiLearn experiment artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
from typing import Any, Mapping


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"created_at", "registered_at"}
    }


class ModelRegistry:
    """Filesystem registry that refuses silent manifest replacement."""

    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run_id(self, prefix: str = "run") -> str:
        if not prefix.strip():
            raise ValueError("prefix must be non-empty")
        return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"

    def save_manifest(self, run_id: str, manifest: Mapping[str, Any]) -> Path:
        if not run_id.strip():
            raise ValueError("run_id must be non-empty")
        run = self.root / run_id
        run.mkdir(parents=True, exist_ok=True)
        path = run / "manifest.json"
        payload = dict(manifest)
        payload.setdefault("run_id", run_id)
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if stable_manifest_payload(existing) != stable_manifest_payload(payload):
                raise FileExistsError(f"immutable run manifest already exists: {path}")
            return path
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def register_artifact(self, run_id: str, path: str | Path, *, name: str | None = None) -> dict[str, Any]:
        artifact = Path(path)
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        manifest_path = self.root / run_id / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"run manifest does not exist: {manifest_path}")
        record = {
            "name": name or artifact.name,
            "path": str(artifact),
            "size_bytes": artifact.stat().st_size,
            "sha256": file_sha256(artifact),
        }
        return record

    def load_manifest(self, run_id: str) -> dict[str, Any]:
        return json.loads((self.root / run_id / "manifest.json").read_text(encoding="utf-8"))

    def list_runs(self) -> list[str]:
        return sorted(
            path.name
            for path in self.root.iterdir()
            if path.is_dir() and (path / "manifest.json").exists()
        )
