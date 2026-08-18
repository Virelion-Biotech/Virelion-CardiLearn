"""Lightweight filesystem model registry with immutable run manifests."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

class ModelRegistry:
    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def create_run_id(self, prefix: str = "run") -> str:
        return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    def save_manifest(self, run_id: str, manifest: dict[str, Any]) -> Path:
        run = self.root / run_id; run.mkdir(parents=True, exist_ok=True)
        path = run / "manifest.json"
        payload = dict(manifest); payload.setdefault("run_id", run_id); payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load_manifest(self, run_id: str) -> dict[str, Any]:
        return json.loads((self.root / run_id / "manifest.json").read_text(encoding="utf-8"))

    def list_runs(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and (p / "manifest.json").exists())
