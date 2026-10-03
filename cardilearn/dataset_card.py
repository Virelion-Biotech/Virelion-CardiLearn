"""Machine-readable dataset cards for provenance and reuse decisions."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import json

@dataclass(frozen=True)
class DatasetCard:
    dataset_id: str
    source: str
    citation: str
    species: str
    tissue: str
    modality: str
    task: str
    n_samples: int
    study_groups: int
    collection_window: str = ""
    preprocessing: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    license: str = "unknown"

    def to_dict(self) -> dict[str, Any]: return asdict(self)
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f: json.dump(self.to_dict(), f, indent=2)
