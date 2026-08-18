"""Model-card representation for communicating intended use and limitations."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json

@dataclass(frozen=True)
class ModelCard:
    name: str
    version: str
    task: str
    training_data: str
    intended_use: str
    out_of_scope_use: str
    evaluation_protocol: str
    validation_metrics: dict[str, float]
    limitations: tuple[str, ...] = ()
    ethical_notes: tuple[str, ...] = ()
    def to_dict(self): return asdict(self)
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f: json.dump(self.to_dict(), f, indent=2)
