"""Benchmark protocol for comparing models without touching the held-out test set during selection."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import json

@dataclass(frozen=True)
class BenchmarkResult:
    dataset: str
    task: str
    model: str
    validation: dict[str, float]
    test: dict[str, float] | None = None
    notes: str = ""
    def to_dict(self) -> dict[str, Any]: return asdict(self)

class BenchmarkTable:
    def __init__(self) -> None: self.results: list[BenchmarkResult] = []
    def add(self, result: BenchmarkResult) -> None: self.results.append(result)
    def best(self, metric: str, *, split: str = "validation") -> BenchmarkResult:
        if not self.results: raise ValueError("benchmark is empty")
        eligible = [r for r in self.results if metric in getattr(r, split)]
        return max(eligible, key=lambda r: getattr(r, split)[metric])
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f: json.dump([r.to_dict() for r in self.results], f, indent=2)
