"""Validated dataset schemas and modality metadata."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Literal

Modality = Literal["tabular", "ecg", "imaging", "omics", "timeseries"]
Task = Literal["classification", "regression", "multilabel"]

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    modality: Modality
    target: str
    id_column: str = "sample_id"
    group_column: str | None = "group_id"
    time_column: str | None = None
    study_column: str | None = "study_id"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip(): raise ValueError("name must be non-empty")
        if not self.target.strip(): raise ValueError("target must be non-empty")
        if not self.id_column.strip(): raise ValueError("id_column must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class FeatureManifest:
    feature_names: tuple[str, ...]
    numeric_features: tuple[str, ...] = ()
    categorical_features: tuple[str, ...] = ()
    excluded_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        overlap = set(self.numeric_features) & set(self.categorical_features)
        if overlap: raise ValueError(f"features cannot be both numeric and categorical: {sorted(overlap)}")
        if not set(self.numeric_features) | set(self.categorical_features) <= set(self.feature_names):
            raise ValueError("typed features must be present in feature_names")

    def to_dict(self) -> dict[str, Any]: return asdict(self)
