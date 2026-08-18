"""Experiment configuration objects.

The configuration is intentionally small and serializable so that a CardiLearn run can be
recreated from a single manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SplitConfig:
    """Leakage-safe dataset split configuration."""

    test_size: float = 0.20
    validation_size: float = 0.20
    random_state: int = 42
    stratify: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")
        if self.test_size + self.validation_size >= 1:
            raise ValueError("test_size + validation_size must be < 1")


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration shared by the first CardiLearn tabular trainers."""

    task: str = "classification"
    model: str = "logistic_regression"
    target_column: str = "target"
    group_column: str | None = "group_id"
    random_state: int = 42
    split: SplitConfig = field(default_factory=SplitConfig)

    def __post_init__(self) -> None:
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        if not self.model:
            raise ValueError("model must be non-empty")
        if not self.target_column:
            raise ValueError("target_column must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
