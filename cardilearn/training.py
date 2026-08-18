"""End-to-end training orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any

import pandas as pd

from .config import TrainingConfig
from .data import Dataset
from .metrics import evaluate
from .models import build_model
from .splitting import SplitIndices, split_frame


@dataclass
class TrainingResult:
    """Fitted model plus split assignments and metrics."""

    model: Any
    splits: SplitIndices
    metrics: dict[str, dict[str, float]]
    feature_columns: list[str]
    target_column: str
    group_column: str | None
    elapsed_seconds: float


def train(dataset: Dataset, config: TrainingConfig) -> TrainingResult:
    """Train one model with leakage-safe splitting and train-only preprocessing."""

    start = time()
    splits = split_frame(
        dataset.frame,
        config.split,
        target=dataset.target,
        groups=dataset.groups,
    )
    X = dataset.features()
    y = dataset.target

    model = build_model(config.task, config.model, X.iloc[splits.train])
    model.fit(X.iloc[splits.train], y.iloc[splits.train])

    metrics = {
        "train": evaluate(model, X.iloc[splits.train], y.iloc[splits.train], config.task),
        "validation": evaluate(
            model, X.iloc[splits.validation], y.iloc[splits.validation], config.task
        ),
        "test": evaluate(model, X.iloc[splits.test], y.iloc[splits.test], config.task),
    }

    return TrainingResult(
        model=model,
        splits=splits,
        metrics=metrics,
        feature_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        group_column=dataset.group_column,
        elapsed_seconds=time() - start,
    )
