"""End-to-end training orchestration with an explicit held-out test boundary."""
from __future__ import annotations
from dataclasses import dataclass
from time import time
from typing import Any
from .config import TrainingConfig
from .data import Dataset
from .metrics import evaluate
from .models import build_model
from .splitting import SplitIndices, split_frame

@dataclass
class TrainingResult:
    model: Any
    splits: SplitIndices
    metrics: dict[str, dict[str, float]]
    feature_columns: list[str]
    target_column: str
    group_column: str | None
    elapsed_seconds: float

def train(dataset: Dataset, config: TrainingConfig) -> TrainingResult:
    """Fit using train only; emit train/validation metrics and keep test untouched."""
    start = time()
    splits = split_frame(dataset.frame, config.split, target=dataset.target, groups=dataset.groups)
    X, y = dataset.features(), dataset.target
    model = build_model(config.task, config.model, X.iloc[splits.train])
    model.fit(X.iloc[splits.train], y.iloc[splits.train])
    metrics = {
        "train": evaluate(model, X.iloc[splits.train], y.iloc[splits.train], config.task),
        "validation": evaluate(model, X.iloc[splits.validation], y.iloc[splits.validation], config.task),
    }
    return TrainingResult(model, splits, metrics, dataset.feature_columns, dataset.target_column, dataset.group_column, time() - start)

def evaluate_held_out_test(result: TrainingResult, dataset: Dataset, task: str) -> dict[str, float]:
    """Evaluate a frozen result on the held-out test set exactly once, on demand."""
    X, y = dataset.features(), dataset.target
    return evaluate(result.model, X.iloc[result.splits.test], y.iloc[result.splits.test], task)
