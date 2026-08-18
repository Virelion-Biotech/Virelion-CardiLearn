"""Benchmark and model-selection primitives shared across cardiac datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold

from .metrics import evaluate


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_size: int
    validation_size: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class CVResult:
    model_name: str
    task: str
    folds: tuple[FoldResult, ...]

    @property
    def aggregate(self) -> dict[str, float]:
        keys = sorted({k for f in self.folds for k in f.metrics})
        return {
            k: float(np.nanmean([f.metrics.get(k, np.nan) for f in self.folds]))
            for k in keys
        }

    @property
    def variability(self) -> dict[str, float]:
        keys = sorted({k for f in self.folds for k in f.metrics})
        return {
            f"{k}_std": float(np.nanstd([f.metrics.get(k, np.nan) for f in self.folds], ddof=1))
            if len(self.folds) > 1 else 0.0
            for k in keys
        }


def cross_validate(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task: str,
    groups: pd.Series | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> CVResult:
    """Cross-validate without allowing subjects/studies to cross folds."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits)
        iterator = splitter.split(X, y, groups=groups)
    elif task == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        iterator = splitter.split(X, y)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        iterator = splitter.split(X)

    results: list[FoldResult] = []
    for fold, (train_idx, val_idx) in enumerate(iterator, start=1):
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        results.append(
            FoldResult(
                fold=fold,
                train_size=len(train_idx),
                validation_size=len(val_idx),
                metrics=evaluate(model, X.iloc[val_idx], y.iloc[val_idx], task),
            )
        )
    name = estimator.__class__.__name__
    return CVResult(name, task, tuple(results))
