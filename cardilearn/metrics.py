"""Metrics emitted by CardiLearn training runs."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def classification_metrics(y_true, y_pred, y_score=None) -> dict[str, float | None]:
    """Return robust classification metrics; AUROC is included when computable."""

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_score is not None:
        try:
            score = np.asarray(y_score)
            if score.ndim == 2 and score.shape[1] == 2:
                score = score[:, 1]
            result["auroc"] = _finite_or_none(roc_auc_score(y_true, score))
        except ValueError:
            result["auroc"] = None
    return result


def regression_metrics(y_true, y_pred) -> dict[str, float | None]:
    """Return standard regression metrics."""

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": _finite_or_none(r2_score(y_true, y_pred)),
    }


def evaluate(estimator, X, y, task: str) -> dict[str, float | None]:
    """Evaluate a fitted estimator without mutating it."""

    predictions = estimator.predict(X)
    if task == "classification":
        scores: Any = estimator.predict_proba(X) if hasattr(estimator, "predict_proba") else None
        return classification_metrics(y, predictions, scores)
    if task == "regression":
        return regression_metrics(y, predictions)
    raise ValueError(f"unsupported task: {task}")
