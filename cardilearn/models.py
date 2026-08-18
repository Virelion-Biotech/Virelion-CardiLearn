"""Baseline model registry."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor


def build_model(task: str, name: str, features) -> Pipeline:
    """Build a reproducible preprocessing + estimator pipeline."""

    preprocessor = build_preprocessor(features)
    if task == "classification":
        estimators: dict[str, Any] = {
            "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
            "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
        }
    elif task == "regression":
        estimators = {
            "ridge": Ridge(alpha=1.0),
            "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
        }
    else:
        raise ValueError(f"unsupported task: {task}")

    if name not in estimators:
        raise KeyError(f"unknown {task} model: {name}; available={sorted(estimators)}")
    return Pipeline([("preprocess", preprocessor), ("model", estimators[name])])


def available_models(task: str) -> tuple[str, ...]:
    if task == "classification":
        return ("logistic_regression", "hist_gradient_boosting")
    if task == "regression":
        return ("ridge", "hist_gradient_boosting")
    raise ValueError(f"unsupported task: {task}")
