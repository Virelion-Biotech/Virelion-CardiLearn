"""Model registry for deterministic and neural cardiac ML baselines."""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from .neural import build_neural_model
from .preprocessing import build_preprocessor


def build_model(task: str, name: str, features) -> Any:
    if name in {"cardilearn", "cardilearn_research"}:
        if task != "representation":
            raise ValueError("CardiLearn research backbones expose representations; use a task head for supervised evaluation")
        try:
            from .research_model import CardiLearnResearch
        except ImportError as exc:
            raise ImportError("install the 'torch' extra to use the CardiLearn research model") from exc
        if not hasattr(features, "shape"):
            raise TypeError("features must expose a shape")
        n_genes = int(features.shape[1])
        return CardiLearnResearch(
            n_genes=n_genes,
            n_species=1,
            n_assays=1,
            n_cell_types=2,
        )

    if name == "mlp":
        if task not in {"classification", "regression"}:
            raise ValueError(f"unsupported task: {task}")
        if any(features[c].dtype == "object" for c in features.columns):
            raise ValueError("mlp requires numeric features; encode categorical variables first")
        return build_neural_model(task)

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
        raise KeyError(f"unknown {task} model: {name}; available={sorted(estimators) + ['mlp']}")
    return Pipeline([("preprocess", preprocessor), ("model", estimators[name])])


def available_models(task: str) -> tuple[str, ...]:
    if task == "classification":
        return (
            "logistic_regression",
            "hist_gradient_boosting",
            "mlp",
        )
    if task == "regression":
        return ("ridge", "hist_gradient_boosting", "mlp")
    if task == "representation":
        return ("pca", "autoencoder", "scvi", "geneformer", "scgpt", "uce", "nicheformer", "scimilarity", "cardilearn_research")
    raise ValueError(f"unsupported task: {task}")
