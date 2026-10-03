"""Optional neural baselines with a stable fit/transform/predict interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class NeuralConfig:
    hidden_layers: tuple[int, ...] = (128, 64)
    max_iter: int = 300
    learning_rate_init: float = 1e-3
    early_stopping: bool = True
    random_state: int = 42


def build_neural_model(task: str, config: NeuralConfig | None = None) -> Pipeline:
    cfg = config or NeuralConfig()
    if cfg.max_iter < 1:
        raise ValueError("max_iter must be positive")
    common: dict[str, Any] = {
        "hidden_layer_sizes": cfg.hidden_layers,
        "max_iter": cfg.max_iter,
        "learning_rate_init": cfg.learning_rate_init,
        "early_stopping": cfg.early_stopping,
        "random_state": cfg.random_state,
    }
    if task == "classification":
        estimator = MLPClassifier(**common)
    elif task == "regression":
        estimator = MLPRegressor(**common)
    else:
        raise ValueError("task must be 'classification' or 'regression'")
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])
