"""Executable baseline registry for the Step 15 benchmark matrix."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    kind: str
    description: str
    representation: str


BASELINES = (
    BaselineSpec("pca_linear", "supervised", "PCA followed by a linear probe", "pca"),
    BaselineSpec("mlp", "supervised", "Plain multilayer perceptron on standardized genes", "identity"),
    BaselineSpec("autoencoder", "reconstruction", "Plain MLP autoencoder representation", "autoencoder"),
    BaselineSpec("cardilearn", "representation", "Structured CardiLearn representation", "cardilearn"),
)


def baseline_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in BASELINES)


def _check_task(task: str) -> None:
    if task not in {"classification", "regression"}:
        raise ValueError("baseline probes currently support classification or regression")


def build_supervised_baseline(
    name: str,
    *,
    task: str,
    n_components: int = 64,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    random_state: int = 42,
) -> Any:
    """Build an explicitly declared baseline for downstream probing."""
    _check_task(task)
    if name == "pca_linear":
        estimator = (
            LogisticRegression(max_iter=2000, random_state=random_state)
            if task == "classification"
            else Ridge(alpha=1.0)
        )
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=random_state)),
                ("probe", estimator),
            ]
        )
    if name == "mlp":
        if task == "classification":
            from sklearn.neural_network import MLPClassifier

            estimator = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes, max_iter=300, random_state=random_state
            )
        else:
            estimator = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes, max_iter=300, random_state=random_state
            )
        return Pipeline([("scale", StandardScaler()), ("probe", estimator)])
    raise KeyError(f"unsupported sklearn baseline: {name}")


def fit_autoencoder(
    x_train: np.ndarray,
    *,
    latent_dim: int = 64,
    hidden_layer_sizes: tuple[int, ...] = (128,),
    random_state: int = 42,
) -> tuple[Pipeline, Callable[[np.ndarray], np.ndarray]]:
    """Fit a plain MLP autoencoder and return its hidden latent transform.

    The estimator is trained only on ``x_train``. The returned encoder manually
    evaluates the fitted hidden layers, avoiding leakage from held-out matrices.
    """
    x_train = np.asarray(x_train, dtype=float)
    if x_train.ndim != 2:
        raise ValueError("x_train must be a 2-D matrix")
    if latent_dim < 1 or latent_dim >= x_train.shape[1]:
        raise ValueError("latent_dim must be positive and smaller than n_features")
    architecture = tuple(hidden_layer_sizes) + (latent_dim,)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "ae",
                MLPRegressor(
                    hidden_layer_sizes=architecture, max_iter=500, random_state=random_state
                ),
            ),
        ]
    )
    model.fit(x_train, x_train)

    def encode(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != x_train.shape[1]:
            raise ValueError("values must have shape [n_samples, n_features]")
        hidden = model.named_steps["scale"].transform(values)
        estimator = model.named_steps["ae"]
        for weights, bias in zip(estimator.coefs_[:-1], estimator.intercepts_[:-1]):
            hidden = hidden @ weights + bias
            if estimator.activation == "identity":
                pass
            elif estimator.activation == "relu":
                hidden = np.maximum(hidden, 0.0)
            elif estimator.activation == "tanh":
                hidden = np.tanh(hidden)
            elif estimator.activation == "logistic":
                hidden = 1.0 / (1.0 + np.exp(-np.clip(hidden, -50, 50)))
            else:
                raise ValueError(f"unsupported MLP activation: {estimator.activation}")
        return hidden

    return model, encode


def benchmark_manifest() -> list[dict[str, str]]:
    """Serializable model manifest used for experiment bookkeeping."""
    return [spec.__dict__.copy() for spec in BASELINES]
