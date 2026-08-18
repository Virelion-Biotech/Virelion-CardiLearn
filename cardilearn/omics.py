"""High-dimensional omics utilities for cardiac transcriptomic/proteomic matrices."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def supervised_reduction(task: str, *, k: int = 1000, n_components: int = 32) -> Pipeline:
    """Fit feature selection and PCA only on the training partition."""
    if k < 1 or n_components < 1:
        raise ValueError("k and n_components must be positive")
    if task == "classification":
        score = f_classif
    elif task == "regression":
        score = f_regression
    else:
        raise ValueError("task must be classification or regression")
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("select", SelectKBest(score_func=score, k=k)),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42)),
    ])


def variance_filter(matrix: np.ndarray, min_variance: float = 0.0) -> np.ndarray:
    """Return features above a variance threshold without learning from labels."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    variance = np.nanvar(matrix, axis=0)
    return matrix[:, variance > min_variance]
